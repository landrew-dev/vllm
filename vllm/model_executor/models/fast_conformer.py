from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.transformers_utils.configs.fastconformer import FastConformerConfig
from vllm.sequence import IntermediateTensors
from typing import Optional

def _split_heads(x, num_heads):
    # x: [B, T, D]
    # Returns: [B, T, H, d_k]
    B, T, D = x.shape
    d_k = D // num_heads
    return x.view(B, T, num_heads, d_k)

def _merge_heads(x):
    # x: [B, T, H, d_k]
    # Returns: [B, T, H*d_k]
    B, T, H, d_k = x.shape
    return x.contiguous().view(B, T, H * d_k)

def _unpack_packed(
    x: torch.Tensor,
    positions: torch.Tensor,
    before_subsample: bool = False,
) -> torch.Tensor:
    # Assume lens is always just 1 for now.
    # x: [N, D]
    # positions: [N//8] (before subsample) or [N] (after subsample)
    # If before_subsample, output shape: [B, 8, D]
    # If after_subsample, output shape: [B, 1, D]
    D = x.shape[1]
    if before_subsample:
        B = positions.shape[0]
        out = x.view(B, 8, D)
    else:
        out = x.unsqueeze(1)  # [B, 1, D]
    return out

def _pack_padded(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # x: [B, T_max, D], positions: [N]
    return x.view(x.shape[0] * x.shape[1], x.shape[2])

class ConvRingBuffer:
    """
    Ring buffer for 1D conv inputs (time-major).
    Stores last (k-1)*d frames of the conv's **input** activation (before stride).
    """
    def __init__(self, receptive: int, channels: int, device=None, dtype=None):
        self.receptive = receptive
        self.channels = channels
        self.buf = None
        self.device = device
        self.dtype = dtype
        self.batch_size = None

    def reset(self, batch_size: int):
        self.batch_size = batch_size
        if self.receptive == 0:
            self.buf = None
            return
        self.buf = torch.zeros(batch_size, self.receptive, self.channels,
                               device=self.device, dtype=self.dtype)

    def concat(self, x_new: torch.Tensor) -> torch.Tensor:
        # x_new: [B, T_new, C]
        if self.receptive == 0 or self.buf is None:
            return x_new
        # If batch size has changed, reset buffer
        if x_new.shape[0] != self.buf.shape[0]:
            self.reset(x_new.shape[0])
            return x_new
        return torch.cat([self.buf, x_new], dim=1)  # [B, T_cached + T_new, C]

    def update(self, x_input: torch.Tensor):
        if self.receptive == 0:
            return
        # If batch size has changed, reset buffer
        if self.buf is not None and x_input.shape[0] != self.buf.shape[0]:
            self.reset(x_input.shape[0])
            return
        T = x_input.shape[1]
        take = min(self.receptive, T)
        self.buf = x_input[:, T - take :, :].detach()

class StreamingSubsample8x(nn.Module):
    """
    Three stride-2 DW-separable conv stages (time dim) with ring-buffered inputs.
    in:  [N, F] (packed)
    out: [N', C_out] (packed, downsampled by 8)
    """
    def __init__(self, feat_dim: int, mid_ch: int, out_dim: int, k: int, device=None, dtype=None):
        super().__init__()
        pad = (k - 1) // 2

        self.in_proj = nn.Linear(feat_dim, mid_ch)

        self.dw1 = nn.Conv1d(mid_ch, mid_ch, k, stride=2, padding=pad, groups=mid_ch)
        self.pw1 = nn.Conv1d(mid_ch, mid_ch, 1)
        self.dw2 = nn.Conv1d(mid_ch, mid_ch, k, stride=2, padding=pad, groups=mid_ch)
        self.pw2 = nn.Conv1d(mid_ch, mid_ch, 1)
        self.dw3 = nn.Conv1d(mid_ch, mid_ch, k, stride=2, padding=pad, groups=mid_ch)
        self.pw3 = nn.Conv1d(mid_ch, mid_ch, 1)

        self.out_proj = nn.Linear(mid_ch, out_dim)

        rec = (k - 1)  # per stage, because dilation=1
        self.rb1 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)
        self.rb2 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)
        self.rb3 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

    def reset_stream(self, batch_size: int):
        self.rb1.reset(batch_size)
        self.rb2.reset(batch_size)
        self.rb3.reset(batch_size)

    def forward_stream(self, feats_chunk: torch.Tensor, positions: torch.Tensor):
        """
        feats_chunk: [N, F] (packed)
        positions: [N]
        returns: [N', D] (packed, downsampled by 8)
        """
        print(f"feats_chunk: {feats_chunk.shape}")
        x = _unpack_packed(feats_chunk, positions, before_subsample=True)
        print(f"x: {x.shape}")
        x = self.in_proj(x)
        x1_in = self.rb1.concat(x)
        y = x1_in.transpose(1, 2)
        y = F.silu(self.pw1(self.dw1(y)))
        y1 = y.transpose(1, 2)
        self.rb1.update(x1_in)

        x2_in = self.rb2.concat(y1)
        y = x2_in.transpose(1, 2)
        y = F.silu(self.pw2(self.dw2(y)))
        y2 = y.transpose(1, 2)
        self.rb2.update(x2_in)

        x3_in = self.rb3.concat(y2)
        y = x3_in.transpose(1, 2)
        y = F.silu(self.pw3(self.dw3(y)))
        y3 = y.transpose(1, 2)
        self.rb3.update(x3_in)

        out = self.out_proj(y3)
        out_packed = _pack_padded(out, positions)
        return out_packed

class AttnKVCache:
    """
    Keeps last W_l timesteps of K,V (after subsampling & projections) per layer.
    """
    def __init__(self, num_heads: int, head_dim: int, W_l: int, device=None, dtype=None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_l = W_l
        self.K = None
        self.V = None
        self.device = device
        self.dtype = dtype
        self.batch_size = None

    def reset(self, batch_size: int):
        self.K = None
        self.V = None
        self.batch_size = batch_size

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        K_new,V_new: [B,H,T_new,d_k]
        """
        B = K_new.shape[0]
        # If batch size has changed, reset cache
        if self.K is not None and self.K.shape[0] != B:
            self.K = None
            self.V = None
            self.batch_size = B
        if self.K is None:
            self.K = K_new
            self.V = V_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)
        if self.K.shape[2] > self.W_l:
            self.K = self.K[:, :, -self.W_l :, :].detach()
            self.V = self.V[:, :, -self.W_l :, :].detach()

    def get(self):
        return self.K, self.V  # possibly None on first chunk

class StreamingFCBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_conv: int, ff_mult: int, attn_win_left: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wl = attn_win_left

        self.ln_ff1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        pad = (k_conv - 1) // 2
        self.ln_conv = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)     # GLU-ish option (we’ll just SiLU here)
        self.dw = nn.Conv1d(2 * d_model, 2 * d_model, k_conv, padding=pad, groups=2 * d_model)
        self.bn = nn.BatchNorm1d(2 * d_model)
        self.pw2 = nn.Conv1d(2 * d_model, d_model, 1)

        self.conv_rb = ConvRingBuffer(receptive=(k_conv - 1), channels=d_model)

        self.kv = AttnKVCache(n_heads, self.d_k, self.Wl)

    def reset_stream(self, batch_size: int):
        self.kv.reset(batch_size)
        self.conv_rb.reset(batch_size)

    def forward_stream(self, x_new: torch.Tensor, positions: torch.Tensor):
        """
        x_new: [N, D] (packed)
        positions: [N]
        Returns: [N, D] (packed)
        """
        x = _unpack_packed(x_new, positions)
        B, T_new, D = x.shape

        y = self.ln_ff1(x)
        y = self.fc2(F.silu(self.fc1(y)))
        x = x + y * 0.5

        y = self.ln_attn(x)
        # x is already [B, T, D] here
        Q = _split_heads(self.q_proj(y), self.n_heads)  # [B, T, H, d_k]
        K_new = _split_heads(self.k_proj(y), self.n_heads)  # [B, T, H, d_k]
        V_new = _split_heads(self.v_proj(y), self.n_heads)  # [B, T, H, d_k]

        # For attention, flatten batch and time for Q, K, V
        Q_flat = Q.reshape(B * T_new, self.n_heads, self.d_k)
        K_new_flat = K_new.reshape(B * T_new, self.n_heads, self.d_k)
        V_new_flat = V_new.reshape(B * T_new, self.n_heads, self.d_k)

        K_ctx, V_ctx = self.kv.get()
        if K_ctx is None:
            K_all, V_all = K_new_flat, V_new_flat
        else:
            # K_ctx, V_ctx: [B, H, T_ctx, d_k]
            # Reshape to [B* T_ctx, H, d_k] for concat
            B_ctx, H, T_ctx, d_k = K_ctx.shape
            # If batch size has changed, ignore old context
            if B_ctx != B:
                K_all, V_all = K_new_flat, V_new_flat
            else:
                K_ctx_flat = K_ctx.permute(0, 2, 1, 3).reshape(B * T_ctx, H, d_k)
                V_ctx_flat = V_ctx.permute(0, 2, 1, 3).reshape(B * T_ctx, H, d_k)
                K_all = torch.cat([K_ctx_flat, K_new_flat], dim=0)
                V_all = torch.cat([V_ctx_flat, V_new_flat], dim=0)

        attn = F.scaled_dot_product_attention(Q_flat, K_all, V_all, is_causal=False)
        # attn: [B * T_new, H, d_k]
        attn = attn.view(B, T_new, self.n_heads, self.d_k)
        attn = _merge_heads(attn)  # [B, T_new, D]
        x = x + self.o_proj(attn)

        # Cache K, V for future context: [B, H, T_new, d_k]
        K_new_4d = K_new.permute(0, 2, 1, 3)  # [B, H, T_new, d_k]
        V_new_4d = V_new.permute(0, 2, 1, 3)  # [B, H, T_new, d_k]
        self.kv.append(K_new_4d, V_new_4d)

        y_in = self.conv_rb.concat(self.ln_conv(x))
        y = y_in.transpose(1, 2)                      # [B,D,T]
        y = self.pw1(y)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y).transpose(1, 2)
        y = y[:, -T_new:, :]
        x = x + y
        self.conv_rb.update(y_in)

        y = self.fc2(F.silu(self.fc1(x)))
        x = x + y * 0.5

        out_packed = _pack_padded(x, positions)
        return out_packed

class StreamingFastConformer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: FastConformerConfig = vllm_config.model_config.hf_config
        self.sub = StreamingSubsample8x(config.feat_dim, mid_ch=256, out_dim=config.d_model, k=config.k_conv)
        self.blocks = nn.ModuleList([
            StreamingFCBlock(
                d_model=config.d_model, n_heads=config.n_heads, k_conv=config.k_conv, ff_mult=config.ff_mult, attn_win_left=config.attn_win_left)
            for _ in range(config.n_layers)
        ])
        self.ln_out = nn.LayerNorm(config.d_model)

    def reset_stream(self, batch_size: int):
        self.sub.reset_stream(batch_size)
        for b in self.blocks:
            b.reset_stream(batch_size)

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None, # always None
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # inputs_embeds: [N, D * 8]
        # positions: [N]
        # # assume pure decode for simplicity, for now
        # x = self.sub.forward_stream(inputs_embeds, positions)
        # for b in self.blocks:
        #     x = b.forward_stream(x, positions)
        # return self.ln_out(x)

        D = inputs_embeds.shape[1] // 8
        return inputs_embeds[:, :D]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # x = self.forward_chunk(hidden_states)
        # for b in self.blocks:
        #     x = b.forward_stream(x)
        # return self.ln_out(x)

        D = hidden_states.shape[1] // 8
        return hidden_states[:, :D]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        pass
        # if not isinstance(weights, dict):
        #     weights = dict(weights)

        # state_dict = {}
        # for k, v in weights.items():
        #     if k.startswith("module."):
        #         state_dict[k[7:]] = v
        #     else:
        #         state_dict[k] = v

        # missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # if missing:
        #     print(f"Warning: Missing keys when loading weights: {missing}")
        # if unexpected:
        #     print(f"Warning: Unexpected keys when loading weights: {unexpected}")
