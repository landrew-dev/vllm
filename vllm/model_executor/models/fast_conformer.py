import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x, num_heads):
    # x: [B, T, D] -> [B, H, T, d_k]
    B, T, D = x.shape
    d_k = D // num_heads
    x = x.view(B, T, num_heads, d_k).transpose(1, 2).contiguous()
    return x

def _merge_heads(x):
    # x: [B, H, T, d_k] -> [B, T, H*d_k]
    B, H, T, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d_k)


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

    def reset(self, batch_size: int):
        if self.receptive == 0:
            self.buf = None
            return
        self.buf = torch.zeros(batch_size, self.receptive, self.channels,
                               device=self.device, dtype=self.dtype)

    def concat(self, x_new: torch.Tensor) -> torch.Tensor:
        # x_new: [B, T_new, C]
        if self.receptive == 0 or self.buf is None:
            return x_new
        return torch.cat([self.buf, x_new], dim=1)  # [B, T_cached + T_new, C]

    def update(self, x_input: torch.Tensor):
        # Keep the last receptive frames of the **input** that produced the latest output.
        if self.receptive == 0:
            return
        T = x_input.shape[1]
        take = min(self.receptive, T)
        self.buf = x_input[:, T - take :, :].detach()


class StreamingSubsample8x(nn.Module):
    """
    Three stride-2 DW-separable conv stages (time dim) with ring-buffered inputs.
    in:  [B, T_in, F]
    out: [B, floor(T_in/8), C_out]
    """
    def __init__(self, feat_dim=80, mid_ch=256, out_dim=512, k=9, device=None, dtype=None):
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

    def forward_stream(self, feats_chunk: torch.Tensor):
        """
        feats_chunk: [B, T_in, F] (new frames only)
        returns: subsampled frames produced by this chunk: [B, T_out, D]
        """
        B, T, Fdim = feats_chunk.shape
        x = self.in_proj(feats_chunk)            # [B,T,C]
        x1_in = self.rb1.concat(x)               # [B,T1_in,C]
        y = x1_in.transpose(1, 2)                # [B,C,T1_in]
        y = F.silu(self.pw1(self.dw1(y)))        # [B,C,T1_out]
        y1 = y.transpose(1, 2)                   # [B,T1_out,C]
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

        out = self.out_proj(y3)                  # [B,T_out,D]
        return out


class AttnKVCache:
    """
    Keeps last W_l timesteps of K,V (after subsampling & projections) per layer.
    """
    def __init__(self, num_heads: int, head_dim: int, W_l: int, device=None, dtype=None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_l = W_l
        self.K = None  # [B, H, T_ctx, d_k]
        self.V = None
        self.device = device
        self.dtype = dtype

    def reset(self, batch_size: int):
        self.K = None
        self.V = None

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        K_new,V_new: [B,H,T_new,d_k]
        """
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
    def __init__(self, d_model: int, n_heads: int, k_conv=9, ff_mult=4, attn_win_left=70, pdrop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wl = attn_win_left

        self.ln_ff1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)
        self.drop = nn.Dropout(pdrop)

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

    def forward_stream(self, x_new: torch.Tensor):
        """
        x_new: [B, T_new, D] (new time steps only for this block)
        Returns: y_new [B, T_new, D]
        """
        B, T_new, D = x_new.shape

        y = self.ln_ff1(x_new)
        y = self.fc2(F.silu(self.fc1(y)))
        x = x_new + self.drop(y) * 0.5

        y = self.ln_attn(x)
        Q = _split_heads(self.q_proj(y), self.n_heads)          # [B,H,T_new,d_k]
        K_new = _split_heads(self.k_proj(y), self.n_heads)
        V_new = _split_heads(self.v_proj(y), self.n_heads)

        K_ctx, V_ctx = self.kv.get()
        if K_ctx is None:
            K_all, V_all = K_new, V_new
        else:
            K_all = torch.cat([K_ctx, K_new], dim=2)           # [B,H,T_ctx+T_new,d_k]
            V_all = torch.cat([V_ctx, V_new], dim=2)

        attn = F.scaled_dot_product_attention(Q, K_all, V_all, is_causal=False)  # non-causal but limited by window
        attn = _merge_heads(attn)                                                # [B,T_new,D]
        x = x + self.drop(self.o_proj(attn))

        self.kv.append(K_new, V_new)

        y_in = self.conv_rb.concat(self.ln_conv(x))   # [B, T_ctx + T_new, D]
        y = y_in.transpose(1, 2)                      # [B,D,T]
        y = self.pw1(y)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y).transpose(1, 2)               # [B,T,D]
        y = y[:, -T_new:, :]
        x = x + self.drop(y)
        self.conv_rb.update(y_in)

        y = self.fc2(F.silu(self.fc1(x)))
        x = x + self.drop(y) * 0.5

        return x  # [B, T_new, D]

class StreamingFastConformer(nn.Module):
    def __init__(self, feat_dim=80, d_model=512, n_layers=17, n_heads=8, k_conv=9, attn_win_left=70):
        super().__init__()
        self.sub = StreamingSubsample8x(feat_dim, mid_ch=256, out_dim=d_model, k=k_conv)
        self.blocks = nn.ModuleList([
            StreamingFCBlock(d_model, n_heads, k_conv=k_conv, attn_win_left=attn_win_left)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def reset_stream(self, batch_size: int):
        self.sub.reset_stream(batch_size)
        for b in self.blocks:
            b.reset_stream(batch_size)

    @torch.inference_mode()
    def forward_chunk(self, feats_chunk: torch.Tensor):
        """
        feats_chunk: [B, T_in, F] new frames only
        returns: enc_out_chunk: [B, T_out, D] for this step
        """
        x = self.sub.forward_stream(feats_chunk)   # [B, T/8, D]
        for b in self.blocks:
            x = b.forward_stream(x)                # keep only new steps all the way
        return self.ln_out(x)                      # [B, T/8, D]
