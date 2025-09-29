# SPDX-License-Identifier: Apache-2.0
# Streaming Fast-Conformer inter-chunk latency benchmark (PyTorch, self-contained)

import argparse
import math
import os
import time
import csv
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt



def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1



def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # [B, T, D] -> [B, H, T, d_k]
    B, T, D = x.shape
    d_k = D // num_heads
    return x.view(B, T, num_heads, d_k).transpose(1, 2).contiguous()

def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # [B, H, T, d_k] -> [B, T, D]
    B, H, T, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

class ConvRingBuffer:
    """
    Ring buffer for 1D conv inputs (time-major).
    Stores last (k-1)*d frames of a conv's **input**.
    Lazy-inits on the first concat() to the incoming tensor's device/dtype.
    """
    def __init__(self, receptive: int, channels: int, device=None, dtype=None):
        self.receptive = int(receptive)
        self.channels = int(channels)
        self.device = device
        self.dtype = dtype
        self.buf = None

    def reset(self, batch_size: int):
        self.buf = None

    def _lazy_init(self, batch_size: int, device, dtype):
        if self.receptive <= 0:
            self.buf = None
            return
        self.buf = torch.zeros(batch_size, self.receptive, self.channels,
                               device=device, dtype=dtype)

    def concat(self, x_new: torch.Tensor) -> torch.Tensor:
        if self.receptive <= 0:
            return x_new
        if self.buf is None:
            self._lazy_init(x_new.shape[0], x_new.device, x_new.dtype)
        return torch.cat([self.buf, x_new], dim=1)  # [B, T_cached+T_new, C]

    def update(self, x_input: torch.Tensor):
        if self.receptive <= 0:
            return
        T = x_input.shape[1]
        take = min(self.receptive, T)
        self.buf = x_input[:, T - take:, :].detach()


class AttnKVCache:
    """
    Keeps last W_l timesteps of K,V for windowed self-attention (per layer).
    K/V format: [B, H, T_ctx, d_k]
    """
    def __init__(self, num_heads: int, head_dim: int, window_left: int, device=None, dtype=None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.Wl = int(window_left)
        self.device = device
        self.dtype = dtype
        self.K = None
        self.V = None

    def reset(self, batch_size: int):
        self.K, self.V = None, None

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        if self.K is None:
            self.K, self.V = K_new, V_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)
        if self.K.shape[2] > self.Wl:
            self.K = self.K[:, :, -self.Wl:, :].detach()
            self.V = self.V[:, :, -self.Wl:, :].detach()

    def get(self):
        return self.K, self.V


class StreamingSubsample8x(nn.Module):
    """
    Three stride-2 DW-separable conv stages with ring buffers.
    in:  [B, T, F]
    out: [B, floor(T/8), D_out]
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

        rec = (k - 1)  # dilation=1
        self.rb1 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)
        self.rb2 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)
        self.rb3 = ConvRingBuffer(rec, mid_ch, device=device, dtype=dtype)

    def reset_stream(self, batch_size: int):
        self.rb1.reset(batch_size)
        self.rb2.reset(batch_size)
        self.rb3.reset(batch_size)

    @torch.inference_mode()
    def forward_stream(self, feats_chunk: torch.Tensor):
        x = self.in_proj(feats_chunk)       # [B,T,C]
        x1_in = self.rb1.concat(x)          # [B,T1_in,C]
        y = x1_in.transpose(1, 2)           # [B,C,T1_in]
        y = F.silu(self.pw1(self.dw1(y)))   # [B,C,T1_out]
        y1 = y.transpose(1, 2)              # [B,T1_out,C]
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

        out = self.out_proj(y3)             # [B,T_out,D]
        return out


class StreamingFCBlock(nn.Module):
    """
    Conformer block in streaming mode:
      FF/2  -> local self-attn (left-window KV cache) -> conv -> FF/2
    """
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
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw  = nn.Conv1d(2 * d_model, 2 * d_model, k_conv, padding=pad, groups=2 * d_model)
        self.bn  = nn.BatchNorm1d(2 * d_model)
        self.pw2 = nn.Conv1d(2 * d_model, d_model, 1)

        self.conv_rb = ConvRingBuffer(receptive=(k_conv - 1), channels=d_model)
        self.kv = AttnKVCache(n_heads, self.d_k, self.Wl)

    def reset_stream(self, batch_size: int):
        self.kv.reset(batch_size)
        self.conv_rb.reset(batch_size)

    @torch.inference_mode()
    def forward_stream(self, x_new: torch.Tensor) -> torch.Tensor:
        """
        x_new: [B, T_new, D] -> returns y_new: [B, T_new, D]
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
            K_all = torch.cat([K_ctx, K_new], dim=2)            # [B,H,T_ctx+T_new,d_k]
            V_all = torch.cat([V_ctx, V_new], dim=2)

        attn = F.scaled_dot_product_attention(Q, K_all, V_all, is_causal=False)
        attn = _merge_heads(attn)                                # [B,T_new,D]
        x = x + self.drop(self.o_proj(attn))

        self.kv.append(K_new, V_new)

        y_in = self.conv_rb.concat(self.ln_conv(x))              # [B, T_ctx+T_new, D]
        y = y_in.transpose(1, 2)                                 # [B,D,T]
        y = self.pw1(y)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y).transpose(1, 2)                          # [B,T,D]
        y = y[:, -T_new:, :]                                     # keep only new steps
        x = x + self.drop(y)
        self.conv_rb.update(y_in)

        y = self.fc2(F.silu(self.fc1(x)))
        x = x + self.drop(y) * 0.5

        return x


class StreamingFastConformer(nn.Module):
    def __init__(self, feat_dim=80, d_model=512, n_layers=17, n_heads=8, k_conv=9, attn_win_left=70, ff_mult=4):
        super().__init__()
        self.sub = StreamingSubsample8x(feat_dim, mid_ch=256, out_dim=d_model, k=k_conv)
        self.blocks = nn.ModuleList([
            StreamingFCBlock(d_model, n_heads, k_conv=k_conv, ff_mult=ff_mult, attn_win_left=attn_win_left)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def reset_stream(self, batch_size: int):
        self.sub.reset_stream(batch_size)
        for b in self.blocks:
            b.reset_stream(batch_size)

    @torch.inference_mode()
    def forward_chunk(self, feats_chunk: torch.Tensor) -> torch.Tensor:
        """
        feats_chunk: [B, T_in, F]
        returns enc_out_chunk: [B, floor(T_in/8), D]
        """
        x = self.sub.forward_stream(feats_chunk)   # [B, T/8, D]
        for b in self.blocks:
            x = b.forward_stream(x)
        return self.ln_out(x)



@torch.inference_mode()
def bench_inter_chunk(
    model: StreamingFastConformer,
    device: str,
    batch_sizes=(1, 8, 32),
    steps=100,
    chunk_frames=40,
    feat_dim=80,
    warmup_steps=10,
    amp="none",
):
    results = []
    for bs in batch_sizes:
        model.reset_stream(bs)
        feats = torch.randn(bs, chunk_frames, feat_dim, device=device)

        with maybe_autocast(device, amp):
            for _ in range(warmup_steps):
                _ = model.forward_chunk(feats)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(steps):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            with maybe_autocast(device, amp):
                _ = model.forward_chunk(feats)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000.0
            times_ms.append(dt_ms)

        out_frames = chunk_frames // 8

        avg = sum(times_ms) / len(times_ms)
        avg_itl = (sum(times_ms) / len(times_ms)) / out_frames
        p50_itl = percentile(times_ms, 0.50) / out_frames
        p90_itl = percentile(times_ms, 0.90) / out_frames
        p99_itl = percentile(times_ms, 0.99) / out_frames

        fps_per_stream = (out_frames) / (avg / 1000.0)
        audio_ms = chunk_frames * 10.0
        rtf = (avg) / audio_ms
        results.append(
            dict(
                batch_size=bs,
                steps=steps,
                chunk_frames=chunk_frames,
                avg_ms=avg_itl,
                p50_ms=p50_itl,
                p90_ms=p90_itl,
                p99_ms=p99_itl,
                fps_per_stream=fps_per_stream,
                rtf=rtf,
            )
        )
    return results

def maybe_autocast(device: str, amp: str):
    if device.startswith("cuda") and amp in ("bf16", "fp16"):
        dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(enabled=False)

def maybe_compile(model: nn.Module, use_compile: bool):
    if use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--feat-dim", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=17)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--k-conv", type=int, default=9)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--attn-win-left", type=int, default=70)
    parser.add_argument("--chunk-frames", type=int, default=40, help="mel frames per chunk (10ms hop ⇒ 400ms default)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--amp", type=str, default="none", choices=["none", "bf16", "fp16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    # Layer sweep controls
    parser.add_argument("--layers-min", type=int, default=1, help="min layers (inclusive) to sweep")
    parser.add_argument("--layers-max", type=int, default=20, help="max layers (inclusive) to sweep")
    parser.add_argument("--layers-step", type=int, default=1, help="step when sweeping layers")
    # Outputs
    parser.add_argument("--save-plot", type=str, default="itl_vs_layers_no_conv.png")
    parser.add_argument("--save-csv", type=str, default="itl_vs_layers_no_conv.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build layer sweep
    layers_list = list(range(args.layers_min, args.layers_max + 1, max(1, args.layers_step)))
    batch_sizes = list(args.batch_sizes)

    print("\n=== Streaming Fast-Conformer Inter-Chunk Latency (ICL) — Layers Sweep ===")
    print(
        f"Model cfg: d_model={args.d_model}, heads={args.heads}, k_conv={args.k_conv}, Wl={args.attn_win_left}, ff_mult={args.ff_mult}"
    )
    print(
        f"Chunk: {args.chunk_frames} mel frames @10ms hop (~{args.chunk_frames*10} ms audio), steps={args.steps}, device={args.device}, amp={args.amp}"
    )
    print(f"Sweeping layers: {layers_list[0]}..{layers_list[-1]} (step {max(1, args.layers_step)}) | batch sizes {batch_sizes}")

    sweep_results = {int(bs): {"layers": [], "avg_itl_ms": []} for bs in batch_sizes}

    for L in layers_list:
        model = StreamingFastConformer(
            feat_dim=args.feat_dim,
            d_model=args.d_model,
            n_layers=L,
            n_heads=args.heads,
            k_conv=args.k_conv,
            attn_win_left=args.attn_win_left,
            ff_mult=args.ff_mult,
        ).to(args.device).eval()

        model = maybe_compile(model, args.compile)

        results_L = bench_inter_chunk(
            model=model,
            device=args.device,
            batch_sizes=tuple(batch_sizes),
            steps=args.steps,
            chunk_frames=args.chunk_frames,
            feat_dim=args.feat_dim,
            warmup_steps=args.warmup_steps,
            amp=args.amp,
        )

        row_bits = []
        for r in results_L:
            bs = int(r["batch_size"])
            sweep_results[bs]["layers"].append(L)
            sweep_results[bs]["avg_itl_ms"].append(float(r["avg_ms"]))
            row_bits.append(f"BS={bs}: {r['avg_ms']:7.3f} ms/it")
        print(f"layers={L:>2} | " + "  ".join(row_bits))

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    try:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            bs_sorted = sorted(sweep_results.keys())
            header = ["layers"] + [f"avg_itl_ms_bs{bs}" for bs in bs_sorted]
            writer.writerow(header)
            base_layers = sweep_results[bs_sorted[0]]["layers"]
            for i, L in enumerate(base_layers):
                row = [L] + [f"{sweep_results[bs]['avg_itl_ms'][i]:.6f}" for bs in bs_sorted]
                writer.writerow(row)
        print(f"Saved CSV to {args.save_csv}")
    except Exception as e:
        print(f"Warning: failed to save CSV at {args.save_csv}: {e}")

    plt.figure(figsize=(8, 5))
    bs_sorted = sorted(sweep_results.keys())
    for bs in bs_sorted:
        plt.plot(
            sweep_results[bs]["layers"],
            sweep_results[bs]["avg_itl_ms"],
            marker="o",
            label=f"BS={bs}",
        )
    plt.xlabel("# layers")
    plt.ylabel("Avg inter-token latency (ms)")
    plt.title(
        f"Fast-Conformer ICL vs Layers | d_model={args.d_model}, heads={args.heads}, chunk={args.chunk_frames}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Saved plot to {args.save_plot}")


if __name__ == "__main__":
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    main()
