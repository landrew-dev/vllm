import argparse
import os
import time
import torch
from types import SimpleNamespace


from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    get_pp_group,
)

from vllm.model_executor.models.fast_conformer import FastConformerEncoder


def make_vllm_config(
    feat_dim=80,
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    layer_norm_epsilon=1e-5,
    conformer_kernel=9,
    intermediate_size_mult=4,
    conv_expansion=2,
    vocab_size=1024,
):
    class Cfg: ...
    cfg = Cfg()
    cfg.feat_dim = feat_dim
    cfg.hidden_size = hidden_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.layer_norm_epsilon = layer_norm_epsilon
    cfg.conformer_kernel = conformer_kernel
    cfg.intermediate_size_mult = intermediate_size_mult
    cfg.conv_expansion = conv_expansion
    cfg.vocab_size = vocab_size

    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=cfg,
            head_dtype=torch.float32,
            pooler_config=None,
        ),
        cache_config=None,
        quant_config=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3000, help="input T (e.g., 30s @100fps)")
    parser.add_argument("--feat-dim", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--port", type=int, default=29500, help="TCP init port for single-proc dist")
    args = parser.parse_args()

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    backend = "nccl" if args.device == "cuda" else "gloo"
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        torch.cuda.set_device(0)

    init_distributed_environment(
        backend=backend,
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{args.port}",
    )

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    assert get_pp_group().is_first_rank and get_pp_group().is_last_rank

    try:
        vcfg = make_vllm_config(
            feat_dim=args.feat_dim,
            hidden_size=args.hidden,
            num_layers=args.layers,
            num_heads=args.heads,
        )
        model = FastConformerEncoder(vllm_config=vcfg).to(args.device).eval()

        B, T, F = args.batch, args.frames, args.feat_dim
        x = torch.randn(B, T, F, device=args.device, dtype=torch.float32)

        with torch.inference_mode():
            y = model(x)
        if args.device == "cuda":
            torch.cuda.synchronize()
        print("Output shape:", tuple(y.shape))  # [B, T/8, D]

        iters = args.iters
        start = time.time()
        with torch.inference_mode():
            for _ in range(iters):
                y = model(x)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        avg = elapsed / iters
        print(f"Avg latency per batch: {avg * 1000:.2f} ms")
        print(f"Throughput: {B / avg:.2f} samples/s")

    finally:
        destroy_model_parallel()


if __name__ == "__main__":
    main()
