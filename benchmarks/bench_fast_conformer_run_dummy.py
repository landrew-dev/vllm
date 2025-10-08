import argparse
import os
import torch

from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    get_pp_group,
)

from vllm.model_executor.models.fast_conformer import StreamingFastConformer
from types import SimpleNamespace
from vllm.transformers_utils.configs.fastconformer import FastConformerConfig

def make_vllm_config(
    feat_dim=80,
    d_model=512,
    n_layers=8,
    n_heads=8,
    k_conv=9,
    ff_mult=4,
    attn_win_left=64,
):
    fc_cfg = FastConformerConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        k_conv=k_conv,
        ff_mult=ff_mult,
        attn_win_left=attn_win_left,
        feat_dim=feat_dim,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=fc_cfg,
            head_dtype=torch.float32,
            pooler_config=None,
        )
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--feat-dim", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--k-conv", type=int, default=9)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--attn-win-left", type=int, default=64)
    parser.add_argument("--port", type=int, default=29500, help="TCP init port for single-proc dist")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for dummy input")
    args = parser.parse_args()

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    backend = "nccl"
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

    vllm_config = make_vllm_config(
        feat_dim=args.feat_dim,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        k_conv=args.k_conv,
        ff_mult=args.ff_mult,
        attn_win_left=args.attn_win_left,
    )

    model = StreamingFastConformer(
        vllm_config=vllm_config,
    ).to("cuda").eval()

    # construct a dummy input tensor and positions
    batch = args.batch
    # seq_len = args.seq_len
    seq_len = 8 # TODO: hardcode decode for now
    feat_dim = args.feat_dim

    # ensure seq_len is a multiple of 8
    if seq_len % 8 != 0:
        seq_len = ((seq_len + 7) // 8) * 8
        print(f"adjusted seq_len to next multiple of 8: {seq_len}")

    # simulate packed input: [N, F] where N = batch * seq_len
    N = batch * seq_len
    inputs_embeds = torch.randn(N, feat_dim, device="cuda", dtype=torch.float32)
    batched_pos = N // 8
    positions = torch.arange(0, batched_pos, device="cuda")

    with torch.no_grad():
        out = model(
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
    print("Forward pass successful.")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    main()
