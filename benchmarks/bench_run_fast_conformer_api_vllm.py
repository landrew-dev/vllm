import argparse
import asyncio
import torch
import os
import json

import shutil
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.config import LogprobsMode

MODEL_DIR = "fastconformer_dir"

def seed_fastconformer_dir(
    feat_dim: int, d_model: int, n_layers: int, n_heads: int, k_conv: int, ff_mult: int, attn_win_left: int,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    weights_path = os.path.join(MODEL_DIR, "pytorch_model.pt")
    torch.save({}, weights_path)

    config = {
        "architectures": ["StreamingFastConformer"],
        "feat_dim": feat_dim,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "k_conv": k_conv,
        "ff_mult": ff_mult,
        "attn_win_left": attn_win_left,
        "model_type": "fastconformer",
        "torch_dtype": "float32"
    }
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Dummy model directory created at {MODEL_DIR}")
    print(f"  Weights: {weights_path}")
    print(f"  Config:  {config_path}")

def rm_fastconformer_dir():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

async def demo_streaming_fast_conformer(
    batch_size: int = 2, feat_dim: int = 80, num_pushes: int = 5, frames_per_push: int = 64
):
    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    req_id = "test"

    q_fut = asyncio.Future()

    async def producer():
        print(f"producer: starting")
        total_in = 0
        q = None
        for i in range(num_pushes):
            frames = torch.randn(batch_size, frames_per_push, feat_dim)
            total_in += frames_per_push
            if i == 0:
                q = llm.generate(frames, request_id=req_id, is_streaming=True)
                q_fut.set_result(q)
            else:
                await llm.append(req_id, frames)
            await asyncio.sleep(0.005)
        await llm.finish(req_id)
        return total_in

    async def consumer():
        print(f"consumer: starting")
        q = await q_fut
        total_out = 0
        async for out in q:
            total_out += out.shape[1]
        return total_out

    prod_task = asyncio.create_task(producer())
    cons_task = asyncio.create_task(consumer())
    total_frames_in, total_frames_out = await asyncio.gather(prod_task, cons_task)
    print(f"streaming demo complete: total input frames={total_frames_in}, total output frames={total_frames_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark fastconformer intertoken latency with vLLM.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size to benchmark.")
    parser.add_argument("--feat_dim", type=int, default=80,
                        help="feature dimension.")
    args = parser.parse_args()

    seed_fastconformer_dir(
        feat_dim=80,
        d_model=512,
        n_layers=8,
        n_heads=8,
        k_conv=9,
        ff_mult=4,
        attn_win_left=64,
    )

    asyncio.run(demo_streaming_fast_conformer(batch_size=args.batch_size, feat_dim=args.feat_dim))

    rm_fastconformer_dir()
