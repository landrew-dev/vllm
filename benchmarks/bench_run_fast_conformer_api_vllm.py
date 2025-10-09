import argparse
import asyncio
from typing import Dict, Optional
import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

async def demo_streaming_fast_conformer(
    batch_size: int = 2, feat_dim: int = 80, num_pushes: int = 5, frames_per_push: int = 64
):
    engine_args = AsyncEngineArgs(model="dummy_model_dir", enforce_eager=True)
    engine = AsyncLLM.from_engine_args(engine_args)

    id_ready = asyncio.Event()
    request_id_holder: Dict[str, Optional[str]] = {"id": None}

    async def producer():
        print(f"producer: starting")
        total_in = 0
        req_id: Optional[str] = None
        for i in range(num_pushes):
            frames = torch.randn(batch_size, frames_per_push, feat_dim)
            total_in += frames_per_push
            if req_id is None:
                req_id = await engine.add_audio_frames_new_request(frames)
                request_id_holder["id"] = req_id
                id_ready.set()
            else:
                await engine.add_audio_frames(req_id, frames)
            await asyncio.sleep(0.005)
        assert req_id is not None
        await engine.finalize_audio_request(req_id)
        return total_in

    async def consumer():
        print(f"consumer: starting")
        await id_ready.wait()
        req_id = request_id_holder["id"]
        assert req_id is not None
        total_out = 0
        async for out in engine.stream_audio(req_id):
            total_out += out.shape[1]
        return total_out

    prod_task = asyncio.create_task(producer())
    total_frames_out = await consumer()
    total_frames_in = await prod_task
    print(f"streaming demo complete: total input frames={total_frames_in}, total output frames={total_frames_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark fastconformer intertoken latency with vLLM.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size to benchmark.")
    parser.add_argument("--feat_dim", type=int, default=80,
                        help="feature dimension.")
    args = parser.parse_args()

    asyncio.run(demo_streaming_fast_conformer(batch_size=args.batch_size, feat_dim=args.feat_dim))
