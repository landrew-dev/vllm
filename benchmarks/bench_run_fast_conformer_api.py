import argparse
import asyncio
from typing import AsyncGenerator, Dict, Optional
import torch

class MockFastConformerEngine:
    def __init__(self):
        self._requests: Dict[str, Dict[str, object]] = {}
        self._next_id: int = 1

    def _new_request_id(self) -> str:
        rid = f"req-{self._next_id}"
        self._next_id += 1
        return rid

    async def start_request(self) -> str:
        request_id = self._new_request_id()
        state = {
            "in_q": asyncio.Queue(),
            "out_q": asyncio.Queue(),
            "worker": None,
        }
        self._requests[request_id] = state
        state["worker"] = asyncio.create_task(self._worker(request_id))
        return request_id

    async def add_frames_new_request(self, frames: torch.Tensor) -> str:
        request_id = await self.start_request()
        await self.add_frames(request_id, frames)
        return request_id

    async def add_frames(self, request_id: str, frames: torch.Tensor) -> None:
        await self._requests[request_id]["in_q"].put(frames)

    async def finalize_request(self, request_id: str) -> None:
        await self._requests[request_id]["in_q"].put(None)

    async def stream(self, request_id: str) -> AsyncGenerator[torch.Tensor, None]:
        out_q: asyncio.Queue = self._requests[request_id]["out_q"]
        while True:
            item = await out_q.get()
            if item is None:
                break
            yield item

    async def _worker(self, request_id: str) -> None:
        state = self._requests[request_id]
        in_q: asyncio.Queue = state["in_q"]
        out_q: asyncio.Queue = state["out_q"]
        try:
            while True:
                item = await in_q.get()
                if item is None:
                    break
                await asyncio.sleep(0.01)
                x = item
                assert x.dim() == 3, "frames must be (B, T, H)"
                B, T, H = x.shape
                T8 = T // 8
                if T8 > 0:
                    x_ds = x[:, : T8 * 8, :].reshape(B, T8, 8, H).mean(dim=2)
                    await out_q.put(x_ds)
        finally:
            await out_q.put(None)

async def demo_streaming_fast_conformer(
    batch_size: int = 2, feat_dim: int = 80, num_pushes: int = 5, frames_per_push: int = 64
):
    engine = MockFastConformerEngine()

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
                req_id = await engine.add_frames_new_request(frames)
                request_id_holder["id"] = req_id
                id_ready.set()
            else:
                await engine.add_frames(req_id, frames)
            await asyncio.sleep(0.005)
        assert req_id is not None
        await engine.finalize_request(req_id)
        return total_in

    async def consumer():
        print(f"consumer: starting")
        await id_ready.wait()
        req_id = request_id_holder["id"]
        assert req_id is not None
        total_out = 0
        async for out in engine.stream(req_id):
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
