import time
import argparse
from vllm import LLM, SamplingParams
import torch

def _generate_fastconformer_inputs(batch_size, frames, feat_dim):
    return torch.randn(batch_size, frames, feat_dim)

def benchmark_fast_conformer(batch_sizes=[1, 8, 32], num_tokens=32, feat_dim=80, num_warmup=4, num_trials=16):
    print("loading model...")
    fastconformer = LLM(model="dummy_model_dir", enforce_eager=True)
    print("model loaded")

    for batch_size in batch_sizes:
        B, T, F = batch_size, num_tokens, feat_dim
        print(f"running fastconformer with batch size: {batch_size}, num tokens: {num_tokens}, feat dim: {feat_dim}")
        inputs = _generate_fastconformer_inputs(B, T, F)
        sampling_params = SamplingParams(max_tokens=num_tokens)

        for _ in range(num_warmup):
            _ = fastconformer.generate(inputs, sampling_params)

        latencies = []
        for _ in range(num_trials):
            t = time.time()
            r = fastconformer.generate(inputs, sampling_params)
            latencies.append(time.time() - t)

        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[int(len(latencies)*0.5)]
        p90 = sorted(latencies)[int(len(latencies)*0.9)]
        print(f"batch size {batch_size}:")
        print(f"  avg inter-token latency: {avg_latency*1000:.2f} ms")
        print(f"  p50 inter-token latency: {p50*1000:.2f} ms")
        print(f"  p90 inter-token latency: {p90*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark fastconformer intertoken latency with vLLM.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32],
                        help="batch sizes to benchmark.")
    parser.add_argument("--num_tokens", type=int, default=32,
                        help="number of tokens to generate per prompt.")
    parser.add_argument("--num_warmup", type=int, default=4,
                        help="number of warmup runs.")
    parser.add_argument("--feat_dim", type=int, default=80,
                        help="feature dimension.")
    parser.add_argument("--num_trials", type=int, default=16,
                        help="number of timed trials.")
    args = parser.parse_args()

    benchmark_fast_conformer(
        batch_sizes=args.batch_sizes,
        num_tokens=args.num_tokens,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials,
        feat_dim=args.feat_dim
    )
