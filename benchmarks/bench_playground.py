import time
import argparse
from vllm import LLM, SamplingParams

def _get_avg_itl(outs):
    """
    The average ITL for a single output is:
      (last_token_time - first_token_time) / (num_generated_tokens - 1)
    (if num_generated_tokens > 1; otherwise, ITL is 0)
    """
    itls = []
    for r in outs:
        m = r.metrics
        num_gen_tokens = len(r.outputs[0].token_ids)
        if num_gen_tokens > 1:
            itl = (m.last_token_time - m.first_token_time) / (num_gen_tokens - 1)
            itls.append(itl)
        else:
            itls.append(0.0)
    if itls:
        return sum(itls) / len(itls)
    else:
        return 0.0


def benchmark_gpt2(batch_sizes=[1, 8, 32], num_tokens=32, num_warmup=4, num_trials=16):
    dummy_prompts = [
        "Once upon a time,",
        "In a world where AI",
        "The quick brown fox",
        "To be or not to be,",
        "As the sun set over",
        "In the beginning,",
        "The meaning of life is",
        "Artificial intelligence will",
        "The future of technology",
        "It was a dark and stormy night,"
    ]

    print("Loading GPT-2 model in vLLM...")
    llm = LLM(model="gpt2")

    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        prompts = [dummy_prompts[i % len(dummy_prompts)] for i in range(batch_size)]
        sampling_params = SamplingParams(max_tokens=num_tokens)

        for _ in range(num_warmup):
            _ = llm.generate(prompts, sampling_params)

        latencies = []
        for _ in range(num_trials):
            r = llm.generate(prompts, sampling_params)
            latencies.append(_get_avg_itl(r))

        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies)//2]
        p90 = sorted(latencies)[int(len(latencies)*0.9)]
        print(f"Batch size {batch_size}:")
        print(f"  Avg inter-token latency: {avg_latency*1000:.2f} ms")
        print(f"  P50 inter-token latency: {p50*1000:.2f} ms")
        print(f"  P90 inter-token latency: {p90*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPT-2 inter-token latency with vLLM.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32],
                        help="Batch sizes to benchmark.")
    parser.add_argument("--num_tokens", type=int, default=32,
                        help="Number of tokens to generate per prompt.")
    parser.add_argument("--num_warmup", type=int, default=4,
                        help="Number of warmup runs.")
    parser.add_argument("--num_trials", type=int, default=16,
                        help="Number of timed trials.")
    args = parser.parse_args()

    benchmark_gpt2(
        batch_sizes=args.batch_sizes,
        num_tokens=args.num_tokens,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials
    )
