"""Serving benchmark for HiCache copy optimization.

Sends requests to a running SGLang server and measures throughput/latency
under different prefix/output length scenarios.

Usage:
  python bench_hicache_serving.py --port 30000 --scenario long_prefix \
      --output results.json
"""

import argparse
import asyncio
import json
import time

import aiohttp


SCENARIOS = {
    "short_prefix": {
        "prefix_len": 128,
        "output_len": 512,
        "num_requests": 200,
        "description": "Short prefix, long output — frequent HiCache eviction",
    },
    "long_prefix": {
        "prefix_len": 4096,
        "output_len": 64,
        "num_requests": 100,
        "description": "Long prefix, short output — frequent HiCache hits",
    },
    "mixed": {
        "prefix_len": 1024,
        "output_len": 256,
        "num_requests": 150,
        "description": "Mixed prefix/output — realistic workload",
    },
}


async def send_request(session, url, prompt, max_tokens):
    """Send a single request and return timing info."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    elapsed = time.perf_counter() - start
    completion_tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "ttft_s": elapsed,  # approximate
    }


async def run_benchmark(port, scenario_name, num_requests=None):
    scenario = SCENARIOS[scenario_name]
    n = num_requests or scenario["num_requests"]
    prefix_len = scenario["prefix_len"]
    output_len = scenario["output_len"]

    # Generate a long prefix (repeated tokens)
    prefix_text = "Hello " * (prefix_len // 6)

    url = f"http://localhost:{port}/v1/chat/completions"

    print(f"  Scenario: {scenario_name}")
    print(f"  Description: {scenario['description']}")
    print(f"  Requests: {n}, Prefix: {prefix_len}, Output: {output_len}")
    print()

    # Warmup
    async with aiohttp.ClientSession() as session:
        await send_request(session, url, prefix_text, 16)

    # Run benchmark
    results = []
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, prefix_text, output_len) for _ in range(n)]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # Compute metrics
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    latencies = [r["elapsed_s"] for r in results]
    latencies.sort()

    metrics = {
        "scenario": scenario_name,
        "num_requests": n,
        "prefix_len": prefix_len,
        "output_len": output_len,
        "total_time_s": total_time,
        "throughput_tok_s": total_completion_tokens / total_time if total_time > 0 else 0,
        "mean_latency_ms": sum(latencies) / len(latencies) * 1000,
        "p50_latency_ms": latencies[len(latencies) // 2] * 1000,
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)] * 1000,
        "total_completion_tokens": total_completion_tokens,
    }

    print(f"  Throughput: {metrics['throughput_tok_s']:.1f} tok/s")
    print(f"  Mean latency: {metrics['mean_latency_ms']:.1f} ms")
    print(f"  P50 latency: {metrics['p50_latency_ms']:.1f} ms")
    print(f"  P99 latency: {metrics['p99_latency_ms']:.1f} ms")
    print()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="HiCache serving benchmark")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics = asyncio.run(run_benchmark(args.port, args.scenario, args.num_requests))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
