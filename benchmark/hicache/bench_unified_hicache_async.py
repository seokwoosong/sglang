"""Compare unified-memory serving with no, synchronous, and asynchronous HiCache.

The benchmark owns the server lifecycle and writes one JSON artifact per launch.
Use independent launches and rotate mode order to reduce allocator/startup drift.

Examples:

    PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
        --mode unified --scenario hot --launch-id hot-unified-1
    PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
        --mode sync --scenario restore --launch-id restore-sync-1
    PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
        --mode async --scenario restore --launch-id restore-async-1

``hot`` keeps the working set in L1 and measures feature overhead. ``restore``
uses independent long prefixes that exceed L1, then replays them to measure real
L2 load-back. The synchronous mode is selected with the production rollback knob
``SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS=1``; all other code is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "Qwen/Qwen3.5-0.8B"
METRIC_NAMES = (
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:load_back_duration_seconds_sum",
    "sglang:load_back_duration_seconds_count",
    "sglang:realtime_tokens_total",
    "sglang:forward_execution_seconds_total",
    "sglang:num_retracted_reqs_total",
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def gpu_snapshot() -> dict[str, Any]:
    query = "index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "query": query,
        "lines": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def build_prompts(scenario: str, count: int, input_len: int) -> list[str]:
    if scenario == "hot":
        shared = "benchmark shared stable prefix for unified memory hierarchy " * 90
        return [
            shared + (f" request-{request_id} payload-{request_id * 7919} " * 20)
            for request_id in range(count)
        ]
    payload_repeats = max(1, input_len // 8)
    return [
        hashlib.sha256(str(request_id).encode()).hexdigest()
        + " "
        + ("payload_" + hashlib.md5(str(request_id).encode()).hexdigest() + " ")
        * payload_repeats
        for request_id in range(count)
    ]


def generate(prompt: str | list[int], output_len: int) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at = None
    final_body = None
    with requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            ("text" if isinstance(prompt, str) else "input_ids"): prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
        },
        timeout=180,
        stream=True,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                continue
            body = json.loads(line)
            final_body = body
            completion_tokens = body.get("meta_info", {}).get("completion_tokens", 0)
            if completion_tokens and first_token_at is None:
                first_token_at = time.perf_counter()

    finished = time.perf_counter()
    if final_body is None or first_token_at is None:
        raise RuntimeError("Streaming response completed without a generated token")
    meta = final_body["meta_info"]
    completion_tokens = int(meta.get("completion_tokens", output_len))
    latency = finished - started
    ttft = first_token_at - started
    return {
        "latency_seconds": latency,
        "ttft_seconds": ttft,
        "tpot_seconds": (
            (latency - ttft) / (completion_tokens - 1) if completion_tokens > 1 else 0.0
        ),
        "prompt_tokens": int(
            meta.get("prompt_tokens", len(prompt) if not isinstance(prompt, str) else 0)
        ),
        "completion_tokens": completion_tokens,
        "cached_tokens": int(meta.get("cached_tokens", 0)),
        "cached_tokens_details": meta.get("cached_tokens_details") or {},
        "output_ids": [int(token) for token in final_body.get("output_ids", [])],
    }


def scrape_metrics() -> dict[str, float]:
    response = requests.get(DEFAULT_URL_FOR_TEST + "/metrics", timeout=30)
    response.raise_for_status()
    totals = {name: 0.0 for name in METRIC_NAMES}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in totals:
            continue
        try:
            totals[name] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            pass
    return totals


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in before}


def run_round(prompts: list[str], output_len: int, workers: int) -> dict[str, Any]:
    metrics_before = scrape_metrics()
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(generate, input_ids, output_len) for input_ids in prompts
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - started
    metrics_after = scrape_metrics()

    latencies = [result["latency_seconds"] for result in results]
    ttfts = [result["ttft_seconds"] for result in results]
    tpots = [
        result["tpot_seconds"] for result in results if result["completion_tokens"] > 1
    ]
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    cached_tokens = sum(result["cached_tokens"] for result in results)
    cache_sources: dict[str, int] = {}
    for result in results:
        for source, tokens in result["cached_tokens_details"].items():
            cache_sources[source] = cache_sources.get(source, 0) + int(tokens)
    signatures = sorted(
        hashlib.sha256(json.dumps(result["output_ids"]).encode()).hexdigest()
        for result in results
    )
    return {
        "wall_seconds": wall_seconds,
        "requests_per_second": len(results) / wall_seconds,
        "input_tokens_per_second": prompt_tokens / wall_seconds,
        "output_tokens_per_second": sum(
            result["completion_tokens"] for result in results
        )
        / wall_seconds,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_max_seconds": max(latencies),
        "ttft_mean_seconds": statistics.mean(ttfts),
        "ttft_p50_seconds": statistics.median(ttfts),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "tpot_mean_seconds": statistics.mean(tpots) if tpots else 0.0,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_token_ratio": cached_tokens / prompt_tokens,
        "cache_sources": cache_sources,
        "metric_delta": metric_delta(metrics_before, metrics_after),
        "output_signature": hashlib.sha256("".join(signatures).encode()).hexdigest(),
    }


def summarize(rounds: list[dict[str, Any]], discard: int) -> dict[str, Any]:
    kept = rounds[discard:]
    fields = (
        "requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "latency_mean_seconds",
        "latency_p95_seconds",
        "ttft_mean_seconds",
        "ttft_p95_seconds",
        "tpot_mean_seconds",
        "cached_token_ratio",
    )
    summary: dict[str, Any] = {
        "discarded_rounds": discard,
        "kept_rounds": len(kept),
    }
    for field in fields:
        values = [float(round_result[field]) for round_result in kept]
        summary[field + "_mean"] = statistics.mean(values)
        summary[field + "_median"] = statistics.median(values)
        summary[field + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
        summary[field + "_min"] = min(values)
        summary[field + "_max"] = max(values)
    summary["metric_delta_total"] = {
        name: sum(result["metric_delta"][name] for result in kept)
        for name in METRIC_NAMES
    }
    summary["cache_sources_total"] = {
        source: sum(result["cache_sources"].get(source, 0) for result in kept)
        for source in sorted(
            {source for result in kept for source in result["cache_sources"]}
        )
    }
    summary["output_signatures"] = [result["output_signature"] for result in kept]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("unified", "sync", "async"), required=True)
    parser.add_argument("--scenario", choices=("hot", "restore"), required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--discard", type=int)
    parser.add_argument("--prompt-count", type=int)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mem-fraction", type=float)
    parser.add_argument("--hicache-size", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/unified-hicache-async-bench")
    args = parser.parse_args()

    defaults = {
        "hot": {
            "rounds": 7,
            "discard": 2,
            "prompt_count": 12,
            "input_len": 512,
            "output_len": 16,
            "mem_fraction": 0.20,
        },
        "restore": {
            "rounds": 5,
            "discard": 1,
            # Matches the correctness stress geometry. ``input_len`` controls
            # payload repetitions approximately (not tokenizer-exact length).
            "prompt_count": 60,
            "input_len": 720,
            "output_len": 1,
            "mem_fraction": 0.075,
        },
    }[args.scenario]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{args.launch_id}.stdout.log"
    stderr_path = output_dir / f"{args.launch_id}.stderr.log"
    result_path = output_dir / f"{args.launch_id}.json"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")

    server_args = [
        "--enable-unified-memory",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--max-running-requests",
        str(args.workers),
        "--mamba-full-memory-ratio",
        "1.5",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
        "--decode-log-interval",
        "1",
        "--log-level",
        "info",
    ]
    if args.mode != "unified":
        server_args += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            str(args.hicache_size),
            "--hicache-write-policy",
            "write_back",
            "--hicache-io-backend",
            "kernel",
        ]

    env = {
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
        "SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS": ("1" if args.mode == "sync" else "0"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "python")
        + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    before_launch = gpu_snapshot()
    process = None
    try:
        process = popen_launch_server(
            args.model,
            DEFAULT_URL_FOR_TEST,
            timeout=180,
            other_args=server_args,
            env=env,
            return_stdout_stderr=(stdout_handle, stderr_handle),
        )
        after_launch = gpu_snapshot()
        prompts = build_prompts(args.scenario, args.prompt_count, args.input_len)

        generate([1000 + index for index in range(32)], 2)  # compile warm-up
        population_started = time.perf_counter()
        population = [generate(prompt, 1) for prompt in prompts]
        population_seconds = time.perf_counter() - population_started
        population_metrics = scrape_metrics()

        rounds = []
        for round_index in range(args.rounds):
            ordered = prompts if round_index % 2 else list(reversed(prompts))
            result = run_round(ordered, args.output_len, args.workers)
            result["round"] = round_index
            rounds.append(result)
            print("ROUND " + json.dumps(result, sort_keys=True), flush=True)

        report = {
            "mode": args.mode,
            "scenario": args.scenario,
            "launch_id": args.launch_id,
            "model": args.model,
            "configuration": {
                "rounds": args.rounds,
                "discard": args.discard,
                "prompt_count": args.prompt_count,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "workers": args.workers,
                "mem_fraction": args.mem_fraction,
                "hicache_size": args.hicache_size,
                "server_args": server_args,
                "environment": env,
            },
            "gpu_before_launch": before_launch,
            "gpu_after_launch": after_launch,
            "gpu_before_shutdown": gpu_snapshot(),
            "population_seconds": population_seconds,
            "population_cached_token_ratio": statistics.mean(
                result["cached_tokens"] / result["prompt_tokens"]
                for result in population
            ),
            "population_metrics": population_metrics,
            "rounds": rounds,
            "summary": summarize(rounds, args.discard),
            "health": requests.get(
                DEFAULT_URL_FOR_TEST + "/health", timeout=30
            ).status_code,
            "server_stdout": str(stdout_path),
            "server_stderr": str(stderr_path),
        }
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print("BENCH_RESULT " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if process is not None:
            kill_process_tree(process.pid)
            process.wait(timeout=120)
        stdout_handle.close()
        stderr_handle.close()


if __name__ == "__main__":
    main()
