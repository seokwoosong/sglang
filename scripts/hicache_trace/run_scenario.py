#!/usr/bin/env python3
"""Drive a small deterministic unified-memory HiCache eviction/load-back run."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

import requests


def metric(base_url: str, name: str) -> float:
    response = requests.get(f"{base_url}/metrics", timeout=10)
    response.raise_for_status()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in response.text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


def wait_metric(base_url: str, name: str, baseline: float, timeout: float) -> float:
    deadline = time.monotonic() + timeout
    value = metric(base_url, name)
    while value <= baseline and time.monotonic() < deadline:
        time.sleep(0.05)
        value = metric(base_url, name)
    return value


def generate(
    base_url: str, index: int, prompt_tokens: int, request_timeout: float
) -> dict:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "rid": f"hicache-trace-{index}-{time.time_ns()}",
            "input_ids": [1000 + index] * prompt_tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
            "return_logprob": True,
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    body = response.json()
    logprobs = body["meta_info"]["output_token_logprobs"]
    return {
        "output_ids": [int(token_id) for token_id in body["output_ids"]],
        "logprob_token_ids": [int(item[1]) for item in logprobs],
        "logprobs": [float(item[0]) for item in logprobs],
        "cached_tokens": int(body["meta_info"].get("cached_tokens", 0)),
    }


def assert_same(expected: dict, actual: dict) -> None:
    assert actual["output_ids"] == expected["output_ids"]
    assert actual["logprob_token_ids"] == expected["logprob_token_ids"]
    assert len(actual["logprobs"]) == len(expected["logprobs"])
    for lhs, rhs in zip(actual["logprobs"], expected["logprobs"]):
        assert math.isfinite(lhs) and math.isfinite(rhs)
        assert abs(lhs - rhs) <= 1e-2, (lhs, rhs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--prompt-tokens", type=int, default=768)
    parser.add_argument("--pressure-requests", type=int, default=5)
    parser.add_argument("--expected-max-total-tokens", type=int, default=1024)
    parser.add_argument("--expected-max-mamba-cache-size", type=int, default=16)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600,
        help="Per-generation HTTP timeout; queued max-running=1 runs can be slow.",
    )
    parser.add_argument(
        "--queue-pressure-requests",
        action="store_true",
        help="Submit all pressure requests concurrently so the server stays busy.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/hicache-scenario.json")
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    info = requests.get(f"{base_url}/server_info", timeout=10).json()
    if info.get("max_total_tokens") != args.expected_max_total_tokens:
        raise AssertionError(
            f"expected max_total_tokens={args.expected_max_total_tokens}, "
            f"got {info.get('max_total_tokens')}"
        )
    if info.get("max_mamba_cache_size") != args.expected_max_mamba_cache_size:
        raise AssertionError(
            "expected max_mamba_cache_size="
            f"{args.expected_max_mamba_cache_size}, "
            f"got {info.get('max_mamba_cache_size')}"
        )

    events = []
    evicted_before = metric(base_url, "sglang:evicted_tokens_total")
    baselines = {}
    if args.queue_pressure_requests:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.pressure_requests
        ) as executor:
            futures = {
                executor.submit(
                    generate,
                    base_url,
                    index,
                    args.prompt_tokens,
                    args.request_timeout,
                ): index
                for index in range(args.pressure_requests)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                baselines[index] = future.result()
        for index in range(args.pressure_requests):
            events.append(
                {"step": "pressure", "index": index, "result": baselines[index]}
            )
    else:
        for index in range(args.pressure_requests):
            result = generate(base_url, index, args.prompt_tokens, args.request_timeout)
            baselines[index] = result
            events.append({"step": "pressure", "index": index, "result": result})

    evicted_after = wait_metric(
        base_url, "sglang:evicted_tokens_total", evicted_before, timeout=5
    )
    if evicted_after <= evicted_before:
        raise AssertionError("fixed L1 capacity did not force eviction")

    restored = None
    for index in range(args.pressure_requests):
        load_before = metric(base_url, "sglang:load_back_tokens_total")
        replay = generate(base_url, index, args.prompt_tokens, args.request_timeout)
        load_after = wait_metric(
            base_url, "sglang:load_back_tokens_total", load_before, timeout=2
        )
        events.append(
            {
                "step": "replay",
                "index": index,
                "load_back_before": load_before,
                "load_back_after": load_after,
                "result": replay,
            }
        )
        if load_after > load_before:
            assert replay["cached_tokens"] > 0
            assert_same(baselines[index], replay)
            restored = index
            break
    if restored is None:
        raise AssertionError("no L2-resident prefix was restored")

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    report = {
        "server": {
            "max_total_tokens": info["max_total_tokens"],
            "max_mamba_cache_size": info["max_mamba_cache_size"],
        },
        "evicted_tokens_delta": evicted_after - evicted_before,
        "restored_index": restored,
        "events": events,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
