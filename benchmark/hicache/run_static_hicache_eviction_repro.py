"""Drive a static HiCache server through KV eviction and host load-back."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

METRICS = (
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_dropped_tokens_total",
    "sglang:kv_evictable_tokens",
)


def metric_snapshot(base_url: str) -> dict[str, list[dict[str, Any]]]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        for name in METRICS:
            if not line.startswith(name):
                continue
            match = re.fullmatch(r"([^ {]+)(\{.*\})? ([^ ]+)", line)
            if match is not None:
                snapshot.setdefault(name, []).append(
                    {
                        "labels": match.group(2) or "",
                        "value": float(match.group(3)),
                    }
                )
            break
    return snapshot


def generate(base_url: str, input_ids: list[int], timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    meta = result.get("meta_info", {})
    return {
        "latency_seconds": time.perf_counter() - start,
        "output_ids": result.get("output_ids"),
        "cached_tokens": meta.get("cached_tokens"),
        "cached_tokens_details": meta.get("cached_tokens_details"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "num_retractions": meta.get("num_retractions"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=10000)
    parser.add_argument("--replay-count", type=int, default=8)
    parser.add_argument("--base-token-id", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_count > args.num_prompts:
        raise ValueError("replay-count must not exceed num-prompts")

    payload: dict[str, Any] = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metrics_before": metric_snapshot(args.base_url),
        "fill": [],
        "replay": [],
    }
    for prompt_id in range(args.num_prompts):
        token_id = args.base_token_id + prompt_id
        result = generate(args.base_url, [token_id] * args.prompt_tokens, args.timeout)
        result["prompt_id"] = prompt_id
        payload["fill"].append(result)
        print(
            f"fill {prompt_id + 1}/{args.num_prompts}: "
            f"{result['latency_seconds']:.3f}s cached={result['cached_tokens']}",
            flush=True,
        )

    payload["metrics_after_fill"] = metric_snapshot(args.base_url)
    for prompt_id in range(args.replay_count):
        token_id = args.base_token_id + prompt_id
        result = generate(args.base_url, [token_id] * args.prompt_tokens, args.timeout)
        result["prompt_id"] = prompt_id
        result["matches_fill_output"] = (
            result["output_ids"] == payload["fill"][prompt_id]["output_ids"]
        )
        payload["replay"].append(result)
        print(
            f"replay {prompt_id + 1}/{args.replay_count}: "
            f"{result['latency_seconds']:.3f}s cached={result['cached_tokens']} "
            f"matches={result['matches_fill_output']}",
            flush=True,
        )

    payload["metrics_after_replay"] = metric_snapshot(args.base_url)
    payload["validation"] = {
        "all_requests_completed": len(payload["fill"]) == args.num_prompts
        and len(payload["replay"]) == args.replay_count,
        "all_replay_outputs_match": all(
            result["matches_fill_output"] for result in payload["replay"]
        ),
        "all_replays_hit_cache": all(
            (result["cached_tokens"] or 0) > 0 for result in payload["replay"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps(payload["validation"], indent=2))


if __name__ == "__main__":
    main()
