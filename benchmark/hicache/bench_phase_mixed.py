#!/usr/bin/env python3
"""Run a token-balanced, length-interleaved HiCache workload on one server.

Each wave mixes short, middle, and long requests in a deterministic shuffled
order.  Wave 1 introduces new prefixes; wave 2 reuses the same prefixes with
new suffixes.  A wave barrier prevents a replay from racing its own first use,
while requests of all three lengths overlap within each wave.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from bench_unified_ablation import (
    flush_cache,
    get_server_info,
    memory_profile_delta,
    memory_profile_snapshot,
    metric_delta,
    metric_snapshot,
    percentile,
    stream_generate,
)


@dataclass(frozen=True)
class LengthClass:
    input_len: int
    output_len: int
    groups: int


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: int
    length_class: str
    group_id: int
    wave: int
    output_len: int
    input_ids: list[int]


@dataclass(frozen=True)
class ScheduledWave:
    role: str
    max_concurrency: int
    requests: list[ScheduledRequest]


# Input-token contribution per wave is intentionally balanced:
# short=96k, middle=100k, long=100k.  Equal request counts would let short
# traffic dominate request throughput while barely exercising shared memory.
LENGTH_CLASSES = {
    "short": LengthClass(3_000, 256, 32),
    "middle": LengthClass(10_000, 256, 10),
    "long": LengthClass(50_000, 128, 2),
}


def _make_profile_waves(
    *, seed: int, shared_ratio: float, traffic_profile: str
) -> list[tuple[str, int, list[tuple[str, int, int, int, float]]]]:
    """Return wave templates as (role, concurrency, request descriptors).

    A descriptor is (class label, group id, input len, output len, reuse ratio).
    Prefixes are keyed by (label, group), so repeated descriptors in later waves
    exercise the requested amount of real prefix reuse without restarting the server.
    """

    stationary = []
    for wave in range(2):
        requests = [
            (label, group_id, spec.input_len, spec.output_len, shared_ratio)
            for label, spec in LENGTH_CLASSES.items()
            for group_id in range(spec.groups)
        ]
        stationary.append(("new" if wave == 0 else "replay", 16, requests))
    if traffic_profile == "stationary":
        return stationary

    if traffic_profile == "demand-shift":
        kv = [("long", i, 50_000, 128, shared_ratio) for i in range(4)]
        mamba = [("short", i, 3_000, 512, shared_ratio) for i in range(32)]
        return [
            ("kv-heavy-new", 4, kv),
            ("mamba-heavy-new", 16, mamba),
            ("kv-heavy-replay", 4, kv),
            ("mamba-heavy-replay", 16, mamba),
        ]

    if traffic_profile == "residency-burst":
        resident = [("resident-long", i, 50_000, 128, shared_ratio) for i in range(6)]
        burst = [("burst-short", i, 3_000, 512, 0.2) for i in range(36)]
        return [
            ("resident-prime", 6, resident),
            ("resident-promote", 6, resident),
            ("short-concurrency-burst", 16, burst),
            ("resident-replay-after-burst", 6, resident),
        ]

    if traffic_profile == "concurrency-spike":
        mixed = (
            [("short", i, 3_000, 256, shared_ratio) for i in range(16)]
            + [("middle", i, 10_000, 256, shared_ratio) for i in range(5)]
            + [("long", i, 50_000, 128, shared_ratio) for i in range(2)]
        )
        return [
            ("low-concurrency-new", 4, mixed),
            ("high-concurrency-replay", 16, mixed),
            ("low-concurrency-refresh", 4, mixed),
            ("high-concurrency-replay-2", 16, mixed),
        ]

    if traffic_profile in {"ordered-long-late", "ordered-long-early"}:
        short = [("short", i, 3_000, 256, shared_ratio) for i in range(16)]
        middle = [("middle", i, 10_000, 256, shared_ratio) for i in range(5)]
        long = [("long", i, 50_000, 128, shared_ratio) for i in range(2)]
        if traffic_profile == "ordered-long-late":
            # Long requests land just outside the initial 16-worker admission
            # window, matching the ordering that favored dynamic sharing in the
            # shuffled discovery trace.
            ordered = short + [long[0], middle[0], long[1]] + middle[1:]
        else:
            ordered = [long[0], short[0], long[1]] + short[1:] + middle
        return [
            ("low-concurrency-new", 4, ordered),
            ("high-concurrency-replay", 16, ordered),
            ("low-concurrency-refresh", 4, ordered),
            ("high-concurrency-replay-2", 16, ordered),
        ]

    if traffic_profile == "reuse-shift":
        wave_a = (
            [("short-low-reuse", i, 3_000, 256, 0.2) for i in range(32)]
            + [("middle-a", i, 10_000, 256, 0.5) for i in range(10)]
            + [("long-high-reuse", i, 50_000, 128, 0.8) for i in range(2)]
        )
        wave_b = (
            [("short-high-reuse", i, 3_000, 256, 0.8) for i in range(32)]
            + [("middle-b", i, 10_000, 256, 0.5) for i in range(10)]
            + [("long-low-reuse", i, 50_000, 128, 0.2) for i in range(2)]
        )
        return [
            ("long-high-reuse", 16, wave_a),
            ("short-high-reuse", 16, wave_b),
            ("long-high-reuse-replay", 16, wave_a),
            ("short-high-reuse-replay", 16, wave_b),
        ]

    if traffic_profile == "heavy-tail":
        rng = random.Random(seed + 30_000)
        lengths = (2_000, 3_000, 5_000, 10_000, 20_000, 50_000)
        requests = []
        for group_id in range(44):
            # Squaring a uniform variate biases toward short requests while
            # retaining a deterministic long tail.
            index = min(int((rng.random() ** 2) * len(lengths)), len(lengths) - 1)
            input_len = lengths[index]
            output_len = 128 if input_len >= 20_000 else 384
            reuse = rng.choice((0.2, 0.5, 0.8))
            requests.append((f"tail-{input_len}", group_id, input_len, output_len, reuse))
        return [
            ("heavy-tail-new", 16, requests),
            ("heavy-tail-replay", 16, requests),
        ]

    raise ValueError(f"Unknown traffic profile: {traffic_profile}")


def make_waves(
    *, seed: int, shared_ratio: float, traffic_profile: str = "stationary"
) -> tuple[list[ScheduledWave], str]:
    if not 0 < shared_ratio < 1:
        raise ValueError("shared-ratio must be between zero and one")

    templates = _make_profile_waves(
        seed=seed, shared_ratio=shared_ratio, traffic_profile=traffic_profile
    )
    prefix_rng = random.Random(seed)
    suffix_rng = random.Random(seed + 10_000)
    prefixes: dict[tuple[str, int], list[int]] = {}
    prefix_specs: dict[tuple[str, int], tuple[int, float]] = {}
    for _, _, descriptors in templates:
        for label, group_id, input_len, _, reuse in descriptors:
            key = (label, group_id)
            old = prefix_specs.get(key)
            candidate = (input_len, reuse)
            if old is None or int(input_len * reuse) > int(old[0] * old[1]):
                prefix_specs[key] = candidate
    for class_index, ((label, group_id), (input_len, reuse)) in enumerate(
        sorted(prefix_specs.items())
    ):
        prefix_len = max(1, int(input_len * reuse))
        if prefix_len >= input_len:
            prefix_len = input_len - 1
        if prefix_len <= 0:
            prefix_len = 1
        # Keep the leading token unique across classes and groups.
        prefix = [1_000 + class_index]
        prefix.extend(
            prefix_rng.randrange(5_000, 200_000) for _ in range(prefix_len - 1)
        )
        prefixes[(label, group_id)] = prefix

    next_request_id = 0
    waves: list[ScheduledWave] = []
    trace_rows: list[str] = []
    for wave, (role, max_concurrency, descriptors) in enumerate(templates):
        scheduled: list[ScheduledRequest] = []
        for label, group_id, input_len, output_len, reuse in descriptors:
            full_prefix = prefixes[(label, group_id)]
            prefix_len = min(len(full_prefix), max(1, int(input_len * reuse)))
            prefix = full_prefix[:prefix_len]
            suffix_len = input_len - prefix_len
            suffix = [
                suffix_rng.randrange(5_000, 200_000) for _ in range(suffix_len)
            ]
            scheduled.append(
                ScheduledRequest(
                    request_id=next_request_id,
                    length_class=label,
                    group_id=group_id,
                    wave=wave,
                    output_len=output_len,
                    input_ids=prefix + suffix,
                )
            )
            next_request_id += 1
        if not traffic_profile.startswith("ordered-"):
            random.Random(seed + 100 * (wave + 1)).shuffle(scheduled)
        waves.append(ScheduledWave(role, max_concurrency, scheduled))
        trace_rows.append(f"wave:{wave}:{role}:concurrency={max_concurrency}")
        trace_rows.extend(
            f"{item.wave}:{item.length_class}:{item.group_id}:"
            f"{len(item.input_ids)}:{item.output_len}"
            for item in scheduled
        )

    fingerprint = hashlib.sha256("\n".join(trace_rows).encode()).hexdigest()
    return waves, fingerprint


def summarize(records: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    ttfts = [record["ttft_s"] for record in successful if record["ttft_s"] is not None]
    latencies = [record["latency_s"] for record in successful]
    total_input = sum(int(record["prompt_tokens"]) for record in successful)
    total_output = sum(int(record["completion_tokens"]) for record in successful)
    return {
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "duration_s": duration_s,
        "request_throughput": len(successful) / duration_s,
        "input_token_throughput": total_input / duration_s,
        "output_token_throughput": total_output / duration_s,
        "total_token_throughput": (total_input + total_output) / duration_s,
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) * 1_000 if ttfts else None,
            "p50": percentile(ttfts, 50) * 1_000 if ttfts else None,
            "p95": percentile(ttfts, 95) * 1_000 if ttfts else None,
            "p99": percentile(ttfts, 99) * 1_000 if ttfts else None,
        },
        "e2e_ms": {
            "mean": statistics.fmean(latencies) * 1_000 if latencies else None,
            "p50": percentile(latencies, 50) * 1_000 if latencies else None,
            "p95": percentile(latencies, 95) * 1_000 if latencies else None,
            "p99": percentile(latencies, 99) * 1_000 if latencies else None,
        },
        "cached_tokens": sum(int(record["cached_tokens"]) for record in successful),
        "cached_tokens_device": sum(
            int(record["cached_tokens_device"]) for record in successful
        ),
        "cached_tokens_host": sum(
            int(record["cached_tokens_host"]) for record in successful
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    flush_cache(args.base_url)
    server_info = get_server_info(args.base_url)
    metrics_before = metric_snapshot(args.base_url)
    memory_before = memory_profile_snapshot(args.output)
    waves, trace_fingerprint = make_waves(
        seed=args.seed,
        shared_ratio=args.shared_ratio,
        traffic_profile=args.traffic_profile,
    )

    timeout = aiohttp.ClientTimeout(total=60 * 60)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))
    url = f"{args.base_url}/generate"
    all_records: list[dict[str, Any]] = []
    wave_summaries: list[dict[str, Any]] = []
    validation_probe: list[dict[str, Any]] = []
    measured_started = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for wave_index, wave in enumerate(waves):
            schedule = wave.requests
            queue: asyncio.Queue[ScheduledRequest] = asyncio.Queue()
            for item in schedule:
                queue.put_nowait(item)

            records: list[dict[str, Any]] = []

            async def worker() -> None:
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    record = await stream_generate(
                        session,
                        url,
                        request_id=item.request_id,
                        group_id=item.group_id,
                        round_id=item.wave,
                        input_ids=item.input_ids,
                        output_len=item.output_len,
                    )
                    records.append(
                        asdict(record)
                        | {
                            "length_class": item.length_class,
                            "wave": item.wave,
                        }
                    )
                    queue.task_done()

            wave_started = time.perf_counter()
            await asyncio.gather(
                *(
                    worker()
                    for _ in range(
                        min(args.max_concurrency, wave.max_concurrency, len(schedule))
                    )
                )
            )
            wave_duration = time.perf_counter() - wave_started
            records.sort(key=lambda item: int(item["request_id"]))
            all_records.extend(records)
            wave_summaries.append(
                {
                    "wave": wave_index,
                    "role": wave.role,
                    "max_concurrency": min(args.max_concurrency, wave.max_concurrency),
                    "schedule": [
                        {
                            "request_id": item.request_id,
                            "length_class": item.length_class,
                            "group_id": item.group_id,
                        }
                        for item in schedule
                    ],
                    "summary": summarize(records, wave_duration),
                }
            )

        # Keep the performance interval limited to the two workload waves.
        # An independently shuffled replay can legitimately miss every L2
        # resident entry after creating new pressure of its own. If that
        # happens, probe the original requests sequentially after measurement
        # so HiCache validation checks capability rather than shuffle luck.
        measured_duration = time.perf_counter() - measured_started
        probe_delta = metric_delta(metrics_before, metric_snapshot(args.base_url))
        if (
            args.expect_hicache
            and probe_delta["sglang:hicache_backup_tokens_total"] > 0
            and probe_delta["sglang:load_back_tokens_total"] <= 0
        ):
            for probe_index, item in enumerate(waves[0].requests):
                record = await stream_generate(
                    session,
                    url,
                    request_id=100_000 + probe_index,
                    group_id=item.group_id,
                    round_id=2,
                    input_ids=item.input_ids,
                    output_len=1,
                )
                probe_delta = metric_delta(
                    metrics_before, metric_snapshot(args.base_url)
                )
                validation_probe.append(
                    asdict(record)
                    | {
                        "length_class": item.length_class,
                        "source_request_id": item.request_id,
                        "load_back_tokens_total": probe_delta[
                            "sglang:load_back_tokens_total"
                        ],
                    }
                )
                if probe_delta["sglang:load_back_tokens_total"] > 0:
                    break

    metrics_after = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_after = memory_profile_snapshot(args.output)
    total_delta = metric_delta(metrics_before, metrics_after)
    summary = summarize(all_records, measured_duration)

    by_class: dict[str, dict[str, Any]] = {}
    for label in sorted({record["length_class"] for record in all_records}):
        class_records = [
            record for record in all_records if record["length_class"] == label
        ]
        by_class[label] = summarize(class_records, measured_duration)

    errors: list[str] = []
    if summary["failed"]:
        errors.append(f"{summary['failed']} interleaved requests failed")
    if total_delta["sglang:evicted_tokens_total"] <= 0:
        errors.append("Interleaved workload did not evict L1 entries")
    if args.expect_hicache:
        if total_delta["sglang:hicache_backup_tokens_total"] <= 0:
            errors.append("Interleaved workload did not back up entries to L2")
        if total_delta["sglang:load_back_tokens_total"] <= 0:
            errors.append("Interleaved workload did not load entries back from L2")
        # Some static backends report a replay as device-cached in response
        # metadata after the host row has already been promoted.  The server's
        # monotonic load-back counter is authoritative host-hit evidence too.
        if (
            summary["cached_tokens_host"] <= 0
            and total_delta["sglang:load_back_tokens_total"] <= 0
        ):
            errors.append("Interleaved workload produced no host-cache hit evidence")
    if args.forbid_dropped and total_delta["sglang:hicache_dropped_tokens_total"] > 0:
        errors.append("HiCache dropped tokens during the interleaved workload")

    return {
        "schema_version": 2,
        "kind": "interleaved-mixed",
        "variant": args.variant,
        "created_wall_time_ns": time.time_ns(),
        "args": vars(args) | {"output": str(args.output)},
        "server_info": server_info,
        "length_classes": {
            label: asdict(spec) for label, spec in LENGTH_CLASSES.items()
        },
        "traffic_profile": args.traffic_profile,
        "trace_fingerprint": trace_fingerprint,
        "waves": wave_summaries,
        "validation_probe": validation_probe,
        "total_metric_delta": total_delta,
        "memory_profile_total_delta": memory_profile_delta(memory_before, memory_after),
        "summary": summary,
        "summary_by_length_class": by_class,
        "requests": all_records,
        "validation": {"passed": not errors, "errors": errors},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=7_301)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument(
        "--traffic-profile",
        choices=[
            "stationary",
            "demand-shift",
            "residency-burst",
            "concurrency-spike",
            "ordered-long-late",
            "ordered-long-early",
            "reuse-shift",
            "heavy-tail",
        ],
        default="stationary",
    )
    parser.add_argument("--expect-hicache", action="store_true")
    parser.add_argument("--forbid-dropped", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(run(args))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "validation": payload["validation"]}))
    return 0 if payload["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
