"""Deterministic correctness and serving workloads for unified-memory HiCache.

The benchmark deliberately distinguishes an enabled feature from an exercised
feature.  Pressure runs fail validation unless L1 eviction happened, and
HiCache runs additionally require an L2 load-back.  Raw per-request records and
Prometheus counter deltas are written to one JSON file for later paired
analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import decimal
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import requests

COUNTERS = (
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:load_back_bytes_total",
    "sglang:eviction_duration_seconds_sum",
    "sglang:eviction_duration_seconds_count",
    "sglang:load_back_duration_seconds_sum",
    "sglang:load_back_duration_seconds_count",
    "sglang:hicache_backup_tokens_total",
    "sglang:hicache_backup_bytes_total",
    "sglang:hicache_backup_duration_seconds_sum",
    "sglang:hicache_backup_duration_seconds_count",
    "sglang:hicache_dropped_tokens_total",
    "sglang:forward_execution_seconds_total",
    "sglang:realtime_tokens_total",
    "sglang:estimated_read_bytes_per_gpu_total",
    "sglang:estimated_write_bytes_per_gpu_total",
)

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def metric_snapshot(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    parsed = parse_prometheus(response.text)
    return {name: parsed.get(name, 0.0) for name in COUNTERS}


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in COUNTERS}


def memory_profile_snapshot(output_path: str) -> dict[str, Any]:
    profile_dir = Path(output_path).parent / "memory_breakdown_profile"
    profiles = []
    for path in sorted(profile_dir.glob("memory_profile.*.json")):
        try:
            profiles.append((str(path), json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError):
            continue

    def aggregate(field: str) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for _, profile in profiles:
            for item in profile.get(field, []):
                key = (item["category"], item["pool"], item["operation"])
                target = merged.setdefault(
                    key,
                    {
                        "category": key[0],
                        "pool": key[1],
                        "operation": key[2],
                        "calls": 0,
                        "errors": 0,
                        "cpu_time_ns": 0,
                        "rows": 0,
                        "bytes": 0,
                    },
                )
                for name in ("calls", "errors", "cpu_time_ns", "rows", "bytes"):
                    target[name] += int(item.get(name, 0))
        return [merged[key] for key in sorted(merged)]

    sample_merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, profile in profiles:
        for item in profile.get("samples", []):
            key = (item["category"], item["pool"], item["operation"])
            target = sample_merged.setdefault(
                key,
                {
                    "category": key[0],
                    "pool": key[1],
                    "operation": key[2],
                    "count": 0,
                    "sum": 0,
                    "min": None,
                    "max": None,
                    "histogram": {},
                },
            )
            target["count"] += int(item.get("count", 0))
            target["sum"] += int(item.get("sum", 0))
            value_min = item.get("min")
            value_max = item.get("max")
            if value_min is not None:
                target["min"] = (
                    value_min
                    if target["min"] is None
                    else min(target["min"], value_min)
                )
            if value_max is not None:
                target["max"] = (
                    value_max
                    if target["max"] is None
                    else max(target["max"], value_max)
                )
            for bucket, count in item.get("histogram", {}).items():
                target["histogram"][bucket] = target["histogram"].get(bucket, 0) + int(
                    count
                )

    layouts: dict[tuple[str, str], dict[str, Any]] = {}
    for _, profile in profiles:
        for item in profile.get("layouts", []):
            layouts[(item["pool"], item["layout_kind"])] = item
    return {
        "files": [path for path, _ in profiles],
        "metrics": aggregate("metrics"),
        "cuda_metrics": aggregate("cuda_metrics"),
        "samples": [sample_merged[key] for key in sorted(sample_merged)],
        "layouts": [layouts[key] for key in sorted(layouts)],
    }


def memory_profile_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    def subtract(field: str) -> list[dict[str, Any]]:
        before_map = {
            (item["category"], item["pool"], item["operation"]): item
            for item in before.get(field, [])
        }
        output = []
        for item in after.get(field, []):
            key = (item["category"], item["pool"], item["operation"])
            prior = before_map.get(key, {})
            result = {
                name: value
                for name, value in zip(("category", "pool", "operation"), key)
            }
            for name in ("calls", "errors", "cpu_time_ns", "rows", "bytes"):
                result[name] = int(item.get(name, 0)) - int(prior.get(name, 0))
            output.append(result)
        return output

    before_samples = {
        (item["category"], item["pool"], item["operation"]): item
        for item in before.get("samples", [])
    }
    sample_delta = []
    for item in after.get("samples", []):
        key = (item["category"], item["pool"], item["operation"])
        prior = before_samples.get(key, {})
        histogram = {}
        for bucket in set(item.get("histogram", {})) | set(prior.get("histogram", {})):
            count = int(item.get("histogram", {}).get(bucket, 0)) - int(
                prior.get("histogram", {}).get(bucket, 0)
            )
            if count:
                histogram[bucket] = count
        sample_delta.append(
            {
                "category": key[0],
                "pool": key[1],
                "operation": key[2],
                "count": int(item.get("count", 0)) - int(prior.get("count", 0)),
                "sum": int(item.get("sum", 0)) - int(prior.get("sum", 0)),
                "histogram": histogram,
            }
        )
    return {
        "metrics": subtract("metrics"),
        "cuda_metrics": subtract("cuda_metrics"),
        "samples": sample_delta,
        "layouts": after.get("layouts", []),
    }


def start_torch_profile(args: argparse.Namespace, phase: str) -> dict[str, Any] | None:
    if args.torch_profile_steps <= 0:
        return None
    if not args.torch_profile_output_dir:
        raise ValueError(
            "--torch-profile-output-dir is required when --torch-profile-steps > 0"
        )
    output_dir = Path(args.torch_profile_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "output_dir": str(output_dir),
        "num_steps": args.torch_profile_steps,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": False,
        "profile_id": f"{args.variant}-{phase}",
    }
    response = requests.post(
        f"{args.base_url}/start_profile", json=request, timeout=120
    )
    response.raise_for_status()
    return {"request": request, "start_response": response.text.strip()}


def stop_torch_profile(
    args: argparse.Namespace, state: dict[str, Any] | None
) -> dict[str, Any] | None:
    if state is None:
        return None
    # num_steps normally stops and exports automatically. The explicit stop is
    # a fallback for short workloads; a response saying that profiling already
    # stopped is therefore informative rather than a benchmark failure.
    output_dir = Path(args.torch_profile_output_dir)
    files: list[Path] = []
    for _ in range(10):
        files = [path for path in sorted(output_dir.glob("**/*")) if path.is_file()]
        if files:
            state["auto_stopped"] = True
            break
        time.sleep(0.1)
    if not files:
        try:
            response = requests.post(f"{args.base_url}/stop_profile", timeout=300)
            state["stop_status"] = response.status_code
            state["stop_response"] = response.text.strip()
        except requests.RequestException as exc:
            state["stop_error"] = repr(exc)
        files = [path for path in sorted(output_dir.glob("**/*")) if path.is_file()]
    state["trace_files"] = [str(path) for path in files]
    return state


def get_server_info(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/server_info", timeout=30)
    response.raise_for_status()
    return response.json()


def flush_cache(base_url: str) -> None:
    response = requests.post(f"{base_url}/flush_cache", timeout=60)
    response.raise_for_status()
    time.sleep(0.25)


def write_json(path: str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def sync_generate(
    base_url: str,
    input_ids: list[int],
    output_len: int,
    *,
    return_logprob: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "return_logprob": return_logprob,
        },
        timeout=60 * 60,
    )
    latency = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    meta = body.get("meta_info") or {}
    logprobs = meta.get("output_token_logprobs") or []
    return {
        "latency_s": latency,
        "output_ids": [int(token_id) for token_id in body.get("output_ids", [])],
        "logprob_token_ids": [int(item[1]) for item in logprobs],
        "logprobs": [float(item[0]) for item in logprobs],
        "cached_tokens": int(meta.get("cached_tokens", 0)),
        "cached_tokens_details": meta.get("cached_tokens_details") or {},
        "prompt_tokens": int(meta.get("prompt_tokens", len(input_ids))),
        "completion_tokens": int(
            meta.get("completion_tokens", len(body.get("output_ids", [])))
        ),
    }


def _chat_result(body: dict[str, Any], latency_s: float) -> dict[str, Any]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = body.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "latency_s": latency_s,
        "text": content or reasoning,
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "cached_tokens": int(prompt_details.get("cached_tokens", 0)),
        "cached_tokens_details": prompt_details,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def sync_generate_text(
    base_url: str, model: str, prompt: str, output_len: int
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": output_len,
            "stop": ["\nQuestion:"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60 * 60,
    )
    latency = time.perf_counter() - started
    response.raise_for_status()
    return _chat_result(response.json(), latency)


def load_gsm8k(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(GSM8K_URL, timeout=120)
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def extract_last_number(text: str) -> str | None:
    marked = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    candidates = marked or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    try:
        return str(decimal.Decimal(candidates[-1].replace(",", "")).normalize())
    except decimal.InvalidOperation:
        return None


def gsm8k_prompt(examples: list[dict[str, str]], question: str) -> str:
    preamble = (
        "Solve each math problem carefully. End the answer with '#### ' followed "
        "by only the final numeric answer.\n\n"
    )
    demonstrations = "\n\n".join(
        f"Question: {item['question']}\nAnswer: {item['answer']}" for item in examples
    )
    return f"{preamble}{demonstrations}\n\nQuestion: {question}\nAnswer:"


async def run_accuracy_async(args: argparse.Namespace) -> dict[str, Any]:
    flush_cache(args.base_url)
    server_info = get_server_info(args.base_url)
    model = str(server_info["served_model_name"])
    metrics_before = metric_snapshot(args.base_url)
    memory_profile_before = memory_profile_snapshot(args.output)
    rows = load_gsm8k(args.dataset_path)
    if args.num_shots + args.num_questions > len(rows):
        raise ValueError("Requested more GSM8K rows than the dataset contains")
    examples = rows[: args.num_shots]
    samples = rows[args.num_shots : args.num_shots + args.num_questions]
    prompts = [gsm8k_prompt(examples, item["question"]) for item in samples]
    gold = [extract_last_number(item["answer"]) for item in samples]

    # Establish an exact greedy baseline, evict it under distinct L1 pressure,
    # then replay it. This makes the semantic benchmark also exercise the
    # implementation being evaluated instead of merely running with HiCache on.
    first_baseline = sync_generate_text(
        args.base_url, model, prompts[0], args.output_len
    )
    for index in range(args.pressure_requests):
        sync_generate(
            args.base_url,
            [1000 + index] * args.input_len,
            1,
            return_logprob=False,
        )
    metrics_after_pressure = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after_pressure = memory_profile_snapshot(args.output)
    torch_profile = start_torch_profile(args, "measured")
    first_replay = sync_generate_text(args.base_url, model, prompts[0], args.output_len)

    semaphore = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=60 * 60)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))

    async def generate_one(
        session: aiohttp.ClientSession, index: int
    ) -> dict[str, Any]:
        async with semaphore:
            started = time.perf_counter()
            try:
                async with session.post(
                    f"{args.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompts[index]}],
                        "temperature": 0,
                        "max_tokens": args.output_len,
                        "stop": ["\nQuestion:"],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ) as response:
                    response.raise_for_status()
                    body = await response.json()
                return {
                    "index": index,
                    "success": True,
                    **_chat_result(body, time.perf_counter() - started),
                }
            except Exception as exc:  # noqa: BLE001 - preserve request failure
                return {
                    "index": index,
                    "success": False,
                    "latency_s": time.perf_counter() - started,
                    "error": repr(exc),
                }

    measured_started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        remaining = await asyncio.gather(
            *(generate_one(session, index) for index in range(1, len(samples)))
        )
    measured_duration = time.perf_counter() - measured_started

    records = [
        {
            "index": 0,
            "success": True,
            **first_replay,
        },
        *remaining,
    ]
    correct = 0
    invalid = 0
    for record in records:
        if not record["success"]:
            record["prediction"] = None
            record["gold"] = gold[record["index"]]
            record["correct"] = False
            continue
        prediction = extract_last_number(record["text"])
        record["prediction"] = prediction
        record["gold"] = gold[record["index"]]
        record["correct"] = prediction == record["gold"]
        correct += int(record["correct"])
        invalid += int(prediction is None)

    metrics_after = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after = memory_profile_snapshot(args.output)
    torch_profile = stop_torch_profile(args, torch_profile)
    total_delta = metric_delta(metrics_before, metrics_after)
    errors: list[str] = []
    failed = sum(not record["success"] for record in records)
    if failed:
        errors.append(f"{failed} GSM8K requests failed")
    if total_delta["sglang:evicted_tokens_total"] <= 0:
        errors.append("L1 eviction did not occur in the semantic run")
    if args.expect_hicache and total_delta["sglang:load_back_tokens_total"] <= 0:
        errors.append("L2 load-back did not occur in the semantic run")
    replay_equal = first_baseline["text"] == first_replay["text"]

    payload = {
        "schema_version": 1,
        "kind": "accuracy",
        "task": "gsm8k",
        "variant": args.variant,
        "created_wall_time_ns": time.time_ns(),
        "args": {**vars(args), "dataset_path": str(args.dataset_path)},
        "dataset_url": GSM8K_URL,
        "server_info": server_info,
        "metrics_before": metrics_before,
        "metrics_after_pressure": metrics_after_pressure,
        "metrics_after": metrics_after,
        "total_metric_delta": total_delta,
        "memory_profile_pressure_delta": memory_profile_delta(
            memory_profile_before, memory_profile_after_pressure
        ),
        "memory_profile_total_delta": memory_profile_delta(
            memory_profile_before, memory_profile_after
        ),
        "torch_profile": torch_profile,
        "eviction_replay": {
            "baseline": first_baseline,
            "replay": first_replay,
            "output_ids_equal": replay_equal,
        },
        "summary": {
            "num_questions": len(records),
            "correct": correct,
            "accuracy": correct / len(records),
            "invalid": invalid,
            "invalid_rate": invalid / len(records),
            "failed": failed,
            "measured_remaining_duration_s": measured_duration,
        },
        "records": records,
        "validation": {"passed": not errors, "errors": errors},
    }
    write_json(args.output, payload)
    return payload


def compare_outputs(
    reference: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    reference_lps = reference["logprobs"]
    actual_lps = actual["logprobs"]
    finite = all(math.isfinite(x) for x in reference_lps + actual_lps)
    same_length = len(reference_lps) == len(actual_lps)
    differences = (
        [abs(a - b) for a, b in zip(reference_lps, actual_lps)] if same_length else []
    )
    return {
        "output_ids_equal": reference["output_ids"] == actual["output_ids"],
        "logprob_token_ids_equal": reference["logprob_token_ids"]
        == actual["logprob_token_ids"],
        "logprobs_finite": finite,
        "logprob_lengths_equal": same_length,
        "max_abs_logprob_diff": max(differences, default=0.0),
        "mean_abs_logprob_diff": statistics.fmean(differences) if differences else 0.0,
    }


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    flush_cache(args.base_url)
    server_info = get_server_info(args.base_url)
    metrics_before = metric_snapshot(args.base_url)
    memory_profile_before = memory_profile_snapshot(args.output)

    baselines: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    for index in range(args.pressure_requests):
        input_ids = [1000 + index] * args.input_len
        baseline = sync_generate(
            args.base_url,
            input_ids,
            args.output_len,
            return_logprob=True,
        )
        # A write-through HiCache entry is promoted on a cache hit. Prime each
        # prefix twice before later prefixes create L1 pressure; otherwise the
        # first eviction may legitimately discard a never-reused cold entry
        # without writing it to L2, making the parity workload test policy
        # rather than transfer correctness.
        promotion = sync_generate(
            args.base_url,
            input_ids,
            args.output_len,
            return_logprob=True,
        )
        baselines.append(baseline)
        promotions.append(
            {
                "index": index,
                "result": promotion,
                "comparison": compare_outputs(baseline, promotion),
            }
        )

    metrics_after_pressure = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after_pressure = memory_profile_snapshot(args.output)
    torch_profile = start_torch_profile(args, "measured")
    restored: dict[str, Any] | None = None
    replay_records: list[dict[str, Any]] = []
    # Priming above scans from oldest to newest. Replay newest entries first so
    # an L2 working set larger than capacity does not cyclically evict every
    # retained entry before the benchmark reaches it.
    for index in reversed(range(args.pressure_requests)):
        loadback_before = metric_snapshot(args.base_url)[
            "sglang:load_back_tokens_total"
        ]
        replay = sync_generate(
            args.base_url,
            [1000 + index] * args.input_len,
            args.output_len,
            return_logprob=True,
        )
        loadback_after = metric_snapshot(args.base_url)["sglang:load_back_tokens_total"]
        comparison = compare_outputs(baselines[index], replay)
        record = {
            "index": index,
            "loadback_delta_tokens": loadback_after - loadback_before,
            "reference": baselines[index],
            "replay": replay,
            "comparison": comparison,
        }
        replay_records.append(record)
        if record["loadback_delta_tokens"] > 0 and restored is None:
            restored = record
            break
        if not args.expect_hicache:
            restored = record
            break

    metrics_after = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after = memory_profile_snapshot(args.output)
    torch_profile = stop_torch_profile(args, torch_profile)
    total_delta = metric_delta(metrics_before, metrics_after)
    pressure_delta = metric_delta(metrics_before, metrics_after_pressure)
    errors: list[str] = []
    for promotion in promotions:
        comparison = promotion["comparison"]
        index = promotion["index"]
        if not comparison["output_ids_equal"]:
            errors.append(f"Output token IDs changed during promotion at {index}")
        if not comparison["logprob_token_ids_equal"]:
            errors.append(f"Logprob token IDs changed during promotion at {index}")
        if not comparison["logprobs_finite"]:
            errors.append(f"Non-finite promotion logprob observed at {index}")
        if comparison["max_abs_logprob_diff"] > args.logprob_atol:
            errors.append(
                "Promotion logprob drift exceeded tolerance at "
                f"{index}: {comparison['max_abs_logprob_diff']} > "
                f"{args.logprob_atol}"
            )
    if pressure_delta["sglang:evicted_tokens_total"] <= 0:
        errors.append("L1 eviction did not occur during pressure phase")
    if restored is None:
        errors.append("No replay record was produced")
    elif args.expect_hicache and restored["loadback_delta_tokens"] <= 0:
        errors.append("No replay was restored from L2")
    if restored is not None:
        comparison = restored["comparison"]
        if not comparison["output_ids_equal"]:
            errors.append("Output token IDs changed after eviction/replay")
        if not comparison["logprob_token_ids_equal"]:
            errors.append("Logprob token IDs changed after eviction/replay")
        if not comparison["logprobs_finite"]:
            errors.append("Non-finite logprob observed")
        if comparison["max_abs_logprob_diff"] > args.logprob_atol:
            errors.append(
                "Logprob drift exceeded tolerance: "
                f"{comparison['max_abs_logprob_diff']} > {args.logprob_atol}"
            )

    payload = {
        "schema_version": 1,
        "kind": "parity",
        "variant": args.variant,
        "created_wall_time_ns": time.time_ns(),
        "args": vars(args),
        "server_info": server_info,
        "metrics_before": metrics_before,
        "metrics_after_pressure": metrics_after_pressure,
        "metrics_after": metrics_after,
        "pressure_metric_delta": pressure_delta,
        "total_metric_delta": total_delta,
        "memory_profile_pressure_delta": memory_profile_delta(
            memory_profile_before, memory_profile_after_pressure
        ),
        "memory_profile_total_delta": memory_profile_delta(
            memory_profile_before, memory_profile_after
        ),
        "torch_profile": torch_profile,
        "promotions": promotions,
        "replays": replay_records,
        "restored": restored,
        "validation": {"passed": not errors, "errors": errors},
    }
    write_json(args.output, payload)
    return payload


@dataclass
class RequestRecord:
    request_id: int
    group_id: int
    round_id: int
    success: bool
    latency_s: float
    ttft_s: float | None
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int
    cached_tokens_device: int
    cached_tokens_host: int
    error: str | None


async def stream_generate(
    session: aiohttp.ClientSession,
    url: str,
    *,
    request_id: int,
    group_id: int,
    round_id: int,
    input_ids: list[int],
    output_len: int,
) -> RequestRecord:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
    }
    started = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0
    prompt_tokens = len(input_ids)
    cached_tokens = 0
    cached_details: dict[str, Any] = {}
    error: str | None = None
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = f"HTTP {response.status}: {(await response.text())[:500]}"
            else:
                while not response.content.at_eof():
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    meta = chunk.get("meta_info") or {}
                    current_completion = int(
                        meta.get("completion_tokens", completion_tokens)
                    )
                    if current_completion > 0 and ttft is None:
                        ttft = time.perf_counter() - started
                    completion_tokens = max(completion_tokens, current_completion)
                    prompt_tokens = int(meta.get("prompt_tokens", prompt_tokens))
                    cached_tokens = int(meta.get("cached_tokens", cached_tokens))
                    details = meta.get("cached_tokens_details")
                    if details:
                        cached_details = details
    except Exception as exc:  # noqa: BLE001 - preserve failure in raw output
        error = repr(exc)
    latency = time.perf_counter() - started
    return RequestRecord(
        request_id=request_id,
        group_id=group_id,
        round_id=round_id,
        success=error is None,
        latency_s=latency,
        ttft_s=ttft,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_device=int(cached_details.get("device", 0)),
        cached_tokens_host=int(cached_details.get("host", 0)),
        error=error,
    )


def make_workload(
    *,
    seed: int,
    input_len: int,
    groups: int,
    rounds: int,
    shared_ratio: float,
    group_order_start: int = 0,
    reverse_group_order: bool = False,
) -> tuple[list[list[int]], list[tuple[int, int, list[int]]]]:
    if not 0 < shared_ratio < 1:
        raise ValueError("shared_ratio must be between zero and one")
    if not 0 <= group_order_start < groups:
        raise ValueError("group_order_start must be in [0, groups)")
    rng = random.Random(seed)
    prefix_len = max(1, int(input_len * shared_ratio))
    suffix_len = input_len - prefix_len
    prefixes: list[list[int]] = []
    for group_id in range(groups):
        prefix = [1000 + group_id]
        prefix.extend(rng.randrange(2000, 200000) for _ in range(prefix_len - 1))
        prefixes.append(prefix)

    group_order = list(range(group_order_start, groups)) + list(
        range(group_order_start)
    )
    if reverse_group_order:
        group_order.reverse()
    schedule: list[tuple[int, int, list[int]]] = []
    for round_id in range(rounds):
        for group_id in group_order:
            prefix = prefixes[group_id]
            suffix = [rng.randrange(2000, 200000) for _ in range(suffix_len)]
            schedule.append((group_id, round_id, prefix + suffix))
    return prefixes, schedule


async def run_steady_async(args: argparse.Namespace) -> dict[str, Any]:
    flush_cache(args.base_url)
    server_info = get_server_info(args.base_url)
    metrics_before = metric_snapshot(args.base_url)
    memory_profile_before = memory_profile_snapshot(args.output)
    prefixes, schedule = make_workload(
        seed=args.seed,
        input_len=args.input_len,
        groups=args.groups,
        rounds=args.rounds,
        shared_ratio=args.shared_ratio,
        group_order_start=args.group_order_start,
        reverse_group_order=args.reverse_group_order,
    )

    prime_started = time.perf_counter()
    for _ in range(args.prime_repeats):
        for prefix in prefixes:
            sync_generate(
                args.base_url,
                prefix,
                args.prime_output_len,
                return_logprob=False,
            )
    prime_duration = time.perf_counter() - prime_started
    metrics_after_prime = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after_prime = memory_profile_snapshot(args.output)
    torch_profile = start_torch_profile(args, "measured")

    timeout = aiohttp.ClientTimeout(total=60 * 60)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))
    semaphore = asyncio.Semaphore(args.max_concurrency)
    url = f"{args.base_url}/generate"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(
            request_id: int, group_id: int, round_id: int, input_ids: list[int]
        ) -> RequestRecord:
            async with semaphore:
                return await stream_generate(
                    session,
                    url,
                    request_id=request_id,
                    group_id=group_id,
                    round_id=round_id,
                    input_ids=input_ids,
                    output_len=args.output_len,
                )

        measured_started = time.perf_counter()
        records = await asyncio.gather(
            *(
                one(request_id, group_id, round_id, input_ids)
                for request_id, (group_id, round_id, input_ids) in enumerate(schedule)
            )
        )
        measured_duration = time.perf_counter() - measured_started

    metrics_after = metric_snapshot(args.base_url)
    time.sleep(0.35)
    memory_profile_after = memory_profile_snapshot(args.output)
    torch_profile = stop_torch_profile(args, torch_profile)
    total_delta = metric_delta(metrics_before, metrics_after)
    measured_delta = metric_delta(metrics_after_prime, metrics_after)
    successful = [record for record in records if record.success]
    ttfts = [record.ttft_s for record in successful if record.ttft_s is not None]
    latencies = [record.latency_s for record in successful]
    tpot_values = [
        (record.latency_s - record.ttft_s) / (record.completion_tokens - 1)
        for record in successful
        if record.ttft_s is not None and record.completion_tokens > 1
    ]
    total_input = sum(record.prompt_tokens for record in successful)
    total_output = sum(record.completion_tokens for record in successful)
    summary = {
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "duration_s": measured_duration,
        "request_throughput": len(successful) / measured_duration,
        "input_token_throughput": total_input / measured_duration,
        "output_token_throughput": total_output / measured_duration,
        "total_token_throughput": (total_input + total_output) / measured_duration,
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) * 1000 if ttfts else None,
            "p50": percentile(ttfts, 50) * 1000 if ttfts else None,
            "p90": percentile(ttfts, 90) * 1000 if ttfts else None,
            "p95": percentile(ttfts, 95) * 1000 if ttfts else None,
            "p99": percentile(ttfts, 99) * 1000 if ttfts else None,
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpot_values) * 1000 if tpot_values else None,
            "p50": percentile(tpot_values, 50) * 1000 if tpot_values else None,
            "p90": percentile(tpot_values, 90) * 1000 if tpot_values else None,
            "p95": percentile(tpot_values, 95) * 1000 if tpot_values else None,
            "p99": percentile(tpot_values, 99) * 1000 if tpot_values else None,
        },
        "e2e_ms": {
            "mean": statistics.fmean(latencies) * 1000 if latencies else None,
            "p50": percentile(latencies, 50) * 1000 if latencies else None,
            "p95": percentile(latencies, 95) * 1000 if latencies else None,
            "p99": percentile(latencies, 99) * 1000 if latencies else None,
        },
        "cached_tokens": sum(record.cached_tokens for record in successful),
        "cached_tokens_device": sum(
            record.cached_tokens_device for record in successful
        ),
        "cached_tokens_host": sum(record.cached_tokens_host for record in successful),
    }

    errors: list[str] = []
    if summary["failed"]:
        errors.append(f"{summary['failed']} measured requests failed")
    if args.require_eviction and total_delta["sglang:evicted_tokens_total"] <= 0:
        errors.append("Required L1 eviction did not occur")
    if args.expect_hicache and args.require_loadback:
        if total_delta["sglang:load_back_tokens_total"] <= 0:
            errors.append("Required L2 load-back did not occur")
    if args.expect_hicache and args.require_backup:
        if total_delta["sglang:hicache_backup_tokens_total"] <= 0:
            errors.append("Required L2 backup did not occur")
    if args.forbid_dropped and total_delta["sglang:hicache_dropped_tokens_total"] > 0:
        errors.append("HiCache dropped tokens under host pressure")
    if args.expect_hicache and args.require_host_hit:
        host_evidence = (
            summary["cached_tokens_host"] > 0
            or total_delta["sglang:load_back_tokens_total"] > 0
        )
        if not host_evidence:
            errors.append("No host-cache hit evidence was observed")

    payload = {
        "schema_version": 1,
        "kind": "steady",
        "variant": args.variant,
        "created_wall_time_ns": time.time_ns(),
        "args": vars(args),
        "server_info": server_info,
        "prime_duration_s": prime_duration,
        "metrics_before": metrics_before,
        "metrics_after_prime": metrics_after_prime,
        "metrics_after": metrics_after,
        "total_metric_delta": total_delta,
        "measured_metric_delta": measured_delta,
        "memory_profile_total_delta": memory_profile_delta(
            memory_profile_before, memory_profile_after
        ),
        "memory_profile_measured_delta": memory_profile_delta(
            memory_profile_after_prime, memory_profile_after
        ),
        "torch_profile": torch_profile,
        "summary": summary,
        "requests": [asdict(record) for record in records],
        "validation": {"passed": not errors, "errors": errors},
    }
    write_json(args.output, payload)
    return payload


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, default=2)
    parser.add_argument("--expect-hicache", action="store_true")
    parser.add_argument("--torch-profile-steps", type=int, default=0)
    parser.add_argument("--torch-profile-output-dir", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parity = subparsers.add_parser("parity")
    add_common(parity)
    parity.add_argument("--pressure-requests", type=int, default=6)
    parity.add_argument("--logprob-atol", type=float, default=1e-2)

    accuracy = subparsers.add_parser("accuracy")
    add_common(accuracy)
    accuracy.add_argument("--pressure-requests", type=int, default=16)
    accuracy.add_argument("--num-questions", type=int, default=200)
    accuracy.add_argument("--num-shots", type=int, default=5)
    accuracy.add_argument("--max-concurrency", type=int, default=4)
    accuracy.add_argument(
        "--dataset-path",
        type=Path,
        default=Path.home() / ".cache/sglang-bench/gsm8k-test.jsonl",
    )

    steady = subparsers.add_parser("steady")
    add_common(steady)
    steady.add_argument("--seed", type=int, default=42)
    steady.add_argument("--groups", type=int, default=8)
    steady.add_argument(
        "--group-order-start",
        type=int,
        default=0,
        help="Rotate measured group order while leaving priming order unchanged.",
    )
    steady.add_argument(
        "--reverse-group-order",
        action="store_true",
        help=(
            "Replay the most recently primed prefixes first. This avoids a "
            "cyclic scan evicting every retained L2 entry before it is reused."
        ),
    )
    steady.add_argument("--rounds", type=int, default=4)
    steady.add_argument("--shared-ratio", type=float, default=0.95)
    steady.add_argument("--prime-output-len", type=int, default=1)
    steady.add_argument(
        "--prime-repeats",
        type=int,
        default=1,
        help=(
            "Number of serial priming passes. Write-through needs two passes "
            "so the second cache hit promotes each prefix to L2 before pressure."
        ),
    )
    steady.add_argument("--max-concurrency", type=int, default=4)
    steady.add_argument("--require-eviction", action="store_true")
    steady.add_argument("--require-loadback", action="store_true")
    steady.add_argument("--require-backup", action="store_true")
    steady.add_argument("--require-host-hit", action="store_true")
    steady.add_argument("--forbid-dropped", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "parity":
        payload = run_parity(args)
    elif args.command == "accuracy":
        payload = asyncio.run(run_accuracy_async(args))
    else:
        payload = asyncio.run(run_steady_async(args))
    print(
        json.dumps(
            {
                "output": args.output,
                "validation": payload["validation"],
                "summary": payload.get("summary"),
                "total_metric_delta": payload.get("total_metric_delta"),
            },
            indent=2,
        )
    )
    if not payload["validation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
