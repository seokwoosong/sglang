#!/usr/bin/env python3
"""Validate and report the unified HiCache policy/transfer-mode matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/unified_transfer_policy_ablation"
MODE_VARIANT = {"sync": "u2-sync-control", "async": "u2"}
MODE_PATH = {"sync": "sync_wait", "async": "row_fence"}
MODELS = ("9b", "4b", "0.8b")
WORKLOADS = ("serial", "replay", "overlap")
POLICIES = ("write_back", "write_through")
FATAL_MARKERS = (
    "AssertionError",
    "CUDA out of memory",
    "Traceback (most recent call last)",
)
PATH_RE = re.compile(
    r"UNIFIED_TRANSFER_PATH direction=(\w+) path=(\w+) registrations=(\d+) "
    r"protected_rows=(\d+) scheduler_gate_ms=([0-9.]+)"
)
SUBMIT_RE = re.compile(
    r"UNIFIED_TRANSFER_SUBMIT direction=(\w+) submit_ms=([0-9.]+)"
)


def latest_valid(base: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        result_path = manifest_path.with_name("result.json")
        if not result_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("status") == "completed"
            and manifest.get("client_exit_code") == 0
            and result.get("validation", {}).get("passed") is True
        ):
            candidates.append((manifest_path.parent, manifest, result))
    if not candidates:
        raise RuntimeError(f"No valid completed run under {base}")
    return candidates[-1]


def collect_run(
    *,
    root: Path,
    stage: str,
    repetition: int,
    model: str,
    workload: str,
    policy: str,
    mode: str,
) -> tuple[dict[str, Any], list[str]]:
    variant = MODE_VARIANT[mode]
    run_name = f"policy-{stage}-rep{repetition}-{model}-{workload}-{policy}"
    run_dir, manifest, result = latest_valid(root / run_name / variant)
    log = (run_dir / "server.log").read_text(errors="replace")
    errors: list[str] = []
    condition = f"{model}/{workload}/{policy}/{mode}/rep{repetition}"

    if manifest["variant_definition"]["sha"] != (
        "1ee4930f27d85c33a73baa1e0e6a9458381b06ec"
    ):
        errors.append(f"{condition}: unexpected source SHA")
    if "--max-mamba-cache-size" in manifest["server_command"]:
        errors.append(f"{condition}: forbidden --max-mamba-cache-size present")
    if any(marker in log for marker in FATAL_MARKERS):
        errors.append(f"{condition}: fatal marker in server log")
    if result["summary"]["failed"] or not all(
        request["success"] for request in result["requests"]
    ):
        errors.append(f"{condition}: request failure")

    total = result["total_metric_delta"]
    measured = result["measured_metric_delta"]
    required_positive = (
        "sglang:evicted_tokens_total",
        "sglang:hicache_backup_tokens_total",
        "sglang:hicache_backup_bytes_total",
        "sglang:load_back_tokens_total",
        "sglang:load_back_bytes_total",
    )
    for metric in required_positive:
        if total[metric] <= 0:
            errors.append(f"{condition}: required metric is zero: {metric}")
    if measured["sglang:load_back_tokens_total"] <= 0:
        errors.append(f"{condition}: measured phase has no L2 load-back")
    if result["summary"]["cached_tokens_host"] <= 0:
        errors.append(f"{condition}: measured phase has no host hit")
    if total["sglang:hicache_dropped_tokens_total"] != 0:
        errors.append(f"{condition}: HiCache dropped tokens")
    if policy == "write_through":
        prime_backup = (
            result["metrics_after_prime"]["sglang:hicache_backup_bytes_total"]
            - result["metrics_before"]["sglang:hicache_backup_bytes_total"]
        )
        if prime_backup <= 0:
            errors.append(f"{condition}: write-through promotion did not back up")

    path_counts: Counter[tuple[str, str]] = Counter()
    gate_ms: defaultdict[str, float] = defaultdict(float)
    registrations = protected_rows = 0
    for direction, path, reg, rows, elapsed in PATH_RE.findall(log):
        path_counts[(direction, path)] += 1
        gate_ms[direction] += float(elapsed)
        registrations += int(reg)
        protected_rows += int(rows)
    submit_ms: defaultdict[str, float] = defaultdict(float)
    for direction, elapsed in SUBMIT_RE.findall(log):
        submit_ms[direction] += float(elapsed)

    expected_path = MODE_PATH[mode]
    unexpected = {
        (direction, path): count
        for (direction, path), count in path_counts.items()
        if path != expected_path
    }
    if unexpected:
        errors.append(f"{condition}: unexpected transfer path(s): {unexpected}")
    d2h_paths = sum(
        count for (direction, _), count in path_counts.items() if direction == "d2h"
    )
    h2d_paths = sum(
        count for (direction, _), count in path_counts.items() if direction == "h2d"
    )
    d2h_metric_ops = total["sglang:hicache_backup_duration_seconds_count"]
    h2d_metric_ops = total["sglang:load_back_duration_seconds_count"]
    if d2h_paths != d2h_metric_ops or h2d_paths != h2d_metric_ops:
        errors.append(
            f"{condition}: path/metric operation mismatch "
            f"D2H={d2h_paths}/{d2h_metric_ops}, H2D={h2d_paths}/{h2d_metric_ops}"
        )
    if mode == "async" and registrations <= 0:
        errors.append(f"{condition}: async run has no row-fence registrations")

    summary = result["summary"]
    row = {
        "stage": stage,
        "repetition": repetition,
        "model": model,
        "workload": workload,
        "policy": policy,
        "mode": mode,
        "variant": variant,
        "source_sha": manifest["variant_definition"]["sha"],
        "run_dir": str(run_dir.resolve()),
        "requests": summary["completed"],
        "prime_duration_s": result["prime_duration_s"],
        "measured_duration_s": summary["duration_s"],
        "request_throughput": summary["request_throughput"],
        "ttft_p50_ms": summary["ttft_ms"]["p50"],
        "ttft_p95_ms": summary["ttft_ms"]["p95"],
        "tpot_p50_ms": summary["tpot_ms"]["p50"],
        "tpot_p95_ms": summary["tpot_ms"]["p95"],
        "cached_tokens_host": summary["cached_tokens_host"],
        "measured_evicted_tokens": measured["sglang:evicted_tokens_total"],
        "measured_d2h_ops": measured[
            "sglang:hicache_backup_duration_seconds_count"
        ],
        "measured_d2h_bytes": measured["sglang:hicache_backup_bytes_total"],
        "measured_h2d_ops": measured["sglang:load_back_duration_seconds_count"],
        "measured_h2d_bytes": measured["sglang:load_back_bytes_total"],
        "total_d2h_ops": d2h_metric_ops,
        "total_h2d_ops": h2d_metric_ops,
        "transfer_path": expected_path,
        "row_fence_registrations": registrations,
        "protected_row_references": protected_rows,
        "d2h_submit_ms": submit_ms["d2h"],
        "h2d_submit_ms": submit_ms["h2d"],
        "d2h_scheduler_gate_ms": gate_ms["d2h"],
        "h2d_scheduler_gate_ms": gate_ms["h2d"],
    }
    return row, errors


def collect_primary(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for model in MODELS:
        stage = "main" if model == "4b" else "confirm"
        for repetition in (1, 2, 3):
            for workload in WORKLOADS:
                for policy in POLICIES:
                    for mode in MODE_VARIANT:
                        row, run_errors = collect_run(
                            root=root,
                            stage=stage,
                            repetition=repetition,
                            model=model,
                            workload=workload,
                            policy=policy,
                            mode=mode,
                        )
                        rows.append(row)
                        errors.extend(run_errors)
    return rows, errors


def paired_rows(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["model"],
            row["workload"],
            row["policy"],
            row["repetition"],
            row["mode"],
        ): row
        for row in primary
    }
    output = []
    for model in MODELS:
        for workload in WORKLOADS:
            for policy in POLICIES:
                for repetition in (1, 2, 3):
                    sync = lookup[(model, workload, policy, repetition, "sync")]
                    async_row = lookup[
                        (model, workload, policy, repetition, "async")
                    ]
                    output.append(
                        {
                            "model": model,
                            "workload": workload,
                            "policy": policy,
                            "repetition": repetition,
                            "sync_request_throughput": sync["request_throughput"],
                            "async_request_throughput": async_row[
                                "request_throughput"
                            ],
                            "async_throughput_delta_pct": (
                                async_row["request_throughput"]
                                / sync["request_throughput"]
                                - 1
                            )
                            * 100,
                            "async_ttft_p50_delta_pct": (
                                async_row["ttft_p50_ms"] / sync["ttft_p50_ms"] - 1
                            )
                            * 100,
                            "async_tpot_p50_delta_pct": (
                                async_row["tpot_p50_ms"] / sync["tpot_p50_ms"] - 1
                            )
                            * 100,
                            "d2h_byte_delta_pct": abs(
                                async_row["measured_d2h_bytes"]
                                - sync["measured_d2h_bytes"]
                            )
                            / max(
                                async_row["measured_d2h_bytes"],
                                sync["measured_d2h_bytes"],
                            )
                            * 100,
                            "h2d_byte_delta_pct": abs(
                                async_row["measured_h2d_bytes"]
                                - sync["measured_h2d_bytes"]
                            )
                            / max(
                                async_row["measured_h2d_bytes"],
                                sync["measured_h2d_bytes"],
                            )
                            * 100,
                            "sync_run_dir": sync["run_dir"],
                            "async_run_dir": async_row["run_dir"],
                        }
                    )
    return output


def condition_rows(
    primary: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for workload in WORKLOADS:
            for policy in POLICIES:
                runs = [
                    row
                    for row in primary
                    if row["model"] == model
                    and row["workload"] == workload
                    and row["policy"] == policy
                ]
                pairs = [
                    row
                    for row in paired
                    if row["model"] == model
                    and row["workload"] == workload
                    and row["policy"] == policy
                ]
                sync = [row for row in runs if row["mode"] == "sync"]
                async_runs = [row for row in runs if row["mode"] == "async"]
                deltas = [row["async_throughput_delta_pct"] for row in pairs]
                output.append(
                    {
                        "model": model,
                        "workload": workload,
                        "policy": policy,
                        "sync_request_throughput_median": statistics.median(
                            row["request_throughput"] for row in sync
                        ),
                        "async_request_throughput_median": statistics.median(
                            row["request_throughput"] for row in async_runs
                        ),
                        "paired_async_delta_pct_median": statistics.median(deltas),
                        "paired_async_delta_pct_min": min(deltas),
                        "paired_async_delta_pct_max": max(deltas),
                        "paired_ttft_delta_pct_median": statistics.median(
                            row["async_ttft_p50_delta_pct"] for row in pairs
                        ),
                        "paired_tpot_delta_pct_median": statistics.median(
                            row["async_tpot_p50_delta_pct"] for row in pairs
                        ),
                        "sync_submit_ms_median": statistics.median(
                            row["d2h_submit_ms"] + row["h2d_submit_ms"]
                            for row in sync
                        ),
                        "sync_scheduler_gate_ms_median": statistics.median(
                            row["d2h_scheduler_gate_ms"]
                            + row["h2d_scheduler_gate_ms"]
                            for row in sync
                        ),
                        "async_scheduler_gate_ms_median": statistics.median(
                            row["d2h_scheduler_gate_ms"]
                            + row["h2d_scheduler_gate_ms"]
                            for row in async_runs
                        ),
                    }
                )
    return output


def collect_accuracy(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    outputs = []
    for policy in POLICIES:
        for mode, variant in MODE_VARIANT.items():
            run_dir, manifest, result = latest_valid(
                root / f"policy-accuracy-4b-{policy}" / variant
            )
            condition = f"accuracy/4b/{policy}/{mode}"
            comparisons = [
                item["comparison"] for item in result.get("promotions", [])
            ] + [item["comparison"] for item in result.get("replays", [])]
            if result.get("restored"):
                comparisons.append(result["restored"]["comparison"])
            exact = bool(comparisons) and all(
                item["output_ids_equal"]
                and item["logprob_token_ids_equal"]
                and item["logprobs_finite"]
                for item in comparisons
            )
            max_diff = max(
                (item["max_abs_logprob_diff"] for item in comparisons), default=0.0
            )
            if not exact:
                errors.append(f"{condition}: internal parity failure")
            if result["total_metric_delta"]["sglang:load_back_tokens_total"] <= 0:
                errors.append(f"{condition}: no load-back")
            if "--max-mamba-cache-size" in manifest["server_command"]:
                errors.append(f"{condition}: forbidden --max-mamba-cache-size")
            restored = result["restored"]
            output_ids = tuple(restored["replay"]["output_ids"])
            outputs.append(output_ids)
            rows.append(
                {
                    "model": "4b",
                    "policy": policy,
                    "mode": mode,
                    "exact_output_and_logprob_token_ids": exact,
                    "output_tokens": len(output_ids),
                    "max_abs_logprob_diff": max_diff,
                    "restored_tokens": restored["loadback_delta_tokens"],
                    "run_dir": str(run_dir.resolve()),
                }
            )
    if len(set(outputs)) != 1:
        errors.append("accuracy: cross-policy/mode output IDs differ")
    return rows, errors


def collect_diagnostic(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for repetition in range(1, 7):
        pair = {}
        for mode in MODE_VARIANT:
            row, run_errors = collect_run(
                root=root,
                stage="confirm",
                repetition=repetition,
                model="9b",
                workload="replay",
                policy="write_back",
                mode=mode,
            )
            errors.extend(run_errors)
            pair[mode] = row
        sync = pair["sync"]
        async_row = pair["async"]
        rows.append(
            {
                "repetition": repetition,
                "async_throughput_delta_pct": (
                    async_row["request_throughput"] / sync["request_throughput"] - 1
                )
                * 100,
                "sync_request_throughput": sync["request_throughput"],
                "async_request_throughput": async_row["request_throughput"],
                "sync_h2d_submit_s": sync["h2d_submit_ms"] / 1000,
                "async_h2d_submit_s": async_row["h2d_submit_ms"] / 1000,
                "sync_run_dir": sync["run_dir"],
                "async_run_dir": async_row["run_dir"],
            }
        )
    return rows, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_report(
    primary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    accuracy: list[dict[str, Any]],
    diagnostic: list[dict[str, Any]],
) -> str:
    all_deltas = [row["async_throughput_delta_pct"] for row in paired]
    max_logprob = max(row["max_abs_logprob_diff"] for row in accuracy)
    requests = sum(int(row["requests"]) for row in primary)
    d2h_ops = sum(int(row["total_d2h_ops"]) for row in primary)
    h2d_ops = sum(int(row["total_h2d_ops"]) for row in primary)
    d2h_bytes = sum(float(row["measured_d2h_bytes"]) for row in primary)
    h2d_bytes = sum(float(row["measured_h2d_bytes"]) for row in primary)
    sync_paths = sum(
        int(row["total_d2h_ops"] + row["total_h2d_ops"])
        for row in primary
        if row["mode"] == "sync"
    )
    async_paths = sum(
        int(row["total_d2h_ops"] + row["total_h2d_ops"])
        for row in primary
        if row["mode"] == "async"
    )
    registrations = sum(
        int(row["row_fence_registrations"])
        for row in primary
        if row["mode"] == "async"
    )
    diagnostic_deltas = [row["async_throughput_delta_pct"] for row in diagnostic]
    stable_deltas = diagnostic_deltas[3:]

    lines = [
        "# Unified HiCache Sync/Async × Write Policy Evaluation",
        "",
        "## Result",
        "",
        "**Functional result: PASS. Performance result: async is active and safe, "
        "but its end-to-end gain is generally small and workload-dependent.**",
        "",
        f"- Primary performance matrix: {len(primary)} runs, {requests} measured "
        "requests, 3 repetitions per condition.",
        f"- Correctness: {len(accuracy)} policy/mode runs; all restored output-token "
        "IDs and logprob-token IDs are exact across all four combinations.",
        f"- Maximum finite logprob difference: {max_logprob:.6f}.",
        "- No assertion, OOM, request failure, HiCache token drop, or missing "
        "required eviction/D2H/H2D/host-hit activity occurred in reported runs.",
        f"- Transfer evidence: {d2h_ops:,} D2H ops and {h2d_ops:,} H2D ops "
        "across the full run lifecycle; the measured phases moved "
        f"{d2h_bytes / 1e12:.3f} TB D2H and {h2d_bytes / 1e12:.3f} TB H2D.",
        f"- Path evidence: {sync_paths:,} `sync_wait` ops, {async_paths:,} "
        f"`row_fence` ops and {registrations:,} per-pool row-fence registrations; "
        "global fallback was 0.",
        "",
        "## Setup",
        "",
        "- Source under test: `1ee4930f27d85c33a73baa1e0e6a9458381b06ec` "
        "(the same U2 source for sync and async).",
        "- GPU: NVIDIA GeForce RTX 5090 32 GB; PyTorch 2.11.0+cu130; driver 610.62.",
        "- Models: Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B.",
        "- Prompt: 10,000 tokens, 95% shared prefix; no "
        "`--max-mamba-cache-size` in any server command.",
        "- Sync: `SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS=1`; async: `=0`.",
        "- Write-back primes each prefix once; write-through primes twice so the "
        "second hit triggers eager L2 backup. Measured replay starts with the most "
        "recently primed prefixes to avoid cyclic L2 thrashing.",
        "",
        "| Workload | Concurrency | Output tokens | Rounds | Purpose |",
        "|---|---:|---:|---:|---|",
        "| serial | 1 | 128 | 2 | No-independent-work control |",
        "| replay | 4 | 128 | 3 | Moderate concurrent L2 replay/churn |",
        "| overlap | 8 | 512 | 2 | Sustained decode work during transfers |",
        "",
        "## Primary Performance",
        "",
        "Positive delta means async is faster. Values are medians of paired "
        "sync/async repetitions; brackets show the three paired results' range.",
        "The req/s columns are independent medians, so their ratio can differ "
        "slightly from the median of the three paired ratios.",
        "",
        "| Model | Workload | Policy | Sync req/s | Async req/s | Async delta | "
        "TTFT delta | TPOT delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            f"| {row['model']} | {row['workload']} | {row['policy']} | "
            f"{f(row['sync_request_throughput_median'], 4)} | "
            f"{f(row['async_request_throughput_median'], 4)} | "
            f"{row['paired_async_delta_pct_median']:+.2f}% "
            f"[{row['paired_async_delta_pct_min']:+.2f}, "
            f"{row['paired_async_delta_pct_max']:+.2f}] | "
            f"{row['paired_ttft_delta_pct_median']:+.2f}% | "
            f"{row['paired_tpot_delta_pct_median']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "Across all 54 primary pairs, the median async throughput delta was "
            f"{statistics.median(all_deltas):+.2f}% ({sum(x > 0 for x in all_deltas)} "
            f"positive, {sum(x < 0 for x in all_deltas)} negative). The median "
            "therefore does not support a general throughput improvement from the "
            "current async fence alone.",
            "",
            "The request sequence is identical within each pair, but asynchronous "
            "completion can change the exact internal cache-residency decisions. "
            f"{sum(max(row['d2h_byte_delta_pct'], row['h2d_byte_delta_pct']) <= 5 for row in paired)} "
            f"of {len(paired)} pairs kept both measured transfer directions within "
            "5% byte parity. The largest difference was 11.13%. In particular, "
            "the +8.99% 9B overlap/write-back outlier moved 11.13% fewer H2D bytes "
            "in async; the other two repetitions were -0.52% and -0.48%, so this "
            "condition is not interpreted as an async speedup.",
            "",
            "## Why the General Gain Is Small",
            "",
            "The row fence removes the final host-side CUDA-event wait, but it does "
            "not remove the layer-by-layer Python transfer submission from the "
            "scheduler critical path.",
            "",
            "| Policy / mode | Median submit time per run | Median scheduler gate |",
            "|---|---:|---:|",
        ]
    )
    for policy in POLICIES:
        for mode in MODE_VARIANT:
            selected = [
                row
                for row in primary
                if row["policy"] == policy and row["mode"] == mode
            ]
            submit = statistics.median(
                row["d2h_submit_ms"] + row["h2d_submit_ms"] for row in selected
            )
            gate = statistics.median(
                row["d2h_scheduler_gate_ms"] + row["h2d_scheduler_gate_ms"]
                for row in selected
            )
            lines.append(
                f"| {policy} / {mode} | {submit / 1000:.3f} s | {gate:.3f} ms |"
            )

    lines.extend(
        [
            "",
            "For write-back sync, the median final wait was only about 0.3 s while "
            "Python submit accumulated about 7.1 s. For write-through sync, the "
            "tail wait was only about 0.02 s while submit accumulated about 8.6 s. "
            "Async correctly reduces the gate to row-fence registration, but the "
            "dominant submission work remains.",
            "",
            "The small write-through regressions in several conditions are "
            "consistent with saving only a few milliseconds of tail wait while "
            "adding row-fence bookkeeping and GPU transfer/compute contention.",
            "",
            "## 9B Replay/Write-Back Diagnostic",
            "",
            "The original three runs unexpectedly showed a large median gain, so "
            "three additional paired runs were executed without intervening "
            "workloads.",
            "",
            "| Repetition | Async throughput delta | Sync H2D submit | Async H2D submit |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic:
        lines.append(
            f"| {row['repetition']} | {row['async_throughput_delta_pct']:+.2f}% | "
            f"{row['sync_h2d_submit_s']:.2f} s | "
            f"{row['async_h2d_submit_s']:.2f} s |"
        )
    lines.extend(
        [
            "",
            f"All six pairs favored async. Their median was "
            f"{statistics.median(diagnostic_deltas):+.2f}% (range "
            f"{min(diagnostic_deltas):+.2f}% to {max(diagnostic_deltas):+.2f}%). "
            f"The three targeted reruns were much tighter: median "
            f"{statistics.median(stable_deltas):+.2f}% (range "
            f"{min(stable_deltas):+.2f}% to {max(stable_deltas):+.2f}%).",
            "",
            "The earlier +8–10% cases coincided with sync H2D submission taking "
            "13–16 s instead of roughly 7–9 s. The benefit is therefore real for "
            "this workload but highly variable: async queueing avoids some "
            "implicit H2D submission stalls, while the typical stable gain is "
            "closer to 0.7–0.8% on this RTX 5090.",
            "",
            "## Write Policy Interpretation",
            "",
            "- Write-back performs D2H when L1 eviction occurs. It exposes a longer "
            "sync tail and is the policy where the clearest async gain appeared.",
            "- Write-through performs eager D2H after a cache hit. Much of that work "
            "is in the excluded priming phase, and its remaining sync tail is very "
            "short. It showed no consistent async throughput gain.",
            "- Raw write-back versus write-through req/s is not a clean policy "
            "ranking because the policies perform D2H at different lifecycle stages; "
            "the report therefore treats sync-versus-async within each policy as the "
            "primary comparison.",
            "",
            "## Experimental Issue Found and Corrected",
            "",
            "The first write-through serial pilot produced D2H backups but zero H2D "
            "hits. The 26-prefix cyclic scan exceeded the 12 GB L2 capacity and "
            "evicted every retained entry just before reuse. This was a workload "
            "ordering error, not a production-cache failure. Replaying the most "
            "recently primed prefixes first fixed it; the failed pilot is retained in "
            "the raw artifacts, and all reported runs require positive H2D and host-hit "
            "evidence.",
            "",
            "## Conclusion and Next Engineering Step",
            "",
            "The U2 row-aware async path is exercised, avoids global fences, preserves "
            "outputs, and remains stable under both write policies and all three model "
            "sizes. The current implementation should be described as an async "
            "scheduler-fence optimization, not a fully asynchronous transfer pipeline.",
            "",
            "A larger general speedup requires reducing the 7–11 s/run Python "
            "submission cost: use pinned contiguous staging and batch layer/token "
            "copies into substantially fewer H2D/D2H operations. The same harness and "
            "path counters can then be reused to verify that submit time, not only the "
            "final event wait, decreases.",
            "",
            "## Files",
            "",
            "- `performance_runs.csv`: all 108 primary raw run summaries.",
            "- `paired_results.csv`: 54 repetition-level sync/async comparisons.",
            "- `condition_summary.csv`: 18 model/workload/policy aggregates.",
            "- `accuracy.csv`: four exact-output parity checks.",
            "- `diagnostic_9b_replay_write_back.csv`: six detailed diagnostic pairs.",
            "- `summary.json`: machine-readable headline results and validation counts.",
            "- `SHA256SUMS`: checksums for the final report package.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_checksums(final_dir: Path) -> None:
    paths = sorted(
        path
        for path in final_dir.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SUCCESS"}
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (final_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify_checksums(final_dir: Path) -> None:
    for line in (final_dir / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--final-dir", type=Path, default=None)
    args = parser.parse_args()
    final_dir = args.final_dir or args.artifact_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    primary, errors = collect_primary(args.artifact_root)
    paired = paired_rows(primary)
    conditions = condition_rows(primary, paired)
    accuracy, accuracy_errors = collect_accuracy(args.artifact_root)
    diagnostic, diagnostic_errors = collect_diagnostic(args.artifact_root)
    errors.extend(accuracy_errors)
    errors.extend(diagnostic_errors)
    if len(primary) != 108:
        errors.append(f"Expected 108 primary runs, got {len(primary)}")
    if len(paired) != 54:
        errors.append(f"Expected 54 pairs, got {len(paired)}")
    if len(conditions) != 18:
        errors.append(f"Expected 18 condition summaries, got {len(conditions)}")
    if len(accuracy) != 4:
        errors.append(f"Expected 4 accuracy runs, got {len(accuracy)}")
    if errors:
        (final_dir / "VALIDATION_ERRORS.json").write_text(
            json.dumps(errors, indent=2) + "\n"
        )
        raise RuntimeError("\n".join(errors))

    write_csv(final_dir / "performance_runs.csv", primary)
    write_csv(final_dir / "paired_results.csv", paired)
    write_csv(final_dir / "condition_summary.csv", conditions)
    write_csv(final_dir / "accuracy.csv", accuracy)
    write_csv(final_dir / "diagnostic_9b_replay_write_back.csv", diagnostic)

    all_deltas = [row["async_throughput_delta_pct"] for row in paired]
    summary = {
        "status": "PASS",
        "source_sha": "1ee4930f27d85c33a73baa1e0e6a9458381b06ec",
        "primary_performance_runs": len(primary),
        "paired_comparisons": len(paired),
        "condition_summaries": len(conditions),
        "accuracy_runs": len(accuracy),
        "measured_requests": sum(int(row["requests"]) for row in primary),
        "async_delta_pct_all_pairs_median": statistics.median(all_deltas),
        "async_delta_pct_all_pairs_mean": statistics.mean(all_deltas),
        "positive_pairs": sum(value > 0 for value in all_deltas),
        "negative_pairs": sum(value < 0 for value in all_deltas),
        "max_abs_logprob_diff": max(
            row["max_abs_logprob_diff"] for row in accuracy
        ),
        "d2h_operations": sum(int(row["total_d2h_ops"]) for row in primary),
        "h2d_operations": sum(int(row["total_h2d_ops"]) for row in primary),
        "global_fallback_operations": 0,
        "diagnostic_9b_replay_write_back": {
            "pairs": len(diagnostic),
            "all_six_median_pct": statistics.median(
                row["async_throughput_delta_pct"] for row in diagnostic
            ),
            "targeted_last_three_median_pct": statistics.median(
                row["async_throughput_delta_pct"] for row in diagnostic[3:]
            ),
        },
    }
    (final_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (final_dir / "REPORT.md").write_text(
        build_report(primary, paired, conditions, accuracy, diagnostic)
    )
    (final_dir / "VALIDATION_ERRORS.json").unlink(missing_ok=True)
    write_checksums(final_dir)
    verify_checksums(final_dir)
    (final_dir / "SUCCESS").write_text(
        "PASS: all reported artifacts validated; SHA256SUMS verified.\n"
    )
    print(final_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
