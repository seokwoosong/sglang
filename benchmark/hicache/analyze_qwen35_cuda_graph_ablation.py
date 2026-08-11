#!/usr/bin/env python3
"""Audit and summarize the Qwen3.5 CUDA graph ablation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/qwen35_unified_hicache_4way"
DEFAULT_VARIANTS = ("eval-s0", "eval-s1", "eval-u0", "eval-u3")
KNOWN_VARIANTS = DEFAULT_VARIANTS + (
    "post-s0",
    "post-s1",
    "post-u0",
    "post-u3",
)
NO_HICACHE_VARIANTS = frozenset(("eval-s0", "eval-u0", "post-s0", "post-u0"))
WORKLOADS = ("short-3k", "middle-10k", "long-50k")
MODES = ("disabled", "enabled")
GRAPH_RUN_RE = re.compile(
    r"graph-r(?P<repetition>\d+)-(?P<model>[^-]+)-"
    r"(?P<workload>short-3k|middle-10k|long-50k)-p(?P<page_size>\d+)-"
    r"(?P<policy>none|write_back)-cg-(?P<mode>enabled|disabled)"
)
CAPTURE_RE = re.compile(
    r"Capture target (prefill|decode) CUDA graph end\. "
    r"elapsed=([0-9.]+) s, mem usage=([0-9.]+) GB"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_results(root: Path, prefix: str) -> list[Path]:
    selected: list[Path] = []
    variant_dirs = list(root.glob(f"{prefix}*/eval-*"))
    variant_dirs.extend(root.glob(f"{prefix}*/post-*"))
    for variant_dir in sorted(variant_dirs):
        candidates = sorted(variant_dir.glob("*/result.json"), reverse=True)
        completed = [
            path
            for path in candidates
            if load_json(path.parent / "manifest.json").get("status") == "completed"
        ]
        if completed:
            selected.append(completed[0])
        elif candidates:
            selected.append(candidates[0])
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def analyze_performance(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for result_path in latest_results(root, "graph-r"):
        run_name = result_path.parents[2].name
        match = GRAPH_RUN_RE.fullmatch(run_name)
        if match is None:
            errors.append(f"Unrecognized graph run name: {run_name}")
            continue
        result = load_json(result_path)
        manifest = load_json(result_path.parent / "manifest.json")
        server_log = (result_path.parent / "server.log").read_text(errors="replace")
        summary = result["summary"]
        metrics = result["total_metric_delta"]
        server_info = result["server_info"]
        captures = {
            phase: {"seconds": float(seconds), "memory_gb": float(memory_gb)}
            for phase, seconds, memory_gb in CAPTURE_RE.findall(server_log)
        }
        graph_true = server_log.count("cuda graph: True")
        graph_false = server_log.count("cuda graph: False")
        graph_total = graph_true + graph_false
        row = {
            "repetition": int(match["repetition"]),
            "model": match["model"],
            "workload": match["workload"],
            "page_size": int(match["page_size"]),
            "policy": match["policy"],
            "variant": result["variant"],
            "cuda_graph_mode": match["mode"],
            "status": manifest["status"],
            "validation_passed": result["validation"]["passed"],
            "failed_requests": summary["failed"],
            "total_token_throughput": summary["total_token_throughput"],
            "input_token_throughput": summary["input_token_throughput"],
            "output_token_throughput": summary["output_token_throughput"],
            "ttft_mean_ms": summary["ttft_ms"]["mean"],
            "tpot_mean_ms": summary["tpot_ms"]["mean"],
            "measured_duration_s": summary["duration_s"],
            "startup_s": (
                manifest["server_ready_wall_time_ns"] - manifest["created_wall_time_ns"]
            )
            / 1e9,
            "graph_true_batches": graph_true,
            "graph_false_batches": graph_false,
            "graph_hit_percent": 100.0 * graph_true / graph_total,
            "prefill_capture_s": captures.get("prefill", {}).get("seconds", 0.0),
            "decode_capture_s": captures.get("decode", {}).get("seconds", 0.0),
            "capture_memory_gb": sum(
                capture["memory_gb"] for capture in captures.values()
            ),
            "evicted_tokens": metrics["sglang:evicted_tokens_total"],
            "backup_tokens": metrics["sglang:hicache_backup_tokens_total"],
            "load_back_tokens": metrics["sglang:load_back_tokens_total"],
            "dropped_tokens": metrics["sglang:hicache_dropped_tokens_total"],
            "result_path": str(result_path.relative_to(REPO_ROOT)),
        }
        rows.append(row)

        label = f"{run_name}/{result['variant']}"
        if manifest["status"] != "completed" or not result["validation"]["passed"]:
            errors.append(f"Run did not pass: {label}")
        if summary["failed"] or row["dropped_tokens"]:
            errors.append(f"Request failure or dropped token: {label}")
        if match["mode"] == "enabled":
            if (
                graph_true == 0
                or set(captures) != {"prefill", "decode"}
                or server_info["cuda_graph_backend_prefill"] != "breakable"
                or server_info["cuda_graph_backend_decode"] != "full"
            ):
                errors.append(f"CUDA graph ON audit failed: {label}")
        elif (
            graph_true != 0
            or captures
            or server_info["cuda_graph_backend_prefill"] != "disabled"
            or server_info["cuda_graph_backend_decode"] != "disabled"
        ):
            errors.append(f"CUDA graph OFF audit failed: {label}")
        if result["variant"] not in NO_HICACHE_VARIANTS and (
            row["backup_tokens"] <= 0 or row["load_back_tokens"] <= 0
        ):
            errors.append(f"HiCache transfer was not exercised: {label}")
    return rows, errors


def aggregate_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["workload"], row["cuda_graph_mode"])].append(row)
    output: list[dict[str, Any]] = []
    for (variant, workload, mode), group in sorted(grouped.items()):
        throughputs = [float(row["total_token_throughput"]) for row in group]
        throughput_mean = statistics.fmean(throughputs)
        throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        output.append(
            {
                "variant": variant,
                "workload": workload,
                "cuda_graph_mode": mode,
                "runs": len(group),
                "throughput_mean": throughput_mean,
                "throughput_std": throughput_std,
                "throughput_cv_percent": 100.0 * throughput_std / throughput_mean,
                "ttft_mean_ms": mean(group, "ttft_mean_ms"),
                "tpot_mean_ms": mean(group, "tpot_mean_ms"),
                "startup_mean_s": mean(group, "startup_s"),
                "graph_hit_mean_percent": mean(group, "graph_hit_percent"),
                "capture_mean_s": mean(group, "prefill_capture_s")
                + mean(group, "decode_capture_s"),
                "capture_memory_mean_gb": mean(group, "capture_memory_gb"),
                "evicted_tokens_mean": mean(group, "evicted_tokens"),
                "backup_tokens_mean": mean(group, "backup_tokens"),
                "load_back_tokens_mean": mean(group, "load_back_tokens"),
            }
        )
    return output


def paired_speedups(
    rows: list[dict[str, Any]], variants: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    indexed = {
        (
            row["variant"],
            row["workload"],
            row["repetition"],
            row["cuda_graph_mode"],
        ): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    errors: list[str] = []
    repetitions = sorted({int(row["repetition"]) for row in rows})
    for variant in variants:
        for workload in WORKLOADS:
            for repetition in repetitions:
                disabled = indexed.get((variant, workload, repetition, "disabled"))
                enabled = indexed.get((variant, workload, repetition, "enabled"))
                if disabled is None or enabled is None:
                    errors.append(
                        f"Missing graph pair: {variant}/{workload}/r{repetition}"
                    )
                    continue
                pairs.append(
                    {
                        "variant": variant,
                        "workload": workload,
                        "repetition": repetition,
                        "throughput_disabled": disabled["total_token_throughput"],
                        "throughput_enabled": enabled["total_token_throughput"],
                        "throughput_speedup": enabled["total_token_throughput"]
                        / disabled["total_token_throughput"],
                        "ttft_disabled_ms": disabled["ttft_mean_ms"],
                        "ttft_enabled_ms": enabled["ttft_mean_ms"],
                        "tpot_disabled_ms": disabled["tpot_mean_ms"],
                        "tpot_enabled_ms": enabled["tpot_mean_ms"],
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["variant"], pair["workload"])].append(pair)
    summaries: list[dict[str, Any]] = []
    for (variant, workload), group in sorted(grouped.items()):
        speedups = [float(row["throughput_speedup"]) for row in group]
        summaries.append(
            {
                "variant": variant,
                "workload": workload,
                "pairs": len(group),
                "throughput_speedup_mean": statistics.fmean(speedups),
                "throughput_speedup_std": (
                    statistics.stdev(speedups) if len(speedups) > 1 else 0.0
                ),
                "throughput_gain_percent": 100.0 * (statistics.fmean(speedups) - 1.0),
                "ttft_change_percent": 100.0
                * (
                    mean(group, "ttft_enabled_ms") / mean(group, "ttft_disabled_ms") - 1
                ),
                "tpot_change_percent": 100.0
                * (
                    mean(group, "tpot_enabled_ms") / mean(group, "tpot_disabled_ms") - 1
                ),
            }
        )
    return pairs, summaries, errors


def analyze_parity(
    root: Path, variants: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fingerprints: dict[tuple[str, str], tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for result_path in latest_results(root, "graph-parity-r"):
        result = load_json(result_path)
        manifest = load_json(result_path.parent / "manifest.json")
        server_log = (result_path.parent / "server.log").read_text(errors="replace")
        mode = manifest["arguments"]["cuda_graph_mode"]
        comparisons = [item["comparison"] for item in result["promotions"]]
        if result.get("restored"):
            comparisons.append(result["restored"]["comparison"])
        promotions = sorted(result["promotions"], key=lambda item: item["index"])
        fingerprint = (
            tuple(
                (int(item["index"]), tuple(item["result"]["output_ids"]))
                for item in promotions
            ),
            tuple(
                (
                    int(item["index"]),
                    tuple(item["result"]["logprob_token_ids"]),
                )
                for item in promotions
            ),
        )
        fingerprints[(result["variant"], mode)] = fingerprint
        row = {
            "variant": result["variant"],
            "cuda_graph_mode": mode,
            "status": manifest["status"],
            "validation_passed": result["validation"]["passed"],
            "comparisons": len(comparisons),
            "output_ids_equal": all(
                comparison["output_ids_equal"] for comparison in comparisons
            ),
            "logprob_token_ids_equal": all(
                comparison["logprob_token_ids_equal"] for comparison in comparisons
            ),
            "cross_mode_output_ids_equal": None,
            "cross_mode_logprob_token_ids_equal": None,
            "cross_variant_output_ids_equal": None,
            "cross_variant_logprob_token_ids_equal": None,
            "max_abs_logprob_diff": max(
                comparison["max_abs_logprob_diff"] for comparison in comparisons
            ),
            "restored_load_back_tokens": (
                result["restored"]["loadback_delta_tokens"]
                if result.get("restored")
                else 0.0
            ),
            "graph_true_batches": server_log.count("cuda graph: True"),
            "graph_false_batches": server_log.count("cuda graph: False"),
            "result_path": str(result_path.relative_to(REPO_ROOT)),
        }
        rows.append(row)
        rows_by_key[(result["variant"], mode)] = row
        if (
            manifest["status"] != "completed"
            or not result["validation"]["passed"]
            or not row["output_ids_equal"]
            or not row["logprob_token_ids_equal"]
        ):
            errors.append(f"Parity failed: {result['variant']}/{mode}")
    for variant in variants:
        enabled = fingerprints.get((variant, "enabled"))
        disabled = fingerprints.get((variant, "disabled"))
        if enabled is None or disabled is None:
            errors.append(f"Missing cross-mode parity result: {variant}")
            continue
        output_equal = enabled[0] == disabled[0]
        token_equal = enabled[1] == disabled[1]
        for mode in MODES:
            row = rows_by_key[(variant, mode)]
            row["cross_mode_output_ids_equal"] = output_equal
            row["cross_mode_logprob_token_ids_equal"] = token_equal
        if not output_equal:
            errors.append(f"Cross-mode output IDs differ: {variant}")
        if not token_equal:
            errors.append(f"Cross-mode logprob token IDs differ: {variant}")

    canonical_variant = "eval-s0" if "eval-s0" in variants else "post-s0"
    canonical = fingerprints.get((canonical_variant, "disabled"))
    if canonical is not None:
        for key, fingerprint in fingerprints.items():
            row = rows_by_key[key]
            row["cross_variant_output_ids_equal"] = fingerprint[0] == canonical[0]
            row["cross_variant_logprob_token_ids_equal"] = (
                fingerprint[1] == canonical[1]
            )
            if not row["cross_variant_output_ids_equal"]:
                errors.append(f"Cross-variant output IDs differ: {key[0]}/{key[1]}")
            if not row["cross_variant_logprob_token_ids_equal"]:
                errors.append(
                    f"Cross-variant logprob token IDs differ: {key[0]}/{key[1]}"
                )
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--expected-repetitions", type=int, default=3)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=KNOWN_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    args = parser.parse_args()
    variants = tuple(args.variants)
    root = args.artifact_root.resolve()
    summary_dir = root / "summary"

    runs, performance_errors = analyze_performance(root)
    aggregates = aggregate_performance(runs)
    pairs, speedups, pair_errors = paired_speedups(runs, variants)
    parity, parity_errors = analyze_parity(root, variants)
    errors = performance_errors + pair_errors + parity_errors
    expected_performance_runs = (
        len(variants) * len(WORKLOADS) * len(MODES) * args.expected_repetitions
    )
    expected_parity_runs = len(variants) * len(MODES)
    if len(runs) != expected_performance_runs:
        errors.append(
            f"Expected {expected_performance_runs} performance runs, got {len(runs)}"
        )
    if len(parity) != expected_parity_runs:
        errors.append(f"Expected {expected_parity_runs} parity runs, got {len(parity)}")

    write_csv(summary_dir / "run_summary.csv", runs)
    write_csv(summary_dir / "aggregate_summary.csv", aggregates)
    write_csv(summary_dir / "paired_runs.csv", pairs)
    write_csv(summary_dir / "paired_speedup_summary.csv", speedups)
    write_csv(summary_dir / "parity_summary.csv", parity)
    audit = {
        "schema_version": 1,
        "performance_runs": len(runs),
        "expected_performance_runs": expected_performance_runs,
        "parity_runs": len(parity),
        "expected_parity_runs": expected_parity_runs,
        "passed": not errors,
        "errors": errors,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(f"performance runs: {len(runs)}")
    print(f"parity runs: {len(parity)}")
    for row in speedups:
        print(
            f"{row['variant']} {row['workload']}: "
            f"{row['throughput_speedup_mean']:.3f}x "
            f"({row['throughput_gain_percent']:+.1f}%)"
        )
    print(f"audit: {'PASS' if not errors else 'FAIL'}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
