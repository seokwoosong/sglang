#!/usr/bin/env python3
"""Compare patched unified-memory runs with pre-patch unified and static results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PATCHED_SHA = "8de3268353c0fbbfaa380003aa808dae38c5ee16"
PREPATCH_SHA = "92eb0737857f4fef0ba46e19bed5cb9bc45816f9"
HOMOGENEOUS_RE = re.compile(
    r"^patched-final-(0\.8b|4b)-unified-triton-(short|middle|long)-"
    r"reuse(\d{3})-r05-rep(\d+)$"
)
MIXED_RE = re.compile(
    r"^patched-mixed-intermixed-(0\.8b|4b)-unified-triton-"
    r"reuse(\d{3})-r05-rep(\d+)$"
)


def pct_delta(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def cv_pct(values: list[float]) -> float:
    if len(values) < 2 or statistics.mean(values) == 0:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_patched_runs(root: Path) -> list[dict]:
    runs = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_name = manifest["arguments"]["run_name"]
        homogeneous_match = HOMOGENEOUS_RE.match(run_name)
        mixed_match = MIXED_RE.match(run_name)
        if not homogeneous_match and not mixed_match:
            continue
        result_path = manifest_path.with_name("result.json")
        if manifest.get("status") != "completed" or not result_path.exists():
            raise RuntimeError(f"Incomplete patched run: {manifest_path.parent}")
        if manifest["variant_definition"].get("sha") != PATCHED_SHA:
            raise RuntimeError(f"Unexpected SHA: {manifest_path}")
        result = json.loads(result_path.read_text())
        if not result.get("validation", {}).get("passed"):
            raise RuntimeError(f"Validation failed: {result_path}")
        if result["summary"].get("failed", 0) != 0:
            raise RuntimeError(f"Failed requests: {result_path}")

        if homogeneous_match:
            model, workload, reuse, repetition = homogeneous_match.groups()
            scenario = "homogeneous"
        else:
            model, reuse, repetition = mixed_match.groups()
            workload = "interleaved-mixed"
            scenario = "mixed"
        summary = result["summary"]
        metric_delta = result["total_metric_delta"]
        runs.append(
            {
                "scenario": scenario,
                "model": model,
                "workload": workload,
                "prefix_reuse_pct": int(reuse),
                "repetition": int(repetition),
                "total_token_throughput": float(summary["total_token_throughput"]),
                "request_throughput": float(summary["request_throughput"]),
                "duration_s": float(summary["duration_s"]),
                "completed": int(summary["completed"]),
                "evicted_tokens": float(
                    metric_delta.get("sglang:evicted_tokens_total", 0.0)
                ),
                "backup_tokens": float(
                    metric_delta.get("sglang:hicache_backup_tokens_total", 0.0)
                ),
                "load_back_tokens": float(
                    metric_delta.get("sglang:load_back_tokens_total", 0.0)
                ),
                "source_sha": PATCHED_SHA,
                "run_dir": str(manifest_path.parent.resolve()),
            }
        )
    if len(runs) != 72:
        raise RuntimeError(f"Expected 72 patched runs, found {len(runs)}")
    return runs


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_patched(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        key = (
            run["scenario"],
            run["model"],
            run["workload"],
            run["prefix_reuse_pct"],
        )
        grouped[key].append(run)
    rows = []
    for key, group in sorted(grouped.items()):
        values = [run["total_token_throughput"] for run in group]
        request_values = [run["request_throughput"] for run in group]
        rows.append(
            {
                "scenario": key[0],
                "model": key[1],
                "workload": key[2],
                "prefix_reuse_pct": key[3],
                "repetitions": len(group),
                "total_token_throughput_median": statistics.median(values),
                "total_token_throughput_min": min(values),
                "total_token_throughput_max": max(values),
                "total_token_throughput_cv_pct": cv_pct(values),
                "request_throughput_median": statistics.median(request_values),
            }
        )
    return rows


def paired_patch_rows(patched_runs: list[dict], old_raw: list[dict]) -> list[dict]:
    old_by_key = {}
    for row in old_raw:
        if row["variant"] != "unified-triton" or row["source_sha"] != PREPATCH_SHA:
            continue
        if row["validation_passed"] != "True":
            continue
        if row["stage"] == "final" and row["kind"] == "steady":
            scenario = "homogeneous"
        elif row["stage"] == "mixed-final" and row["kind"] == "interleaved-mixed":
            scenario = "mixed"
        else:
            continue
        key = (
            scenario,
            row["model"],
            row["workload"],
            int(row["prefix_reuse_pct"]),
            int(row["repetition"]),
        )
        old_by_key[key] = row

    paired = []
    for run in patched_runs:
        key = (
            run["scenario"],
            run["model"],
            run["workload"],
            run["prefix_reuse_pct"],
            run["repetition"],
        )
        old = old_by_key.get(key)
        if old is None:
            continue
        old_throughput = float(old["total_token_throughput"])
        new_throughput = run["total_token_throughput"]
        old_evicted = float(old["evicted_tokens"])
        old_backup = float(old["backup_tokens"])
        old_load_back = float(old["load_back_tokens"])
        paired.append(
            {
                "scenario": run["scenario"],
                "model": run["model"],
                "workload": run["workload"],
                "prefix_reuse_pct": run["prefix_reuse_pct"],
                "repetition": run["repetition"],
                "prepatch_total_token_throughput": old_throughput,
                "patched_total_token_throughput": new_throughput,
                "patched_vs_prepatch_pct": pct_delta(new_throughput, old_throughput),
                "prepatch_evicted_tokens": old_evicted,
                "patched_evicted_tokens": run["evicted_tokens"],
                "evicted_tokens_delta_pct": (
                    pct_delta(run["evicted_tokens"], old_evicted)
                    if old_evicted
                    else ""
                ),
                "prepatch_backup_tokens": old_backup,
                "patched_backup_tokens": run["backup_tokens"],
                "backup_tokens_delta_pct": (
                    pct_delta(run["backup_tokens"], old_backup) if old_backup else ""
                ),
                "prepatch_load_back_tokens": old_load_back,
                "patched_load_back_tokens": run["load_back_tokens"],
                "load_back_tokens_delta_pct": (
                    pct_delta(run["load_back_tokens"], old_load_back)
                    if old_load_back
                    else ""
                ),
                "prepatch_run_dir": old["run_dir"],
                "patched_run_dir": run["run_dir"],
            }
        )
    return paired


def static_comparison_rows(
    patched_summary: list[dict], homogeneous_summary: list[dict], mixed_summary: list[dict]
) -> list[dict]:
    static_by_key = {}
    for scenario, source_rows in (
        ("homogeneous", homogeneous_summary),
        ("mixed", mixed_summary),
    ):
        for row in source_rows:
            if row["variant"] not in {"static-auto", "static-triton"}:
                continue
            key = (
                scenario,
                row["model"],
                row["workload"],
                int(row["prefix_reuse_pct"]),
                row["variant"],
            )
            static_by_key[key] = row

    output = []
    for patched in patched_summary:
        base_key = (
            patched["scenario"],
            patched["model"],
            patched["workload"],
            patched["prefix_reuse_pct"],
        )
        auto = static_by_key[base_key + ("static-auto",)]
        triton = static_by_key[base_key + ("static-triton",)]
        unified_value = float(patched["total_token_throughput_median"])
        auto_value = float(auto["total_token_throughput_median"])
        triton_value = float(triton["total_token_throughput_median"])
        best_name, best_value = max(
            (("static-auto", auto_value), ("static-triton", triton_value)),
            key=lambda item: item[1],
        )
        candidates = {
            "unified-triton": unified_value,
            "static-auto": auto_value,
            "static-triton": triton_value,
        }
        rank = sorted(candidates, key=candidates.get, reverse=True).index("unified-triton") + 1
        output.append(
            {
                "scenario": patched["scenario"],
                "model": patched["model"],
                "workload": patched["workload"],
                "prefix_reuse_pct": patched["prefix_reuse_pct"],
                "patched_unified_median": unified_value,
                "patched_unified_cv_pct": patched["total_token_throughput_cv_pct"],
                "static_triton_median": triton_value,
                "unified_vs_static_triton_pct": pct_delta(unified_value, triton_value),
                "static_triton_l1_fraction": triton["mamba_full_memory_ratio"],
                "static_auto_median": auto_value,
                "unified_vs_static_auto_pct": pct_delta(unified_value, auto_value),
                "static_auto_l1_fraction": auto["mamba_full_memory_ratio"],
                "best_static_variant": best_name,
                "best_static_median": best_value,
                "unified_vs_best_static_pct": pct_delta(unified_value, best_value),
                "unified_rank_of_3": rank,
            }
        )
    return output


def grouped_delta_lines(rows: list[dict], field: str, keys: list[str]) -> list[str]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[field]))
    lines = []
    for key, values in sorted(grouped.items()):
        label = ", ".join(f"{name}={value}" for name, value in zip(keys, key))
        lines.append(
            f"- {label}: median {fmt_pct(statistics.median(values))}, "
            f"mean {fmt_pct(statistics.mean(values))}, n={len(values)}"
        )
    return lines


def build_report(paired: list[dict], comparison: list[dict]) -> str:
    paired_values = [float(row["patched_vs_prepatch_pct"]) for row in paired]
    homogeneous_pairs = [row for row in paired if row["scenario"] == "homogeneous"]
    mixed_pairs = [row for row in paired if row["scenario"] == "mixed"]
    homogeneous_eviction = [
        float(row["evicted_tokens_delta_pct"])
        for row in homogeneous_pairs
        if row["evicted_tokens_delta_pct"] != ""
    ]
    homogeneous_backup = [
        float(row["backup_tokens_delta_pct"])
        for row in homogeneous_pairs
        if row["backup_tokens_delta_pct"] != ""
    ]

    vs_triton = [float(row["unified_vs_static_triton_pct"]) for row in comparison]
    vs_auto = [float(row["unified_vs_static_auto_pct"]) for row in comparison]
    vs_best = [float(row["unified_vs_best_static_pct"]) for row in comparison]
    wins_triton = sum(value > 0 for value in vs_triton)
    wins_auto = sum(value > 0 for value in vs_auto)
    wins_best = sum(value > 0 for value in vs_best)
    ties_best = sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in vs_best)
    biggest_wins = sorted(
        comparison, key=lambda row: float(row["unified_vs_best_static_pct"]), reverse=True
    )[:5]
    biggest_losses = sorted(
        comparison, key=lambda row: float(row["unified_vs_best_static_pct"])
    )[:5]
    homogeneous_comparison = [
        row for row in comparison if row["scenario"] == "homogeneous"
    ]
    mixed_comparison = [row for row in comparison if row["scenario"] == "mixed"]

    def median_field(rows: list[dict], field: str) -> float:
        return statistics.median(float(row[field]) for row in rows)

    def condition(row: dict) -> str:
        return (
            f"{row['scenario']}/{row['model']}/{row['workload']}/"
            f"reuse{row['prefix_reuse_pct']}"
        )

    lines = [
        "# Patched unified-memory performance analysis",
        "",
        f"- Patched SHA: `{PATCHED_SHA}`",
        f"- Pre-patch comparison SHA: `{PREPATCH_SHA}`",
        "- Patched matrix: 72/72 completed and validated (54 homogeneous + 18 mixed)",
        "- The pre-patch unified matrix completed 61/72 comparable conditions "
        "(54 homogeneous + 7 mixed); patched unified completed all 72/72",
        "- Metric: median total-token throughput over three repetitions per condition",
        "",
        "## Patch impact on unified-triton",
        "",
        f"Exact pre/post pairs available: {len(paired)} "
        f"({len(homogeneous_pairs)} homogeneous, {len(mixed_pairs)} mixed).",
        f"Across exact pairs: median {fmt_pct(statistics.median(paired_values))}, "
        f"mean {fmt_pct(statistics.mean(paired_values))}.",
        "This A/B covers the cumulative five-commit range from pre-patch SHA to "
        "patched SHA; it does not isolate the final commit by itself.",
        "",
        *grouped_delta_lines(paired, "patched_vs_prepatch_pct", ["scenario", "model"]),
        "",
        f"For homogeneous exact pairs, eviction volume increased by a median "
        f"{fmt_pct(statistics.median(homogeneous_eviction))} and backup volume by "
        f"{fmt_pct(statistics.median(homogeneous_backup))}. This is consistent with "
        "the patched shared-pool fallback evicting a Mamba row and its Full-side "
        "byte donor together. It is evidence of changed cache behavior, not a "
        "standalone causal proof for every throughput loss.",
        "",
        "The mixed patch-impact estimate only covers pre-patch runs that completed and "
        "validated; failed pre-patch runs have no throughput value and are excluded.",
        "",
        "## Patched unified vs static",
        "",
        f"Across {len(comparison)} conditions, unified beats static-triton in "
        f"{wins_triton}/{len(comparison)} and static-auto in {wins_auto}/{len(comparison)}.",
        f"Against the better static score in each condition, unified wins "
        f"{wins_best}/{len(comparison)}, ties {ties_best}, and loses "
        f"{len(comparison) - wins_best - ties_best}.",
        f"Median delta: vs static-triton {fmt_pct(statistics.median(vs_triton))}; "
        f"vs static-auto {fmt_pct(statistics.median(vs_auto))}; "
        f"vs best-static {fmt_pct(statistics.median(vs_best))}.",
        f"Homogeneous median: vs static-triton "
        f"{fmt_pct(median_field(homogeneous_comparison, 'unified_vs_static_triton_pct'))}; "
        f"vs static-auto "
        f"{fmt_pct(median_field(homogeneous_comparison, 'unified_vs_static_auto_pct'))}.",
        f"Mixed median: vs static-triton "
        f"{fmt_pct(median_field(mixed_comparison, 'unified_vs_static_triton_pct'))}; "
        f"vs static-auto "
        f"{fmt_pct(median_field(mixed_comparison, 'unified_vs_static_auto_pct'))}.",
        "",
        *grouped_delta_lines(
            comparison, "unified_vs_best_static_pct", ["scenario", "model"]
        ),
        "",
        "### Closest unified results to best static",
        "",
    ]
    for row in biggest_wins:
        lines.append(
            f"- {condition(row)}: {fmt_pct(float(row['unified_vs_best_static_pct']))} "
            f"({row['best_static_variant']})"
        )
    lines.extend(["", "### Largest unified losses vs best static", ""])
    for row in biggest_losses:
        lines.append(
            f"- {condition(row)}: {fmt_pct(float(row['unified_vs_best_static_pct']))} "
            f"({row['best_static_variant']})"
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Static rows reuse the completed static experiments because the allocator "
            "patch is confined to unified memory. The static L1 fraction shown in the CSV "
            "is the selected best fraction for that static variant and condition.",
            "- Static and patched unified were measured at different wall-clock times, so "
            "small differences near run-to-run CV should not be treated as decisive.",
            "- `paired_patch_impact.csv` is the appropriate evidence for patch overhead; "
            "`static_comparison.csv` is the architecture comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patched-root", type=Path, default=Path("artifacts/unified_patched_8de3268")
    )
    parser.add_argument(
        "--old-root", type=Path, default=Path("artifacts/static_unified_oracle_92eb073")
    )
    args = parser.parse_args()
    output_dir = args.patched_root / "analysis"

    patched_runs = load_patched_runs(args.patched_root)
    patched_summary = summarize_patched(patched_runs)
    old_summary_dir = args.old_root / "summary"
    old_raw = load_csv(old_summary_dir / "raw_runs.csv")
    paired = paired_patch_rows(patched_runs, old_raw)
    comparison = static_comparison_rows(
        patched_summary,
        load_csv(old_summary_dir / "homogeneous_summary.csv"),
        load_csv(old_summary_dir / "mixed_summary.csv"),
    )

    write_csv(output_dir / "patched_runs.csv", patched_runs)
    write_csv(output_dir / "patched_summary.csv", patched_summary)
    write_csv(output_dir / "paired_patch_impact.csv", paired)
    write_csv(output_dir / "static_comparison.csv", comparison)
    report = build_report(paired, comparison)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
