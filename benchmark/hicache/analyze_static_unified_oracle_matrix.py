#!/usr/bin/env python3
"""Validate and summarize the static-oracle versus unified HiCache matrix."""

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
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/static_unified_oracle_92eb073"
EXPECTED_SHA = "92eb0737857f4fef0ba46e19bed5cb9bc45816f9"
VARIANTS = ("static-auto", "static-triton", "unified-triton")
REUSES = (20, 50, 80)
MODELS = ("0.8b", "4b")
WORKLOADS = ("short", "middle", "long")

HOMOGENEOUS_RE = re.compile(
    r"^(?P<stage>tune|final)-(?P<model>0\.8b|4b)-"
    r"(?P<variant>static-auto|static-triton|unified-triton)-"
    r"(?P<workload>short|middle|long)-reuse(?P<reuse>\d{3})-"
    r"r(?P<ratio>\d{2})-rep(?P<repetition>\d+)$"
)
MIXED_RE = re.compile(
    r"^mixed-(?P<model>0\.8b|4b)-"
    r"(?P<variant>static-auto|static-triton|unified-triton)-"
    r"reuse(?P<reuse>\d{3})-r(?P<ratio>\d{2})-"
    r"rep(?P<repetition>\d+)$"
)
INTERMIXED_RE = re.compile(
    r"^(?P<stage>mixed-tune|mixed-final)-intermixed-"
    r"(?P<model>0\.8b|4b)-"
    r"(?P<variant>static-auto|static-triton|unified-triton)-"
    r"reuse(?P<reuse>\d{3})-r(?P<ratio>\d{2})-"
    r"rep(?P<repetition>\d+)$"
)
PREFLIGHT_RE = re.compile(
    r"^preflight-(?P<model>0\.8b|4b)-"
    r"(?P<variant>static-auto|static-triton|unified-triton)$"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_complete_runs(root: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    latest: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for manifest_path in sorted(root.glob("**/manifest.json")):
        result_path = manifest_path.with_name("result.json")
        if not result_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            manifest.get("status") != "completed"
            or manifest.get("client_exit_code") != 0
        ):
            continue
        run_name = manifest.get("arguments", {}).get("run_name")
        variant = manifest.get("arguments", {}).get("variant")
        if run_name and variant:
            latest[(run_name, variant)] = (manifest, result)
    return list(latest.values())


def validate_preflight(root: Path) -> list[str]:
    found: set[tuple[str, str]] = set()
    errors: list[str] = []
    for manifest, result in latest_complete_runs(root):
        run_name = manifest.get("arguments", {}).get("run_name", "")
        match = PREFLIGHT_RE.match(run_name)
        if match is None:
            continue
        key = (match.group("model"), match.group("variant"))
        found.add(key)
        if not result.get("validation", {}).get("passed", False):
            errors.append(f"preflight {key} failed client validation")
        if manifest.get("variant_definition", {}).get("sha") != EXPECTED_SHA:
            errors.append(f"preflight {key} used an unexpected source SHA")
        total_metrics = result.get("total_metric_delta", {})
        if float(total_metrics.get("sglang:evicted_tokens_total", 0.0)) <= 0:
            errors.append(f"preflight {key} did not evict L1 entries")
        if float(total_metrics.get("sglang:load_back_tokens_total", 0.0)) <= 0:
            errors.append(f"preflight {key} did not load entries back from L2")
    expected = {(model, variant) for model in MODELS for variant in VARIANTS}
    errors.extend(f"missing preflight: {key}" for key in sorted(expected - found))
    return errors


def metric(result: dict[str, Any], name: str) -> float:
    return float(result.get("total_metric_delta", {}).get(name, 0.0))


def raw_row(
    manifest: dict[str, Any], result: dict[str, Any], match: re.Match[str]
) -> dict[str, Any]:
    groups = match.groupdict()
    summary = result["summary"]
    row = {
        "kind": result.get("kind", ""),
        "stage": groups.get("stage") or "mixed",
        "model": groups["model"],
        "variant": groups["variant"],
        "workload": groups.get("workload")
        or (
            "interleaved-mixed"
            if result.get("kind") == "interleaved-mixed"
            else "phase-mixed"
        ),
        "prefix_reuse_pct": int(groups["reuse"]),
        "mamba_full_memory_ratio": int(groups["ratio"]) / 10,
        "repetition": int(groups["repetition"]),
        "total_token_throughput": summary["total_token_throughput"],
        "input_token_throughput": summary["input_token_throughput"],
        "output_token_throughput": summary["output_token_throughput"],
        "request_throughput": summary["request_throughput"],
        "duration_s": summary["duration_s"],
        "completed": summary["completed"],
        "failed": summary["failed"],
        "evicted_tokens": metric(result, "sglang:evicted_tokens_total"),
        "backup_tokens": metric(result, "sglang:hicache_backup_tokens_total"),
        "load_back_tokens": metric(result, "sglang:load_back_tokens_total"),
        "dropped_tokens": metric(result, "sglang:hicache_dropped_tokens_total"),
        "cached_tokens_host": summary.get("cached_tokens_host", 0),
        "validation_passed": result.get("validation", {}).get("passed", False),
        "source_sha": manifest.get("variant_definition", {}).get("sha", ""),
        "run_dir": manifest.get("run_dir", ""),
    }
    by_length = result.get("summary_by_length_class", {})
    for label in ("short", "middle", "long"):
        item = by_length.get(label, {})
        row[f"{label}_total_token_throughput"] = item.get("total_token_throughput")
        row[f"{label}_ttft_p95_ms"] = item.get("ttft_ms", {}).get("p95")
        row[f"{label}_e2e_p95_ms"] = item.get("e2e_ms", {}).get("p95")
        row[f"{label}_cached_tokens_device"] = item.get("cached_tokens_device")
        row[f"{label}_cached_tokens_host"] = item.get("cached_tokens_host")
    return row


def aggregate_mixed_lengths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for label in ("short", "middle", "long"):
            if row.get(f"{label}_total_token_throughput") is not None:
                groups[
                    (row["model"], row["prefix_reuse_pct"], row["variant"], label)
                ].append(row)

    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):

        def median_field(suffix: str) -> float:
            return statistics.median(
                float(item[f"{key[3]}_{suffix}"]) for item in items
            )

        output.append(
            {
                "model": key[0],
                "prefix_reuse_pct": key[1],
                "variant": key[2],
                "length_class": key[3],
                "repetitions": len(items),
                "total_token_throughput_median": median_field("total_token_throughput"),
                "ttft_p95_ms_median": median_field("ttft_p95_ms"),
                "e2e_p95_ms_median": median_field("e2e_p95_ms"),
                "cached_tokens_device_median": median_field("cached_tokens_device"),
                "cached_tokens_host_median": median_field("cached_tokens_host"),
            }
        )
    return output


def collect(
    root: Path, selections: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for manifest, result in latest_complete_runs(root):
        run_name = manifest["arguments"]["run_name"]
        match = (
            HOMOGENEOUS_RE.match(run_name)
            or INTERMIXED_RE.match(run_name)
            or MIXED_RE.match(run_name)
        )
        if match is None:
            continue
        row = raw_row(manifest, result, match)
        condition = (
            f"{row['model']}/{row['variant']}/{row['workload']}/"
            f"reuse{row['prefix_reuse_pct']}/rep{row['repetition']}"
        )
        if not row["validation_passed"]:
            errors.append(f"{condition}: client validation failed")
        if row["failed"]:
            errors.append(f"{condition}: {row['failed']} requests failed")
        if row["source_sha"] != EXPECTED_SHA:
            errors.append(f"{condition}: unexpected source SHA {row['source_sha']}")
        if row["evicted_tokens"] <= 0:
            errors.append(f"{condition}: no L1 eviction")
        if row["backup_tokens"] <= 0:
            errors.append(f"{condition}: no L2 backup")
        if row["load_back_tokens"] <= 0:
            errors.append(f"{condition}: no L2 load-back")
        if row["dropped_tokens"] > 0:
            errors.append(f"{condition}: HiCache dropped tokens")
        rows.append(row)

    # Keep all screening rows in raw.csv, but summary selection happens later.
    if not selections:
        errors.append("Static ratio selection file is missing or empty")
    return rows, errors


def selected_homogeneous_rows(
    rows: list[dict[str, Any]], selections: dict[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["workload"] == "phase-mixed":
            continue
        reuse = row["prefix_reuse_pct"]
        variant = row["variant"]
        if variant == "unified-triton":
            if row["stage"] == "final":
                selected.append(row)
            continue
        key = f"{row['model']}/{variant}/{row['workload']}"
        if key not in selections:
            continue
        chosen = float(selections[key]["selected_ratio"])
        if abs(row["mamba_full_memory_ratio"] - chosen) > 1e-9:
            continue
        if reuse == 80 and row["stage"] == "tune":
            selected.append(row)
        elif reuse in (20, 50) and row["stage"] == "final":
            selected.append(row)
    return selected


def selected_mixed_rows(
    rows: list[dict[str, Any]], selections: dict[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["workload"] != "interleaved-mixed":
            continue
        variant = row["variant"]
        reuse = row["prefix_reuse_pct"]
        if variant == "unified-triton":
            if row["stage"] == "mixed-final":
                selected.append(row)
            continue
        key = f"{row['model']}/{variant}/interleaved-mixed"
        if key not in selections:
            continue
        chosen = float(selections[key]["selected_ratio"])
        if abs(row["mamba_full_memory_ratio"] - chosen) > 1e-9:
            continue
        if reuse == 80 and row["stage"] == "mixed-tune":
            selected.append(row)
        elif reuse in (20, 50) and row["stage"] == "mixed-final":
            selected.append(row)
    return selected


def aggregate(rows: list[dict[str, Any]], *, mixed: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        is_mixed = row["workload"] in ("phase-mixed", "interleaved-mixed")
        if is_mixed != mixed:
            continue
        groups[
            (
                row["model"],
                row["workload"],
                row["prefix_reuse_pct"],
                row["variant"],
            )
        ].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        throughputs = [float(item["total_token_throughput"]) for item in items]
        mean = statistics.mean(throughputs)
        aggregated.append(
            {
                "model": key[0],
                "workload": key[1],
                "prefix_reuse_pct": key[2],
                "variant": key[3],
                "mamba_full_memory_ratio": items[0]["mamba_full_memory_ratio"],
                "repetitions": len(items),
                "total_token_throughput_median": statistics.median(throughputs),
                "total_token_throughput_min": min(throughputs),
                "total_token_throughput_max": max(throughputs),
                "total_token_throughput_cv_pct": (
                    statistics.pstdev(throughputs) / mean * 100 if mean else 0.0
                ),
                "request_throughput_median": statistics.median(
                    float(item["request_throughput"]) for item in items
                ),
                "load_back_tokens_median": statistics.median(
                    float(item["load_back_tokens"]) for item in items
                ),
                "cached_tokens_host_median": statistics.median(
                    float(item["cached_tokens_host"]) for item in items
                ),
            }
        )
    return aggregated


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["model"], row["workload"], row["prefix_reuse_pct"], row["variant"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    conditions = sorted(
        {(row["model"], row["workload"], row["prefix_reuse_pct"]) for row in rows}
    )
    for model, workload, reuse in conditions:
        values: dict[str, float] = {}
        for variant in VARIANTS:
            row = indexed.get((model, workload, reuse, variant))
            if row is None:
                break
            values[variant] = float(row["total_token_throughput_median"])
        if len(values) != len(VARIANTS):
            continue
        ceiling = max(values["static-auto"], values["static-triton"])
        unified = values["unified-triton"]
        output.append(
            {
                "model": model,
                "workload": workload,
                "prefix_reuse_pct": reuse,
                "static_auto": values["static-auto"],
                "static_triton": values["static-triton"],
                "unified_triton": unified,
                "unified_vs_static_triton_pct": (unified / values["static-triton"] - 1)
                * 100,
                "static_ceiling": ceiling,
                "unified_vs_static_ceiling_pct": (unified / ceiling - 1) * 100,
            }
        )
    return output


def write_report(
    path: Path,
    homogeneous: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    errors: list[str],
) -> None:
    lines = [
        "# Static oracle vs unified HiCache",
        "",
        f"Validation: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Homogeneous workloads",
        "",
        "| Model | Workload | Reuse | U/S-T | U/static ceiling |",
        "|---|---|---:|---:|---:|",
    ]
    for row in homogeneous:
        lines.append(
            f"| {row['model']} | {row['workload']} | {row['prefix_reuse_pct']}% "
            f"| {row['unified_vs_static_triton_pct']:+.2f}% "
            f"| {row['unified_vs_static_ceiling_pct']:+.2f}% |"
        )
    lines.extend(
        (
            "",
            "## Length-interleaved mixed workload",
            "",
            "| Model | Reuse | U/S-T | U/static ceiling |",
            "|---|---:|---:|---:|",
        )
    )
    for row in mixed:
        lines.append(
            f"| {row['model']} | {row['prefix_reuse_pct']}% "
            f"| {row['unified_vs_static_triton_pct']:+.2f}% "
            f"| {row['unified_vs_static_ceiling_pct']:+.2f}% |"
        )
    if errors:
        lines.extend(("", "## Validation errors", ""))
        lines.extend(f"- {error}" for error in errors)
    path.write_text("\n".join(lines) + "\n")


def validate_completeness(
    homogeneous: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    selections: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_selections = {
        f"{model}/{variant}/{workload}"
        for model in MODELS
        for variant in ("static-auto", "static-triton")
        for workload in WORKLOADS
    }
    expected_selections.update(
        f"{model}/{variant}/interleaved-mixed"
        for model in MODELS
        for variant in ("static-auto", "static-triton")
    )
    missing_selections = sorted(expected_selections - set(selections))
    errors.extend(f"missing static selection: {key}" for key in missing_selections)

    homogeneous_index = {
        (row["model"], row["workload"], row["prefix_reuse_pct"], row["variant"]): row
        for row in homogeneous
    }
    for model in MODELS:
        for workload in WORKLOADS:
            for reuse in REUSES:
                for variant in VARIANTS:
                    key = (model, workload, reuse, variant)
                    row = homogeneous_index.get(key)
                    if row is None:
                        errors.append(f"missing homogeneous group: {key}")
                    elif row["repetitions"] != 3:
                        errors.append(
                            f"homogeneous group {key} has {row['repetitions']} reps, expected 3"
                        )

    mixed_index = {
        (row["model"], row["prefix_reuse_pct"], row["variant"]): row for row in mixed
    }
    for model in MODELS:
        for reuse in REUSES:
            for variant in VARIANTS:
                key = (model, reuse, variant)
                row = mixed_index.get(key)
                if row is None:
                    errors.append(f"missing interleaved-mixed group: {key}")
                elif row["repetitions"] != 3:
                    errors.append(
                        f"interleaved-mixed group {key} has {row['repetitions']} reps, expected 3"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    selection_path = root / "static_ratio_selection.json"
    selections = (
        json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    )

    raw, errors = collect(root, selections)
    errors.extend(validate_preflight(root))
    homogeneous_raw = selected_homogeneous_rows(raw, selections)
    homogeneous_agg = aggregate(homogeneous_raw, mixed=False)
    # Historical phase-ordered runs and unselected ratio-screening candidates
    # remain in raw_runs.csv. The comparison uses the selected static
    # finalist's three tuning runs at 80%, fixed-ratio static controls at 20/50%,
    # and unified final runs at all reuse levels.
    mixed_selected_raw = selected_mixed_rows(raw, selections)
    mixed_agg = aggregate(mixed_selected_raw, mixed=True)
    mixed_length_agg = aggregate_mixed_lengths(mixed_selected_raw)
    errors.extend(validate_completeness(homogeneous_agg, mixed_agg, selections))
    homogeneous_compare = comparisons(homogeneous_agg)
    mixed_compare = comparisons(mixed_agg)

    write_csv(output / "raw_runs.csv", raw)
    write_csv(output / "homogeneous_summary.csv", homogeneous_agg)
    write_csv(output / "mixed_summary.csv", mixed_agg)
    write_csv(output / "mixed_length_summary.csv", mixed_length_agg)
    write_csv(output / "homogeneous_comparison.csv", homogeneous_compare)
    write_csv(output / "mixed_comparison.csv", mixed_compare)
    write_report(output / "REPORT.md", homogeneous_compare, mixed_compare, errors)
    (output / "validation_errors.txt").write_text(
        "\n".join(errors) + ("\n" if errors else "")
    )
    print(
        json.dumps(
            {
                "raw_runs": len(raw),
                "homogeneous_groups": len(homogeneous_agg),
                "mixed_groups": len(mixed_agg),
                "mixed_length_groups": len(mixed_length_agg),
                "validation_errors": len(errors),
                "output_dir": str(output),
            },
            indent=2,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
