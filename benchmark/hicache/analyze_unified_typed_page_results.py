#!/usr/bin/env python3
"""Aggregate unified typed-page HiCache micro and server experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_micro(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    for path in sorted(root.glob("full-run*/results.json")):
        measurements.extend(_read_json(path)["measurements"])
    if not measurements:
        raise ValueError(f"No full microbenchmark results found under {root}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in measurements:
        key = (
            row["page_size"],
            row["tokens"],
            row["pattern"],
            row["direction"],
            row["variant"],
        )
        grouped[key].append(row)

    summary = []
    medians: dict[tuple[Any, ...], float] = {}
    for key, rows in sorted(grouped.items()):
        page_size, tokens, pattern, direction, variant = key
        elapsed = [float(row["elapsed_ms"]) for row in rows]
        throughput = [float(row["gib_per_s"]) for row in rows]
        medians[key] = _median(elapsed)
        summary.append(
            {
                "page_size": page_size,
                "tokens": tokens,
                "pattern": pattern,
                "direction": direction,
                "variant": variant,
                "samples": len(rows),
                "median_ms": _median(elapsed),
                "min_ms": min(elapsed),
                "max_ms": max(elapsed),
                "median_gib_s": _median(throughput),
            }
        )

    speedups = []
    conditions = sorted(
        {
            (row["page_size"], row["tokens"], row["pattern"], row["direction"])
            for row in measurements
        }
    )
    for page_size, tokens, pattern, direction in conditions:
        prefix = (page_size, tokens, pattern, direction)
        baseline_ms = medians[prefix + ("baseline-static",)]
        ours_ms = medians[prefix + ("ours-unified-typed",)]
        speedups.append(
            {
                "page_size": page_size,
                "tokens": tokens,
                "pattern": pattern,
                "direction": direction,
                "baseline_median_ms": baseline_ms,
                "ours_median_ms": ours_ms,
                "ours_speedup": baseline_ms / ours_ms,
            }
        )
    return summary, speedups


def aggregate_server(root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("paired-r*-p*/paged-*/**/result.json")):
        match = re.fullmatch(r"paired-r(\d+)-p(\d+)", path.parents[2].name)
        if match is None:
            continue
        result = _read_json(path)
        if not result["validation"]["passed"]:
            continue
        grouped[(int(match.group(2)), result["variant"])].append(result)

    rows = []
    metric_prefix = "sglang:"
    for (page_size, variant), results in sorted(grouped.items()):
        summaries = [result["summary"] for result in results]
        metrics = [result["total_metric_delta"] for result in results]

        def summary_values(*keys: str) -> list[float]:
            values: list[float] = []
            for summary in summaries:
                value: Any = summary
                for key in keys:
                    value = value[key]
                values.append(float(value))
            return values

        def metric_values(key: str) -> list[float]:
            return [float(metric[metric_prefix + key]) for metric in metrics]

        backup_bytes = metric_values("hicache_backup_bytes_total")
        backup_seconds = metric_values("hicache_backup_duration_seconds_sum")
        load_bytes = metric_values("load_back_bytes_total")
        load_seconds = metric_values("load_back_duration_seconds_sum")
        rows.append(
            {
                "page_size": page_size,
                "variant": variant,
                "runs": len(results),
                "input_tok_s_median": _median(summary_values("input_token_throughput")),
                "input_tok_s_min": min(summary_values("input_token_throughput")),
                "input_tok_s_max": max(summary_values("input_token_throughput")),
                "ttft_p50_ms_median": _median(summary_values("ttft_ms", "p50")),
                "ttft_p95_ms_median": _median(summary_values("ttft_ms", "p95")),
                "tpot_p50_ms_median": _median(summary_values("tpot_ms", "p50")),
                "tpot_p95_ms_median": _median(summary_values("tpot_ms", "p95")),
                "backup_gib_s_median": _median(
                    [
                        size / duration / (1024**3)
                        for size, duration in zip(backup_bytes, backup_seconds)
                    ]
                ),
                "load_gib_s_median": _median(
                    [
                        size / duration / (1024**3)
                        for size, duration in zip(load_bytes, load_seconds)
                    ]
                ),
                "backup_gb_median": _median(backup_bytes) / 1e9,
                "load_gb_median": _median(load_bytes) / 1e9,
                "evicted_tokens_median": _median(metric_values("evicted_tokens_total")),
            }
        )
    if not rows:
        raise ValueError(f"No valid server results found under {root}")
    return rows


def aggregate_parity(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("parity02-p*/paged-*/**/result.json")):
        result = _read_json(path)
        match = re.fullmatch(r"parity02-p(\d+)", path.parents[2].name)
        if match is None:
            continue
        restored = result.get("restored")
        comparison = restored["comparison"] if restored is not None else {}
        promotion_diffs = [
            float(row["comparison"]["max_abs_logprob_diff"])
            for row in result["promotions"]
        ]
        rows.append(
            {
                "page_size": int(match.group(1)),
                "variant": result["variant"],
                "passed": result["validation"]["passed"],
                "restored_index": restored["index"] if restored else None,
                "loadback_tokens": (
                    restored["loadback_delta_tokens"] if restored else 0
                ),
                "output_ids_equal": comparison.get("output_ids_equal", False),
                "logprob_token_ids_equal": comparison.get(
                    "logprob_token_ids_equal", False
                ),
                "restored_max_abs_logprob_diff": comparison.get("max_abs_logprob_diff"),
                "promotion_max_abs_logprob_diff": max(promotion_diffs, default=0.0),
            }
        )
    if not rows:
        raise ValueError(f"No parity results found under {root}")
    return sorted(rows, key=lambda row: (row["page_size"], row["variant"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--micro-root",
        type=Path,
        default=Path("artifacts/unified_typed_page_transfer"),
    )
    parser.add_argument(
        "--server-root",
        type=Path,
        default=Path("artifacts/unified_typed_page_server/matrix"),
    )
    parser.add_argument(
        "--parity-root",
        type=Path,
        default=Path("artifacts/unified_typed_page_server/parity"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/unified_typed_page_summary"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    micro, speedups = aggregate_micro(args.micro_root)
    server = aggregate_server(args.server_root)
    parity = aggregate_parity(args.parity_root)
    _write_csv(args.output_dir / "micro_summary.csv", micro)
    _write_csv(args.output_dir / "micro_speedups.csv", speedups)
    _write_csv(args.output_dir / "server_summary.csv", server)
    _write_csv(args.output_dir / "parity_summary.csv", parity)
    print(args.output_dir)


if __name__ == "__main__":
    main()
