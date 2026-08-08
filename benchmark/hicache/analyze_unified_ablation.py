#!/usr/bin/env python3
"""Summarize the final unified-memory HiCache ablation artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

VARIANTS = ("u0", "u1", "u2", "u3")
STEADY_RUNS = {
    "short-3k": "final2-short-3k-r3-rep*",
    "long-32k": "final2-long-32k-r3-rep*",
}
METRICS = (
    ("req/s", ("summary", "request_throughput")),
    ("input tok/s", ("summary", "input_token_throughput")),
    ("TTFT p50 ms", ("summary", "ttft_ms", "p50")),
    ("TTFT p95 ms", ("summary", "ttft_ms", "p95")),
    ("TPOT p50 ms", ("summary", "tpot_ms", "p50")),
    ("TPOT p95 ms", ("summary", "tpot_ms", "p95")),
    ("evicted tokens", ("measured_metric_delta", "sglang:evicted_tokens_total")),
    (
        "load-back tokens",
        ("measured_metric_delta", "sglang:load_back_tokens_total"),
    ),
)


def nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


def load_json(path: str) -> dict[str, Any]:
    with Path(path).open() as file:
        return json.load(file)


def final_results(root: Path, run_glob: str, variant: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(root / run_glob / variant / "*" / "result.json")))
    results = [load_json(path) for path in paths]
    if len(results) != 3:
        raise RuntimeError(
            f"Expected three {run_glob}/{variant} results, found {len(results)}"
        )
    for result in results:
        if not result["validation"]["passed"] or result["summary"]["failed"]:
            raise RuntimeError(f"Invalid final result in {run_glob}/{variant}")
    return results


def compact(values: list[float]) -> str:
    median = statistics.median(values)
    return f"{median:.3f} [{min(values):.3f}, {max(values):.3f}]"


def print_steady(root: Path, label: str, run_glob: str) -> None:
    print(f"\n## {label}\n")
    print("| variant | " + " | ".join(name for name, _ in METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    for variant in VARIANTS:
        results = final_results(root, run_glob, variant)
        cells = []
        for _, path in METRICS:
            cells.append(compact([float(nested(result, path)) for result in results]))
        print(f"| {variant} | " + " | ".join(cells) + " |")


def print_accuracy(root: Path) -> None:
    print("\n## GSM8K accuracy\n")
    print(
        "| variant | correct | accuracy | failed | invalid | duration s | eviction | load-back |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    records: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        paths = sorted(
            glob.glob(str(root / "final2-accuracy-200" / variant / "*" / "result.json"))
        )
        if len(paths) != 1:
            raise RuntimeError(f"Expected one final accuracy result for {variant}")
        result = load_json(paths[0])
        if not result["validation"]["passed"]:
            raise RuntimeError(f"Accuracy validation failed for {variant}")
        summary = result["summary"]
        delta = result["total_metric_delta"]
        records[variant] = result["records"]
        print(
            f"| {variant} | {summary['correct']}/{summary['num_questions']} | "
            f"{summary['accuracy'] * 100:.1f}% | {summary['failed']} | "
            f"{summary['invalid']} | {summary['measured_remaining_duration_s']:.3f} | "
            f"{delta['sglang:evicted_tokens_total']:.0f} | "
            f"{delta['sglang:load_back_tokens_total']:.0f} |"
        )

    base = records["u0"]
    for variant in VARIANTS[1:]:
        prediction_diff = sum(
            left["prediction"] != right["prediction"]
            for left, right in zip(base, records[variant])
        )
        correctness_diff = sum(
            left["correct"] != right["correct"]
            for left, right in zip(base, records[variant])
        )
        print(
            f"- {variant} vs u0: prediction differences={prediction_diff}, "
            f"correctness differences={correctness_diff}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/unified_ablation"),
    )
    args = parser.parse_args()
    for label, run_glob in STEADY_RUNS.items():
        print_steady(args.artifact_root, label, run_glob)
    print_accuracy(args.artifact_root)


if __name__ == "__main__":
    main()
