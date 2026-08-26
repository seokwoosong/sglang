"""Validate and summarize the recorded NVFP4 experiment results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

CONFIGS = ("bf16", "w4a4_bf16_kv", "w4a4_nvfp4_kv")
WORKLOADS = ("prefill_heavy", "balanced", "decode_heavy")
CONCURRENCIES = (1, 8, 32)
PROMPTS_BY_CONCURRENCY = {1: 4, 8: 16, 32: 64}
SERVING_METRICS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "p99_ttft_ms",
    "p99_tpot_ms",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def median(values) -> float:
    return float(statistics.median(values))


def wilson_interval(successes: int, cases: int, z: float = 1.96) -> list[float]:
    proportion = successes / cases
    denominator = 1 + z**2 / cases
    center = (proportion + z**2 / (2 * cases)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / cases + z**2 / (4 * cases**2))
        / denominator
    )
    return [center - radius, center + radius]


def summarize_gemm(results_dir: Path) -> dict[str, Any]:
    payload = read_json(results_dir / "operator/gemm_full_matrix.json")
    rows = payload["results"]
    expected = {
        (projection, distribution, m, seed)
        for projection in ("qkv", "attention_output", "gate_up", "down")
        for distribution in ("gaussian", "outlier")
        for m in (1, 16, 128, 512, 2048)
        for seed in (0, 1, 2)
    }
    actual = {
        (row["projection"], row["distribution"], row["m"], row["seed"]) for row in rows
    }
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("GEMM result matrix is incomplete or contains duplicates")

    groups = []
    for projection in ("qkv", "attention_output", "gate_up", "down"):
        for distribution in ("gaussian", "outlier"):
            for m in (1, 16, 128, 512, 2048):
                subset = [
                    row
                    for row in rows
                    if row["projection"] == projection
                    and row["distribution"] == distribution
                    and row["m"] == m
                ]
                groups.append(
                    {
                        "projection": projection,
                        "distribution": distribution,
                        "m": m,
                        "n": subset[0]["n"],
                        "k": subset[0]["k"],
                        "seeds": len(subset),
                        "bf16_ms": median(row["bf16"]["median_ms"] for row in subset),
                        "activation_quant_ms": median(
                            row["activation_quant"]["median_ms"] for row in subset
                        ),
                        "prequantized_fp4_gemm_ms": median(
                            row["fp4_gemm"]["median_ms"] for row in subset
                        ),
                        "quantize_plus_fp4_gemm_ms": median(
                            row["fp4_end_to_end"]["median_ms"] for row in subset
                        ),
                        "prequantized_speedup": median(
                            row["fp4_gemm_speedup"] for row in subset
                        ),
                        "quantize_plus_gemm_speedup": median(
                            row["fp4_end_to_end_speedup"] for row in subset
                        ),
                        "diagnostic_linear_quant_ms": median(
                            row["linear_layout_quant"]["median_ms"] for row in subset
                        ),
                        "diagnostic_linear_dequant_ms": median(
                            row["linear_layout_dequant"]["median_ms"] for row in subset
                        ),
                        "relative_rmse": median(row["relative_rmse"] for row in subset),
                        "cosine_similarity": median(
                            row["cosine_similarity"] for row in subset
                        ),
                        "nan_count": sum(row["nan_count"] for row in subset),
                        "inf_count": sum(row["inf_count"] for row in subset),
                    }
                )
    return {
        "source": "operator/gemm_full_matrix.json",
        "raw_cases": len(rows),
        "groups": groups,
    }


def summarize_attention(results_dir: Path) -> dict[str, Any]:
    payload = read_json(results_dir / "operator/attention_full_matrix.json")
    rows = payload["results"]
    expected = {
        (head_dim, seq_len, seed)
        for head_dim in (64, 128)
        for seq_len in (512, 1024, 2048)
        for seed in (0, 1, 2)
    }
    actual = {(row["head_dim"], row["seq_len"], row["seed"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("attention result matrix is incomplete or contains duplicates")

    groups = []
    for head_dim in (64, 128):
        for seq_len in (512, 1024, 2048):
            subset = [
                row
                for row in rows
                if row["head_dim"] == head_dim and row["seq_len"] == seq_len
            ]
            groups.append(
                {
                    "head_dim": head_dim,
                    "seq_len": seq_len,
                    "seeds": len(subset),
                    "bf16_ms": median(row["baseline_ms"] for row in subset),
                    "prequantized_fp4_attention_ms": median(
                        row["fp4_attention_ms"] for row in subset
                    ),
                    "quantize_plus_fp4_attention_ms": median(
                        row["fp4_end_to_end_ms"] for row in subset
                    ),
                    "prequantized_speedup": median(
                        row["baseline_ms"] / row["fp4_attention_ms"] for row in subset
                    ),
                    "quantize_plus_attention_speedup": median(
                        row["baseline_ms"] / row["fp4_end_to_end_ms"] for row in subset
                    ),
                    "relative_rmse": median(row["relative_rmse"] for row in subset),
                    "cosine_similarity": median(
                        row["cosine_similarity"] for row in subset
                    ),
                    "nan_count": sum(row["nan_count"] for row in subset),
                    "inf_count": sum(row["inf_count"] for row in subset),
                }
            )
    return {
        "source": "operator/attention_full_matrix.json",
        "raw_cases": len(rows),
        "groups": groups,
    }


def summarize_serving(results_dir: Path) -> dict[str, Any]:
    speed_files = sorted((results_dir / "serving").glob("*/speed/*.jsonl"))
    rows = []
    for path in speed_files:
        result = read_json(path)
        metadata = result["nvfp4_experiment"]
        concurrency = metadata["target_concurrency"]
        expected_prompts = PROMPTS_BY_CONCURRENCY[concurrency]
        if result["completed"] != expected_prompts:
            raise ValueError(f"incomplete request count: {path}")
        if (
            result["total_input_tokens"]
            != expected_prompts * metadata["target_input_tokens"]
        ):
            raise ValueError(f"input token mismatch: {path}")
        if (
            result["total_output_tokens"]
            != expected_prompts * metadata["target_output_tokens"]
        ):
            raise ValueError(f"output token mismatch: {path}")
        if not all(math.isfinite(result[metric]) for metric in SERVING_METRICS):
            raise ValueError(f"non-finite serving metric: {path}")
        rows.append((result, metadata))

    groups = []
    indexed = {}
    for config in CONFIGS:
        for workload in WORKLOADS:
            for concurrency in CONCURRENCIES:
                subset = [
                    result
                    for result, metadata in rows
                    if metadata["config"] == config
                    and metadata["workload"] == workload
                    and metadata["target_concurrency"] == concurrency
                ]
                if len(subset) != 3:
                    raise ValueError(
                        f"expected three serving repeats for {config}/{workload}/"
                        f"c{concurrency}, found {len(subset)}"
                    )
                group = {
                    "config": config,
                    "workload": workload,
                    "concurrency": concurrency,
                    "repeats": len(subset),
                    **{
                        metric: median(result[metric] for result in subset)
                        for metric in SERVING_METRICS
                    },
                }
                groups.append(group)
                indexed[(config, workload, concurrency)] = group

    comparisons = []
    for workload in WORKLOADS:
        for concurrency in CONCURRENCIES:
            bf16 = indexed[("bf16", workload, concurrency)]
            w4a4 = indexed[("w4a4_bf16_kv", workload, concurrency)]
            nvfp4_kv = indexed[("w4a4_nvfp4_kv", workload, concurrency)]
            comparisons.append(
                {
                    "workload": workload,
                    "concurrency": concurrency,
                    "w4a4_output_throughput_vs_bf16": (
                        w4a4["output_throughput"] / bf16["output_throughput"]
                    ),
                    "w4a4_ttft_speedup_vs_bf16": (
                        bf16["mean_ttft_ms"] / w4a4["mean_ttft_ms"]
                    ),
                    "w4a4_tpot_speedup_vs_bf16": (
                        bf16["mean_tpot_ms"] / w4a4["mean_tpot_ms"]
                    ),
                    "nvfp4_kv_output_throughput_vs_w4a4_bf16_kv": (
                        nvfp4_kv["output_throughput"] / w4a4["output_throughput"]
                    ),
                    "nvfp4_kv_ttft_speedup_vs_w4a4_bf16_kv": (
                        w4a4["mean_ttft_ms"] / nvfp4_kv["mean_ttft_ms"]
                    ),
                    "nvfp4_kv_tpot_speedup_vs_w4a4_bf16_kv": (
                        w4a4["mean_tpot_ms"] / nvfp4_kv["mean_tpot_ms"]
                    ),
                    "full_output_throughput_vs_bf16": (
                        nvfp4_kv["output_throughput"] / bf16["output_throughput"]
                    ),
                }
            )

    memory = []
    for config in CONFIGS:
        metadata_rows = [
            metadata for _, metadata in rows if metadata["config"] == config
        ]
        memory.append(
            {
                "config": config,
                "initial_gpu_memory_mib_median": median(
                    row["initial_gpu_memory_mib"] for row in metadata_rows
                ),
                "peak_gpu_memory_mib_median": median(
                    row["peak_gpu_memory_mib"] for row in metadata_rows
                ),
                "peak_gpu_memory_mib_max": max(
                    row["peak_gpu_memory_mib"] for row in metadata_rows
                ),
            }
        )

    return {
        "source": "serving/*/speed/*.jsonl",
        "raw_cases": len(rows),
        "groups": groups,
        "comparisons": comparisons,
        "memory": memory,
    }


def summarize_accuracy(results_dir: Path) -> dict[str, Any]:
    configs = []
    for config in CONFIGS:
        accuracy_dir = results_dir / "serving" / config / "accuracy"
        gsm8k = read_json(accuracy_dir / "gsm8k.json")
        passkey = read_json(accuracy_dir / "passkey.json")
        examples = gsm8k["nvfp4_experiment"]["examples"]
        if examples != 200:
            raise ValueError(f"expected 200 GSM8K examples for {config}")
        successes = round(gsm8k["score"] * examples)
        if len(passkey["results"]) != 60:
            raise ValueError(f"expected 60 passkey cases for {config}")
        if any(
            result["server_prompt_tokens"] != result["context_tokens"]
            for result in passkey["results"]
        ):
            raise ValueError(f"passkey prompt token mismatch for {config}")
        passkey_summary = {}
        for context_length in (4096, 8192):
            result = passkey["summary"][str(context_length)]
            passkey_summary[str(context_length)] = {
                **result,
                "wilson_95_interval": wilson_interval(
                    result["exact_matches"], result["cases"]
                ),
            }
        configs.append(
            {
                "config": config,
                "gsm8k_examples": examples,
                "gsm8k_correct": successes,
                "gsm8k_accuracy": gsm8k["score"],
                "gsm8k_wilson_95_interval": wilson_interval(successes, examples),
                "passkey": passkey_summary,
            }
        )
    return {
        "source": "serving/*/accuracy/{gsm8k,passkey}.json",
        "configs": configs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = args.output or results_dir / "summary.json"
    payload = {
        "schema_version": 1,
        "results_dir": str(results_dir),
        "gemm": summarize_gemm(results_dir),
        "attention": summarize_attention(results_dir),
        "serving": summarize_serving(results_dir),
        "accuracy": summarize_accuracy(results_dir),
        "notes": [
            "All grouped performance values are medians of three repeats or seeds.",
            "Standalone linear-layout dequantization is diagnostic; the production FP4 GEMM epilogue converts to BF16 without a separate dequant kernel.",
            "Serving request_rate is intentionally infinite and is excluded from finite-metric validation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Validated and summarized results to {output}")


if __name__ == "__main__":
    main()
