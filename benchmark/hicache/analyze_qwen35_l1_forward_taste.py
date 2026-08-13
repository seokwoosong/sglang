#!/usr/bin/env python3
"""Audit and summarize a Qwen3.5 LF/PF forward-path matrix."""

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
SERVER_SHA = "743cae224c5bc28687457558a074736776350392"
VARIANTS = ("l1-lf", "l1-pf-static", "l1-lf-auto")
RUN_RE = re.compile(
    r"(?P<prefix>l1-forward-(?:taste|scale))-r(?P<repetition>\d+)-"
    r"(?P<model>0\.8b|4b|9b)-"
    r"(?P<workload>prefill-50k|decode-10k-512)-p(?P<page>\d+)"
)
BACKEND_FLAGS = (
    "--attention-backend",
    "--linear-attn-backend",
    "--mamba-backend",
)
NO_HICACHE_METRICS = (
    "sglang:hicache_backup_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:hicache_dropped_tokens_total",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def latest_results(
    root: Path, *, run_prefix: str, models: set[str]
) -> list[tuple[re.Match[str], str, dict[str, Any], dict[str, Any], Path]]:
    selected: dict[
        tuple[str, int, str, int, str],
        tuple[re.Match[str], str, dict[str, Any], dict[str, Any], Path],
    ] = {}
    for path in sorted(root.glob(f"{run_prefix}-*/l1-*/*/result.json")):
        match = RUN_RE.fullmatch(path.parents[2].name)
        if (
            match is None
            or match["prefix"] != run_prefix
            or match["model"] not in models
        ):
            continue
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        result = load_json(path)
        manifest = load_json(manifest_path)
        variant = str(result["variant"])
        identity = (
            match["model"],
            int(match["repetition"]),
            match["workload"],
            int(match["page"]),
            variant,
        )
        if manifest.get("status") == "completed":
            selected[identity] = (match, variant, result, manifest, path)
    return list(selected.values())


def validate_configuration(
    *, variant: str, result: dict[str, Any], manifest: dict[str, Any], label: str
) -> list[str]:
    errors: list[str] = []
    info = result["server_info"]
    command = list(manifest.get("server_command", []))
    expected_pf = variant == "l1-pf-static"
    if manifest.get("variant_definition", {}).get("sha") != SERVER_SHA:
        errors.append(f"wrong server SHA: {label}")
    if bool(info["enable_page_major_kv_layout"]) != expected_pf:
        errors.append(f"wrong page-major setting: {label}")
    if info["enable_unified_memory"]:
        errors.append(f"unified memory enabled: {label}")
    if info["enable_hierarchical_cache"]:
        errors.append(f"HiCache enabled: {label}")
    if not info["disable_radix_cache"]:
        errors.append(f"radix cache enabled: {label}")
    if int(info["max_total_num_tokens"]) != 120000:
        errors.append(f"wrong token capacity: {label}")
    for phase in ("decode", "prefill"):
        if info[f"cuda_graph_backend_{phase}"] != "disabled":
            errors.append(f"CUDA graph {phase} enabled: {label}")
    if variant == "l1-lf-auto":
        present = [flag for flag in BACKEND_FLAGS if flag in command]
        if present:
            errors.append(f"auto run has explicit backend flags {present}: {label}")
    else:
        for name in ("attention_backend", "linear_attn_backend", "mamba_backend"):
            if info[name] != "triton":
                errors.append(f"{name} is not Triton: {label}")
        missing = [flag for flag in BACKEND_FLAGS if flag not in command]
        if missing:
            errors.append(f"Triton run lacks backend flags {missing}: {label}")
    if not result.get("validation", {}).get("passed"):
        errors.append(f"client validation failed: {label}")
    if int(result.get("summary", {}).get("failed", 0)):
        errors.append(f"measured request failed: {label}")
    metrics = result.get("measured_metric_delta", {})
    for name in NO_HICACHE_METRICS:
        if float(metrics.get(name, 0.0)) != 0.0:
            errors.append(f"unexpected HiCache metric {name}: {label}")
    if float(metrics.get("sglang:evicted_tokens_total", 0.0)) != 0.0:
        errors.append(f"unexpected eviction: {label}")
    return errors


def build_rows(
    root: Path, *, run_prefix: str, models: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for match, variant, result, manifest, path in latest_results(
        root, run_prefix=run_prefix, models=models
    ):
        run_name = path.parents[2].name
        label = f"{run_name}/{variant}"
        errors.extend(
            validate_configuration(
                variant=variant, result=result, manifest=manifest, label=label
            )
        )
        categories = result.get("measured_forward_seconds_by_category")
        if not isinstance(categories, dict) or not categories:
            errors.append(f"missing forward categories: {label}")
            continue
        extend_s = sum(
            float(categories.get(name, 0.0))
            for name in ("extend", "mixed", "split_prefill")
        )
        decode_s = float(categories.get("decode", 0.0))
        category_total_s = sum(float(value) for value in categories.values())
        metric_total_s = float(
            result["measured_metric_delta"].get(
                "sglang:forward_execution_seconds_total", 0.0
            )
        )
        if metric_total_s <= 0 or category_total_s <= 0:
            errors.append(f"zero forward GPU time: {label}")
        elif abs(metric_total_s - category_total_s) > max(0.02, metric_total_s * 1e-3):
            errors.append(f"forward category sum mismatch: {label}")
        successful = [item for item in result["requests"] if item["success"]]
        prompt_tokens = sum(int(item["prompt_tokens"]) for item in successful)
        completion_tokens = sum(int(item["completion_tokens"]) for item in successful)
        decode_positions = sum(
            max(int(item["completion_tokens"]) - 1, 0) for item in successful
        )
        workload = match["workload"]
        if workload == "prefill-50k":
            if extend_s <= 0:
                errors.append(f"missing extend GPU time: {label}")
                primary_tokens = prompt_tokens
                primary_s = 0.0
            else:
                primary_tokens = prompt_tokens
                primary_s = extend_s
        else:
            if decode_s <= 0 or decode_positions <= 0:
                errors.append(f"missing decode GPU work: {label}")
                primary_tokens = decode_positions
                primary_s = 0.0
            else:
                primary_tokens = decode_positions
                primary_s = decode_s
        info = result["server_info"]
        rows.append(
            {
                "model": match["model"],
                "repetition": int(match["repetition"]),
                "page_size": int(match["page"]),
                "workload": workload,
                "variant": variant,
                "attention_backend": info["attention_backend"],
                "linear_attn_backend": info["linear_attn_backend"],
                "mamba_backend": info["mamba_backend"],
                "completed_requests": len(successful),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "decode_positions": decode_positions,
                "extend_gpu_s": extend_s,
                "decode_gpu_s": decode_s,
                "total_forward_gpu_s": metric_total_s,
                "primary_forward_tokens": primary_tokens,
                "primary_forward_gpu_s": primary_s,
                "primary_forward_tok_s": (
                    primary_tokens / primary_s if primary_s > 0 else 0.0
                ),
                "wall_total_tok_s": float(result["summary"]["total_token_throughput"]),
                "result_path": str(path.relative_to(REPO_ROOT)),
            }
        )
    return rows, errors


def paired_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    indexed = {
        (
            str(row["model"]),
            int(row["repetition"]),
            int(row["page_size"]),
            str(row["workload"]),
            str(row["variant"]),
        ): row
        for row in rows
    }
    identities = sorted(
        {
            (
                str(row["model"]),
                int(row["repetition"]),
                int(row["page_size"]),
                str(row["workload"]),
            )
            for row in rows
            if row["variant"] in ("l1-lf", "l1-pf-static")
        }
    )
    paired: list[dict[str, Any]] = []
    errors: list[str] = []
    for model, repetition, page, workload in identities:
        lf = indexed.get((model, repetition, page, workload, "l1-lf"))
        pf = indexed.get((model, repetition, page, workload, "l1-pf-static"))
        auto = indexed.get((model, repetition, page, workload, "l1-lf-auto"))
        if lf is None or pf is None:
            errors.append(
                f"missing LF/PF pair: {model}/r{repetition}/p{page}/{workload}"
            )
            continue
        lf_rate = float(lf["primary_forward_tok_s"])
        pf_rate = float(pf["primary_forward_tok_s"])
        auto_rate = float(auto["primary_forward_tok_s"]) if auto else None
        paired.append(
            {
                "model": model,
                "repetition": repetition,
                "page_size": page,
                "workload": workload,
                "lf_triton_forward_tok_s": lf_rate,
                "pf_triton_forward_tok_s": pf_rate,
                "layout_pf_over_lf": pf_rate / lf_rate,
                "lf_auto_forward_tok_s": auto_rate,
                "backend_auto_over_triton": (
                    auto_rate / lf_rate if auto_rate is not None else None
                ),
            }
        )

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[
            (str(row["model"]), int(row["page_size"]), str(row["workload"]))
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, page, workload), group in sorted(grouped.items()):
        layout = [float(row["layout_pf_over_lf"]) for row in group]
        backend = [
            float(row["backend_auto_over_triton"])
            for row in group
            if row["backend_auto_over_triton"] is not None
        ]
        summaries.append(
            {
                "model": model,
                "page_size": page,
                "workload": workload,
                "layout_pairs": len(layout),
                "layout_pf_over_lf_mean": statistics.fmean(layout),
                "layout_pf_over_lf_std": (
                    statistics.stdev(layout) if len(layout) > 1 else 0.0
                ),
                "layout_pf_wins": sum(value > 1 for value in layout),
                "backend_pairs": len(backend),
                "backend_auto_over_triton_mean": (
                    statistics.fmean(backend) if backend else None
                ),
                "backend_auto_over_triton_std": (
                    statistics.stdev(backend) if len(backend) > 1 else 0.0
                ),
                "backend_auto_wins": sum(value > 1 for value in backend),
            }
        )
    return paired, summaries, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_l1_forward_taste_743cae2",
    )
    parser.add_argument(
        "--run-prefix",
        choices=["l1-forward-taste", "l1-forward-scale"],
        default="l1-forward-taste",
    )
    parser.add_argument(
        "--models", nargs="+", choices=["0.8b", "4b", "9b"], default=["0.8b"]
    )
    parser.add_argument("--triton-repetitions", type=int, default=5)
    parser.add_argument("--auto-repetitions", type=int, default=3)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output = root / "summary"
    models = set(args.models)
    rows, errors = build_rows(root, run_prefix=args.run_prefix, models=models)
    paired, summaries, pair_errors = paired_rows(rows)
    errors.extend(pair_errors)
    expected = (
        len(models)
        * 3
        * 2
        * (2 * args.triton_repetitions + args.auto_repetitions)
    )
    if not args.allow_partial and len(rows) != expected:
        errors.append(f"expected {expected} runs, got {len(rows)}")
    expected_layout_pairs = len(models) * 3 * 2 * args.triton_repetitions
    if not args.allow_partial and len(paired) != expected_layout_pairs:
        errors.append(
            f"expected {expected_layout_pairs} layout pairs, got {len(paired)}"
        )
    write_csv(output / "run_summary.csv", rows)
    write_csv(output / "paired_runs.csv", paired)
    write_csv(output / "paired_summary.csv", summaries)
    audit = {
        "schema_version": 1,
        "server_sha": SERVER_SHA,
        "run_prefix": args.run_prefix,
        "models": sorted(models),
        "runs": len(rows),
        "expected_runs": expected,
        "layout_pairs": len(paired),
        "expected_layout_pairs": expected_layout_pairs,
        "complete": len(rows) == expected and len(paired) == expected_layout_pairs,
        "passed": not errors,
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if errors and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
