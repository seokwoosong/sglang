#!/usr/bin/env python3
"""Run the final Qwen3.5 unified-memory + HiCache evaluation matrix.

The matrix is intentionally resumable: a completed launcher manifest skips the
matching task, while a failed task stops the matrix so it can be investigated
before later measurements consume GPU time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "benchmark/hicache/run_unified_ablation.py"
PYTHON = Path("/home/sukwoo24/.venv_sglang/bin/python")
MODELS = {
    "0.8b": Path(
        "/home/sukwoo24/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17"
    ),
    "4b": Path(
        "/home/sukwoo24/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ),
}


@dataclass(frozen=True)
class Workload:
    label: str
    input_len: int
    output_len: int
    groups: int
    rounds: int
    concurrency: int


WORKLOADS = (
    Workload("short-3k", 3000, 256, 120, 2, 8),
    Workload("middle-10k", 10000, 256, 40, 2, 8),
    Workload("long-50k", 50000, 128, 8, 2, 4),
)
VARIANT_POLICIES = {
    "eval-s1": ("write_back", "write_through"),
    "eval-u0": ("none",),
    "eval-u2": ("write_back", "write_through"),
    "eval-u3": ("write_back", "write_through"),
}
GRAPH_VARIANTS = ("eval-s1", "eval-u0", "eval-u3")
GRAPH_WORKLOADS = ("short-3k", "long-50k")


def task_completed(artifact_root: Path, run_name: str, variant: str) -> bool:
    manifests = sorted((artifact_root / run_name / variant).glob("*/manifest.json"))
    for path in reversed(manifests):
        try:
            if json.loads(path.read_text()).get("status") == "completed":
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def common_command(
    args: argparse.Namespace,
    *,
    variant: str,
    policy: str,
    page_size: int,
    run_name: str,
    scenario: str,
    input_len: int,
    output_len: int,
    cuda_graph_mode: str,
    profile: bool,
) -> list[str]:
    command = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        variant,
        "--scenario",
        scenario,
        "--run-name",
        run_name,
        "--artifact-root",
        str(args.artifact_root),
        "--model",
        str(MODELS[args.model_size]),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--page-size",
        str(page_size),
        "--max-running-requests",
        str(args.max_running_requests),
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--hicache-size",
        str(args.hicache_size),
        "--hicache-write-policy",
        "write_back" if policy == "none" else policy,
        "--cuda-graph-mode",
        cuda_graph_mode,
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={args.mem_fraction_static}",
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--server-timeout",
        "1200",
        "--client-timeout",
        "3600",
        "--forbid-dropped",
    ]
    command.append(
        "--profile-memory-breakdown" if profile else "--no-profile-memory-breakdown"
    )
    return command


def run_task(
    args: argparse.Namespace,
    *,
    command: list[str],
    run_name: str,
    variant: str,
) -> None:
    if args.resume and task_completed(args.artifact_root, run_name, variant):
        print(f"skip completed: {run_name}/{variant}", flush=True)
        return
    print(f"start: {run_name}/{variant}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed = time.monotonic() - started
    print(
        f"finish: {run_name}/{variant} exit={completed.returncode} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Matrix task failed: {run_name}/{variant}")


def iter_variant_policies(args: argparse.Namespace):
    selected = set(args.variants)
    for variant, policies in VARIANT_POLICIES.items():
        if variant not in selected:
            continue
        for policy in policies:
            yield variant, policy


def run_clean(args: argparse.Namespace) -> None:
    for page_size in args.pages:
        for variant, policy in iter_variant_policies(args):
            for workload in WORKLOADS:
                concurrency = (
                    args.long_concurrency
                    if workload.label == "long-50k"
                    else workload.concurrency
                )
                concurrency_suffix = (
                    f"-c{concurrency}"
                    if workload.label == "long-50k"
                    and concurrency != workload.concurrency
                    else ""
                )
                run_name = (
                    f"clean-r{args.repetition}-{args.model_size}-{workload.label}-"
                    f"p{page_size}-{policy}{concurrency_suffix}"
                )
                command = common_command(
                    args,
                    variant=variant,
                    policy=policy,
                    page_size=page_size,
                    run_name=run_name,
                    scenario="steady",
                    input_len=workload.input_len,
                    output_len=workload.output_len,
                    cuda_graph_mode="enabled",
                    profile=False,
                )
                command.extend(
                    [
                        "--groups",
                        str(workload.groups),
                        "--rounds",
                        str(workload.rounds),
                        "--shared-ratio",
                        "0.95",
                        "--prime-output-len",
                        "1",
                        "--prime-repeats",
                        "1",
                        "--max-concurrency",
                        str(concurrency),
                        "--reverse-group-order",
                        "--require-eviction",
                    ]
                )
                if variant != "eval-u0":
                    command.extend(
                        ["--require-loadback", "--require-backup", "--require-host-hit"]
                    )
                run_task(
                    args,
                    command=command,
                    run_name=run_name,
                    variant=variant,
                )


def run_profile(args: argparse.Namespace) -> None:
    workload = next(item for item in WORKLOADS if item.label == "middle-10k")
    for page_size in args.pages:
        for variant, policy in iter_variant_policies(args):
            run_name = (
                f"profile-r{args.repetition}-{args.model_size}-{workload.label}-"
                f"p{page_size}-{policy}"
            )
            command = common_command(
                args,
                variant=variant,
                policy=policy,
                page_size=page_size,
                run_name=run_name,
                scenario="steady",
                input_len=workload.input_len,
                output_len=64,
                cuda_graph_mode="disabled",
                profile=True,
            )
            command.extend(
                [
                    "--groups",
                    str(workload.groups),
                    "--rounds",
                    str(workload.rounds),
                    "--shared-ratio",
                    "0.95",
                    "--prime-output-len",
                    "1",
                    "--prime-repeats",
                    "1",
                    "--max-concurrency",
                    "4",
                    "--reverse-group-order",
                    "--require-eviction",
                ]
            )
            if variant != "eval-u0":
                command.extend(
                    ["--require-loadback", "--require-backup", "--require-host-hit"]
                )
            run_task(args, command=command, run_name=run_name, variant=variant)


def run_parity(args: argparse.Namespace) -> None:
    for page_size in args.pages:
        for variant, policy in iter_variant_policies(args):
            run_name = (
                f"parity-r{args.repetition}-{args.model_size}-p{page_size}-{policy}"
            )
            command = common_command(
                args,
                variant=variant,
                policy=policy,
                page_size=page_size,
                run_name=run_name,
                scenario="parity",
                input_len=10000,
                output_len=64,
                cuda_graph_mode="enabled",
                profile=False,
            )
            command.extend(["--pressure-requests", "40", "--logprob-atol", "0.02"])
            run_task(args, command=command, run_name=run_name, variant=variant)


def graph_variant_order(args: argparse.Namespace) -> list[str]:
    selected = set(args.variants)
    variants = [variant for variant in GRAPH_VARIANTS if variant in selected]
    if not variants:
        raise ValueError(f"Graph stages require one of {GRAPH_VARIANTS}")
    offset = (args.repetition - 1) % len(variants)
    return variants[offset:] + variants[:offset]


def graph_mode_order(args: argparse.Namespace) -> list[str]:
    modes = list(args.cuda_graph_modes)
    return list(reversed(modes)) if args.repetition % 2 == 0 else modes


def run_graph(args: argparse.Namespace) -> None:
    workloads = [item for item in WORKLOADS if item.label in args.graph_workloads]
    if args.repetition % 2 == 0:
        workloads.reverse()
    for variant in graph_variant_order(args):
        policy = "none" if variant == "eval-u0" else "write_back"
        for workload in workloads:
            concurrency = (
                args.long_concurrency
                if workload.label == "long-50k"
                else workload.concurrency
            )
            for cuda_graph_mode in graph_mode_order(args):
                run_name = (
                    f"graph-r{args.repetition}-{args.model_size}-{workload.label}-"
                    f"p{args.pages[0]}-{policy}-cg-{cuda_graph_mode}"
                )
                command = common_command(
                    args,
                    variant=variant,
                    policy=policy,
                    page_size=args.pages[0],
                    run_name=run_name,
                    scenario="steady",
                    input_len=workload.input_len,
                    output_len=workload.output_len,
                    cuda_graph_mode=cuda_graph_mode,
                    profile=False,
                )
                command.extend(
                    [
                        "--groups",
                        str(workload.groups),
                        "--rounds",
                        str(workload.rounds),
                        "--shared-ratio",
                        "0.95",
                        "--prime-output-len",
                        "1",
                        "--prime-repeats",
                        "1",
                        "--max-concurrency",
                        str(concurrency),
                        "--reverse-group-order",
                        "--require-eviction",
                    ]
                )
                if variant != "eval-u0":
                    command.extend(
                        ["--require-loadback", "--require-backup", "--require-host-hit"]
                    )
                run_task(args, command=command, run_name=run_name, variant=variant)


def run_graph_parity(args: argparse.Namespace) -> None:
    for variant in graph_variant_order(args):
        policy = "none" if variant == "eval-u0" else "write_back"
        for cuda_graph_mode in graph_mode_order(args):
            run_name = (
                f"graph-parity-r{args.repetition}-{args.model_size}-"
                f"p{args.pages[0]}-{policy}-cg-{cuda_graph_mode}"
            )
            command = common_command(
                args,
                variant=variant,
                policy=policy,
                page_size=args.pages[0],
                run_name=run_name,
                scenario="parity",
                input_len=10000,
                output_len=64,
                cuda_graph_mode=cuda_graph_mode,
                profile=False,
            )
            command.extend(["--pressure-requests", "40", "--logprob-atol", "0.02"])
            run_task(args, command=command, run_name=run_name, variant=variant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["clean", "profile", "parity", "graph", "graph-parity"]
    )
    parser.add_argument("--model-size", choices=sorted(MODELS), default="0.8b")
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANT_POLICIES),
        default=list(VARIANT_POLICIES),
    )
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_hicache_matrix",
    )
    parser.add_argument("--max-total-tokens", type=int, default=120000)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument(
        "--long-concurrency",
        type=int,
        default=4,
        help="Client concurrency for the 50k-token workload.",
    )
    parser.add_argument("--hicache-size", type=int, default=12)
    parser.add_argument("--mem-fraction-static", type=float, default=0.27)
    parser.add_argument(
        "--cuda-graph-modes",
        nargs="+",
        choices=["enabled", "disabled"],
        default=["enabled", "disabled"],
        help="CUDA graph modes used by the graph ablation stages.",
    )
    parser.add_argument(
        "--graph-workloads",
        nargs="+",
        choices=list(GRAPH_WORKLOADS),
        default=list(GRAPH_WORKLOADS),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.artifact_root = args.artifact_root.resolve()
    started = time.monotonic()
    if args.stage in {"graph", "graph-parity"} and len(args.pages) != 1:
        raise ValueError("Graph ablation stages require exactly one page size")
    {
        "clean": run_clean,
        "profile": run_profile,
        "parity": run_parity,
        "graph": run_graph,
        "graph-parity": run_graph_parity,
    }[args.stage](args)
    print(f"matrix elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
