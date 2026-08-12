#!/usr/bin/env python3
"""Run the Qwen3.5 L1 layer-first versus page-first evaluation matrix.

The primary comparison keeps the static allocator fixed and toggles only the
device-pool layout.  A third page-first unified variant separates allocator and
dynamic-sharing effects from the layout effect.  HiCache is disabled in every
variant.
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
MODEL = Path(
    "/home/sukwoo24/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
SERVER_SHA = "743cae224c5bc28687457558a074736776350392"
VARIANTS = ("l1-lf", "l1-pf-static", "l1-pf-unified")
GRAPH_MODES = ("enabled", "disabled")


@dataclass(frozen=True)
class Workload:
    label: str
    input_len: int
    output_len: int
    groups: int
    rounds: int
    resident_concurrency: int
    pressure_concurrency: int


WORKLOADS = (
    Workload("short-3k", 3000, 256, 120, 2, 8, 8),
    Workload("middle-10k", 10000, 256, 40, 2, 8, 8),
    # Two simultaneous 50K requests fit within the common 120K resident budget.
    Workload("long-50k", 50000, 128, 8, 2, 2, 4),
)


def task_completed(root: Path, run_name: str, variant: str) -> bool:
    for path in reversed(sorted((root / run_name / variant).glob("*/manifest.json"))):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("status") == "completed"
            and manifest.get("variant_definition", {}).get("sha") == SERVER_SHA
        ):
            return True
    return False


def rotated(items: tuple[str, ...], repetition: int) -> list[str]:
    offset = (repetition - 1) % len(items)
    values = list(items)
    return values[offset:] + values[:offset]


def ordered_pages(args: argparse.Namespace) -> list[int]:
    return list(reversed(args.pages)) if args.repetition % 2 == 0 else args.pages


def ordered_workloads(args: argparse.Namespace) -> list[Workload]:
    values = list(WORKLOADS)
    return list(reversed(values)) if args.repetition % 2 == 0 else values


def ordered_graph_modes(args: argparse.Namespace) -> list[str]:
    values = list(args.graph_modes)
    return list(reversed(values)) if args.repetition % 2 == 0 else values


def common_command(
    args: argparse.Namespace,
    *,
    variant: str,
    run_name: str,
    page_size: int,
    scenario: str,
    input_len: int,
    output_len: int,
    graph_mode: str,
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
        str(MODEL),
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
        "--cuda-graph-mode",
        graph_mode,
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
    args: argparse.Namespace, *, command: list[str], run_name: str, variant: str
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
        raise RuntimeError(f"L1 layout task failed: {run_name}/{variant}")


def run_parity(args: argparse.Namespace) -> None:
    for page_size in ordered_pages(args):
        for graph_mode in ordered_graph_modes(args):
            for variant in rotated(VARIANTS, args.repetition):
                run_name = (
                    f"l1-parity-r{args.repetition}-0.8b-p{page_size}-cg-{graph_mode}"
                )
                command = common_command(
                    args,
                    variant=variant,
                    run_name=run_name,
                    page_size=page_size,
                    scenario="parity",
                    input_len=10000,
                    output_len=64,
                    graph_mode=graph_mode,
                    profile=False,
                )
                command.extend(["--pressure-requests", "40", "--logprob-atol", "0.02"])
                run_task(args, command=command, run_name=run_name, variant=variant)


def run_resident(args: argparse.Namespace) -> None:
    """Measure fresh resident L1 reads/writes without radix reuse or eviction."""
    for page_size in ordered_pages(args):
        for workload in ordered_workloads(args):
            for graph_mode in ordered_graph_modes(args):
                for variant in rotated(VARIANTS, args.repetition):
                    run_name = (
                        f"l1-resident-r{args.repetition}-0.8b-{workload.label}-"
                        f"p{page_size}-cg-{graph_mode}"
                    )
                    command = common_command(
                        args,
                        variant=variant,
                        run_name=run_name,
                        page_size=page_size,
                        scenario="steady",
                        input_len=workload.input_len,
                        output_len=workload.output_len,
                        graph_mode=graph_mode,
                        profile=False,
                    )
                    command.extend(
                        [
                            "--server-extra-arg=--disable-radix-cache",
                            "--groups",
                            str(workload.groups),
                            "--rounds",
                            str(workload.rounds),
                            "--shared-ratio",
                            "0.95",
                            "--prime-output-len",
                            "1",
                            "--prime-repeats",
                            "0",
                            "--max-concurrency",
                            str(workload.resident_concurrency),
                            "--reverse-group-order",
                        ]
                    )
                    run_task(args, command=command, run_name=run_name, variant=variant)


def pressure_command(
    args: argparse.Namespace,
    *,
    variant: str,
    page_size: int,
    workload: Workload,
    run_name: str,
    output_len: int,
    graph_mode: str,
    profile: bool,
) -> list[str]:
    command = common_command(
        args,
        variant=variant,
        run_name=run_name,
        page_size=page_size,
        scenario="steady",
        input_len=workload.input_len,
        output_len=output_len,
        graph_mode=graph_mode,
        profile=profile,
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
            str(workload.pressure_concurrency if not profile else 4),
            "--reverse-group-order",
            "--require-eviction",
        ]
    )
    return command


def run_pressure(args: argparse.Namespace) -> None:
    """Measure L1 radix retention and eviction with no host cache."""
    for page_size in ordered_pages(args):
        for workload in ordered_workloads(args):
            for variant in rotated(VARIANTS, args.repetition):
                run_name = (
                    f"l1-pressure-r{args.repetition}-0.8b-{workload.label}-p{page_size}"
                )
                command = pressure_command(
                    args,
                    variant=variant,
                    page_size=page_size,
                    workload=workload,
                    run_name=run_name,
                    output_len=workload.output_len,
                    graph_mode="enabled",
                    profile=False,
                )
                run_task(args, command=command, run_name=run_name, variant=variant)


def run_profile(args: argparse.Namespace) -> None:
    workload = next(item for item in WORKLOADS if item.label == "middle-10k")
    for page_size in ordered_pages(args):
        for variant in rotated(VARIANTS, args.repetition):
            run_name = (
                f"l1-profile-r{args.repetition}-0.8b-{workload.label}-p{page_size}"
            )
            command = pressure_command(
                args,
                variant=variant,
                page_size=page_size,
                workload=workload,
                run_name=run_name,
                output_len=64,
                graph_mode="disabled",
                profile=True,
            )
            run_task(args, command=command, run_name=run_name, variant=variant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("parity", "resident", "pressure", "profile"))
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--graph-modes",
        nargs="+",
        choices=GRAPH_MODES,
        default=list(GRAPH_MODES),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_l1_layout_743cae2",
    )
    parser.add_argument("--max-total-tokens", type=int, default=120000)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.27)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.artifact_root = args.artifact_root.resolve()
    started = time.monotonic()
    {
        "parity": run_parity,
        "resident": run_resident,
        "pressure": run_pressure,
        "profile": run_profile,
    }[args.stage](args)
    print(f"matrix elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
