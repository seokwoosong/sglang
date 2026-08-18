#!/usr/bin/env python3
"""Run the two-hour Qwen3.5 LF/PF forward-path taste matrix."""

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
TRITON_VARIANTS = ("l1-lf", "l1-pf-static")
AUTO_VARIANT = "l1-lf-auto"


@dataclass(frozen=True)
class Workload:
    label: str
    input_len: int
    output_len: int
    groups: int
    rounds: int
    concurrency: int


WORKLOADS = (
    Workload("prefill-50k", 50000, 1, 8, 2, 2),
    Workload("decode-10k-512", 10000, 512, 16, 2, 8),
)


def task_completed(root: Path, run_name: str, variant: str) -> bool:
    run_root = root / run_name / variant
    for path in reversed(sorted(run_root.glob("*/manifest.json"))):
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


def variants_for_repetition(repetition: int) -> list[str]:
    if repetition <= 3:
        return rotated((*TRITON_VARIANTS, AUTO_VARIANT), repetition)
    return rotated(TRITON_VARIANTS, repetition)


def command(
    args: argparse.Namespace,
    *,
    variant: str,
    workload: Workload,
    page_size: int,
    repetition: int,
) -> tuple[str, list[str]]:
    run_name = f"l1-forward-taste-r{repetition}-0.8b-{workload.label}-p{page_size}"
    invocation = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        variant,
        "--scenario",
        "steady",
        "--run-name",
        run_name,
        "--artifact-root",
        str(args.artifact_root),
        "--model",
        str(MODEL),
        "--max-total-tokens",
        "120000",
        "--page-size",
        str(page_size),
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--cuda-graph-mode",
        "disabled",
        "--server-extra-arg=--mem-fraction-static",
        "--server-extra-arg=0.27",
        "--server-extra-arg=--disable-radix-cache",
        "--server-env=SGLANG_ENABLE_METRICS_DEVICE_TIMER=true",
        "--input-len",
        str(workload.input_len),
        "--output-len",
        str(workload.output_len),
        "--groups",
        str(workload.groups),
        "--rounds",
        str(workload.rounds),
        "--shared-ratio",
        "0.95",
        "--prime-repeats",
        "0",
        "--max-concurrency",
        str(workload.concurrency),
        "--reverse-group-order",
        "--forbid-dropped",
        "--no-profile-memory-breakdown",
        "--server-timeout",
        "1200",
        "--client-timeout",
        "3600",
    ]
    return run_name, invocation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--triton-repetitions", type=int, default=5)
    parser.add_argument("--auto-repetitions", type=int, default=3)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_l1_forward_taste_743cae2",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=6900,
        help="Stop launching new tasks after this elapsed wall time.",
    )
    parser.add_argument(
        "--deadline-reserve-seconds",
        type=float,
        default=180,
        help="Do not start a task if less than this much deadline remains.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.artifact_root = args.artifact_root.resolve()
    if args.auto_repetitions > args.triton_repetitions:
        raise ValueError("auto repetitions cannot exceed Triton repetitions")
    started = time.monotonic()
    completed = 0
    skipped = 0
    planned = (
        len(args.pages)
        * len(WORKLOADS)
        * (2 * args.triton_repetitions + args.auto_repetitions)
    )
    for repetition in range(1, args.triton_repetitions + 1):
        pages = list(args.pages)
        workloads = list(WORKLOADS)
        if repetition % 2 == 0:
            pages.reverse()
            workloads.reverse()
        variants = [
            variant
            for variant in variants_for_repetition(repetition)
            if variant != AUTO_VARIANT or repetition <= args.auto_repetitions
        ]
        for page_size in pages:
            for workload in workloads:
                for variant in variants:
                    run_name, invocation = command(
                        args,
                        variant=variant,
                        workload=workload,
                        page_size=page_size,
                        repetition=repetition,
                    )
                    if args.resume and task_completed(
                        args.artifact_root, run_name, variant
                    ):
                        skipped += 1
                        print(f"skip completed: {run_name}/{variant}", flush=True)
                        continue
                    elapsed = time.monotonic() - started
                    remaining = args.deadline_seconds - elapsed
                    if remaining < args.deadline_reserve_seconds:
                        print(
                            "deadline reached: "
                            f"completed={completed} skipped={skipped} planned={planned} "
                            f"elapsed={elapsed:.1f}s remaining={remaining:.1f}s",
                            flush=True,
                        )
                        return
                    print(
                        f"start: {run_name}/{variant} remaining={remaining:.1f}s",
                        flush=True,
                    )
                    task_started = time.monotonic()
                    result = subprocess.run(invocation, cwd=REPO_ROOT, check=False)
                    task_elapsed = time.monotonic() - task_started
                    print(
                        f"finish: {run_name}/{variant} exit={result.returncode} "
                        f"elapsed={task_elapsed:.1f}s",
                        flush=True,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"forward taste task failed: {run_name}/{variant}"
                        )
                    completed += 1
    elapsed = time.monotonic() - started
    print(
        f"matrix complete: completed={completed} skipped={skipped} "
        f"planned={planned} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
