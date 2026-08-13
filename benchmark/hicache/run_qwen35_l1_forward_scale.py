#!/usr/bin/env python3
"""Run the Qwen3.5 4B/9B LF/PF forward-only scaling matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "benchmark/hicache/run_unified_ablation.py"
ANALYZER = REPO_ROOT / "benchmark/hicache/analyze_qwen35_l1_forward_taste.py"
PYTHON = Path("/home/sukwoo24/.venv_sglang/bin/python")
SERVER_SHA = "743cae224c5bc28687457558a074736776350392"

MODELS = {
    "4b": Path(
        "/home/sukwoo24/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ),
    "9b": Path(
        "/home/sukwoo24/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-9B/snapshots/"
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    ),
}
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


def run_name(
    *, model_label: str, workload: Workload, page_size: int, repetition: int
) -> str:
    return (
        f"l1-forward-scale-r{repetition}-{model_label}-"
        f"{workload.label}-p{page_size}"
    )


def task_completed(
    root: Path,
    *,
    name: str,
    variant: str,
    model_path: Path,
    page_size: int,
) -> bool:
    run_root = root / name / variant
    for path in reversed(sorted(run_root.glob("*/manifest.json"))):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        arguments = manifest.get("arguments", {})
        if (
            manifest.get("status") == "completed"
            and manifest.get("variant_definition", {}).get("sha") == SERVER_SHA
            and Path(arguments.get("model", "")) == model_path
            and int(arguments.get("page_size", -1)) == page_size
            and int(arguments.get("max_total_tokens", -1)) == 120000
        ):
            return True
    return False


def rotated(items: tuple[str, ...], repetition: int) -> list[str]:
    offset = (repetition - 1) % len(items)
    values = list(items)
    return values[offset:] + values[:offset]


def variants_for_repetition(repetition: int, auto_repetitions: int) -> list[str]:
    if repetition <= auto_repetitions:
        return rotated((*TRITON_VARIANTS, AUTO_VARIANT), repetition)
    return rotated(TRITON_VARIANTS, repetition)


def invocation(
    args: argparse.Namespace,
    *,
    model_label: str,
    model_path: Path,
    variant: str,
    workload: Workload,
    page_size: int,
    repetition: int,
) -> tuple[str, list[str]]:
    name = run_name(
        model_label=model_label,
        workload=workload,
        page_size=page_size,
        repetition=repetition,
    )
    command = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        variant,
        "--scenario",
        "steady",
        "--run-name",
        name,
        "--artifact-root",
        str(args.artifact_root),
        "--model",
        str(model_path),
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
    return name, command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", choices=sorted(MODELS), nargs="+", default=["4b", "9b"]
    )
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--triton-repetitions", type=int, default=5)
    parser.add_argument("--auto-repetitions", type=int, default=3)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_l1_forward_scale_743cae2",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--analyze-on-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.artifact_root = args.artifact_root.resolve()
    if args.auto_repetitions > args.triton_repetitions:
        raise ValueError("auto repetitions cannot exceed Triton repetitions")
    if args.max_attempts < 1:
        raise ValueError("max attempts must be positive")

    planned = (
        len(args.models)
        * len(args.pages)
        * len(WORKLOADS)
        * (2 * args.triton_repetitions + args.auto_repetitions)
    )
    started = time.monotonic()
    completed = 0
    skipped = 0

    for repetition in range(1, args.triton_repetitions + 1):
        model_labels = list(args.models)
        pages = list(args.pages)
        workloads = list(WORKLOADS)
        if repetition % 2 == 0:
            model_labels.reverse()
            pages.reverse()
            workloads.reverse()
        variants = variants_for_repetition(repetition, args.auto_repetitions)
        for model_label in model_labels:
            model_path = MODELS[model_label]
            if not model_path.is_dir():
                raise FileNotFoundError(f"model snapshot not found: {model_path}")
            for page_size in pages:
                for workload in workloads:
                    for variant in variants:
                        name, command = invocation(
                            args,
                            model_label=model_label,
                            model_path=model_path,
                            variant=variant,
                            workload=workload,
                            page_size=page_size,
                            repetition=repetition,
                        )
                        if args.resume and task_completed(
                            args.artifact_root,
                            name=name,
                            variant=variant,
                            model_path=model_path,
                            page_size=page_size,
                        ):
                            skipped += 1
                            print(
                                f"skip completed [{completed + skipped}/{planned}]: "
                                f"{name}/{variant}",
                                flush=True,
                            )
                            continue
                        if args.dry_run:
                            print(" ".join(command), flush=True)
                            continue

                        for attempt in range(1, args.max_attempts + 1):
                            print(
                                f"start [{completed + skipped + 1}/{planned}] "
                                f"attempt={attempt}/{args.max_attempts}: "
                                f"{name}/{variant}",
                                flush=True,
                            )
                            task_started = time.monotonic()
                            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
                            task_elapsed = time.monotonic() - task_started
                            print(
                                f"finish exit={result.returncode} "
                                f"elapsed={task_elapsed:.1f}s: {name}/{variant}",
                                flush=True,
                            )
                            if result.returncode == 0:
                                completed += 1
                                break
                            if attempt == args.max_attempts:
                                raise RuntimeError(
                                    f"task failed after {args.max_attempts} attempts: "
                                    f"{name}/{variant}"
                                )
                            time.sleep(5)

    if args.dry_run:
        print(f"dry run complete: planned={planned}", flush=True)
        return

    elapsed = time.monotonic() - started
    print(
        f"matrix complete: completed={completed} skipped={skipped} "
        f"planned={planned} elapsed={elapsed:.1f}s",
        flush=True,
    )
    if args.analyze_on_completion:
        analyze = [
            str(PYTHON),
            str(ANALYZER),
            "--artifact-root",
            str(args.artifact_root),
            "--run-prefix",
            "l1-forward-scale",
            "--models",
            *args.models,
            "--triton-repetitions",
            str(args.triton_repetitions),
            "--auto-repetitions",
            str(args.auto_repetitions),
        ]
        subprocess.run(analyze, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
