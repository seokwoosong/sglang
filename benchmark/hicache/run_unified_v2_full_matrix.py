#!/usr/bin/env python3
"""Run the complete patched unified-v9 matrix against the existing static oracle."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import run_static_unified_oracle_matrix as matrix


DEFAULT_ARTIFACT_ROOT = matrix.REPO_ROOT / "artifacts/unified_v9_full_5b2d430"


@dataclass(frozen=True)
class Task:
    label: str
    name: str
    launcher_variant: str
    command: list[str]
    mixed: bool = False


def homogeneous_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for repetition in args.repetitions:
        for model in args.models:
            for workload in args.workloads:
                for reuse in args.prefix_reuse:
                    name, launcher_variant, command = matrix.build_steady_command(
                        args,
                        stage="unified-v9-final",
                        model_name=model,
                        variant_name="unified-triton",
                        workload_name=workload,
                        reuse=reuse,
                        ratio=0.5,
                        repetition=repetition,
                    )
                    tasks.append(
                        Task(
                            label=(
                                f"homogeneous/{model}/{workload}/"
                                f"{matrix.reuse_label(reuse)}/rep{repetition}"
                            ),
                            name=name,
                            launcher_variant=launcher_variant,
                            command=command,
                        )
                    )
    random.Random(25_108).shuffle(tasks)
    return tasks


def mixed_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for repetition in args.repetitions:
        for model in args.models:
            for reuse in args.prefix_reuse:
                name, launcher_variant, command = matrix.build_mixed_command(
                    args,
                    model_name=model,
                    variant_name="unified-triton",
                    reuse=reuse,
                    ratio=0.5,
                    repetition=repetition,
                    stage="unified-v9-mixed-final",
                )
                tasks.append(
                    Task(
                        label=(
                            f"mixed/{model}/{matrix.reuse_label(reuse)}/"
                            f"rep{repetition}"
                        ),
                        name=name,
                        launcher_variant=launcher_variant,
                        command=command,
                        mixed=True,
                    )
                )
    random.Random(25_109).shuffle(tasks)
    return tasks


def run_tasks(args: argparse.Namespace, tasks: list[Task]) -> None:
    total = len(tasks)
    for index, task in enumerate(tasks, 1):
        print(f"MATRIX {index}/{total} {task.label}", flush=True)
        result = matrix.execute(
            args,
            label=task.label,
            name=task.name,
            launcher_variant=task.launcher_variant,
            command=task.command,
            allow_reproducible_failure=False,
        )
        if result is None and not args.dry_run:
            raise RuntimeError(f"No valid result for {task.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "homogeneous", "mixed", "all")
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--models", default="0.8b,4b")
    parser.add_argument("--workloads", default="short,middle,long")
    parser.add_argument("--prefix-reuse", default="0.2,0.5,0.8")
    parser.add_argument("--repetitions", default="1,2,3")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.models = matrix.csv_choices(args.models, matrix.MODELS)
    args.workloads = matrix.csv_choices(args.workloads, matrix.WORKLOADS)
    args.prefix_reuse = [float(value) for value in args.prefix_reuse.split(",")]
    if not set(args.prefix_reuse).issubset(matrix.ALL_PREFIX_REUSE):
        raise ValueError(f"Unsupported prefix reuse values: {args.prefix_reuse}")
    args.repetitions = [int(value) for value in args.repetitions.split(",")]
    if not set(args.repetitions).issubset(matrix.FINAL_REPETITIONS):
        raise ValueError(f"Unsupported repetitions: {args.repetitions}")
    args.retry_failures = True

    matrix.ALL_VARIANTS["unified-triton"] = "oracle-u3-v9"
    homogeneous = homogeneous_tasks(args)
    mixed = mixed_tasks(args)
    started = time.monotonic()

    if args.stage == "preflight":
        smoke = [
            next(task for task in homogeneous if f"/{model}/short/reuse050/" in task.label)
            for model in args.models
        ]
        run_tasks(args, smoke)
    else:
        if args.stage in {"homogeneous", "all"}:
            run_tasks(args, homogeneous)
        if args.stage in {"mixed", "all"}:
            run_tasks(args, mixed)
    print(f"COMPLETE stage={args.stage} elapsed_s={time.monotonic() - started:.3f}")


if __name__ == "__main__":
    main()
