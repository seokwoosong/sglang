#!/usr/bin/env python3
"""Re-evaluate every unified-triton condition on the patched source SHA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import run_static_unified_oracle_matrix as matrix


DEFAULT_ARTIFACT_ROOT = matrix.REPO_ROOT / "artifacts/unified_patched_8de3268"


def csv_values(raw: str, allowed: set[str]) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown values: {sorted(unknown)}")
    return values


def run_homogeneous(args: argparse.Namespace) -> None:
    for repetition in args.repetitions:
        for model in args.models:
            for workload in args.workloads:
                for reuse in matrix.ALL_PREFIX_REUSE:
                    name, launcher_variant, command = matrix.build_steady_command(
                        args,
                        stage="patched-final",
                        model_name=model,
                        variant_name="unified-triton",
                        workload_name=workload,
                        reuse=reuse,
                        ratio=0.5,
                        repetition=repetition,
                    )
                    matrix.execute(
                        args,
                        label=(
                            f"patched/homogeneous/{model}/{workload}/"
                            f"{matrix.reuse_label(reuse)}/rep{repetition}"
                        ),
                        name=name,
                        launcher_variant=launcher_variant,
                        command=command,
                    )


def run_mixed(args: argparse.Namespace) -> None:
    for repetition in args.repetitions:
        for model in args.models:
            for reuse in matrix.ALL_PREFIX_REUSE:
                name, launcher_variant, command = matrix.build_mixed_command(
                    args,
                    model_name=model,
                    variant_name="unified-triton",
                    reuse=reuse,
                    ratio=0.5,
                    repetition=repetition,
                    stage="patched-mixed",
                )
                matrix.execute(
                    args,
                    label=(
                        f"patched/mixed/{model}/{matrix.reuse_label(reuse)}/"
                        f"rep{repetition}"
                    ),
                    name=name,
                    launcher_variant=launcher_variant,
                    command=command,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("homogeneous", "mixed", "all"), default="all", nargs="?"
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--models", default="0.8b,4b")
    parser.add_argument("--workloads", default="short,middle,long")
    parser.add_argument("--repetitions", default="1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.models = csv_values(args.models, set(matrix.MODELS))
    args.workloads = csv_values(args.workloads, set(matrix.WORKLOADS))
    args.repetitions = [int(value) for value in args.repetitions.split(",")]
    args.retry_failures = False

    started = time.monotonic()
    if args.stage in ("homogeneous", "all"):
        run_homogeneous(args)
    if args.stage in ("mixed", "all"):
        run_mixed(args)
    print(f"COMPLETE stage={args.stage} elapsed_s={time.monotonic() - started:.3f}")


if __name__ == "__main__":
    main()
