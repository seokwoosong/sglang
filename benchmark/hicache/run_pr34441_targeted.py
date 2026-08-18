#!/usr/bin/env python3
"""Run a small PR #34441-only unified performance isolation matrix."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import run_static_unified_oracle_matrix as matrix


DEFAULT_ARTIFACT_ROOT = matrix.REPO_ROOT / "artifacts/unified_pr34441_0e3deb8_targeted"
HOMOGENEOUS_CASES = (
    ("4b", "short", 0.2),
    ("4b", "short", 0.5),
    ("4b", "short", 0.8),
    ("0.8b", "short", 0.5),
    ("4b", "middle", 0.5),
)
MIXED_CASES = (("4b", 0.5),)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--repetitions", default="1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.repetitions = [int(value) for value in args.repetitions.split(",")]
    args.retry_failures = False

    matrix.ALL_VARIANTS["unified-pr34441"] = "pr34441-u3"
    started = time.monotonic()
    for repetition in args.repetitions:
        for model, workload, reuse in HOMOGENEOUS_CASES:
            name, launcher_variant, command = matrix.build_steady_command(
                args,
                stage="pr34441-targeted",
                model_name=model,
                variant_name="unified-pr34441",
                workload_name=workload,
                reuse=reuse,
                ratio=0.5,
                repetition=repetition,
            )
            matrix.execute(
                args,
                label=f"pr34441/{model}/{workload}/reuse{int(reuse * 100)}/rep{repetition}",
                name=name,
                launcher_variant=launcher_variant,
                command=command,
            )

        for model, reuse in MIXED_CASES:
            name, launcher_variant, command = matrix.build_mixed_command(
                args,
                model_name=model,
                variant_name="unified-pr34441",
                reuse=reuse,
                ratio=0.5,
                repetition=repetition,
                stage="pr34441-targeted-mixed",
            )
            matrix.execute(
                args,
                label=f"pr34441/mixed/{model}/reuse{int(reuse * 100)}/rep{repetition}",
                name=name,
                launcher_variant=launcher_variant,
                command=command,
                allow_reproducible_failure=True,
            )
    print(f"COMPLETE elapsed_s={time.monotonic() - started:.3f}")


if __name__ == "__main__":
    main()
