"""Run the static/before/after lazy-compaction mapping benchmark matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "benchmark/hicache/run_unified_ablation.py"
DEFAULT_MODEL = Path(
    "/home/sukwoo24/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
DEFAULT_SERVER_PYTHON = Path("/home/sukwoo24/.venv_sglang_upstream_full/bin/python")
VARIANTS = ("mapping-static", "mapping-before", "mapping-after")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server-python", type=Path, default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--pressures", nargs="+", default=["0.30", "0.15"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=4)
    return parser.parse_args()


def variant_order(repetition: int) -> tuple[str, ...]:
    offset = (repetition - 1) % len(VARIANTS)
    return VARIANTS[offset:] + VARIANTS[:offset]


def main() -> None:
    args = parse_args()
    for pressure in args.pressures:
        pressure_tag = pressure.replace(".", "")
        for repetition in range(1, args.repetitions + 1):
            for variant in variant_order(repetition):
                command = [
                    sys.executable,
                    str(LAUNCHER),
                    "--variant",
                    variant,
                    "--scenario",
                    "steady",
                    "--run-name",
                    f"mf{pressure_tag}-r{repetition}",
                    "--artifact-root",
                    str(args.artifact_root),
                    "--python",
                    str(args.server_python),
                    "--model",
                    str(args.model),
                    "--max-total-tokens",
                    "60000",
                    "--max-running-requests",
                    "16",
                    "--chunked-prefill-size",
                    "4096",
                    "--context-length",
                    "65536",
                    "--server-extra-arg=--mem-fraction-static",
                    f"--server-extra-arg={pressure}",
                    "--input-len",
                    "1024",
                    "--output-len",
                    "4",
                    "--groups",
                    str(args.groups),
                    "--rounds",
                    str(args.rounds),
                    "--shared-ratio",
                    "0.95",
                    "--prime-output-len",
                    "1",
                    "--prime-repeats",
                    "2",
                    "--max-concurrency",
                    "16",
                    "--reverse-group-order",
                    "--require-eviction",
                    "--forbid-dropped",
                    "--no-profile-memory-breakdown",
                ]
                print(
                    f"running pressure={pressure} repetition={repetition} "
                    f"variant={variant}",
                    flush=True,
                )
                subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
