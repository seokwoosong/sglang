#!/usr/bin/env python3
"""Run the Qwen3.5 unified-memory/HiCache evaluation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "benchmark/hicache/run_unified_ablation.py"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/unified_ablation_qwen35_v2"


@dataclass(frozen=True)
class ModelCase:
    path: Path
    mem_fraction_static: float
    hicache_size: int
    artifact_suffix: str
    groups: dict[str, int]


@dataclass(frozen=True)
class LengthCase:
    input_len: int
    output_len: int
    chunked_prefill_size: int
    max_concurrency: int


MODELS = {
    "9b": ModelCase(
        path=Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-9B/snapshots/"
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        ),
        mem_fraction_static=0.75,
        hicache_size=8,
        artifact_suffix="",
        groups={"long": 4, "middle": 18, "short": 60},
    ),
    "4b": ModelCase(
        path=Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
        mem_fraction_static=0.48,
        hicache_size=12,
        artifact_suffix="-h12",
        groups={"long": 6, "middle": 26, "short": 84},
    ),
    "0.8b": ModelCase(
        path=Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-0.8B/snapshots/"
            "2fc06364715b967f1860aea9cf38778875588b17"
        ),
        mem_fraction_static=0.27,
        hicache_size=12,
        artifact_suffix="-h12",
        groups={"long": 8, "middle": 40, "short": 140},
    ),
}

LENGTHS = {
    "long": LengthCase(
        input_len=50_000,
        output_len=64,
        chunked_prefill_size=8192,
        max_concurrency=2,
    ),
    "middle": LengthCase(
        input_len=10_000,
        output_len=128,
        chunked_prefill_size=4096,
        max_concurrency=4,
    ),
    "short": LengthCase(
        input_len=3_000,
        output_len=128,
        chunked_prefill_size=2048,
        max_concurrency=8,
    ),
}

VARIANT_ORDERS = {
    1: ("u0", "u1", "u2", "u3"),
    2: ("u2", "u3", "u0", "u1"),
    3: ("u1", "u0", "u3", "u2"),
}


def csv_choices(raw: str, choices: dict[str, object] | tuple[str, ...]) -> list[str]:
    allowed = set(choices)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown choices: {sorted(unknown)}")
    return values


def successful_result_exists(root: Path, run_name: str, variant: str) -> bool:
    manifests = sorted(
        (root / run_name / variant).glob("*/manifest.json"), reverse=True
    )
    for manifest_path in manifests:
        result_path = manifest_path.with_name("result.json")
        if not result_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            manifest.get("status") == "completed"
            and result.get("validation", {}).get("passed") is True
        ):
            return True
    return False


def build_command(
    *,
    args: argparse.Namespace,
    model_name: str,
    length_name: str,
    variant: str,
) -> tuple[str, list[str]]:
    model = MODELS[model_name]
    length = LENGTHS[length_name]
    suffix = f"rep{args.repetition}" if args.phase == "performance" else "accuracy"
    run_name = (
        f"qwen35-{args.phase}-{suffix}-{model_name}-{length_name}"
        f"{model.artifact_suffix}"
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--variant",
        variant,
        "--scenario",
        "steady" if args.phase == "performance" else "parity",
        "--run-name",
        run_name,
        "--artifact-root",
        str(args.artifact_root),
        "--python",
        sys.executable,
        "--model",
        str(model.path),
        "--input-len",
        str(length.input_len),
        "--output-len",
        str(length.output_len if args.phase == "performance" else 32),
        "--max-total-tokens",
        "120000",
        "--max-running-requests",
        str(length.max_concurrency),
        "--chunked-prefill-size",
        str(length.chunked_prefill_size),
        "--context-length",
        "65536",
        "--hicache-size",
        str(model.hicache_size),
        "--hicache-write-policy",
        "write_back",
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={model.mem_fraction_static}",
        "--server-extra-arg=--language-only",
        "--server-extra-arg=--mm-feature-transport",
        "--server-extra-arg=cpu",
    ]
    if args.phase == "performance":
        command.extend(
            [
                "--groups",
                str(model.groups[length_name]),
                "--rounds",
                "3",
                "--shared-ratio",
                "0.95",
                "--max-concurrency",
                str(length.max_concurrency),
                "--group-order-start",
                str((args.repetition - 1) % model.groups[length_name]),
                "--require-eviction",
                "--require-loadback",
                "--require-host-hit",
            ]
        )
    else:
        command.extend(
            [
                "--pressure-requests",
                str(model.groups[length_name]),
                "--logprob-atol",
                "0.05",
            ]
        )
    return run_name, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("performance", "accuracy"), required=True)
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--models", default="9b,4b,0.8b")
    parser.add_argument("--lengths", default="long,middle,short")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_names = csv_choices(args.models, MODELS)
    length_names = csv_choices(args.lengths, LENGTHS)
    default_variants = VARIANT_ORDERS[args.repetition]
    variants = (
        csv_choices(args.variants, default_variants)
        if args.variants
        else list(default_variants)
    )

    for model_name in model_names:
        if not MODELS[model_name].path.is_dir():
            raise FileNotFoundError(MODELS[model_name].path)

    started = time.monotonic()
    completed = 0
    skipped = 0
    total = len(model_names) * len(length_names) * len(variants)
    for model_name in model_names:
        for length_name in length_names:
            for variant in variants:
                run_name, command = build_command(
                    args=args,
                    model_name=model_name,
                    length_name=length_name,
                    variant=variant,
                )
                label = f"{model_name}/{length_name}/{variant}"
                if successful_result_exists(args.artifact_root, run_name, variant):
                    skipped += 1
                    print(f"SKIP {label}: valid artifact already exists", flush=True)
                    continue
                print(
                    f"RUN {completed + skipped + 1}/{total} {label}: "
                    + " ".join(command),
                    flush=True,
                )
                if args.dry_run:
                    continue
                result = subprocess.run(command, cwd=REPO_ROOT, check=False)
                if result.returncode != 0:
                    print(
                        f"STOP {label}: runner exited {result.returncode}; "
                        "inspect the newest artifact before retrying",
                        file=sys.stderr,
                        flush=True,
                    )
                    return result.returncode
                completed += 1

    elapsed = time.monotonic() - started
    print(
        f"MATRIX COMPLETE phase={args.phase} repetition={args.repetition} "
        f"completed={completed} skipped={skipped} elapsed_s={elapsed:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
