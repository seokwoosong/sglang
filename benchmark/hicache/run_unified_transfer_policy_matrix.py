#!/usr/bin/env python3
"""Run sync/async unified HiCache transfer tests under both write policies."""

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
DEFAULT_ROOT = REPO_ROOT / "artifacts/unified_transfer_policy_ablation"
PYTHON = Path("/home/sukwoo24/.venv_sglang/bin/python")


@dataclass(frozen=True)
class ModelCase:
    path: Path
    mem_fraction_static: float
    hicache_size_gb: int
    groups: int


@dataclass(frozen=True)
class Workload:
    output_len: int
    rounds: int
    concurrency: int


MODELS = {
    "9b": ModelCase(
        Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-9B/snapshots/"
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        ),
        0.75,
        8,
        18,
    ),
    "4b": ModelCase(
        Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
        0.48,
        12,
        26,
    ),
    "0.8b": ModelCase(
        Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-0.8B/snapshots/"
            "2fc06364715b967f1860aea9cf38778875588b17"
        ),
        0.27,
        12,
        40,
    ),
}

WORKLOADS = {
    # Negative control: no independent request is available while a transfer gates
    # scheduling, so async should not materially improve throughput.
    "serial": Workload(output_len=128, rounds=2, concurrency=1),
    # Prefix replay under moderate concurrency, close to the original matrix.
    "replay": Workload(output_len=128, rounds=3, concurrency=4),
    # Longer decode and more independent requests expose transfer/compute overlap.
    "overlap": Workload(output_len=512, rounds=2, concurrency=8),
}

POLICIES = ("write_back", "write_through")
MODES = {"sync": "u2-sync-control", "async": "u2"}


def csv_values(raw: str, allowed: set[str]) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown values: {sorted(unknown)}")
    return values


def complete_result_exists(root: Path, run_name: str, variant: str) -> bool:
    for manifest_path in sorted((root / run_name / variant).glob("*/manifest.json")):
        result_path = manifest_path.with_name("result.json")
        if not result_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("status") == "completed"
            and manifest.get("client_exit_code") == 0
            and result.get("validation", {}).get("passed") is True
        ):
            return True
    return False


def common_server_args(model: ModelCase, policy: str, max_running: int) -> list[str]:
    return [
        "--allow-dirty-worktree",
        "--python",
        str(PYTHON),
        "--model",
        str(model.path),
        "--max-total-tokens",
        "120000",
        "--max-running-requests",
        str(max_running),
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--hicache-size",
        str(model.hicache_size_gb),
        "--hicache-write-policy",
        policy,
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={model.mem_fraction_static}",
        "--server-extra-arg=--language-only",
        "--server-extra-arg=--mm-feature-transport",
        "--server-extra-arg=cpu",
    ]


def performance_command(
    *,
    root: Path,
    stage: str,
    repetition: int,
    model_name: str,
    workload_name: str,
    policy: str,
    mode: str,
) -> tuple[str, str, list[str]]:
    model = MODELS[model_name]
    workload = WORKLOADS[workload_name]
    variant = MODES[mode]
    run_name = f"policy-{stage}-rep{repetition}-{model_name}-{workload_name}-{policy}"
    command = [
        str(PYTHON),
        str(RUNNER),
        "--variant",
        variant,
        "--scenario",
        "steady",
        "--run-name",
        run_name,
        "--artifact-root",
        str(root),
        "--input-len",
        "10000",
        "--output-len",
        str(workload.output_len),
        "--groups",
        str(model.groups),
        "--rounds",
        str(workload.rounds),
        "--shared-ratio",
        "0.95",
        "--prime-output-len",
        "1",
        "--prime-repeats",
        "2" if policy == "write_through" else "1",
        "--max-concurrency",
        str(workload.concurrency),
        "--group-order-start",
        str((repetition - 1) % model.groups),
        "--reverse-group-order",
        "--require-eviction",
        "--require-loadback",
        "--require-backup",
        "--require-host-hit",
        "--forbid-dropped",
        *common_server_args(model, policy, workload.concurrency),
    ]
    return run_name, variant, command


def accuracy_command(
    *, root: Path, model_name: str, policy: str, mode: str
) -> tuple[str, str, list[str]]:
    model = MODELS[model_name]
    variant = MODES[mode]
    run_name = f"policy-accuracy-{model_name}-{policy}"
    command = [
        str(PYTHON),
        str(RUNNER),
        "--variant",
        variant,
        "--scenario",
        "parity",
        "--run-name",
        run_name,
        "--artifact-root",
        str(root),
        "--input-len",
        "10000",
        "--output-len",
        "32",
        "--pressure-requests",
        str(model.groups),
        "--logprob-atol",
        "0.05",
        *common_server_args(model, policy, 4),
    ]
    return run_name, variant, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("pilot", "main", "confirm", "accuracy"), required=True
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", default="4b")
    parser.add_argument("--workloads", default="serial,replay,overlap")
    parser.add_argument("--repetitions", default="1")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--modes", default="sync,async")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = csv_values(args.models, set(MODELS))
    workloads = csv_values(args.workloads, set(WORKLOADS))
    policies = csv_values(args.policies, set(POLICIES))
    modes = csv_values(args.modes, set(MODES))
    repetitions = [int(value) for value in args.repetitions.split(",")]
    if any(value < 1 for value in repetitions):
        raise ValueError("Repetitions must be positive")
    for model_name in models:
        if not MODELS[model_name].path.is_dir():
            raise FileNotFoundError(MODELS[model_name].path)

    tasks: list[tuple[str, str, list[str], str]] = []
    if args.stage == "accuracy":
        for model_name in models:
            for policy in policies:
                for mode in modes:
                    run_name, variant, command = accuracy_command(
                        root=args.artifact_root,
                        model_name=model_name,
                        policy=policy,
                        mode=mode,
                    )
                    tasks.append(
                        (run_name, variant, command, f"{model_name}/{policy}/{mode}")
                    )
    else:
        for repetition in repetitions:
            # Reverse both axes on alternating repetitions to reduce time-order bias.
            ordered_policies = policies if repetition % 2 else list(reversed(policies))
            ordered_modes = modes if repetition % 2 else list(reversed(modes))
            for model_name in models:
                for workload_name in workloads:
                    for policy in ordered_policies:
                        for mode in ordered_modes:
                            run_name, variant, command = performance_command(
                                root=args.artifact_root,
                                stage=args.stage,
                                repetition=repetition,
                                model_name=model_name,
                                workload_name=workload_name,
                                policy=policy,
                                mode=mode,
                            )
                            label = (
                                f"rep{repetition}/{model_name}/{workload_name}/"
                                f"{policy}/{mode}"
                            )
                            tasks.append((run_name, variant, command, label))

    started = time.monotonic()
    completed = skipped = 0
    for index, (run_name, variant, command, label) in enumerate(tasks, start=1):
        if complete_result_exists(args.artifact_root, run_name, variant):
            skipped += 1
            print(f"SKIP {index}/{len(tasks)} {label}", flush=True)
            continue
        print(f"RUN {index}/{len(tasks)} {label}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            print(
                f"STOP {label}: exit={result.returncode}", file=sys.stderr, flush=True
            )
            return result.returncode
        completed += 1

    print(
        f"MATRIX COMPLETE stage={args.stage} completed={completed} skipped={skipped} "
        f"elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
