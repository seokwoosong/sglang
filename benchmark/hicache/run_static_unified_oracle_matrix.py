#!/usr/bin/env python3
"""Run the Qwen3.5 static-oracle versus unified-memory HiCache matrix.

Static partition ratios are tuned at 80% prefix reuse.  Homogeneous and
interleaved-mixed workloads have independent static oracles.  The selected
ratios are then held fixed for the 20% and 50% reuse controls.  Each invocation
of the underlying launcher starts a fresh server and records a self-contained
artifact, so this controller is safely resumable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "benchmark/hicache/run_unified_ablation.py"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/static_unified_oracle_92eb073"
PYTHON = Path("/home/sukwoo24/.venv_sglang/bin/python")


@dataclass(frozen=True)
class ModelCase:
    path: Path
    mem_fraction_static: float


@dataclass(frozen=True)
class Workload:
    input_len: int
    output_len: int
    groups: int
    max_concurrency: int
    max_running_requests: int
    chunked_prefill_size: int


MODELS = {
    "0.8b": ModelCase(
        path=Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-0.8B/snapshots/"
            "2fc06364715b967f1860aea9cf38778875588b17"
        ),
        mem_fraction_static=0.27,
    ),
    "4b": ModelCase(
        path=Path(
            "/home/sukwoo24/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
        mem_fraction_static=0.55,
    ),
}

WORKLOADS = {
    "short": Workload(3_000, 256, 120, 32, 32, 2_048),
    "middle": Workload(10_000, 256, 40, 8, 8, 4_096),
    "long": Workload(50_000, 128, 8, 2, 8, 4_096),
}

STATIC_VARIANTS = {
    "static-auto": "oracle-s1-default",
    "static-triton": "oracle-s1",
}
ALL_VARIANTS = {**STATIC_VARIANTS, "unified-triton": "oracle-u3"}
PREFIX_REUSE_CONTROLS = (0.2, 0.5)
TUNING_REUSE = 0.8
ALL_PREFIX_REUSE = (*PREFIX_REUSE_CONTROLS, TUNING_REUSE)
INITIAL_RATIOS = (0.5, 0.4, 0.6)
MIN_RATIO = 0.1
MAX_RATIO = 0.9
RATIO_STEP = 0.1
FINAL_REPETITIONS = (1, 2, 3)
MIXED_WORKLOAD_NAME = "interleaved-mixed"


def ratio_label(value: float) -> str:
    return f"r{int(round(value * 10)):02d}"


def reuse_label(value: float) -> str:
    return f"reuse{int(round(value * 100)):03d}"


def selection_key(model: str, variant: str, workload: str) -> str:
    return f"{model}/{variant}/{workload}"


def latest_valid_result(
    root: Path, run_name: str, launcher_variant: str
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    base = root / run_name / launcher_variant
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
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
            and manifest.get("client_exit_code") == 0
            and result.get("validation", {}).get("passed") is True
        ):
            candidates.append((manifest_path.parent, manifest, result))
    return candidates[-1] if candidates else None


def failed_attempts(root: Path, run_name: str, launcher_variant: str) -> list[Path]:
    """Return completed launcher attempts that failed before a valid result.

    Final mixed conditions deliberately retain reproducible OOMs as outcomes
    instead of changing concurrency for unified only. Two independent attempts
    are enough to classify a condition as reproducibly failed and move on.
    """
    base = root / run_name / launcher_variant
    failures: list[Path] = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if manifest.get("status") == "failed":
            failures.append(manifest_path.parent)
    return failures


def run_name(
    *,
    stage: str,
    model: str,
    variant: str,
    workload: str,
    reuse: float,
    ratio: float,
    repetition: int,
) -> str:
    return "-".join(
        (
            stage,
            model,
            variant,
            workload,
            reuse_label(reuse),
            ratio_label(ratio),
            f"rep{repetition}",
        )
    )


def build_steady_command(
    args: argparse.Namespace,
    *,
    stage: str,
    model_name: str,
    variant_name: str,
    workload_name: str,
    reuse: float,
    ratio: float,
    repetition: int,
) -> tuple[str, str, list[str]]:
    model = MODELS[model_name]
    workload = WORKLOADS[workload_name]
    launcher_variant = ALL_VARIANTS[variant_name]
    name = run_name(
        stage=stage,
        model=model_name,
        variant=variant_name,
        workload=workload_name,
        reuse=reuse,
        ratio=ratio,
        repetition=repetition,
    )
    command = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        launcher_variant,
        "--scenario",
        "steady",
        "--run-name",
        name,
        "--artifact-root",
        str(args.artifact_root),
        "--python",
        str(PYTHON),
        "--model",
        str(model.path),
        "--input-len",
        str(workload.input_len),
        "--output-len",
        str(workload.output_len),
        "--max-total-tokens",
        "120000",
        "--page-size",
        "8",
        "--max-running-requests",
        str(workload.max_running_requests),
        "--chunked-prefill-size",
        str(workload.chunked_prefill_size),
        "--context-length",
        "65536",
        "--hicache-size",
        "12",
        "--hicache-write-policy",
        "write_back",
        "--cuda-graph-mode",
        "enabled",
        "--groups",
        str(workload.groups),
        "--rounds",
        "2",
        "--shared-ratio",
        str(reuse),
        "--prime-repeats",
        "1",
        "--max-concurrency",
        str(workload.max_concurrency),
        "--group-order-start",
        str((repetition - 1) % workload.groups),
        "--reverse-group-order",
        "--require-eviction",
        "--require-backup",
        "--forbid-dropped",
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={model.mem_fraction_static}",
        "--server-extra-arg=--mamba-full-memory-ratio",
        f"--server-extra-arg={ratio}",
        "--server-extra-arg=--language-only",
        "--server-extra-arg=--mm-feature-transport",
        "--server-extra-arg=cpu",
        "--no-profile-memory-breakdown",
    ]
    command.extend(("--require-loadback", "--require-host-hit"))
    return name, launcher_variant, command


def build_preflight_command(
    args: argparse.Namespace, model_name: str, variant_name: str
) -> tuple[str, str, list[str]]:
    model = MODELS[model_name]
    launcher_variant = ALL_VARIANTS[variant_name]
    name = f"preflight-{model_name}-{variant_name}"
    command = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        launcher_variant,
        "--scenario",
        "parity",
        "--run-name",
        name,
        "--artifact-root",
        str(args.artifact_root),
        "--python",
        str(PYTHON),
        "--model",
        str(model.path),
        "--input-len",
        "10000",
        "--output-len",
        "32",
        "--pressure-requests",
        "40",
        "--max-total-tokens",
        "120000",
        "--page-size",
        "8",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--hicache-size",
        "12",
        "--hicache-write-policy",
        "write_back",
        "--cuda-graph-mode",
        "enabled",
        "--logprob-atol",
        "0.02",
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={model.mem_fraction_static}",
        "--server-extra-arg=--mamba-full-memory-ratio",
        "--server-extra-arg=0.5",
        "--server-extra-arg=--language-only",
        "--server-extra-arg=--mm-feature-transport",
        "--server-extra-arg=cpu",
        "--no-profile-memory-breakdown",
    ]
    return name, launcher_variant, command


def build_mixed_command(
    args: argparse.Namespace,
    *,
    model_name: str,
    variant_name: str,
    reuse: float,
    ratio: float,
    repetition: int,
    stage: str,
) -> tuple[str, str, list[str]]:
    model = MODELS[model_name]
    launcher_variant = ALL_VARIANTS[variant_name]
    name = "-".join(
        (
            stage,
            "intermixed",
            model_name,
            variant_name,
            reuse_label(reuse),
            ratio_label(ratio),
            f"rep{repetition}",
        )
    )
    command = [
        str(PYTHON),
        str(LAUNCHER),
        "--variant",
        launcher_variant,
        "--scenario",
        "mixed",
        "--run-name",
        name,
        "--artifact-root",
        str(args.artifact_root),
        "--python",
        str(PYTHON),
        "--model",
        str(model.path),
        "--input-len",
        "50000",
        "--output-len",
        "128",
        "--max-total-tokens",
        "120000",
        "--page-size",
        "8",
        "--max-running-requests",
        "32",
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--hicache-size",
        "12",
        "--hicache-write-policy",
        "write_back",
        "--cuda-graph-mode",
        "enabled",
        "--shared-ratio",
        str(reuse),
        "--seed",
        str(7_300 + repetition),
        "--max-concurrency",
        "16",
        "--forbid-dropped",
        "--server-extra-arg=--mem-fraction-static",
        f"--server-extra-arg={model.mem_fraction_static}",
        "--server-extra-arg=--mamba-full-memory-ratio",
        f"--server-extra-arg={ratio}",
        "--server-extra-arg=--language-only",
        "--server-extra-arg=--mm-feature-transport",
        "--server-extra-arg=cpu",
        "--no-profile-memory-breakdown",
    ]
    return name, launcher_variant, command


def execute(
    args: argparse.Namespace,
    *,
    label: str,
    name: str,
    launcher_variant: str,
    command: list[str],
    allow_reproducible_failure: bool = False,
) -> dict[str, Any] | None:
    existing = latest_valid_result(args.artifact_root, name, launcher_variant)
    if existing is not None and args.resume:
        print(f"SKIP {label}: {existing[0]}", flush=True)
        return existing[2]
    prior_failures = failed_attempts(args.artifact_root, name, launcher_variant)
    if (
        allow_reproducible_failure
        and args.resume
        and not args.retry_failures
        and len(prior_failures) >= 2
    ):
        print(
            f"SKIP REPRODUCIBLE FAILURE {label}: "
            f"{len(prior_failures)} failed attempts; latest={prior_failures[-1]}",
            flush=True,
        )
        return None
    print(f"RUN {label}: {' '.join(command)}", flush=True)
    if args.dry_run:
        return None
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        if allow_reproducible_failure:
            failures = failed_attempts(args.artifact_root, name, launcher_variant)
            if len(failures) < 2:
                print(
                    f"RETRY {label}: first failed attempt={failures[-1]}",
                    flush=True,
                )
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                failures = failed_attempts(args.artifact_root, name, launcher_variant)
                print(
                    f"RECORD REPRODUCIBLE FAILURE {label}: "
                    f"{len(failures)} attempts; continuing matrix",
                    flush=True,
                )
                return None
        else:
            raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    result = latest_valid_result(args.artifact_root, name, launcher_variant)
    if result is None:
        raise RuntimeError(f"{label} completed without a valid result artifact")
    return result[2]


def throughput(result: dict[str, Any]) -> float:
    return float(result["summary"]["total_token_throughput"])


def load_selections(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def write_selections(path: Path, selections: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_preflight(args: argparse.Namespace) -> None:
    for model in args.models:
        for variant in ALL_VARIANTS:
            name, launcher_variant, command = build_preflight_command(
                args, model, variant
            )
            execute(
                args,
                label=f"preflight/{model}/{variant}",
                name=name,
                launcher_variant=launcher_variant,
                command=command,
            )


def measure_static_candidate(
    args: argparse.Namespace,
    *,
    model: str,
    variant: str,
    workload: str,
    ratio: float,
    repetition: int,
) -> float | None:
    name, launcher_variant, command = build_steady_command(
        args,
        stage="tune",
        model_name=model,
        variant_name=variant,
        workload_name=workload,
        reuse=TUNING_REUSE,
        ratio=ratio,
        repetition=repetition,
    )
    result = execute(
        args,
        label=(
            f"tune/{model}/{variant}/{workload}/{ratio_label(ratio)}/rep{repetition}"
        ),
        name=name,
        launcher_variant=launcher_variant,
        command=command,
    )
    return throughput(result) if result is not None else None


def tune_one(
    args: argparse.Namespace,
    *,
    model: str,
    variant: str,
    workload: str,
) -> dict[str, Any] | None:
    screening: dict[float, float] = {}
    for ratio in INITIAL_RATIOS:
        score = measure_static_candidate(
            args,
            model=model,
            variant=variant,
            workload=workload,
            ratio=ratio,
            repetition=1,
        )
        if score is None:
            return None
        screening[ratio] = score

    best_initial = max(screening, key=screening.get)
    direction = 0
    if best_initial == 0.4:
        direction = -1
    elif best_initial == 0.6:
        direction = 1

    previous_ratio = best_initial
    previous_score = screening[best_initial]
    while direction:
        next_ratio = round(previous_ratio + direction * RATIO_STEP, 1)
        if not MIN_RATIO <= next_ratio <= MAX_RATIO:
            break
        score = measure_static_candidate(
            args,
            model=model,
            variant=variant,
            workload=workload,
            ratio=next_ratio,
            repetition=1,
        )
        if score is None:
            return None
        screening[next_ratio] = score
        if score <= previous_score:
            break
        previous_ratio = next_ratio
        previous_score = score

    finalists = sorted(screening, key=screening.get, reverse=True)[:2]
    repeated: dict[float, list[float]] = {
        ratio: [screening[ratio]] for ratio in finalists
    }
    for ratio in finalists:
        for repetition in (2, 3):
            score = measure_static_candidate(
                args,
                model=model,
                variant=variant,
                workload=workload,
                ratio=ratio,
                repetition=repetition,
            )
            if score is None:
                return None
            repeated[ratio].append(score)

    medians = {ratio: statistics.median(scores) for ratio, scores in repeated.items()}
    selected = max(medians, key=medians.get)
    return {
        "selected_ratio": selected,
        "selected_median_total_token_throughput": medians[selected],
        "screening": {f"{ratio:.1f}": score for ratio, score in screening.items()},
        "finalists": {
            f"{ratio:.1f}": {
                "scores": repeated[ratio],
                "median": medians[ratio],
            }
            for ratio in finalists
        },
    }


def run_tuning(args: argparse.Namespace) -> None:
    selections = load_selections(args.selection_file)
    for model in args.models:
        for variant in STATIC_VARIANTS:
            for workload in args.workloads:
                key = selection_key(model, variant, workload)
                if key in selections and args.resume:
                    print(
                        f"SKIP selection {key}: ratio="
                        f"{selections[key]['selected_ratio']}",
                        flush=True,
                    )
                    continue
                selected = tune_one(
                    args, model=model, variant=variant, workload=workload
                )
                if selected is None:
                    print(f"DRY selection pending: {key}", flush=True)
                    continue
                selections[key] = selected
                write_selections(args.selection_file, selections)
                print(
                    f"SELECT {key}: ratio={selected['selected_ratio']:.1f} "
                    f"median={selected['selected_median_total_token_throughput']:.3f}",
                    flush=True,
                )


def selected_ratio(
    selections: dict[str, Any], model: str, variant: str, workload: str
) -> float:
    key = selection_key(model, variant, workload)
    if key not in selections:
        raise RuntimeError(
            f"Missing {key} in the selection file; run the tune stage first"
        )
    return float(selections[key]["selected_ratio"])


def run_final(args: argparse.Namespace) -> None:
    selections = load_selections(args.selection_file)
    required = [
        selection_key(model, variant, workload)
        for model in args.models
        for variant in STATIC_VARIANTS
        for workload in args.workloads
    ]
    missing = [key for key in required if key not in selections]
    if missing:
        raise RuntimeError(
            "Run the tune stage first; missing selections: " + ", ".join(missing)
        )

    # U-T at the 80% tuning workload. Static 80% repetitions already live in
    # the tune artifacts, so rerunning them here would duplicate measurements.
    for model in args.models:
        for workload in args.workloads:
            for repetition in FINAL_REPETITIONS:
                name, launcher_variant, command = build_steady_command(
                    args,
                    stage="final",
                    model_name=model,
                    variant_name="unified-triton",
                    workload_name=workload,
                    reuse=TUNING_REUSE,
                    ratio=0.5,
                    repetition=repetition,
                )
                execute(
                    args,
                    label=(
                        f"final/{model}/unified-triton/{workload}/"
                        f"reuse080/rep{repetition}"
                    ),
                    name=name,
                    launcher_variant=launcher_variant,
                    command=command,
                )

    # Low- and medium-reuse controls use the ratio selected at 80%.
    for reuse in PREFIX_REUSE_CONTROLS:
        for model in args.models:
            for workload in args.workloads:
                for variant in ALL_VARIANTS:
                    ratio = (
                        selected_ratio(selections, model, variant, workload)
                        if variant in STATIC_VARIANTS
                        else 0.5
                    )
                    for repetition in FINAL_REPETITIONS:
                        name, launcher_variant, command = build_steady_command(
                            args,
                            stage="final",
                            model_name=model,
                            variant_name=variant,
                            workload_name=workload,
                            reuse=reuse,
                            ratio=ratio,
                            repetition=repetition,
                        )
                        execute(
                            args,
                            label=(
                                f"final/{model}/{variant}/{workload}/"
                                f"{reuse_label(reuse)}/rep{repetition}"
                            ),
                            name=name,
                            launcher_variant=launcher_variant,
                            command=command,
                        )


def single_static_ratio(
    selections: dict[str, Any], model: str, variant: str, workloads: list[str]
) -> float:
    ratios = sorted(
        selected_ratio(selections, model, variant, workload) for workload in workloads
    )
    return ratios[len(ratios) // 2]


def mixed_selection_key(model: str, variant: str) -> str:
    return selection_key(model, variant, MIXED_WORKLOAD_NAME)


def measure_mixed_static_candidate(
    args: argparse.Namespace,
    *,
    model: str,
    variant: str,
    ratio: float,
    repetition: int,
) -> float | None:
    name, launcher_variant, command = build_mixed_command(
        args,
        model_name=model,
        variant_name=variant,
        reuse=TUNING_REUSE,
        ratio=ratio,
        repetition=repetition,
        stage="mixed-tune",
    )
    result = execute(
        args,
        label=f"mixed-tune/{model}/{variant}/{ratio_label(ratio)}/rep{repetition}",
        name=name,
        launcher_variant=launcher_variant,
        command=command,
    )
    return throughput(result) if result is not None else None


def tune_one_mixed(
    args: argparse.Namespace, *, model: str, variant: str
) -> dict[str, Any] | None:
    screening: dict[float, float] = {}
    for ratio in INITIAL_RATIOS:
        score = measure_mixed_static_candidate(
            args,
            model=model,
            variant=variant,
            ratio=ratio,
            repetition=1,
        )
        if score is None:
            return None
        screening[ratio] = score

    best_initial = max(screening, key=screening.get)
    direction = -1 if best_initial == 0.4 else 1 if best_initial == 0.6 else 0
    previous_ratio = best_initial
    previous_score = screening[best_initial]
    while direction:
        next_ratio = round(previous_ratio + direction * RATIO_STEP, 1)
        if not MIN_RATIO <= next_ratio <= MAX_RATIO:
            break
        score = measure_mixed_static_candidate(
            args,
            model=model,
            variant=variant,
            ratio=next_ratio,
            repetition=1,
        )
        if score is None:
            return None
        screening[next_ratio] = score
        if score <= previous_score:
            break
        previous_ratio, previous_score = next_ratio, score

    finalists = sorted(screening, key=screening.get, reverse=True)[:2]
    repeated = {ratio: [screening[ratio]] for ratio in finalists}
    for ratio in finalists:
        for repetition in (2, 3):
            score = measure_mixed_static_candidate(
                args,
                model=model,
                variant=variant,
                ratio=ratio,
                repetition=repetition,
            )
            if score is None:
                return None
            repeated[ratio].append(score)
    medians = {ratio: statistics.median(scores) for ratio, scores in repeated.items()}
    selected = max(medians, key=medians.get)
    return {
        "selected_ratio": selected,
        "selected_median_total_token_throughput": medians[selected],
        "screening": {f"{ratio:.1f}": score for ratio, score in screening.items()},
        "finalists": {
            f"{ratio:.1f}": {"scores": repeated[ratio], "median": medians[ratio]}
            for ratio in finalists
        },
    }


def run_mixed(args: argparse.Namespace) -> None:
    selections = load_selections(args.selection_file)

    # A median of independently tuned homogeneous ratios is not a best-static
    # baseline for genuinely interleaved traffic.  Tune each static backend on
    # this exact mixed trace at 80% reuse before comparing it with unified.
    for model in args.models:
        for variant in STATIC_VARIANTS:
            key = mixed_selection_key(model, variant)
            if key in selections and args.resume:
                print(
                    f"SKIP selection {key}: ratio={selections[key]['selected_ratio']}",
                    flush=True,
                )
                continue
            selected = tune_one_mixed(args, model=model, variant=variant)
            if selected is None:
                print(f"DRY selection pending: {key}", flush=True)
                continue
            selections[key] = selected
            write_selections(args.selection_file, selections)
            print(
                f"SELECT {key}: ratio={selected['selected_ratio']:.1f} "
                f"median={selected['selected_median_total_token_throughput']:.3f}",
                flush=True,
            )

    for reuse in ALL_PREFIX_REUSE:
        for model in args.models:
            for variant in ALL_VARIANTS:
                # The selected static ratio already has three 80%-reuse
                # repetitions in mixed-tune. Reuse those measurements exactly
                # as the homogeneous stage does; only unified needs fresh 80% runs.
                if reuse == TUNING_REUSE and variant in STATIC_VARIANTS:
                    continue
                ratio = (
                    selected_ratio(selections, model, variant, MIXED_WORKLOAD_NAME)
                    if variant in STATIC_VARIANTS
                    else 0.5
                )
                for repetition in FINAL_REPETITIONS:
                    name, launcher_variant, command = build_mixed_command(
                        args,
                        model_name=model,
                        variant_name=variant,
                        reuse=reuse,
                        ratio=ratio,
                        repetition=repetition,
                        stage="mixed-final",
                    )
                    execute(
                        args,
                        label=(
                            f"intermixed/{model}/{variant}/{reuse_label(reuse)}/"
                            f"rep{repetition}"
                        ),
                        name=name,
                        launcher_variant=launcher_variant,
                        command=command,
                        allow_reproducible_failure=True,
                    )


def print_plan(args: argparse.Namespace) -> None:
    searches = len(args.models) * len(STATIC_VARIANTS) * len(args.workloads)
    preflight = len(args.models) * len(ALL_VARIANTS)
    tune_min = searches * 7
    tune_expected = searches * 9
    tune_max = searches * 10
    unified_80 = len(args.models) * len(args.workloads) * 3
    controls = (
        len(PREFIX_REUSE_CONTROLS)
        * len(args.models)
        * len(args.workloads)
        * len(ALL_VARIANTS)
        * 3
    )
    mixed_searches = len(args.models) * len(STATIC_VARIANTS)
    mixed_tune_min = mixed_searches * 7
    mixed_tune_expected = mixed_searches * 9
    mixed_tune_max = mixed_searches * 10
    mixed = (
        len(PREFIX_REUSE_CONTROLS) * len(args.models) * len(ALL_VARIANTS) * 3
        + len(args.models) * 3  # unified at 80%; static reuses tuning finalists
    )
    payload = {
        "source_sha": "92eb0737857f4fef0ba46e19bed5cb9bc45816f9",
        "models": args.models,
        "workloads": {name: asdict(WORKLOADS[name]) for name in args.workloads},
        "prefix_reuse": [0.2, 0.5, 0.8],
        "counts": {
            "preflight": preflight,
            "static_tuning_min": tune_min,
            "static_tuning_expected": tune_expected,
            "static_tuning_max": tune_max,
            "unified_80_final": unified_80,
            "reuse_20_50_controls": controls,
            "interleaved_mixed_tuning_min": mixed_tune_min,
            "interleaved_mixed_tuning_expected": mixed_tune_expected,
            "interleaved_mixed_tuning_max": mixed_tune_max,
            "interleaved_mixed_final": mixed,
            "total_min": preflight
            + tune_min
            + unified_80
            + controls
            + mixed_tune_min
            + mixed,
            "total_expected": preflight
            + tune_expected
            + unified_80
            + controls
            + mixed_tune_expected
            + mixed,
            "total_max": preflight
            + tune_max
            + unified_80
            + controls
            + mixed_tune_max
            + mixed,
        },
        "artifact_root": str(args.artifact_root),
        "selection_file": str(args.selection_file),
    }
    print(json.dumps(payload, indent=2))


def csv_choices(raw: str, allowed: dict[str, Any]) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown choices: {sorted(unknown)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("plan", "preflight", "tune", "final", "mixed", "all")
    )
    parser.add_argument("--models", default="0.8b,4b")
    parser.add_argument("--workloads", default="short,middle,long")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry unresolved conditions even after two reproducible failures.",
    )
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.selection_file = (
        args.selection_file.resolve()
        if args.selection_file
        else args.artifact_root / "static_ratio_selection.json"
    )
    args.models = csv_choices(args.models, MODELS)
    args.workloads = csv_choices(args.workloads, WORKLOADS)

    for model in args.models:
        if not MODELS[model].path.is_dir():
            raise FileNotFoundError(MODELS[model].path)

    started = time.monotonic()
    if args.stage == "plan":
        print_plan(args)
        return 0
    if args.stage in {"preflight", "all"}:
        run_preflight(args)
    if args.stage in {"tune", "all"}:
        run_tuning(args)
    if args.stage in {"final", "all"}:
        run_final(args)
    if args.stage in {"mixed", "all"}:
        run_mixed(args)
    print(f"COMPLETE stage={args.stage} elapsed_s={time.monotonic() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
