#!/usr/bin/env python3
"""Launch one pinned unified-memory ablation server and run one workload.

The launcher records the exact command, environment overrides, git revision,
hardware snapshot, server log, client log, and benchmark JSON in a unique run
directory.  It always terminates the server process group on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "benchmark/hicache/bench_unified_ablation.py"
DEFAULT_MODEL = Path(
    "/home/sukwoo24/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

VARIANTS = {
    # Diagnostic control: the same historical source as U0, with the static
    # (non-unified) hybrid pools. It is not one of the four reported variants.
    "plain": {
        "sha": "8279702e0b8f159c29ed201d05503a0548fefef9",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u0"),
        "unified": False,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "u0": {
        "sha": "b6c9c14037ccabf263f7cdae73c31a3ad1b609cc",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u0"),
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "u1": {
        "sha": "bf41c36e09375559cf25177df8038007db2611d0",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u1"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": None,
    },
    "u2": {
        "sha": "1ee4930f27d85c33a73baa1e0e6a9458381b06ec",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u2"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
    },
    "u2-sync-control": {
        "sha": "1ee4930f27d85c33a73baa1e0e6a9458381b06ec",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u2"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "1",
    },
    "u3": {
        "sha": "0912b3824558982396cd2e18315867a81549a9cf",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/u3"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
    },
}


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )
    return completed.stdout.strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_variant(variant: dict[str, Any], *, allow_dirty: bool) -> str:
    worktree = variant["worktree"]
    if not worktree.is_dir():
        raise FileNotFoundError(f"Missing worktree: {worktree}")
    actual = run_text(["git", "rev-parse", "HEAD"], cwd=worktree)
    if actual != variant["sha"]:
        raise RuntimeError(
            f"Worktree {worktree} is at {actual}, expected {variant['sha']}"
        )
    dirty = run_text(["git", "status", "--short"], cwd=worktree)
    if dirty and not allow_dirty:
        raise RuntimeError(f"Evaluation worktree is dirty: {worktree}\n{dirty}")
    return dirty


def server_command(args: argparse.Namespace, variant: dict[str, Any]) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp",
        "1",
        "--dtype",
        "bfloat16",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-mamba-cache-size",
        str(args.max_mamba_cache_size),
        "--max-running-requests",
        str(args.max_running_requests),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--context-length",
        str(args.context_length),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
        "--log-level",
        args.log_level,
    ]
    if variant["unified"]:
        command.append("--enable-unified-memory")
    if variant["hicache"]:
        command.extend(
            [
                "--enable-hierarchical-cache",
                "--hicache-size",
                str(args.hicache_size),
                "--hicache-write-policy",
                args.hicache_write_policy,
                "--hicache-io-backend",
                "kernel",
                "--hicache-mem-layout",
                "page_first",
            ]
        )
    command.extend(args.server_extra_arg)
    return command


def benchmark_command(
    args: argparse.Namespace, variant: dict[str, Any], output: Path
) -> list[str]:
    command = [
        str(args.python),
        str(BENCHMARK),
        args.scenario,
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--variant",
        args.variant,
        "--output",
        str(output),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
    ]
    if variant["hicache"]:
        command.append("--expect-hicache")
    if args.scenario == "parity":
        command.extend(
            [
                "--pressure-requests",
                str(args.pressure_requests),
                "--logprob-atol",
                str(args.logprob_atol),
            ]
        )
    elif args.scenario == "accuracy":
        command.extend(
            [
                "--pressure-requests",
                str(args.pressure_requests),
                "--num-questions",
                str(args.accuracy_questions),
                "--num-shots",
                str(args.accuracy_shots),
                "--max-concurrency",
                str(args.max_concurrency),
                "--dataset-path",
                str(args.accuracy_dataset),
            ]
        )
    else:
        command.extend(
            [
                "--seed",
                str(args.seed),
                "--groups",
                str(args.groups),
                "--group-order-start",
                str(args.group_order_start),
                "--rounds",
                str(args.rounds),
                "--shared-ratio",
                str(args.shared_ratio),
                "--prime-output-len",
                str(args.prime_output_len),
                "--max-concurrency",
                str(args.max_concurrency),
            ]
        )
        if args.require_eviction:
            command.append("--require-eviction")
        if args.require_loadback and variant["hicache"]:
            command.append("--require-loadback")
        if args.require_host_hit and variant["hicache"]:
            command.append("--require-host-hit")
    return command


def wait_for_server(
    base_url: str, process: subprocess.Popen[str], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy: {last_error}")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def sample_runtime(base_url: str, elapsed_s: float) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "wall_time_ns": time.time_ns(),
        "elapsed_s": elapsed_s,
        "gpu": run_text(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        response.raise_for_status()
        wanted = {
            "sglang:num_requests_total",
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
            "sglang:evicted_tokens_total",
            "sglang:load_back_tokens_total",
        }
        metrics: dict[str, float] = {}
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            name = fields[0].split("{", 1)[0]
            if name not in wanted or len(fields) < 2:
                continue
            metrics[name] = metrics.get(name, 0.0) + float(fields[1])
        sample["metrics"] = metrics
    except Exception as exc:  # noqa: BLE001 - monitoring must not fail the run
        sample["metrics_error"] = repr(exc)
    return sample


def run_client_with_monitor(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    monitor_path: Path,
    base_url: str,
    timeout_s: int,
    monitor_interval_s: float,
) -> tuple[int, bool]:
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as client_log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            text=True,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_sample = started
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_s:
                timed_out = True
                terminate_process_group(process)
                break
            if now >= next_sample:
                samples.append(sample_runtime(base_url, now - started))
                atomic_json(monitor_path, {"samples": samples})
                next_sample = now + monitor_interval_s
            time.sleep(min(1.0, monitor_interval_s))
        if process.poll() is None:
            terminate_process_group(process)
        exit_code = 124 if timed_out else int(process.returncode or 0)
    samples.append(sample_runtime(base_url, time.monotonic() - started))
    atomic_json(monitor_path, {"samples": samples, "timed_out": timed_out})
    return exit_code, timed_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Permit a diagnostic source patch and record its diff in the manifest.",
    )
    parser.add_argument(
        "--scenario", choices=["parity", "accuracy", "steady"], required=True
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=REPO_ROOT / "artifacts/unified_ablation"
    )
    parser.add_argument(
        "--python", type=Path, default=Path("/home/sukwoo24/.venv_sglang/bin/python")
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--client-timeout", type=int, default=1800)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--max-mamba-cache-size", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--hicache-size", type=int, default=4)
    parser.add_argument(
        "--hicache-write-policy",
        choices=["write_through", "write_back"],
        default="write_through",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the server (repeatable).",
    )
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, default=2)
    parser.add_argument("--pressure-requests", type=int, default=6)
    parser.add_argument("--logprob-atol", type=float, default=1e-2)
    parser.add_argument("--accuracy-questions", type=int, default=200)
    parser.add_argument("--accuracy-shots", type=int, default=5)
    parser.add_argument(
        "--accuracy-dataset",
        type=Path,
        default=Path.home() / ".cache/sglang-bench/gsm8k-test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-order-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--shared-ratio", type=float, default=0.95)
    parser.add_argument("--prime-output-len", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--require-eviction", action="store_true")
    parser.add_argument("--require-loadback", action="store_true")
    parser.add_argument("--require-host-hit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variant = VARIANTS[args.variant]
    dirty_status = verify_variant(variant, allow_dirty=args.allow_dirty_worktree)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.artifact_root / args.run_name / args.variant / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    server_log_path = run_dir / "server.log"
    client_log_path = run_dir / "client.log"
    monitor_path = run_dir / "monitor.json"
    manifest_path = run_dir / "manifest.json"

    command = server_command(args, variant)
    client_command = benchmark_command(args, variant, result_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(variant["worktree"] / "python")
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.pop("SGLANG_HICACHE_TRACE_PATH", None)
    sync_value = variant["sync_unified_transfers"]
    if sync_value is None:
        environment.pop("SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS", None)
    else:
        environment["SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS"] = sync_value
    explicit_environment: dict[str, str] = {}
    for item in args.server_env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--server-env must be KEY=VALUE, got {item!r}")
        explicit_environment[key] = value
        environment[key] = value

    manifest: dict[str, Any] = {
        "status": "starting",
        "created_wall_time_ns": time.time_ns(),
        "run_dir": str(run_dir),
        "variant": args.variant,
        "variant_definition": {**variant, "worktree": str(variant["worktree"])},
        "worktree_status": dirty_status,
        "worktree_diff": run_text(["git", "diff", "--binary"], cwd=variant["worktree"]),
        "arguments": vars(args)
        | {
            "artifact_root": str(args.artifact_root),
            "python": str(args.python),
            "model": str(args.model),
            "accuracy_dataset": str(args.accuracy_dataset),
        },
        "server_command": command,
        "client_command": client_command,
        "environment_overrides": {
            "PYTHONPATH": environment["PYTHONPATH"],
            "TOKENIZERS_PARALLELISM": environment["TOKENIZERS_PARALLELISM"],
            "SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS": sync_value,
            **explicit_environment,
        },
        "hardware_before": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.free,driver_version,pstate,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ]
        ),
        "python_version": run_text([str(args.python), "--version"]),
    }
    atomic_json(manifest_path, manifest)

    process: subprocess.Popen[str] | None = None
    exit_code = 1
    try:
        with server_log_path.open("w") as server_log:
            process = subprocess.Popen(
                command,
                cwd=variant["worktree"],
                env=environment,
                text=True,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            manifest["server_pid"] = process.pid
            atomic_json(manifest_path, manifest)
            wait_for_server(
                f"http://127.0.0.1:{args.port}", process, args.server_timeout
            )
            manifest["status"] = "server_ready"
            manifest["server_ready_wall_time_ns"] = time.time_ns()
            atomic_json(manifest_path, manifest)

            exit_code, timed_out = run_client_with_monitor(
                client_command,
                cwd=REPO_ROOT,
                log_path=client_log_path,
                monitor_path=monitor_path,
                base_url=f"http://127.0.0.1:{args.port}",
                timeout_s=args.client_timeout,
                monitor_interval_s=args.monitor_interval,
            )
            manifest["client_exit_code"] = exit_code
            manifest["client_timed_out"] = timed_out
            manifest["status"] = "completed" if exit_code == 0 else "failed"
    except Exception as exc:  # noqa: BLE001 - persist orchestration failure
        manifest["status"] = "failed"
        manifest["orchestration_error"] = repr(exc)
        raise
    finally:
        if process is not None:
            terminate_process_group(process)
        manifest["finished_wall_time_ns"] = time.time_ns()
        manifest["hardware_after"] = run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.free,driver_version,pstate,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ]
        )
        atomic_json(manifest_path, manifest)
        print(run_dir)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
