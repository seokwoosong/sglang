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
EVAL_WORKTREE_ROOT = Path(
    os.environ.get(
        "SGLANG_ABLATION_WORKTREE_ROOT",
        "/home/sukwoo24/sglang-eval-worktrees/ablation-clean",
    )
)
DEFAULT_MODEL = Path(
    "/home/sukwoo24/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
EVAL_SERVER_SHA = "34b8af8f2bee3b769b07bceb6410bda1441aa507"
EVAL_SERVER_WORKTREE = Path("/home/sukwoo24/sglang-eval-worktrees/qwen08-eval-server")

VARIANTS = {
    # Fair final-source evaluation: all four variants use the same allocator,
    # compaction, transfer, and profiling code. Only the advertised memory/L2
    # configuration changes.
    "eval-s1": {
        "sha": EVAL_SERVER_SHA,
        "worktree": EVAL_SERVER_WORKTREE,
        "unified": False,
        "hicache": True,
        "sync_unified_transfers": None,
        "hicache_mem_layout": "page_first",
        "server_env": {"SGLANG_HICACHE_UNIFIED_TYPED_L2": "1"},
    },
    "eval-u0": {
        "sha": EVAL_SERVER_SHA,
        "worktree": EVAL_SERVER_WORKTREE,
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
        "server_env": {"SGLANG_HICACHE_UNIFIED_TYPED_L2": "1"},
    },
    "eval-u2": {
        "sha": EVAL_SERVER_SHA,
        "worktree": EVAL_SERVER_WORKTREE,
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
        "hicache_mem_layout": "page_first",
        "server_env": {"SGLANG_HICACHE_UNIFIED_TYPED_L2": "0"},
    },
    "eval-u3": {
        "sha": EVAL_SERVER_SHA,
        "worktree": EVAL_SERVER_WORKTREE,
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
        "hicache_mem_layout": "page_first",
        "server_env": {"SGLANG_HICACHE_UNIFIED_TYPED_L2": "1"},
    },
    # Multi-token typed-L2 evaluation. Both variants use identical source;
    # only --enable-unified-memory selects OURS.
    "paged-baseline": {
        "sha": "4c886ec5978bbe778ea3199b5fef18ca92f39c18",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/paged-l2-server"),
        "unified": False,
        "hicache": True,
        "sync_unified_transfers": None,
        "hicache_mem_layout": "page_first",
    },
    "paged-ours": {
        "sha": "4c886ec5978bbe778ea3199b5fef18ca92f39c18",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/paged-l2-server"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
        "hicache_mem_layout": "page_first",
    },
    # Standalone lazy-compaction mapping benchmark. These variants deliberately
    # disable HiCache so the comparison isolates unified-memory compaction.
    "mapping-static": {
        "sha": "52afe87a08c6aa049c52f9507b4f0ca26cecb562",
        "worktree": Path(
            "/home/sukwoo24/sglang-eval-worktrees/upstream-compaction-before"
        ),
        "unified": False,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "mapping-before": {
        "sha": "52afe87a08c6aa049c52f9507b4f0ca26cecb562",
        "worktree": Path(
            "/home/sukwoo24/sglang-eval-worktrees/upstream-compaction-before"
        ),
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "mapping-after": {
        "sha": "66535cef39775b13a589668fc1429d57e8f8ff03",
        "worktree": Path(
            "/home/sukwoo24/sglang-eval-worktrees/upstream-compaction-batch-lookup"
        ),
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    # Direct-copy experiment: all three variants use the same production
    # source. Only the GPU layout and scheduler-side transfer fence differ.
    "direct-plain": {
        "sha": "d173689debcd1b9c54159deb220fb0b60099474a",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/direct-copy"),
        "unified": False,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "direct-u0": {
        "sha": "d173689debcd1b9c54159deb220fb0b60099474a",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/direct-copy"),
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "direct-static": {
        "sha": "d173689debcd1b9c54159deb220fb0b60099474a",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/direct-copy"),
        "unified": False,
        "hicache": True,
        "sync_unified_transfers": None,
        # Qwen3.5's Mamba host pool supports page-first storage. "Static
        # layer-first" refers to the non-unified GPU pool, not this L2 layout.
        "hicache_mem_layout": "page_first",
    },
    "direct-u1": {
        "sha": "d173689debcd1b9c54159deb220fb0b60099474a",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/direct-copy"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "1",
        "hicache_mem_layout": "page_first",
    },
    "direct-u2": {
        "sha": "d173689debcd1b9c54159deb220fb0b60099474a",
        "worktree": Path("/home/sukwoo24/sglang-eval-worktrees/direct-copy"),
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
        "hicache_mem_layout": "page_first",
    },
    # Diagnostic control: the same historical source as U0, with the static
    # (non-unified) hybrid pools. It is not one of the four reported variants.
    "plain": {
        "sha": "8279702e0b8f159c29ed201d05503a0548fefef9",
        "worktree": EVAL_WORKTREE_ROOT / "plain",
        "unified": False,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "u0": {
        "sha": "b6c9c14037ccabf263f7cdae73c31a3ad1b609cc",
        "worktree": EVAL_WORKTREE_ROOT / "u0",
        "unified": True,
        "hicache": False,
        "sync_unified_transfers": None,
    },
    "u1": {
        "sha": "61b05e439c72e729cf1f2b967dc22f5f8f8e18a9",
        "worktree": EVAL_WORKTREE_ROOT / "u1",
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": None,
    },
    "u2": {
        "sha": "969f4bf4b3fa97fed2e3c0b624158f9f44b267fe",
        "worktree": EVAL_WORKTREE_ROOT / "u2",
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "0",
    },
    "u2-sync-control": {
        "sha": "969f4bf4b3fa97fed2e3c0b624158f9f44b267fe",
        "worktree": EVAL_WORKTREE_ROOT / "u2",
        "unified": True,
        "hicache": True,
        "sync_unified_transfers": "1",
    },
    "u3": {
        "sha": "f6982dadc336666774d7d7eff44446c6c6f85999",
        "worktree": EVAL_WORKTREE_ROOT / "u3",
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


def collect_memory_profiles(profile_dir: Path) -> list[dict[str, Any]]:
    profiles = []
    if not profile_dir.is_dir():
        return profiles
    for path in sorted(profile_dir.glob("memory_profile.*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            profiles.append({"path": str(path), "read_error": repr(exc)})
            continue
        profiles.append({"path": str(path), "profile": payload})
    return profiles


def attach_memory_profiles(
    result_path: Path,
    manifest: dict[str, Any],
    profile_dir: Path,
) -> None:
    profiles = collect_memory_profiles(profile_dir)
    manifest["memory_breakdown_profiles"] = profiles
    if not result_path.is_file():
        return
    payload = json.loads(result_path.read_text())
    payload["memory_breakdown_profiles"] = profiles
    atomic_json(result_path, payload)


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
        str(args.page_size),
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
        "--max-running-requests",
        str(args.max_running_requests),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--context-length",
        str(args.context_length),
        "--enable-metrics",
        "--log-level",
        args.log_level,
    ]
    if args.cuda_graph_mode == "enabled":
        command.extend(
            [
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-backend-prefill",
                "breakable",
            ]
        )
    else:
        command.extend(
            [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ]
        )
    if args.max_mamba_cache_size is not None:
        command.extend(["--max-mamba-cache-size", str(args.max_mamba_cache_size)])
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
                variant.get("hicache_mem_layout", "page_first"),
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
    if args.torch_profile_steps > 0:
        command.extend(
            [
                "--torch-profile-steps",
                str(args.torch_profile_steps),
                "--torch-profile-output-dir",
                str((output.parent / "torch_profile").resolve()),
            ]
        )
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
                "--prime-repeats",
                str(args.prime_repeats),
                "--max-concurrency",
                str(args.max_concurrency),
            ]
        )
        if args.reverse_group_order:
            command.append("--reverse-group-order")
        if args.require_eviction:
            command.append("--require-eviction")
        if args.require_loadback and variant["hicache"]:
            command.append("--require-loadback")
        if args.require_backup and variant["hicache"]:
            command.append("--require-backup")
        if args.require_host_hit and variant["hicache"]:
            command.append("--require-host-hit")
        if args.forbid_dropped:
            command.append("--forbid-dropped")
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
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--max-mamba-cache-size",
        type=int,
        default=None,
        help=(
            "Optional explicit Mamba slot cap. Omit it to exercise the server's "
            "automatic unified-memory sizing."
        ),
    )
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
    parser.add_argument(
        "--cuda-graph-mode",
        choices=["enabled", "disabled"],
        default="disabled",
        help=(
            "Use decode=full and prefill=breakable for clean/correctness runs, "
            "or disable both graphs for component profiling."
        ),
    )
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
    parser.add_argument("--reverse-group-order", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--shared-ratio", type=float, default=0.95)
    parser.add_argument("--prime-output-len", type=int, default=1)
    parser.add_argument("--prime-repeats", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--require-eviction", action="store_true")
    parser.add_argument("--require-loadback", action="store_true")
    parser.add_argument("--require-backup", action="store_true")
    parser.add_argument("--require-host-hit", action="store_true")
    parser.add_argument("--forbid-dropped", action="store_true")
    parser.add_argument(
        "--profile-memory-breakdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Collect allocator/translation/compaction/fence and L1 layout "
            "profiles into the run artifact (enabled by default)."
        ),
    )
    parser.add_argument(
        "--torch-profile-steps",
        type=int,
        default=0,
        help=(
            "Optionally capture the first N measured forward steps with the "
            "built-in CPU/GPU torch profiler. This is diagnostic and affects "
            "the profiled requests, so leave it at zero for clean performance runs."
        ),
    )
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
    # The server runs from the pinned variant worktree, while the benchmark
    # client runs from this repository. Use one absolute path so both processes
    # observe the same snapshots and result.json can embed their deltas.
    memory_profile_dir = (run_dir / "memory_breakdown_profile").resolve()

    command = server_command(args, variant)
    client_command = benchmark_command(args, variant, result_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(variant["worktree"] / "python")
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.pop("SGLANG_HICACHE_TRACE_PATH", None)
    if args.profile_memory_breakdown:
        environment["SGLANG_MEMORY_BREAKDOWN_PROFILE_DIR"] = str(memory_profile_dir)
        # SGLang's existing async CUDA-event timer exposes aggregate model
        # forward GPU time without a device synchronize. It complements the
        # allocator/layout profiler and makes the residual model-path cost
        # visible in the same result JSON.
        environment["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "true"
    else:
        environment.pop("SGLANG_MEMORY_BREAKDOWN_PROFILE_DIR", None)
        environment.pop("SGLANG_ENABLE_METRICS_DEVICE_TIMER", None)
    sync_value = variant["sync_unified_transfers"]
    if sync_value is None:
        environment.pop("SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS", None)
    else:
        environment["SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS"] = sync_value
    variant_environment = dict(variant.get("server_env", {}))
    environment.update(variant_environment)
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
            "SGLANG_MEMORY_BREAKDOWN_PROFILE_DIR": (
                str(memory_profile_dir) if args.profile_memory_breakdown else None
            ),
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER": (
                "true" if args.profile_memory_breakdown else None
            ),
            **variant_environment,
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
            if args.profile_memory_breakdown:
                # Allow the aggregate writer's final periodic snapshot to include
                # the last workload operations without synchronizing the server.
                time.sleep(0.5)
    except Exception as exc:  # noqa: BLE001 - persist orchestration failure
        manifest["status"] = "failed"
        manifest["orchestration_error"] = repr(exc)
        raise
    finally:
        if process is not None:
            terminate_process_group(process)
        if args.profile_memory_breakdown:
            attach_memory_profiles(result_path, manifest, memory_profile_dir)
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
