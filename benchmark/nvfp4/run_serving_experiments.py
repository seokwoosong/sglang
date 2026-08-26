"""Run the reproducible NVFP4 serving comparison on one SM120 GPU.

The runner is intentionally resumable: a completed, structurally valid result
is retained, while failed or partial measurements stop the run and must be
diagnosed before continuing.  Run one server configuration at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PASSKEY_SCRIPT = Path(__file__).with_name("eval_passkey_retrieval.py")


@dataclass(frozen=True)
class ServingConfig:
    name: str
    model_kind: str
    kv_cache_dtype: str
    extra_args: tuple[str, ...]


CONFIGS = {
    "bf16": ServingConfig("bf16", "bf16", "bfloat16", ()),
    "w4a4_bf16_kv": ServingConfig(
        "w4a4_bf16_kv",
        "nvfp4",
        "bfloat16",
        ("--quantization", "modelopt_fp4", "--fp4-gemm-backend", "auto"),
    ),
    "w4a4_nvfp4_kv": ServingConfig(
        "w4a4_nvfp4_kv",
        "nvfp4",
        "nvfp4",
        ("--quantization", "modelopt_fp4", "--fp4-gemm-backend", "auto"),
    ),
}

WORKLOADS = {
    "prefill_heavy": (2048, 128),
    "balanced": (1024, 1024),
    "decode_heavy": (128, 2048),
}
PROMPTS_BY_CONCURRENCY = {1: 4, 8: 16, 32: 64}


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def run_capture(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def query_gpu_memory_mib() -> int:
    output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    return int(output.splitlines()[0].strip())


class GpuMemoryMonitor:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_mib = max(self.peak_mib, query_gpu_memory_mib())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def wait_for_server(base_url: str, process: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/model_info", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"server did not become ready in {timeout_s}s: {last_error}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_logged(
    command: list[str], log_path: Path, timeout_s: int | None = None
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w") as log_file, GpuMemoryMonitor() as monitor:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    duration_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with code {completed.returncode}; see {log_path}"
        )
    return monitor.peak_mib, duration_s


def valid_speed_result(
    path: Path,
    expected_prompts: int,
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> bool:
    if not path.exists():
        return False
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            return False
        result = json.loads(lines[0])
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("completed") == expected_prompts
        and result.get("total_input_tokens") == expected_prompts * expected_input_tokens
        and result.get("total_output_tokens")
        == expected_prompts * expected_output_tokens
    )


def annotate_speed_result(path: Path, metadata: dict[str, Any]) -> None:
    result = json.loads(path.read_text().strip())
    result["nvfp4_experiment"] = metadata
    path.write_text(json.dumps(result) + "\n")


def valid_gsm8k_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("score"), (int, float))


def valid_passkey_result(path: Path, cases_per_length: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    summaries = payload.get("summary", {})
    return all(
        summaries.get(str(length), {}).get("cases") == cases_per_length
        for length in (4096, 8192)
    )


def write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    try:
        torch_version = run_capture(
            [sys.executable, "-c", "import torch; print(torch.__version__)"]
        )
        flashinfer_version = run_capture(
            [sys.executable, "-c", "import flashinfer; print(flashinfer.__version__)"]
        )
    except subprocess.SubprocessError:
        torch_version = flashinfer_version = "unknown"
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": run_capture(["git", "rev-parse", "HEAD"]),
        "git_branch": run_capture(["git", "branch", "--show-current"]),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch_version,
        "flashinfer": flashinfer_version,
        "nvidia_smi": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
                "--id=0",
            ]
        ),
        "models": {"bf16": args.bf16_model, "nvfp4": args.nvfp4_model},
        "serving_invariants": {
            "dtype": "bfloat16",
            "prefill_attention_backend": "flashinfer",
            "decode_attention_backend": "trtllm_mha",
            "page_size": 64,
            "mem_fraction_static": 0.90,
            "max_total_tokens": 73728,
            "cuda_graph_max_bs_decode": 32,
            "random_seed": 42,
            "mm_feature_transport": "cpu",
            "max_initial_gpu_memory_mib": args.max_initial_gpu_memory_mib,
        },
        "speed": {
            "workloads": WORKLOADS,
            "concurrency": args.concurrency,
            "prompts_by_concurrency": PROMPTS_BY_CONCURRENCY,
            "repeats": args.repeats,
            "request_rate": "inf",
            "random_range_ratio": 1.0,
            "temperature": 0.0,
            "cache_flushed_after_warmup": True,
        },
        "accuracy": {
            "gsm8k_examples": args.gsm8k_examples,
            "gsm8k_num_shots": 5,
            "gsm8k_temperature": 0.0,
            "passkey_context_lengths": [4096, 8192],
            "passkey_cases_per_length": args.passkey_cases,
        },
        "design_note": (
            "max_total_tokens is 73728 because 32 concurrent requests at the "
            "largest 2176-token workload require 69632 resident tokens; 65536 "
            "would silently reduce effective concurrency."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def server_command(
    args: argparse.Namespace, config: ServingConfig, model_path: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        "nvfp4-eval",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        config.kv_cache_dtype,
        "--prefill-attention-backend",
        "flashinfer",
        "--decode-attention-backend",
        "trtllm_mha",
        "--page-size",
        "64",
        "--mem-fraction-static",
        "0.90",
        "--max-total-tokens",
        "73728",
        "--cuda-graph-max-bs-decode",
        "32",
        "--random-seed",
        "42",
        "--mm-feature-transport",
        "cpu",
        "--trust-remote-code",
        *config.extra_args,
    ]


def run_speed_suite(
    args: argparse.Namespace,
    config: ServingConfig,
    model_path: str,
    config_dir: Path,
    initial_memory_mib: int,
) -> None:
    speed_dir = config_dir / "speed"
    log_dir = config_dir / "logs" / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    for workload_name in args.workloads:
        input_len, output_len = WORKLOADS[workload_name]
        for concurrency in args.concurrency:
            prompts = PROMPTS_BY_CONCURRENCY[concurrency]
            for repeat in range(1, args.repeats + 1):
                stem = f"{workload_name}_c{concurrency}_r{repeat}"
                output_path = speed_dir / f"{stem}.jsonl"
                if valid_speed_result(output_path, prompts, input_len, output_len):
                    print(
                        f"Retaining completed speed result: {output_path}", flush=True
                    )
                    continue
                output_path.unlink(missing_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "sglang.benchmark.serving",
                    "--backend",
                    "sglang",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--model",
                    model_path,
                    "--served-model-name",
                    "nvfp4-eval",
                    "--tokenizer",
                    args.bf16_model,
                    "--dataset-name",
                    "random",
                    "--num-prompts",
                    str(prompts),
                    "--random-input-len",
                    str(input_len),
                    "--random-output-len",
                    str(output_len),
                    "--random-range-ratio",
                    "1",
                    "--request-rate",
                    "inf",
                    "--max-concurrency",
                    str(concurrency),
                    "--seed",
                    "42",
                    "--temperature",
                    "0",
                    "--warmup-requests",
                    str(min(concurrency, 8)),
                    "--flush-cache",
                    "--tokenize-prompt",
                    "--disable-tqdm",
                    "--output-file",
                    str(output_path),
                ]
                print(
                    f"Running {config.name} {workload_name} c={concurrency} "
                    f"repeat={repeat}/{args.repeats}",
                    flush=True,
                )
                peak_mib, duration_s = run_logged(command, log_dir / f"{stem}.log")
                if not valid_speed_result(output_path, prompts, input_len, output_len):
                    raise RuntimeError(f"invalid or incomplete result: {output_path}")
                annotate_speed_result(
                    output_path,
                    {
                        "config": config.name,
                        "workload": workload_name,
                        "target_input_tokens": input_len,
                        "target_output_tokens": output_len,
                        "target_concurrency": concurrency,
                        "repeat": repeat,
                        "initial_gpu_memory_mib": initial_memory_mib,
                        "peak_gpu_memory_mib": peak_mib,
                        "peak_gpu_memory_delta_mib": peak_mib - initial_memory_mib,
                        "client_duration_s": duration_s,
                        "command": command,
                    },
                )


def run_accuracy_suite(
    args: argparse.Namespace,
    config: ServingConfig,
    config_dir: Path,
    initial_memory_mib: int,
) -> None:
    accuracy_dir = config_dir / "accuracy"
    log_dir = config_dir / "logs" / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_path = accuracy_dir / "gsm8k.json"
    if not valid_gsm8k_result(gsm8k_path):
        temporary_result = Path("/tmp/gsm8k_nvfp4-eval.json")
        temporary_result.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "sglang.test.run_eval",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            "nvfp4-eval",
            "--eval-name",
            "gsm8k",
            "--api",
            "generate",
            "--num-examples",
            str(args.gsm8k_examples),
            "--num-threads",
            "32",
            "--num-shots",
            "5",
            "--max-tokens",
            "512",
            "--temperature",
            "0",
        ]
        print(f"Running {config.name} GSM8K", flush=True)
        peak_mib, duration_s = run_logged(
            command, log_dir / "gsm8k.log", timeout_s=3600
        )
        if not temporary_result.exists():
            raise RuntimeError(f"GSM8K did not write {temporary_result}")
        payload = json.loads(temporary_result.read_text())
        payload["nvfp4_experiment"] = {
            "config": config.name,
            "examples": args.gsm8k_examples,
            "initial_gpu_memory_mib": initial_memory_mib,
            "peak_gpu_memory_mib": peak_mib,
            "peak_gpu_memory_delta_mib": peak_mib - initial_memory_mib,
            "client_duration_s": duration_s,
            "command": command,
        }
        gsm8k_path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(f"Retaining completed GSM8K result: {gsm8k_path}", flush=True)

    passkey_path = accuracy_dir / "passkey.json"
    if not valid_passkey_result(passkey_path, args.passkey_cases):
        passkey_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(PASSKEY_SCRIPT),
            "--base-url",
            f"http://{args.host}:{args.port}",
            "--tokenizer",
            args.bf16_model,
            "--context-lengths",
            "4096,8192",
            "--cases-per-length",
            str(args.passkey_cases),
            "--parallel",
            "8",
            "--seed",
            "42",
            "--output",
            str(passkey_path),
        ]
        print(f"Running {config.name} passkey retrieval", flush=True)
        peak_mib, duration_s = run_logged(
            command, log_dir / "passkey.log", timeout_s=3600
        )
        if not valid_passkey_result(passkey_path, args.passkey_cases):
            raise RuntimeError(f"invalid or incomplete passkey result: {passkey_path}")
        payload = json.loads(passkey_path.read_text())
        payload["nvfp4_experiment"] = {
            "config": config.name,
            "initial_gpu_memory_mib": initial_memory_mib,
            "peak_gpu_memory_mib": peak_mib,
            "peak_gpu_memory_delta_mib": peak_mib - initial_memory_mib,
            "client_duration_s": duration_s,
            "command": command,
        }
        passkey_path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(f"Retaining completed passkey result: {passkey_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-model", required=True)
    parser.add_argument("--nvfp4-model", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark/nvfp4/results/2026-08-26-rtx5090",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--configs", type=parse_csv, default=list(CONFIGS), help="comma-separated"
    )
    parser.add_argument(
        "--workloads", type=parse_csv, default=list(WORKLOADS), help="comma-separated"
    )
    parser.add_argument("--concurrency", type=parse_csv, default=["1", "8", "32"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gsm8k-examples", type=int, default=200)
    parser.add_argument("--passkey-cases", type=int, default=30)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument(
        "--max-initial-gpu-memory-mib",
        type=int,
        default=4096,
        help=(
            "abort if baseline GPU memory exceeds this value; the RTX 5090 test "
            "host reserves about 2.3 GiB for its display/driver with no compute process"
        ),
    )
    args = parser.parse_args()
    args.concurrency = [int(value) for value in args.concurrency]
    # The first FlashInfer SM120 CUTLASS JIT build uses roughly 4-5 GiB of host
    # RAM per translation unit.  Two jobs avoid OOM on the 32-GiB test host.
    os.environ.setdefault("MAX_JOBS", "2")

    unknown_configs = sorted(set(args.configs) - set(CONFIGS))
    unknown_workloads = sorted(set(args.workloads) - set(WORKLOADS))
    unknown_concurrency = sorted(set(args.concurrency) - set(PROMPTS_BY_CONCURRENCY))
    if unknown_configs or unknown_workloads or unknown_concurrency:
        raise ValueError(
            f"unknown selections: configs={unknown_configs}, "
            f"workloads={unknown_workloads}, concurrency={unknown_concurrency}"
        )
    for model_path in (args.bf16_model, args.nvfp4_model):
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)

    args.output_dir = args.output_dir.resolve()
    write_manifest(args, args.output_dir)
    base_url = f"http://{args.host}:{args.port}"
    for config_name in args.configs:
        config = CONFIGS[config_name]
        model_path = (
            args.bf16_model if config.model_kind == "bf16" else args.nvfp4_model
        )
        config_dir = args.output_dir / config.name
        config_dir.mkdir(parents=True, exist_ok=True)
        server_log = config_dir / "logs" / "server.log"
        server_log.parent.mkdir(parents=True, exist_ok=True)
        command = server_command(args, config, model_path)
        initial_memory = query_gpu_memory_mib()
        if initial_memory > args.max_initial_gpu_memory_mib:
            raise RuntimeError(
                f"GPU has {initial_memory} MiB in use before server launch; "
                f"limit is {args.max_initial_gpu_memory_mib} MiB; refusing a "
                "contaminated measurement"
            )
        print(f"Launching {config.name}: {' '.join(command)}", flush=True)
        with server_log.open("a") as log_file:
            log_file.write("\nCOMMAND: " + " ".join(command) + "\n\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                wait_for_server(base_url, process, args.server_timeout)
                if not args.skip_speed:
                    run_speed_suite(
                        args, config, model_path, config_dir, initial_memory
                    )
                if not args.skip_accuracy:
                    run_accuracy_suite(args, config, config_dir, initial_memory)
            finally:
                stop_server(process)
                time.sleep(5)


if __name__ == "__main__":
    main()
