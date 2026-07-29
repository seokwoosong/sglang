#!/usr/bin/env python3
"""Run and compare default/Marconi Mamba admission on a real SGLang server.

Example:
    python scripts/verify_marconi_mamba_admission.py \
        --model-path Qwen/Qwen3.5-0.8B

The script launches this checkout twice with unified memory enabled, sends the
same deterministic branch-producing workload, parses the structured
MAMBA_ADMISSION_STATS records, and writes a machine-readable comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

POLICIES = ("default", "marconi")
STATS_MARKER = "MAMBA_ADMISSION_STATS "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare default and Marconi Mamba state admission."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--output-dir", default="/tmp/sglang-marconi-verification")
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Attempt launch even when this Python environment cannot see CUDA.",
    )
    parser.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments after this flag are appended to both server commands.",
    )
    return parser.parse_args()


def check_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. Run the script "
            "from the same CUDA environment used to run SGLang."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. --enable-unified-memory uses the CUDA/Triton "
            "hybrid-cache path, so this end-to-end check requires an NVIDIA GPU."
        )


def http_json(url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_until_ready(
    base_url: str, process: subprocess.Popen[str], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited during startup with code {return_code}")
        try:
            with urllib.request.urlopen(
                f"{base_url}/health_generate", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server startup timed out: {last_error}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def workload() -> list[tuple[str, str]]:
    # A long common prefix makes chunked prefill generate intermediate
    # candidates. Distinct suffixes split the radix path and expose an aligned
    # branch point on a later request.
    common = (
        "This is a deterministic cache-admission ledger. "
        "Preserve every earlier entry before answering the final instruction. "
        "Entry values are alpha, beta, gamma, and delta. "
    ) * 220
    return [
        ("seed", common + "\nBranch A: Reply with the word ALPHA."),
        ("fork", common + "\nBranch B: Reply with the word BETA."),
        ("reuse", common + "\nBranch C: Reply with the word GAMMA."),
    ]


def extract_text(response: dict[str, Any]) -> str:
    text = response.get("text")
    if isinstance(text, list):
        return str(text[0])
    return str(text)


def parse_stats(log_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        marker_position = line.find(STATS_MARKER)
        if marker_position < 0:
            continue
        payload = line[marker_position + len(STATS_MARKER) :]
        records.append(json.loads(payload))
    return records


def run_policy(
    args: argparse.Namespace, repo_root: Path, policy: str, port: int
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{policy}.log"
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-unified-memory",
        "--mamba-state-admission-policy",
        policy,
        "--log-level",
        "debug",
        "--chunked-prefill-size",
        "256",
        "--mamba-track-interval",
        "64",
        "--disable-cuda-graph",
        *args.extra_server_args,
    ]
    env = os.environ.copy()
    python_path = str(repo_root / "python")
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else python_path + os.pathsep + env["PYTHONPATH"]
    )

    responses: dict[str, str] = {}
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            wait_until_ready(base_url, process, args.startup_timeout)
            for name, prompt in workload():
                response = http_json(
                    f"{base_url}/generate",
                    {
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 80,
                        },
                    },
                    args.request_timeout,
                )
                responses[name] = extract_text(response)
        finally:
            stop_process(process)

    return {
        "command": command,
        "log": str(log_path),
        "responses": responses,
        "stats": parse_stats(log_path),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "branch_candidates",
        "final_candidates",
        "intermediate_candidates",
        "branch_admitted",
        "final_admitted",
        "intermediate_admitted",
        "duplicate_candidates",
        "intermediate_skipped",
    )
    return {
        field: sum(int(record.get(field, 0)) for record in records) for field in fields
    }


def validate(results: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    default_stats = aggregate(results["default"]["stats"])
    marconi_stats = aggregate(results["marconi"]["stats"])

    if not results["default"]["stats"] or not results["marconi"]["stats"]:
        failures.append("one or both runs emitted no admission statistics")
    if default_stats["intermediate_admitted"] <= 0:
        failures.append("default admitted no intermediate checkpoint")
    if marconi_stats["intermediate_candidates"] <= 0:
        failures.append("workload produced no Marconi intermediate candidate")
    if marconi_stats["intermediate_skipped"] <= 0:
        failures.append("Marconi skipped no intermediate checkpoint")
    if marconi_stats["intermediate_admitted"] != 0:
        failures.append("Marconi admitted an intermediate checkpoint")
    if marconi_stats["branch_candidates"] <= 0:
        failures.append("workload exposed no aligned branch candidate")

    for record in results["marconi"]["stats"]:
        persistent_candidates = int(record["branch_candidates"]) + int(
            record["final_candidates"]
        )
        if persistent_candidates > 2:
            failures.append(
                f"request {record['rid']} had {persistent_candidates} "
                "persistent candidates (expected at most 2)"
            )

    if results["default"]["responses"] != results["marconi"]["responses"]:
        failures.append("deterministic generated text differs between policies")
    return failures


def main() -> int:
    args = parse_args()
    if not args.skip_cuda_check:
        try:
            check_cuda()
        except RuntimeError as exc:
            skipped = {
                "model_path": args.model_path,
                "verdict": "SKIP",
                "reason": str(exc),
            }
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "comparison.json"
            output_path.write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n")
            print(f"ERROR: {exc}", file=sys.stderr)
            print(f"Machine-readable result: {output_path}", file=sys.stderr)
            return 2

    repo_root = Path(__file__).resolve().parents[1]
    results = {
        policy: run_policy(args, repo_root, policy, args.port + index)
        for index, policy in enumerate(POLICIES)
    }
    comparison = {
        "model_path": args.model_path,
        "default": {
            **results["default"],
            "aggregate": aggregate(results["default"]["stats"]),
        },
        "marconi": {
            **results["marconi"],
            "aggregate": aggregate(results["marconi"]["stats"]),
        },
    }
    failures = validate(results)
    comparison["verdict"] = "PASS" if not failures else "FAIL"
    comparison["failures"] = failures

    output_path = Path(args.output_dir) / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"\nMachine-readable result: {output_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
