#!/usr/bin/env python3
"""Run the Qwen3.5 static/unified speculative-decoding validation matrix.

The runner intentionally keeps CUDA graph, scheduler mode, token budget, and
workload fixed while varying only memory mode and speculative algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests
from transformers import AutoTokenizer

CONFIGS = {
    "S0": [],
    "U0": ["--enable-unified-memory"],
    "S-N": [
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-num-draft-tokens",
        "4",
    ],
    "U-N": [
        "--enable-unified-memory",
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-num-draft-tokens",
        "4",
    ],
    "S-M": [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ],
    "U-M": [
        "--enable-unified-memory",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=128)
    return parser.parse_args()


def wait_ready(base_url: str, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            if requests.get(f"{base_url}/v1/models", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 10)):
        os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def download_gsm8k(output_dir: Path) -> list[dict]:
    path = output_dir / "gsm8k_test.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "master/grade_school_math/data/test.jsonl",
            path,
        )
    with path.open() as file:
        return [json.loads(line) for line in file]


def answer_value(text: str):
    normalized = text.replace(",", "")
    # GSM8K generations can continue into another few-shot example after the
    # first answer. Grade the first explicit answer marker, not an unrelated
    # number in that trailing continuation.
    marked = re.search(r"####\s*(-?\d+(?:\.\d+)?)", normalized)
    if marked:
        return marked.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    return numbers[-1] if numbers else None


def accuracy_cases(tokenizer, gsm8k: list[dict]) -> list[dict]:
    cases = [
        {"name": "capital", "text": "The capital of France is", "max_new_tokens": 32},
        {
            "name": "arithmetic",
            "text": "Compute 37 * 19. Answer:",
            "max_new_tokens": 48,
        },
        {
            "name": "reasoning",
            "text": "A box has 12 red and 8 blue balls. Give the total and explain briefly.",
            "max_new_tokens": 64,
        },
        {
            "name": "ngram_repetition",
            "text": "alpha beta gamma delta " * 128 + "alpha beta",
            "max_new_tokens": 64,
        },
    ]
    filler = "The archive contains routine status notes with no secret number. "
    prefix_ids = tokenizer.encode(
        "Remember this exact retrieval key: ORCHID-7319. ",
        add_special_tokens=False,
    )
    suffix_ids = tokenizer.encode(
        "What is the exact retrieval key? Answer with only the key.",
        add_special_tokens=False,
    )
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    filler_budget = 8192 - len(prefix_ids) - len(suffix_ids)
    long_ids = prefix_ids + (filler_ids * 1800)[:filler_budget] + suffix_ids
    cases.append(
        {
            "name": "retrieval_8k",
            "input_ids": long_ids,
            "max_new_tokens": 32,
        }
    )

    few_shot = ""
    for example in gsm8k[:4]:
        few_shot += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
    for index, example in enumerate(gsm8k[4:20]):
        cases.append(
            {
                "name": f"gsm8k_{index:02d}",
                "text": few_shot + f"Question: {example['question']}\nAnswer:",
                "max_new_tokens": 128,
                "label": answer_value(example["answer"]),
            }
        )
    return cases


def run_accuracy(base_url: str, cases: list[dict]) -> list[dict]:
    requests.post(f"{base_url}/flush_cache", timeout=30).raise_for_status()
    records = []
    for case in cases:
        payload = {
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": case["max_new_tokens"],
            },
            "return_logprob": True,
        }
        if "input_ids" in case:
            payload["input_ids"] = case["input_ids"]
        else:
            payload["text"] = case["text"]
        response = requests.post(f"{base_url}/generate", json=payload, timeout=300)
        response.raise_for_status()
        body = response.json()
        token_ids = [
            int(item[1]) for item in body["meta_info"]["output_token_logprobs"]
        ]
        text = body["text"]
        records.append(
            {
                "name": case["name"],
                "token_ids": token_ids,
                "token_sha256": hashlib.sha256(
                    json.dumps(token_ids).encode()
                ).hexdigest(),
                "text": text,
                "prediction": answer_value(text) if "label" in case else None,
                "label": case.get("label"),
                "correct": (
                    answer_value(text) == case["label"] if "label" in case else None
                ),
                "prompt_tokens": body["meta_info"]["prompt_tokens"],
                "completion_tokens": body["meta_info"]["completion_tokens"],
            }
        )
    return records


def run_perf(args, config: str, base_url: str, perf_file: Path) -> None:
    for input_len in (3000, 8000):
        for repetition in range(1, args.repetitions + 1):
            cmd = [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--backend",
                "sglang",
                "--base-url",
                base_url,
                "--dataset-name",
                "random",
                "--tokenizer",
                args.model,
                "--num-prompts",
                str(args.num_prompts),
                "--random-input-len",
                str(input_len),
                "--random-output-len",
                str(args.output_len),
                "--random-range-ratio",
                "1",
                "--request-rate",
                "inf",
                "--max-concurrency",
                str(args.max_concurrency),
                "--seed",
                "20260809",
                "--flush-cache",
                "--warmup-requests",
                "1",
                "--output-file",
                str(perf_file),
                "--tag",
                f"{config}_{input_len}_r{repetition}",
            ]
            subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_file = args.output_dir / "perf_final_raw.jsonl"
    accuracy_file = args.output_dir / "accuracy_final_raw.json"
    if perf_file.exists():
        perf_file.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = accuracy_cases(tokenizer, download_gsm8k(args.output_dir))
    accuracy_results = {}
    base_url = f"http://127.0.0.1:{args.port}"

    common = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
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
        "60000",
        "--max-running-requests",
        "16",
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "65536",
        "--mem-fraction-static",
        "0.5",
        "--trust-remote-code",
        "--mm-feature-transport",
        "cpu",
        "--cuda-graph-max-bs-decode",
        "16",
        "--disable-overlap-schedule",
    ]

    for config in args.configs:
        log_path = args.output_dir / f"server_{config}.log"
        print(f"\n=== {config}: launch ===", flush=True)
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                common + CONFIGS[config],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_ready(base_url, process)
                accuracy_results[config] = run_accuracy(base_url, cases)
                accuracy_file.write_text(
                    json.dumps(accuracy_results, indent=2, ensure_ascii=False)
                )
                run_perf(args, config, base_url, perf_file)
            finally:
                stop_server(process)

    baseline = {record["name"]: record for record in accuracy_results["S0"]}
    comparisons = {}
    for config, records in accuracy_results.items():
        comparisons[config] = {
            "exact_token_cases": sum(
                record["token_ids"] == baseline[record["name"]]["token_ids"]
                for record in records
            ),
            "total_cases": len(records),
            "gsm8k_correct": sum(record["correct"] is True for record in records),
            "gsm8k_total": sum(record["correct"] is not None for record in records),
        }
    (args.output_dir / "accuracy_summary.json").write_text(
        json.dumps(
            {
                "against_S0": comparisons,
                "static_unified_pairs": {
                    f"{static}_vs_{unified}": {
                        "exact_token_cases": sum(
                            lhs["token_ids"] == rhs["token_ids"]
                            for lhs, rhs in zip(
                                accuracy_results[static], accuracy_results[unified]
                            )
                        ),
                        "total_cases": len(accuracy_results[static]),
                        "gsm8k_primary_answer_matches": sum(
                            lhs["prediction"] == rhs["prediction"]
                            for lhs, rhs in zip(
                                accuracy_results[static], accuracy_results[unified]
                            )
                            if lhs["label"] is not None
                        ),
                        "gsm8k_total": sum(
                            record["label"] is not None
                            for record in accuracy_results[static]
                        ),
                    }
                    for static, unified in (
                        ("S0", "U0"),
                        ("S-N", "U-N"),
                        ("S-M", "U-M"),
                    )
                },
            },
            indent=2,
        )
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
