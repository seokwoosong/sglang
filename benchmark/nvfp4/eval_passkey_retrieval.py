"""Run a lightweight, exact-length passkey retrieval evaluation.

This evaluator uses SGLang's native ``/generate`` API with token IDs so the
requested 4K/8K prompt lengths are exact and identical between serving
configurations.  It is intentionally a rough KV-cache correctness signal,
not a comprehensive long-context benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def repeat_to_length(pattern: list[int], length: int) -> list[int]:
    if not pattern:
        raise ValueError("filler pattern tokenized to an empty sequence")
    return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]


def build_prompt_ids(tokenizer, target_tokens: int, passkey: str) -> list[int]:
    start = tokenizer.encode(
        "Memory test. Read all records and remember the passkey.\n",
        add_special_tokens=True,
    )
    marker = tokenizer.encode(
        f"\nImportant record: the passkey is {passkey}. Remember {passkey}.\n",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nQuestion: What is the eight-digit passkey? Answer with only the passkey.\nAnswer:",
        add_special_tokens=False,
    )
    filler_pattern = tokenizer.encode(
        "Record: this neutral sentence is unrelated filler for the memory test.\n",
        add_special_tokens=False,
    )
    fixed_tokens = len(start) + len(marker) + len(question)
    if target_tokens <= fixed_tokens:
        raise ValueError(
            f"target length {target_tokens} must exceed fixed prompt length {fixed_tokens}"
        )
    filler_tokens = target_tokens - fixed_tokens
    before = filler_tokens // 2
    after = filler_tokens - before
    prompt_ids = (
        start
        + repeat_to_length(filler_pattern, before)
        + marker
        + repeat_to_length(filler_pattern, after)
        + question
    )
    assert len(prompt_ids) == target_tokens
    return prompt_ids


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"request failed after four attempts: {last_error}")


def run_case(
    base_url: str,
    tokenizer,
    target_tokens: int,
    case_index: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    rng = random.Random(seed + target_tokens * 10_000 + case_index)
    passkey = f"{rng.randrange(10_000_000, 100_000_000):08d}"
    prompt_ids = build_prompt_ids(tokenizer, target_tokens, passkey)
    started = time.perf_counter()
    response = post_json(
        f"{base_url.rstrip('/')}/generate",
        {
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 32,
                "stop": ["\n"],
                "skip_special_tokens": True,
            },
            "stream": False,
        },
        timeout=timeout,
    )
    latency = time.perf_counter() - started
    generated = response.get("text") or ""
    exact_match = (
        re.search(rf"(?<!\d){re.escape(passkey)}(?!\d)", generated) is not None
    )
    meta_info = response.get("meta_info") or {}
    return {
        "context_tokens": target_tokens,
        "case_index": case_index,
        "passkey": passkey,
        "generated_text": generated,
        "exact_match": exact_match,
        "latency_s": latency,
        "server_prompt_tokens": meta_info.get("prompt_tokens"),
        "completion_tokens": meta_info.get("completion_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--context-lengths", type=parse_int_list, default=[4096, 8192])
    parser.add_argument("--cases-per-length", type=int, default=30)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    work = [
        (context_tokens, case_index)
        for context_tokens in args.context_lengths
        for case_index in range(args.cases_per_length)
    ]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        future_to_case = {
            pool.submit(
                run_case,
                args.base_url,
                tokenizer,
                context_tokens,
                case_index,
                args.seed,
                args.request_timeout,
            ): (context_tokens, case_index)
            for context_tokens, case_index in work
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_case), start=1
        ):
            context_tokens, case_index = future_to_case[future]
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(work)}] context={context_tokens} "
                f"case={case_index} exact_match={result['exact_match']}",
                flush=True,
            )

    results.sort(key=lambda item: (item["context_tokens"], item["case_index"]))
    by_context: dict[str, dict[str, Any]] = {}
    for context_tokens in args.context_lengths:
        subset = [item for item in results if item["context_tokens"] == context_tokens]
        matches = sum(item["exact_match"] for item in subset)
        by_context[str(context_tokens)] = {
            "cases": len(subset),
            "exact_matches": matches,
            "accuracy": matches / len(subset),
            "mean_latency_s": sum(item["latency_s"] for item in subset) / len(subset),
        }
    payload = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "cases_per_length": args.cases_per_length,
        "parallel": args.parallel,
        "duration_s": time.perf_counter() - started,
        "summary": by_context,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(by_context, indent=2))


if __name__ == "__main__":
    main()
