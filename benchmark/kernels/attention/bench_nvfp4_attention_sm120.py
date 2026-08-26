"""Benchmark FlashInfer's dense NVFP4 attention kernel on SM120 GPUs.

This is a kernel microbenchmark, not an SGLang serving benchmark.  The current
FlashInfer kernel requires dense self-attention with equal Q/K/V shapes, so it
does not exercise paged KV cache, GQA, radix cache, or single-token decode.

Examples:
    python benchmark/kernels/attention/bench_nvfp4_attention_sm120.py
    python benchmark/kernels/attention/bench_nvfp4_attention_sm120.py \
        --seq-lens 512,1024,2048 --head-dims 64,128 --causal
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.nvfp4_attention_sm120 import (
    nvfp4_attention_sm120_fwd,
    nvfp4_attention_sm120_quantize_qkv,
)
from flashinfer.testing import bench_gpu_time
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class Result:
    seed: int
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    baseline_ms: float
    fp4_attention_ms: float
    fp4_end_to_end_ms: float
    relative_rmse: float
    cosine_similarity: float
    sqnr_db: float
    p99_absolute_error: float
    max_absolute_error: float
    nan_count: int
    inf_count: int


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def median_gpu_time(fn, args: argparse.Namespace) -> float:
    times = bench_gpu_time(
        fn=fn,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.measure_ms,
        cold_l2_cache=not args.warm_l2,
    )
    return float(statistics.median(times))


def benchmark_shape(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
    args: argparse.Namespace,
) -> Result:
    torch.manual_seed(seed)
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    def bf16_attention():
        return F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)

    # Force the BF16 baseline onto PyTorch's fused FlashAttention backend rather
    # than allowing a math-backend fallback that would distort the comparison.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        reference = bf16_attention()
        baseline_ms = median_gpu_time(fn=bf16_attention, args=args)

    quantized_qkv = nvfp4_attention_sm120_quantize_qkv(
        q=q, k=k, v=v, per_block_mean=args.per_block_mean
    )
    out = torch.empty_like(q)
    lse = torch.empty(
        (batch_size, num_heads, seq_len), device="cuda", dtype=torch.float32
    )

    def fp4_attention():
        return nvfp4_attention_sm120_fwd(
            *quantized_qkv,
            causal=args.causal,
            per_block_mean=args.per_block_mean,
            out=out,
            lse=lse,
        )

    fp4_attention()
    torch.cuda.synchronize()
    difference = out.float() - reference.float()
    reference_rms = torch.sqrt(reference.float().square().mean())
    error_rms = torch.sqrt(difference.square().mean())
    relative_rmse = float(
        error_rms / reference_rms.clamp_min(torch.finfo(torch.float32).tiny)
    )
    cosine_similarity = float(
        F.cosine_similarity(out.float().flatten(), reference.float().flatten(), dim=0)
    )
    sqnr_db = (
        math.inf
        if float(error_rms) == 0.0
        else 20.0 * math.log10(float(reference_rms / error_rms))
    )
    absolute_error = difference.abs().flatten()
    p99_absolute_error = float(torch.quantile(absolute_error, 0.99))
    max_absolute_error = float(absolute_error.max())
    nan_count = int(torch.isnan(out).sum())
    inf_count = int(torch.isinf(out).sum())
    fp4_attention_ms = median_gpu_time(fn=fp4_attention, args=args)

    def fp4_end_to_end():
        packed = nvfp4_attention_sm120_quantize_qkv(
            q=q, k=k, v=v, per_block_mean=args.per_block_mean
        )
        return nvfp4_attention_sm120_fwd(
            *packed,
            causal=args.causal,
            per_block_mean=args.per_block_mean,
            out=out,
            lse=lse,
        )

    fp4_end_to_end_ms = median_gpu_time(fn=fp4_end_to_end, args=args)
    return Result(
        seed=seed,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        baseline_ms=baseline_ms,
        fp4_attention_ms=fp4_attention_ms,
        fp4_end_to_end_ms=fp4_end_to_end_ms,
        relative_rmse=relative_rmse,
        cosine_similarity=cosine_similarity,
        sqnr_db=sqnr_db,
        p99_absolute_error=p99_absolute_error,
        max_absolute_error=max_absolute_error,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def print_results(results: list[Result], dtype: torch.dtype) -> None:
    baseline_label = "BF16" if dtype == torch.bfloat16 else "FP16"
    print()
    print(
        f"| seed | S | D | {baseline_label} Flash SDPA (ms) | "
        "NVFP4 attention (ms) | speedup | NVFP4 end-to-end (ms) | speedup | "
        "rel. RMSE | cosine | SQNR (dB) | p99 abs. | NaN/Inf |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result.seed} | {result.seq_len} | {result.head_dim} | "
            f"{result.baseline_ms:.4f} | "
            f"{result.fp4_attention_ms:.4f} | "
            f"{result.baseline_ms / result.fp4_attention_ms:.2f}x | "
            f"{result.fp4_end_to_end_ms:.4f} | "
            f"{result.baseline_ms / result.fp4_end_to_end_ms:.2f}x | "
            f"{result.relative_rmse:.4f} | {result.cosine_similarity:.6f} | "
            f"{result.sqnr_db:.2f} | {result.p99_absolute_error:.5f} | "
            f"{result.nan_count}/{result.inf_count} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seq-lens", type=parse_int_list, default=[512, 1024, 2048])
    parser.add_argument("--head-dims", type=parse_int_list, default=[128])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per-block-mean", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--measure-ms", type=int, default=100)
    parser.add_argument(
        "--warm-l2",
        action="store_true",
        help="do not flush L2 between iterations (the default measures cold L2)",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "FlashInfer NVFP4 attention requires SM120; "
            f"found compute capability {capability[0]}.{capability[1]}"
        )
    invalid_head_dims = [dim for dim in args.head_dims if dim not in (64, 128)]
    if invalid_head_dims:
        raise ValueError(f"head dimensions must be 64 or 128: {invalid_head_dims}")
    invalid_seq_lens = [length for length in args.seq_lens if length % 128 != 0]
    if invalid_seq_lens:
        raise ValueError(
            "sequence lengths must be multiples of 128 to keep the BF16 and "
            f"NVFP4 workloads identical: {invalid_seq_lens}"
        )

    dtype = getattr(torch, args.dtype)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"B={args.batch_size}, H={args.num_heads}, dtype={args.dtype}, "
        f"causal={args.causal}, L2={'warm' if args.warm_l2 else 'cold'}"
    )
    print(
        "Scope: dense MHA prefill/self-attention only; this is not paged serving decode."
    )

    results: list[Result] = []
    for head_dim in args.head_dims:
        for seq_len in args.seq_lens:
            for seed in args.seeds:
                print(
                    f"Benchmarking S={seq_len}, D={head_dim}, seed={seed} ...",
                    flush=True,
                )
                results.append(
                    benchmark_shape(
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        dtype=dtype,
                        seed=seed,
                        args=args,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()
    print_results(results=results, dtype=dtype)
    if args.output_json:
        payload = {
            "schema_version": 1,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "dtype": args.dtype,
            "causal": args.causal,
            "per_block_mean": args.per_block_mean,
            "l2": "warm" if args.warm_l2 else "cold",
            "warmup_ms": args.warmup_ms,
            "measure_ms": args.measure_ms,
            "scope": "dense MHA self-attention; not SGLang paged serving",
            "results": [asdict(result) for result in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
