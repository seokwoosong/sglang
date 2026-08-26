"""Decompose BF16 and NVFP4 linear GEMM costs on an SM120 GPU.

The benchmark uses the projection shapes of Llama 3.1 8B and separately times
PyTorch and FlashInfer BF16 baselines, GEMM-layout activation quantization,
prequantized FP4 GEMM, and the combined FP4 runtime path.  It also records a
diagnostic NVFP4 quantize/dequantize roundtrip using FlashInfer's linear KV
layout.  That standalone dequantization is not part of the FP4 GEMM hot path:
scale application and BF16 output conversion are fused into the GEMM epilogue.

Weight quantization and static global-scale calibration happen outside the
timed runtime path, matching a serialized ModelOpt NVFP4 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# FlashInfer's first SM120 CUTLASS build launches several translation units
# whose peak host memory is roughly 4-5 GiB each.  Keep the benchmark usable on
# 32-GiB workstations while still allowing callers to override the limit.
os.environ.setdefault("MAX_JOBS", "2")

import flashinfer
import torch
import torch.nn.functional as F
from flashinfer.autotuner import autotune
from flashinfer.testing import bench_gpu_time

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
MAX_ERROR_QUANTILE_SAMPLES = 1_000_000
PROJECTIONS = {
    "qkv": (6144, 4096),
    "attention_output": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
}
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TimingStats:
    median_ms: float
    p95_ms: float
    samples: int


@dataclass
class Result:
    projection: str
    distribution: str
    seed: int
    m: int
    n: int
    k: int
    bf16: TimingStats
    flashinfer_bf16: TimingStats
    activation_quant: TimingStats
    fp4_gemm: TimingStats
    fp4_end_to_end: TimingStats
    linear_layout_quant: TimingStats
    linear_layout_dequant: TimingStats
    flashinfer_bf16_speedup_vs_torch: float
    fp4_gemm_speedup: float
    fp4_end_to_end_speedup: float
    fp4_gemm_speedup_vs_flashinfer_bf16: float
    fp4_end_to_end_speedup_vs_flashinfer_bf16: float
    pipeline_residual_ms: float
    bf16_tflops: float
    flashinfer_bf16_tflops: float
    fp4_gemm_tflops: float
    flashinfer_bf16_max_absolute_difference: float
    relative_rmse: float
    cosine_similarity: float
    sqnr_db: float
    p99_absolute_error: float
    error_quantile_samples: int
    error_total_values: int
    max_absolute_error: float
    nan_count: int
    inf_count: int
    linear_roundtrip_relative_rmse: float
    linear_roundtrip_cosine_similarity: float
    linear_roundtrip_nan_count: int
    linear_roundtrip_inf_count: int


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def measure_gpu_time(fn, args: argparse.Namespace) -> TimingStats:
    times = bench_gpu_time(
        fn=fn,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.measure_ms,
        cold_l2_cache=not args.warm_l2,
    )
    return TimingStats(
        median_ms=float(statistics.median(times)),
        p95_ms=percentile(times, 0.95),
        samples=len(times),
    )


def make_tensor(
    shape: tuple[int, int], distribution: str, generator: torch.Generator
) -> torch.Tensor:
    tensor = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if distribution == "outlier":
        # Deterministic heavy tails: amplify roughly 0.1% of values by 16x.
        stride = max(tensor.numel() // max(tensor.numel() // 1000, 1), 1)
        tensor.view(-1)[::stride].mul_(16)
    return tensor


def make_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    max_abs = tensor.abs().float().amax().clamp_min(1e-6)
    return (
        torch.tensor(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX,
            device=tensor.device,
            dtype=torch.float32,
        )
        / max_abs
    )


def normalize_scale_dtype(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return scale.view(torch.float8_e4m3fn)
    return scale


def benchmark_case(
    projection: str,
    distribution: str,
    seed: int,
    m: int,
    n: int,
    k: int,
    args: argparse.Namespace,
) -> Result:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    activation = make_tensor((m, k), distribution, generator)
    weight = make_tensor((n, k), distribution, generator)
    activation_global_scale = make_global_scale(activation)
    weight_global_scale = make_global_scale(weight)
    # fp4_quantize/mm_fp4 consume the inverse global scale, while the public
    # NVFP4 KV quantize/dequantize pair consumes the direct reconstruction
    # scale.  Keep the two API contracts explicit instead of reusing a tensor
    # with the opposite semantic meaning.
    linear_global_scale = activation_global_scale.reciprocal()
    alpha = (1.0 / (activation_global_scale * weight_global_scale)).to(torch.float32)

    activation_fp4, activation_sf = flashinfer.fp4_quantize(
        activation, activation_global_scale
    )
    weight_fp4, weight_sf = flashinfer.fp4_quantize(weight, weight_global_scale)
    activation_sf = normalize_scale_dtype(activation_sf)
    weight_sf = normalize_scale_dtype(weight_sf)
    fp4_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    bf16_out = torch.empty_like(fp4_out)
    flashinfer_bf16_out = torch.empty_like(fp4_out)

    def bf16_gemm():
        return torch.mm(activation, weight.T, out=bf16_out)

    def flashinfer_bf16_gemm():
        return flashinfer.gemm.mm_bf16(
            activation,
            weight.T,
            out=flashinfer_bf16_out,
            backend=args.bf16_backend,
        )

    def activation_quantize():
        return flashinfer.fp4_quantize(activation, activation_global_scale)

    def fp4_gemm():
        return flashinfer.mm_fp4(
            activation_fp4,
            weight_fp4.T,
            activation_sf,
            weight_sf.T,
            alpha,
            torch.bfloat16,
            fp4_out,
            backend=args.backend,
        )

    def fp4_end_to_end():
        dynamic_activation_fp4, dynamic_activation_sf = flashinfer.fp4_quantize(
            activation, activation_global_scale
        )
        dynamic_activation_sf = normalize_scale_dtype(dynamic_activation_sf)
        return flashinfer.mm_fp4(
            dynamic_activation_fp4,
            weight_fp4.T,
            dynamic_activation_sf,
            weight_sf.T,
            alpha,
            torch.bfloat16,
            fp4_out,
            backend=args.backend,
        )

    # FlashInfer's public NVFP4 KV helpers use a linear scale layout and expose
    # a real standalone dequantization kernel.  This is a useful conversion
    # diagnostic, but it is deliberately kept separate from the swizzled scale
    # layout consumed by FP4 GEMM.
    linear_fp4, linear_sf = flashinfer.nvfp4_kv_quantize(
        activation, linear_global_scale
    )

    def linear_layout_quantize():
        return flashinfer.nvfp4_kv_quantize(activation, linear_global_scale)

    def linear_layout_dequantize():
        return flashinfer.nvfp4_kv_dequantize(
            linear_fp4,
            linear_sf,
            linear_global_scale,
            output_dtype=torch.bfloat16,
        )

    # Trigger JIT/autotuning outside timed regions.
    activation_quantize()
    linear_layout_quantize()
    linear_reconstructed = linear_layout_dequantize()
    with autotune():
        fp4_gemm()
    bf16_gemm()
    flashinfer_bf16_gemm()
    torch.cuda.synchronize()

    reference = bf16_out.float().clone()
    flashinfer_bf16_difference = flashinfer_bf16_out.float() - reference
    flashinfer_bf16_max_absolute_difference = float(
        flashinfer_bf16_difference.abs().max()
    )
    candidate = fp4_out.float().clone()
    difference = candidate - reference
    reference_rms = torch.sqrt(reference.square().mean())
    error_rms = torch.sqrt(difference.square().mean())
    relative_rmse = float(
        error_rms / reference_rms.clamp_min(torch.finfo(torch.float32).tiny)
    )
    cosine_similarity = float(
        F.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0)
    )
    sqnr_db = (
        math.inf
        if float(error_rms) == 0.0
        else 20.0 * math.log10(float(reference_rms / error_rms))
    )
    absolute_error = difference.abs().flatten()
    error_total_values = absolute_error.numel()
    error_stride = max(
        math.ceil(error_total_values / MAX_ERROR_QUANTILE_SAMPLES),
        1,
    )
    error_quantile_input = absolute_error[::error_stride]
    p99_absolute_error = float(torch.quantile(error_quantile_input, 0.99))
    error_quantile_samples = error_quantile_input.numel()
    max_absolute_error = float(absolute_error.max())
    nan_count = int(torch.isnan(candidate).sum())
    inf_count = int(torch.isinf(candidate).sum())

    linear_difference = linear_reconstructed.float() - activation.float()
    linear_reference_rms = torch.sqrt(activation.float().square().mean())
    linear_error_rms = torch.sqrt(linear_difference.square().mean())
    linear_roundtrip_relative_rmse = float(
        linear_error_rms
        / linear_reference_rms.clamp_min(torch.finfo(torch.float32).tiny)
    )
    linear_roundtrip_cosine_similarity = float(
        F.cosine_similarity(
            linear_reconstructed.float().flatten(),
            activation.float().flatten(),
            dim=0,
        )
    )
    linear_roundtrip_nan_count = int(torch.isnan(linear_reconstructed).sum())
    linear_roundtrip_inf_count = int(torch.isinf(linear_reconstructed).sum())

    bf16 = measure_gpu_time(bf16_gemm, args)
    flashinfer_bf16 = measure_gpu_time(flashinfer_bf16_gemm, args)
    activation_quant = measure_gpu_time(activation_quantize, args)
    fp4_gemm_timing = measure_gpu_time(fp4_gemm, args)
    fp4_end_to_end = measure_gpu_time(fp4_end_to_end, args)
    linear_layout_quant = measure_gpu_time(linear_layout_quantize, args)
    linear_layout_dequant = measure_gpu_time(linear_layout_dequantize, args)
    pipeline_residual_ms = (
        fp4_end_to_end.median_ms
        - activation_quant.median_ms
        - fp4_gemm_timing.median_ms
    )
    flops = 2.0 * m * n * k
    return Result(
        projection=projection,
        distribution=distribution,
        seed=seed,
        m=m,
        n=n,
        k=k,
        bf16=bf16,
        flashinfer_bf16=flashinfer_bf16,
        activation_quant=activation_quant,
        fp4_gemm=fp4_gemm_timing,
        fp4_end_to_end=fp4_end_to_end,
        linear_layout_quant=linear_layout_quant,
        linear_layout_dequant=linear_layout_dequant,
        flashinfer_bf16_speedup_vs_torch=(bf16.median_ms / flashinfer_bf16.median_ms),
        fp4_gemm_speedup=bf16.median_ms / fp4_gemm_timing.median_ms,
        fp4_end_to_end_speedup=bf16.median_ms / fp4_end_to_end.median_ms,
        fp4_gemm_speedup_vs_flashinfer_bf16=(
            flashinfer_bf16.median_ms / fp4_gemm_timing.median_ms
        ),
        fp4_end_to_end_speedup_vs_flashinfer_bf16=(
            flashinfer_bf16.median_ms / fp4_end_to_end.median_ms
        ),
        pipeline_residual_ms=pipeline_residual_ms,
        bf16_tflops=flops / (bf16.median_ms * 1e-3) / 1e12,
        flashinfer_bf16_tflops=(flops / (flashinfer_bf16.median_ms * 1e-3) / 1e12),
        fp4_gemm_tflops=flops / (fp4_gemm_timing.median_ms * 1e-3) / 1e12,
        flashinfer_bf16_max_absolute_difference=(
            flashinfer_bf16_max_absolute_difference
        ),
        relative_rmse=relative_rmse,
        cosine_similarity=cosine_similarity,
        sqnr_db=sqnr_db,
        p99_absolute_error=p99_absolute_error,
        error_quantile_samples=error_quantile_samples,
        error_total_values=error_total_values,
        max_absolute_error=max_absolute_error,
        nan_count=nan_count,
        inf_count=inf_count,
        linear_roundtrip_relative_rmse=linear_roundtrip_relative_rmse,
        linear_roundtrip_cosine_similarity=linear_roundtrip_cosine_similarity,
        linear_roundtrip_nan_count=linear_roundtrip_nan_count,
        linear_roundtrip_inf_count=linear_roundtrip_inf_count,
    )


def print_results(results: list[Result]) -> None:
    print()
    print(
        "| projection | dist. | seed | M | N | K | Torch BF16 ms | "
        "FlashInfer BF16 ms | FI speedup | GEMM quant ms | FP4 GEMM ms | "
        "vs Torch | vs FI BF16 | FP4 e2e ms | vs Torch | vs FI BF16 | "
        "linear quant ms | linear dequant ms | rel. RMSE | cosine |"
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|---:|---:|"
    )
    for result in results:
        print(
            f"| {result.projection} | {result.distribution} | {result.seed} | "
            f"{result.m} | {result.n} | {result.k} | {result.bf16.median_ms:.4f} | "
            f"{result.flashinfer_bf16.median_ms:.4f} | "
            f"{result.flashinfer_bf16_speedup_vs_torch:.2f}x | "
            f"{result.activation_quant.median_ms:.4f} | "
            f"{result.fp4_gemm.median_ms:.4f} | {result.fp4_gemm_speedup:.2f}x | "
            f"{result.fp4_gemm_speedup_vs_flashinfer_bf16:.2f}x | "
            f"{result.fp4_end_to_end.median_ms:.4f} | "
            f"{result.fp4_end_to_end_speedup:.2f}x | "
            f"{result.fp4_end_to_end_speedup_vs_flashinfer_bf16:.2f}x | "
            f"{result.linear_layout_quant.median_ms:.4f} | "
            f"{result.linear_layout_dequant.median_ms:.4f} | "
            f"{result.relative_rmse:.4f} | {result.cosine_similarity:.6f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=[1, 16, 128, 512, 2048]
    )
    parser.add_argument("--projections", type=parse_str_list, default=list(PROJECTIONS))
    parser.add_argument(
        "--distributions", type=parse_str_list, default=["gaussian", "outlier"]
    )
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--backend", choices=("cutlass", "auto"), default="cutlass")
    parser.add_argument(
        "--bf16-backend",
        choices=("cublaslt", "cudnn", "cutile", "tinygemm", "auto"),
        default="cublaslt",
    )
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--measure-ms", type=int, default=30)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            f"this benchmark targets SM120; found {capability[0]}.{capability[1]}"
        )
    unknown_projections = sorted(set(args.projections) - set(PROJECTIONS))
    unknown_distributions = sorted(set(args.distributions) - {"gaussian", "outlier"})
    if unknown_projections or unknown_distributions:
        raise ValueError(
            f"unknown selections: projections={unknown_projections}, "
            f"distributions={unknown_distributions}"
        )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"FP4 backend={args.backend}, BF16 backend={args.bf16_backend}, "
        f"L2={'warm' if args.warm_l2 else 'cold'}, "
        "dtype=BF16 input/output"
    )
    results: list[Result] = []
    for projection in args.projections:
        n, k = PROJECTIONS[projection]
        for distribution in args.distributions:
            for m in args.batch_sizes:
                for seed in args.seeds:
                    print(
                        f"Benchmarking {projection} {distribution} "
                        f"M={m}, N={n}, K={k}, seed={seed} ...",
                        flush=True,
                    )
                    results.append(
                        benchmark_case(
                            projection,
                            distribution,
                            seed,
                            m,
                            n,
                            k,
                            args,
                        )
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
    print_results(results)
    if args.output_json:
        payload = {
            "schema_version": 3,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": [sys.executable, *sys.argv],
            "git_commit": run_capture(["git", "rev-parse", "HEAD"]),
            "git_branch": run_capture(["git", "branch", "--show-current"]),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__,
            "nvidia_smi": run_capture(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                    "--id=0",
                ]
            ),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "backend": args.backend,
            "bf16_backend": args.bf16_backend,
            "dtype": "bfloat16",
            "l2": "warm" if args.warm_l2 else "cold",
            "warmup_ms": args.warmup_ms,
            "measure_ms": args.measure_ms,
            "global_scale_calibration": "per synthetic tensor, outside timing",
            "weight_quantization": "outside timing (serialized-checkpoint analogue)",
            "gemm_activation_quantization": (
                "FlashInfer FP4 quantization with swizzled block-scale layout"
            ),
            "standalone_dequantization": (
                "diagnostic FlashInfer NVFP4 KV linear-layout kernel; not part of "
                "the FP4 GEMM hot path, whose output scaling is fused"
            ),
            "pipeline_residual": (
                "FP4 end-to-end median minus separately timed GEMM-layout "
                "quantization and prequantized GEMM medians; cache handoff and "
                "independent sampling mean this value is diagnostic, not additive"
            ),
            "p99_absolute_error": (
                "exact when the output has at most 1,000,000 values; otherwise "
                "computed from at most 1,000,000 deterministically strided samples"
            ),
            "max_jit_jobs": os.environ.get("MAX_JOBS"),
            "results": [asdict(result) for result in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
