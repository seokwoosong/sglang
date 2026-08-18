#!/usr/bin/env python3
"""Diagnose unified Mamba H2D host-gather amplification.

This is an experimental benchmark, not a production-code patch.  It compares
the U1 implementation with the minimal equivalent ordering change below:

    src.index_select(rows)[:, layer]  ->  src[:, layer].index_select(rows)

The latter copies only the requested layer instead of gathering every layer on
every per-layer load call.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from bench_static_vs_unified_transfer import (
    SHAPES,
    Measurement,
    _make_indices,
    _make_mamba_device,
    _make_mamba_host,
    _measure,
    _validate_mamba,
)

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost


def _reordered_copy_tensor_pf_lf(
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    num_layers,
    io_backend,
):
    if src_indices.numel() == 0:
        return
    item_size = MambaPoolHost._item_size_per_index(dst)
    dst_stride = dst.stride(0) * dst.dtype.itemsize
    if io_backend == "kernel" and dst_stride != item_size:
        host_indices = src_indices.to(device="cpu", dtype=torch.long)
        selected = src[:, layer_id, 0].index_select(0, host_indices)
        dst.index_copy_(
            0,
            dst_indices.to(dtype=torch.long),
            selected.to(dst.device),
        )
        return
    return ORIGINAL_COPY(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        num_layers,
        io_backend,
    )


ORIGINAL_COPY = MambaPoolHost._copy_tensor_pf_lf
ORIGINAL_COPY_DESCRIPTOR = MambaPoolHost.__dict__["_copy_tensor_pf_lf"]


def _run_variant(pattern: str, variant: str, warmup: int, repetitions: int):
    shape = SHAPES["4b_9b"]
    capacity, host_indices, source, restore = _make_indices(4, pattern)
    host_indices = host_indices.cuda(non_blocking=True)
    device_pool = _make_mamba_device(shape, capacity, "unified")
    host = _make_mamba_host(shape, capacity, device_pool)

    if variant == "current":
        MambaPoolHost._copy_tensor_pf_lf = ORIGINAL_COPY_DESCRIPTOR
    elif variant == "select_layer_first":
        MambaPoolHost._copy_tensor_pf_lf = staticmethod(_reordered_copy_tensor_pf_lf)
    else:
        raise ValueError(variant)

    def transfer():
        for layer in range(shape.mamba_layers):
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                restore,
                layer,
                io_backend="kernel",
            )

    _validate_mamba(
        host,
        device_pool,
        host_indices,
        source,
        restore,
        "h2d",
    )
    return _measure(
        transfer,
        profile="4b_9b",
        component="mamba",
        direction="h2d",
        layout=variant,
        pattern=pattern,
        count=4,
        logical_bytes=shape.mamba_entry_bytes * 4,
        warmup=warmup,
        repetitions=repetitions,
        nvtx=False,
        torch_profile_dir=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Measurement] = []
    try:
        for pattern in ("contiguous", "fragmented"):
            for variant in ("current", "select_layer_first"):
                print(f"RUN {pattern} {variant}", flush=True)
                rows.extend(
                    _run_variant(pattern, variant, args.warmup, args.repetitions)
                )
    finally:
        MambaPoolHost._copy_tensor_pf_lf = ORIGINAL_COPY_DESCRIPTOR

    with (args.output_dir / "raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    with (args.output_dir / "summary.csv").open("w", newline="") as file:
        fieldnames = ["pattern", "variant", "complete_ms_median", "speedup"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pattern in ("contiguous", "fragmented"):
            medians = {}
            for variant in ("current", "select_layer_first"):
                medians[variant] = statistics.median(
                    row.complete_ms
                    for row in rows
                    if row.pattern == pattern and row.layout == variant
                )
            for variant, median in medians.items():
                writer.writerow(
                    {
                        "pattern": pattern,
                        "variant": variant,
                        "complete_ms_median": median,
                        "speedup": medians["current"] / median,
                    }
                )

    print(args.output_dir)


if __name__ == "__main__":
    main()
