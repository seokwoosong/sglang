#!/usr/bin/env python3
"""Benchmark production Qwen3.5 Mamba HiCache transfer layouts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.layout.page_major import build_page_major_mamba_views
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
from sglang.srt.mem_cache.pool_host.unified_chunk import UnifiedChunkMambaPoolHost
from sglang.srt.mem_cache.typed_chunk_host import SharedTypedChunkHostArena

NUM_LAYERS = 18
TEMPORAL_SHAPE = (16, 128, 128)
TEMPORAL_DTYPE = torch.float32
CONV_SHAPE = (18_432,)
CONV_DTYPE = torch.bfloat16
TEMPORAL_ROW_BYTES = 1_048_576
CONV_ROW_BYTES = 36_864
SLOT_BYTES = NUM_LAYERS * (TEMPORAL_ROW_BYTES + CONV_ROW_BYTES)
KV_PAGE_BYTES_P8 = 98_304


@dataclass(frozen=True)
class Measurement:
    path: str
    layout: str
    pattern: str
    batch_slots: int
    direction: str
    repetition: int
    payload_bytes: int
    elapsed_ms: float
    enqueue_ms: float
    gib_per_s: float
    source_stride_bytes: int
    host_stride_bytes: int
    kernel_calls: int


def make_device_pool(layout: str, num_slots: int):
    if layout == "static":
        temporal = torch.empty(
            (NUM_LAYERS, num_slots, *TEMPORAL_SHAPE),
            dtype=TEMPORAL_DTYPE,
            device="cuda",
        )
        conv = torch.empty(
            (NUM_LAYERS, num_slots, *CONV_SHAPE),
            dtype=CONV_DTYPE,
            device="cuda",
        )
        backing = None
    elif layout == "unified":
        backing = torch.empty(num_slots * SLOT_BYTES, dtype=torch.uint8, device="cuda")
        conv_views, temporal = build_page_major_mamba_views(
            backing,
            layer_num=NUM_LAYERS,
            conv_state_shapes=(CONV_SHAPE,),
            conv_dtype=CONV_DTYPE,
            temporal_state_shape=TEMPORAL_SHAPE,
            temporal_dtype=TEMPORAL_DTYPE,
            max_slots=num_slots,
        )
        conv = conv_views[0]
    else:
        raise ValueError(f"unknown layout: {layout}")
    unified = (
        SimpleNamespace(_raw=backing, total_bytes=backing.numel())
        if backing is not None
        else None
    )
    return SimpleNamespace(
        _unified_buffer=unified,
        _backing=backing,
        mamba_cache=SimpleNamespace(temporal=temporal, conv=[conv]),
        num_mamba_layers=NUM_LAYERS,
        size=num_slots,
        device="cuda",
    )


def page_sets(batch_slots: int, pattern: str):
    if pattern == "contiguous":
        source = torch.arange(1, batch_slots + 1, dtype=torch.int64)
        host = torch.arange(batch_slots + 1, 2 * batch_slots + 1, dtype=torch.int64)
        destination = torch.arange(
            2 * batch_slots + 1, 3 * batch_slots + 1, dtype=torch.int64
        )
    elif pattern == "fragmented":
        source = torch.arange(1, 2 * batch_slots + 1, 2, dtype=torch.int64)
        host = torch.arange(2, 2 * batch_slots + 1, 2, dtype=torch.int64)
        destination = torch.arange(
            2 * batch_slots + 1, 4 * batch_slots + 1, 2, dtype=torch.int64
        )
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    capacity = int(max(source.max(), host.max(), destination.max())) + 2
    return capacity, source, host, destination


def initialize_source(device_pool, source_indices: torch.Tensor) -> None:
    for layer in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer, source_indices].fill_(layer + 1)
        device_pool.mamba_cache.conv[0][layer, source_indices].fill_(layer + 101)


def validate_d2h(
    host: MambaPoolHost,
    device_pool,
    host_indices: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:
    for layer in (0, NUM_LAYERS - 1):
        torch.testing.assert_close(
            host.temporal_buffer[host_indices, layer, 0],
            device_pool.mamba_cache.temporal[layer, source_indices].cpu(),
        )
        torch.testing.assert_close(
            host.conv_buffer[0][host_indices, layer, 0],
            device_pool.mamba_cache.conv[0][layer, source_indices].cpu(),
        )


def validate_h2d(
    host: MambaPoolHost,
    device_pool,
    host_indices: torch.Tensor,
    destination_indices: torch.Tensor,
) -> None:
    for layer in (0, NUM_LAYERS - 1):
        torch.testing.assert_close(
            device_pool.mamba_cache.temporal[layer, destination_indices],
            host.temporal_buffer[host_indices, layer, 0].cuda(),
        )
        torch.testing.assert_close(
            device_pool.mamba_cache.conv[0][layer, destination_indices],
            host.conv_buffer[0][host_indices, layer, 0].cuda(),
        )


def measure(fn, warmup: int, repetitions: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    output = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_start = time.perf_counter()
        fn()
        enqueue_ms = (time.perf_counter() - cpu_start) * 1e3
        end.record()
        end.synchronize()
        output.append((float(start.elapsed_time(end)), enqueue_ms))
    return output


def release_host(host: MambaPoolHost) -> None:
    for buffer in (host.temporal_buffer, *host.conv_buffer):
        _cuda_host_unregister(buffer)


def run_condition(
    *,
    path: str,
    batch_slots: int,
    pattern: str,
    warmup: int,
    repetitions: int,
) -> list[Measurement]:
    layout = "static" if path == "baseline-static" else "unified"
    capacity, source_cpu, host_indices, destination_cpu = page_sets(
        batch_slots, pattern
    )
    device_pool = make_device_pool(layout, capacity)
    arena = None
    if layout == "unified":
        chunk_bytes = (
            (SLOT_BYTES + KV_PAGE_BYTES_P8 - 1) // KV_PAGE_BYTES_P8
        ) * KV_PAGE_BYTES_P8
        arena = SharedTypedChunkHostArena(
            total_bytes=capacity * chunk_bytes,
            kv_page_bytes=KV_PAGE_BYTES_P8,
            mamba_slot_bytes=SLOT_BYTES,
            host_device="cpu",
            accelerator_device="cuda",
            pin_memory=True,
            allocator_type="default",
        )
        host = UnifiedChunkMambaPoolHost(device_pool, arena)
    else:
        host = MambaPoolHost(
            device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            pin_memory=True,
            device="cpu",
            allocator_type="default",
            layout="page_first",
        )
    try:
        source = source_cpu.cuda()
        destination = destination_cpu.cuda()
        initialize_source(device_pool, source)

        if path == "ours-component-direct":
            d2h = lambda: MambaPoolHost.backup_from_device_all_layer(
                host, device_pool, host_indices, source, "kernel"
            )
        else:
            d2h = lambda: host.backup_from_device_all_layer(
                device_pool, host_indices, source, "kernel"
            )
        h2d = lambda: [
            host.load_to_device_per_layer(
                device_pool, host_indices, destination, layer, "kernel"
            )
            for layer in range(NUM_LAYERS)
        ]

        d2h()
        torch.cuda.synchronize()
        validate_d2h(host, device_pool, host_indices, source)
        h2d()
        torch.cuda.synchronize()
        validate_h2d(host, device_pool, host_indices, destination)

        payload = batch_slots * SLOT_BYTES
        source_stride = (
            device_pool.mamba_cache.temporal[0].stride(0)
            * device_pool.mamba_cache.temporal.element_size()
        )
        host_stride = (
            host.temporal_buffer.stride(0) * host.temporal_buffer.element_size()
        )
        rows = []
        for direction, fn in (("d2h", d2h), ("h2d", h2d)):
            for repetition, (elapsed_ms, enqueue_ms) in enumerate(
                measure(fn, warmup, repetitions)
            ):
                rows.append(
                    Measurement(
                        path=path,
                        layout=layout,
                        pattern=pattern,
                        batch_slots=batch_slots,
                        direction=direction,
                        repetition=repetition,
                        payload_bytes=payload,
                        elapsed_ms=elapsed_ms,
                        enqueue_ms=enqueue_ms,
                        gib_per_s=payload / (elapsed_ms * 1e-3) / 2**30,
                        source_stride_bytes=source_stride,
                        host_stride_bytes=host_stride,
                        kernel_calls=(
                            1
                            if direction == "d2h" and path == "ours-raw-slot"
                            else 2 if direction == "d2h" else 2 * NUM_LAYERS
                        ),
                    )
                )
        return rows
    finally:
        torch.cuda.synchronize()
        if arena is None:
            release_host(host)
        else:
            arena.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=[
            "baseline-static",
            "ours-component-direct",
            "ours-raw-slot",
        ],
        default=["baseline-static", "ours-component-direct", "ours-raw-slot"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["contiguous", "fragmented"],
        default=["contiguous"],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hicache_component_transfer/mamba"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[Measurement] = []
    started = time.time()
    for path in args.paths:
        for pattern in args.patterns:
            for batch_slots in args.batches:
                condition = run_condition(
                    path=path,
                    batch_slots=batch_slots,
                    pattern=pattern,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                rows.extend(condition)
                for direction in ("d2h", "h2d"):
                    selected = [
                        row.gib_per_s for row in condition if row.direction == direction
                    ]
                    print(
                        f"{path:>22} {pattern:>10} batch={batch_slots:>2} "
                        f"{direction.upper()}={statistics.median(selected):.2f} GiB/s",
                        flush=True,
                    )

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_layers": NUM_LAYERS,
        "temporal_row_bytes": TEMPORAL_ROW_BYTES,
        "conv_row_bytes": CONV_ROW_BYTES,
        "slot_bytes": SLOT_BYTES,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "results.json").write_text(
        json.dumps(
            {"environment": metadata, "measurements": [asdict(row) for row in rows]},
            indent=2,
        )
        + "\n"
    )
    with (args.output / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()
