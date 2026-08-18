#!/usr/bin/env python3
"""Microbenchmark static-pool versus unified-pool HiCache transfers.

This benchmark deliberately imports the production host-pool adapters from the
selected source tree.  It keeps the logical payload, host layout, dtype, and
indices equal while changing only the device layout:

* static: one tightly packed allocation per layer;
* unified: page-major views over one byte arena, with a full-envelope row stride.

The shapes match Qwen3.5-0.8B and Qwen3.5-4B/9B at TP=1.  Results include the
Python call-return time, end-to-end completion time, and CUDA stream span.  The
``--nvtx`` option labels every measured transfer for Nsight Systems.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import torch

from sglang.kernels.ops.kvcache import hicache as hicache_module
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    mamba_entry_bytes,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

DIRECT_STRIDED_IO = hasattr(hicache_module, "_to_2d_view")


@dataclass(frozen=True)
class CacheShape:
    name: str
    kv_layers: int
    kv_heads: int
    kv_head_dim: int
    mamba_layers: int
    mamba_heads: int
    mamba_head_dim: int = 128
    mamba_state_dim: int = 128
    conv_dim: int = 8192
    conv_width: int = 3

    @property
    def temporal_shape(self) -> tuple[int, int, int]:
        return (self.mamba_heads, self.mamba_head_dim, self.mamba_state_dim)

    @property
    def conv_shape(self) -> tuple[int, int]:
        return (self.conv_dim, self.conv_width)

    @property
    def kv_entry_bytes(self) -> int:
        return mha_entry_bytes(
            layer_num=self.kv_layers,
            head_num=self.kv_heads,
            head_dim=self.kv_head_dim,
            v_head_dim=self.kv_head_dim,
            itemsize=torch.bfloat16.itemsize,
        )

    @property
    def mamba_entry_bytes(self) -> int:
        return mamba_entry_bytes(
            layer_num=self.mamba_layers,
            conv_state_shapes=[self.conv_shape],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=self.temporal_shape,
            temporal_dtype=torch.float32,
        )


SHAPES = {
    "0.8b": CacheShape(
        name="Qwen3.5-0.8B",
        kv_layers=6,
        kv_heads=2,
        kv_head_dim=256,
        mamba_layers=18,
        mamba_heads=16,
        conv_dim=6144,
    ),
    "4b_9b": CacheShape(
        name="Qwen3.5-4B/9B",
        kv_layers=8,
        kv_heads=4,
        kv_head_dim=256,
        mamba_layers=24,
        mamba_heads=32,
        conv_dim=8192,
    ),
}


@dataclass
class Measurement:
    profile: str
    component: str
    direction: str
    layout: str
    pattern: str
    count: int
    repetition: int
    logical_bytes: int
    call_return_ms: float
    complete_ms: float
    stream_span_ms: float

    @property
    def bandwidth_gbps(self) -> float:
        if self.complete_ms <= 0:
            return math.nan
        return self.logical_bytes / self.complete_ms / 1e6


def _device_ptrs(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.uint64,
        device="cuda",
    )


def _make_indices(count: int, pattern: str):
    capacity = 2 * count + 1
    if pattern == "contiguous":
        host = torch.arange(count, dtype=torch.int64)
        source = torch.arange(count, dtype=torch.int64, device="cuda")
        restore = torch.arange(count, 2 * count, dtype=torch.int64, device="cuda")
    elif pattern == "fragmented":
        host = torch.arange(0, 2 * count, 2, dtype=torch.int64)
        source = torch.arange(1, 2 * count + 1, 2, dtype=torch.int64, device="cuda")
        restore = torch.arange(0, 2 * count, 2, dtype=torch.int64, device="cuda")
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return capacity, host, source, restore


def _make_kv_device(shape: CacheShape, capacity: int, layout: str):
    dtype = torch.bfloat16
    if layout == "static":
        k_buffer = [
            torch.empty(
                (capacity, shape.kv_heads, shape.kv_head_dim),
                dtype=dtype,
                device="cuda",
            )
            for _ in range(shape.kv_layers)
        ]
        v_buffer = [torch.empty_like(layer) for layer in k_buffer]
        backing = None
    elif layout == "unified":
        total_bytes = capacity * shape.kv_entry_bytes
        backing = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        k_buffer, v_buffer = build_page_major_mha_views(
            backing,
            layer_num=shape.kv_layers,
            head_num=shape.kv_heads,
            head_dim=shape.kv_head_dim,
            v_head_dim=shape.kv_head_dim,
            store_dtype=dtype,
            page_size=1,
            num_pages=capacity,
        )
    else:
        raise ValueError(layout)
    pool = SimpleNamespace(
        device=torch.device("cuda"),
        size=capacity - 1,
        page_size=1,
        head_num=shape.kv_heads,
        head_dim=shape.kv_head_dim,
        v_head_dim=shape.kv_head_dim,
        layer_num=shape.kv_layers,
        dtype=dtype,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=_device_ptrs(k_buffer),
        v_data_ptrs=_device_ptrs(v_buffer),
        _benchmark_backing=backing,
    )
    if layout == "unified":
        pool._unified_buffer = backing
    return pool


def _make_kv_host(shape: CacheShape, capacity: int, device_pool):
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.device_pool = device_pool
    host.layout = "page_first"
    host.page_size = 1
    host.page_num = capacity
    host.size = capacity
    host.device = "cpu"
    host.pin_memory = True
    host.layer_num = shape.kv_layers
    host.head_num = shape.kv_heads
    host.head_dim = shape.kv_head_dim
    host.dtype = torch.bfloat16
    host.element_dim = shape.kv_heads * shape.kv_head_dim
    host.token_stride_size = host.element_dim * host.dtype.itemsize
    host.layout_dim = host.token_stride_size * host.layer_num
    host.kv_buffer = (
        torch.empty(
            (capacity, shape.kv_layers, shape.kv_heads, shape.kv_head_dim),
            dtype=host.dtype,
            pin_memory=True,
        ),
        torch.empty(
            (capacity, shape.kv_layers, shape.kv_heads, shape.kv_head_dim),
            dtype=host.dtype,
            pin_memory=True,
        ),
    )
    k_transposed = host.k_buffer.transpose(0, 1)
    v_transposed = host.v_buffer.transpose(0, 1)
    host.k_data_refs = [k_transposed[i] for i in range(shape.kv_layers)]
    host.v_data_refs = [v_transposed[i] for i in range(shape.kv_layers)]
    host.k_data_ptrs = _device_ptrs(host.k_data_refs)
    host.v_data_ptrs = _device_ptrs(host.v_data_refs)
    host.use_unified_direct_io = DIRECT_STRIDED_IO and hasattr(
        device_pool, "_unified_buffer"
    )
    host.can_use_jit = True
    host.can_use_write_back_jit = not host.use_unified_direct_io
    host.staging_page_capacity = 0 if host.use_unified_direct_io else min(capacity, 64)
    host.staging_token_capacity = host.staging_page_capacity
    if host.use_unified_direct_io:
        host.staging_k_buffer = None
        host.staging_v_buffer = None
    else:
        host.staging_k_buffer = torch.empty(
            (
                host.staging_token_capacity,
                shape.kv_layers,
                shape.kv_heads,
                shape.kv_head_dim,
            ),
            dtype=host.dtype,
            device="cuda",
        )
        host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
    return host


def _make_mamba_device(shape: CacheShape, capacity: int, layout: str):
    if layout == "static":
        temporal = torch.empty(
            (shape.mamba_layers, capacity, *shape.temporal_shape),
            dtype=torch.float32,
            device="cuda",
        )
        conv = [
            torch.empty(
                (shape.mamba_layers, capacity, *shape.conv_shape),
                dtype=torch.bfloat16,
                device="cuda",
            )
        ]
        backing = None
    elif layout == "unified":
        backing = torch.empty(
            capacity * shape.mamba_entry_bytes, dtype=torch.uint8, device="cuda"
        )
        conv, temporal = build_page_major_mamba_views(
            backing,
            layer_num=shape.mamba_layers,
            conv_state_shapes=[shape.conv_shape],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=shape.temporal_shape,
            temporal_dtype=torch.float32,
            max_slots=capacity,
        )
    else:
        raise ValueError(layout)
    pool = SimpleNamespace(
        device="cuda",
        size=capacity,
        num_mamba_layers=shape.mamba_layers,
        mamba_cache=SimpleNamespace(temporal=temporal, conv=conv),
        _benchmark_backing=backing,
    )
    if layout == "unified":
        pool._unified_buffer = backing
    return pool


def _make_mamba_host(shape: CacheShape, capacity: int, device_pool):
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.device_pool = device_pool
    host.layout = "page_first"
    host.page_size = 1
    host.page_num = capacity
    host.size = capacity
    host.pin_memory = True
    host.device = "cpu"
    host.num_mamba_layers = shape.mamba_layers
    host.conv_state_shapes = [shape.conv_shape]
    host.temporal_state_shape = shape.temporal_shape
    host.temporal_state_elem_size = math.prod(shape.temporal_shape)
    host.conv_state_elem_sizes = [math.prod(shape.conv_shape)]
    host.conv_dtype = torch.bfloat16
    host.temporal_dtype = torch.float32
    host.dtype = host.conv_dtype
    host.size_per_token = host.get_size_per_token()
    host.temporal_buffer = torch.empty(
        (capacity, shape.mamba_layers, 1, *shape.temporal_shape),
        dtype=host.temporal_dtype,
        pin_memory=True,
    )
    host.conv_buffer = [
        torch.empty(
            (capacity, shape.mamba_layers, 1, *shape.conv_shape),
            dtype=host.conv_dtype,
            pin_memory=True,
        )
    ]
    host.temporal_staging_buffer = None
    host.conv_staging_buffers = [None]
    host.can_use_write_back_jit = not DIRECT_STRIDED_IO
    host._temporal_can_use_jit = False
    host._conv_can_use_jit = [False]
    host.temporal_device_ptrs = _device_ptrs(
        [device_pool.mamba_cache.temporal[i] for i in range(shape.mamba_layers)]
    )
    host.conv_device_ptrs = [
        _device_ptrs(
            [device_pool.mamba_cache.conv[0][i] for i in range(shape.mamba_layers)]
        )
    ]
    return host


def _measure(
    fn: Callable[[], None],
    *,
    profile: str,
    component: str,
    direction: str,
    layout: str,
    pattern: str,
    count: int,
    logical_bytes: int,
    warmup: int,
    repetitions: int,
    nvtx: bool,
    torch_profile_dir: Path | None,
) -> list[Measurement]:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    label = f"{profile}/{component}/{direction}/{layout}/{pattern}/n{count}"
    if torch_profile_dir is not None:
        torch_profile_dir.mkdir(parents=True, exist_ok=True)
        stem = label.replace("/", "_")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(torch_profile_dir / f"{stem}.json.gz"))
        (torch_profile_dir / f"{stem}.txt").write_text(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=200
            )
            + "\n"
        )

    rows = []
    for repetition in range(repetitions):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        call_start = time.perf_counter_ns()
        if nvtx:
            torch.cuda.nvtx.range_push(label)
        try:
            fn()
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
        call_end = time.perf_counter_ns()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        rows.append(
            Measurement(
                profile=profile,
                component=component,
                direction=direction,
                layout=layout,
                pattern=pattern,
                count=count,
                repetition=repetition,
                logical_bytes=logical_bytes,
                call_return_ms=(call_end - call_start) / 1e6,
                complete_ms=(wall_end - wall_start) / 1e6,
                stream_span_ms=start.elapsed_time(end),
            )
        )
    return rows


def _validate_kv(host, device_pool, host_indices, source, restore, direction):
    sample_host = host_indices[:1].cpu()
    sample_source = source[:1]
    sample_restore = restore[:1]
    if direction == "d2h":
        device_pool.k_buffer[0].index_fill_(0, sample_source, 3)
        device_pool.v_buffer[0].index_fill_(0, sample_source, 4)
        host.backup_from_device_all_layer(
            device_pool, host_indices, source, io_backend="kernel"
        )
        torch.cuda.synchronize()
        assert torch.all(host.k_data_refs[0][sample_host] == 3)
        assert torch.all(host.v_data_refs[0][sample_host] == 4)
    else:
        host.k_data_refs[0].index_fill_(0, sample_host, 5)
        host.v_data_refs[0].index_fill_(0, sample_host, 6)
        for layer in range(host.layer_num):
            host.load_to_device_per_layer(
                device_pool, host_indices, restore, layer, io_backend="kernel"
            )
        torch.cuda.synchronize()
        assert torch.all(device_pool.k_buffer[0][sample_restore] == 5)
        assert torch.all(device_pool.v_buffer[0][sample_restore] == 6)


def _validate_mamba(host, device_pool, host_indices, source, restore, direction):
    sample_host = host_indices[:1].cpu()
    sample_source = source[:1]
    sample_restore = restore[:1]
    if direction == "d2h":
        device_pool.mamba_cache.temporal[0].index_fill_(0, sample_source, 3)
        device_pool.mamba_cache.conv[0][0].index_fill_(0, sample_source, 4)
        host.backup_from_device_all_layer(
            device_pool, host_indices, source, io_backend="kernel"
        )
        torch.cuda.synchronize()
        assert torch.all(host.temporal_buffer[sample_host, 0] == 3)
        assert torch.all(host.conv_buffer[0][sample_host, 0] == 4)
    else:
        host.temporal_buffer[:, 0].index_fill_(0, sample_host, 5)
        host.conv_buffer[0][:, 0].index_fill_(0, sample_host, 6)
        for layer in range(host.num_mamba_layers):
            host.load_to_device_per_layer(
                device_pool, host_indices, restore, layer, io_backend="kernel"
            )
        torch.cuda.synchronize()
        assert torch.all(device_pool.mamba_cache.temporal[0][sample_restore] == 5)
        assert torch.all(device_pool.mamba_cache.conv[0][0][sample_restore] == 6)


def _run_case(
    *,
    shape_key: str,
    component: str,
    direction: str,
    layout: str,
    pattern: str,
    count: int,
    warmup: int,
    repetitions: int,
    nvtx: bool,
    torch_profile_dir: Path | None,
) -> list[Measurement]:
    shape = SHAPES[shape_key]
    capacity, host_indices, source, restore = _make_indices(count, pattern)
    # The production controller supplies CUDA host indices for kernel H2D,
    # while staged page-first D2H normalizes destination indices to CPU.
    if direction == "h2d":
        host_indices = host_indices.cuda(non_blocking=True)
    if component == "kv":
        device_pool = _make_kv_device(shape, capacity, layout)
        host = _make_kv_host(shape, capacity, device_pool)
        if direction == "d2h" and host.use_unified_direct_io:
            host_indices = host_indices.cuda(non_blocking=True)
        logical_bytes = count * shape.kv_entry_bytes
        if direction == "d2h":
            fn = lambda: host.backup_from_device_all_layer(
                device_pool, host_indices, source, io_backend="kernel"
            )
        else:

            def fn():
                for layer in range(shape.kv_layers):
                    host.load_to_device_per_layer(
                        device_pool,
                        host_indices,
                        restore,
                        layer,
                        io_backend="kernel",
                    )

        _validate_kv(host, device_pool, host_indices, source, restore, direction)
    elif component == "mamba":
        device_pool = _make_mamba_device(shape, capacity, layout)
        host = _make_mamba_host(shape, capacity, device_pool)
        if direction == "d2h" and DIRECT_STRIDED_IO:
            host_indices = host_indices.cuda(non_blocking=True)
        logical_bytes = count * shape.mamba_entry_bytes
        if direction == "d2h":
            fn = lambda: host.backup_from_device_all_layer(
                device_pool, host_indices, source, io_backend="kernel"
            )
        else:

            def fn():
                for layer in range(shape.mamba_layers):
                    host.load_to_device_per_layer(
                        device_pool,
                        host_indices,
                        restore,
                        layer,
                        io_backend="kernel",
                    )

        _validate_mamba(host, device_pool, host_indices, source, restore, direction)
    else:
        raise ValueError(component)

    return _measure(
        fn,
        profile=shape_key,
        component=component,
        direction=direction,
        layout=layout,
        pattern=pattern,
        count=count,
        logical_bytes=logical_bytes,
        warmup=warmup,
        repetitions=repetitions,
        nvtx=nvtx,
        torch_profile_dir=torch_profile_dir,
    )


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def _summary(rows: list[Measurement]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (
            row.profile,
            row.component,
            row.direction,
            row.layout,
            row.pattern,
            row.count,
            row.logical_bytes,
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        (
            profile,
            component,
            direction,
            layout,
            pattern,
            count,
            logical_bytes,
        ) = key
        complete = _median(row.complete_ms for row in group)
        out.append(
            {
                "profile": profile,
                "component": component,
                "direction": direction,
                "layout": layout,
                "pattern": pattern,
                "count": count,
                "logical_bytes": logical_bytes,
                "call_return_ms_median": _median(row.call_return_ms for row in group),
                "complete_ms_median": complete,
                "stream_span_ms_median": _median(row.stream_span_ms for row in group),
                "bandwidth_gbps_median": logical_bytes / complete / 1e6,
                "repetitions": len(group),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument(
        "--components", nargs="+", choices=("kv", "mamba"), default=["kv", "mamba"]
    )
    parser.add_argument(
        "--directions", nargs="+", choices=("d2h", "h2d"), default=["d2h", "h2d"]
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("static", "unified"),
        default=["static", "unified"],
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("contiguous", "fragmented"),
        default=["contiguous", "fragmented"],
    )
    parser.add_argument("--kv-counts", nargs="+", type=int, default=[64, 512, 4096])
    parser.add_argument("--mamba-counts", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--torch-profile-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(inspect.getfile(MHATokenToKVPoolHost)).resolve()
    metadata = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_file": str(source_file),
        "shapes": {key: asdict(value) for key, value in SHAPES.items()},
        "shape_bytes": {
            key: {
                "kv_entry_bytes": value.kv_entry_bytes,
                "mamba_entry_bytes": value.mamba_entry_bytes,
            }
            for key, value in SHAPES.items()
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    rows: list[Measurement] = []
    for shape_key in args.profiles:
        for component in args.components:
            counts = args.kv_counts if component == "kv" else args.mamba_counts
            for pattern in args.patterns:
                for count in counts:
                    for direction in args.directions:
                        # Alternate the first layout by count to reduce ordering bias.
                        layouts = list(args.layouts)
                        if count % 2 == 0:
                            layouts.reverse()
                        for layout in layouts:
                            print(
                                f"RUN {shape_key} {component} {direction} {layout} "
                                f"{pattern} n={count}",
                                flush=True,
                            )
                            rows.extend(
                                _run_case(
                                    shape_key=shape_key,
                                    component=component,
                                    direction=direction,
                                    layout=layout,
                                    pattern=pattern,
                                    count=count,
                                    warmup=args.warmup,
                                    repetitions=args.repetitions,
                                    nvtx=args.nvtx,
                                    torch_profile_dir=args.torch_profile_dir,
                                )
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            raw_dicts = [
                                {**asdict(row), "bandwidth_gbps": row.bandwidth_gbps}
                                for row in rows
                            ]
                            _write_csv(args.output_dir / "raw.csv", raw_dicts)
                            _write_csv(args.output_dir / "summary.csv", _summary(rows))
    print(args.output_dir)


if __name__ == "__main__":
    main()
