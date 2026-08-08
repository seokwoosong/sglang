#!/usr/bin/env python3
"""Compare static HiCache transfer with unified typed-page L2 transfer.

The baseline uses the production non-unified MHA device pool and page-first
HiCache host adapter.  OURS uses a unified page-major L1 and the shared typed
L2 adapter.  Both move the same logical Qwen3.5-0.8B KV payload.
"""

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

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified_chunk import UnifiedChunkMHAPoolHost
from sglang.srt.mem_cache.typed_chunk_host import (
    SharedTypedChunkHostArena,
    build_shared_kv_envelope_view,
)

LAYERS = 6
HEADS = 2
HEAD_DIM = 256
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Measurement:
    page_size: int
    tokens: int
    pages: int
    pattern: str
    direction: str
    variant: str
    repetition: int
    elapsed_ms: float
    payload_bytes: int
    gib_per_s: float
    transfer_calls: int


def _ptrs(layers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [layer.data_ptr() for layer in layers], dtype=torch.uint64, device="cuda"
    )


def _expand_pages(page_ids: torch.Tensor, page_size: int) -> torch.Tensor:
    offsets = torch.arange(page_size, dtype=torch.int64, device=page_ids.device)
    return (page_ids[:, None] * page_size + offsets).reshape(-1)


def _page_sets(num_pages: int, pattern: str):
    if pattern == "contiguous":
        src = torch.arange(1, num_pages + 1, dtype=torch.int64)
        host = torch.arange(num_pages + 1, 2 * num_pages + 1, dtype=torch.int64)
        dst = torch.arange(2 * num_pages + 1, 3 * num_pages + 1, dtype=torch.int64)
    elif pattern == "fragmented":
        src = torch.arange(1, 2 * num_pages + 1, 2, dtype=torch.int64)
        host = torch.arange(2, 2 * num_pages + 1, 2, dtype=torch.int64)
        dst = torch.arange(2 * num_pages + 1, 4 * num_pages + 1, 2, dtype=torch.int64)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    capacity_pages = int(max(src.max(), host.max(), dst.max())) + 2
    capacity_pages = (capacity_pages + 3) // 4 * 4
    return capacity_pages, src, host, dst


def _make_static(capacity_pages: int, page_size: int):
    capacity_tokens = capacity_pages * page_size
    k_layers = [
        torch.empty(capacity_tokens, HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        for _ in range(LAYERS)
    ]
    v_layers = [torch.empty_like(layer) for layer in k_layers]
    pool = SimpleNamespace(
        device=torch.device("cuda"),
        size=capacity_tokens,
        page_size=page_size,
        store_dtype=DTYPE,
        dtype=DTYPE,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        layer_num=LAYERS,
        start_layer=0,
        end_layer=LAYERS,
        layer_shard_enabled=False,
        k_buffer=k_layers,
        v_buffer=v_layers,
        k_data_ptrs=_ptrs(k_layers),
        v_data_ptrs=_ptrs(v_layers),
    )
    host = MHATokenToKVPoolHost(
        pool,
        host_to_device_ratio=1.0,
        host_size=0,
        page_size=page_size,
        layout="page_first",
        pin_memory=True,
        device="cpu",
        allocator_type="default",
    )
    return pool, host


def _make_ours(capacity_pages: int, page_size: int):
    token_bytes = 2 * LAYERS * HEADS * HEAD_DIM * DTYPE.itemsize
    page_bytes = page_size * token_bytes
    raw = torch.empty(capacity_pages * page_bytes, dtype=torch.uint8, device="cuda")
    envelope = build_shared_kv_envelope_view(
        raw,
        num_pages=capacity_pages,
        page_size=page_size,
        layer_num=LAYERS,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )
    k_layers = [envelope[0, :, layer] for layer in range(LAYERS)]
    v_layers = [envelope[1, :, layer] for layer in range(LAYERS)]
    unified = SimpleNamespace(total_bytes=raw.numel(), _raw=raw)
    pool = SimpleNamespace(
        _unified_buffer=unified,
        device=torch.device("cuda"),
        size=capacity_pages * page_size,
        page_size=page_size,
        store_dtype=DTYPE,
        dtype=DTYPE,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        layer_num=LAYERS,
        start_layer=0,
        end_layer=LAYERS,
        k_buffer=k_layers,
        v_buffer=v_layers,
        k_data_ptrs=_ptrs(k_layers),
        v_data_ptrs=_ptrs(v_layers),
    )
    arena = SharedTypedChunkHostArena(
        total_bytes=capacity_pages * page_bytes,
        kv_page_bytes=page_bytes,
        mamba_slot_bytes=4 * page_bytes,
        host_device="cpu",
        accelerator_device="cuda",
        pin_memory=True,
        allocator_type="default",
    )
    return pool, UnifiedChunkMHAPoolHost(pool, arena), arena


def _initialize_sources(static_pool, ours_pool, src_pages, page_size):
    src_tokens = _expand_pages(src_pages, page_size).cuda()
    src_pages_gpu = src_pages.cuda()
    page_values = src_pages_gpu.to(DTYPE).reshape(-1, 1, 1, 1)
    for layer in range(LAYERS):
        k_values = (page_values + layer * 100).expand(-1, page_size, HEADS, HEAD_DIM)
        v_values = k_values + 50
        static_pool.k_buffer[layer][src_tokens] = k_values.reshape(-1, HEADS, HEAD_DIM)
        static_pool.v_buffer[layer][src_tokens] = v_values.reshape(-1, HEADS, HEAD_DIM)
        ours_pool.k_buffer[layer][src_pages_gpu] = k_values
        ours_pool.v_buffer[layer][src_pages_gpu] = v_values


def _measure(fn, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _validate(
    static_pool,
    static_host,
    ours_pool,
    ours_host,
    src_pages,
    host_pages,
    dst_pages,
    page_size,
):
    src_tokens_gpu = _expand_pages(src_pages, page_size).cuda()
    host_tokens_cpu = _expand_pages(host_pages, page_size)
    dst_tokens_gpu = _expand_pages(dst_pages, page_size).cuda()
    src_pages_gpu = src_pages.cuda()
    dst_pages_gpu = dst_pages.cuda()

    static_host.backup_from_device_all_layer(
        static_pool, host_tokens_cpu, src_tokens_gpu, "kernel"
    )
    ours_host.backup_from_device_all_layer(
        ours_pool, host_tokens_cpu, src_tokens_gpu, "kernel"
    )
    torch.cuda.synchronize()
    for layer in (0, LAYERS - 1):
        expected_k = ours_pool.k_buffer[layer][src_pages_gpu].cpu()
        expected_v = ours_pool.v_buffer[layer][src_pages_gpu].cpu()
        torch.testing.assert_close(
            static_host.k_data_refs[layer][host_tokens_cpu].reshape_as(expected_k),
            expected_k,
        )
        torch.testing.assert_close(
            static_host.v_data_refs[layer][host_tokens_cpu].reshape_as(expected_v),
            expected_v,
        )
        torch.testing.assert_close(ours_host.k_data_refs[layer][host_pages], expected_k)
        torch.testing.assert_close(ours_host.v_data_refs[layer][host_pages], expected_v)

    for layer in range(LAYERS):
        static_pool.k_buffer[layer][dst_tokens_gpu] = -1
        static_pool.v_buffer[layer][dst_tokens_gpu] = -1
        ours_pool.k_buffer[layer][dst_pages_gpu] = -1
        ours_pool.v_buffer[layer][dst_pages_gpu] = -1
        static_host.load_to_device_per_layer(
            static_pool,
            host_tokens_cpu.cuda(),
            dst_tokens_gpu,
            layer,
            "kernel",
        )
        ours_host.load_to_device_per_layer(
            ours_pool, host_tokens_cpu, dst_tokens_gpu, layer, "kernel"
        )
    torch.cuda.synchronize()
    for layer in (0, LAYERS - 1):
        torch.testing.assert_close(
            static_pool.k_buffer[layer][dst_tokens_gpu].reshape(
                -1, page_size, HEADS, HEAD_DIM
            ),
            ours_pool.k_buffer[layer][dst_pages_gpu],
        )
        torch.testing.assert_close(
            static_pool.v_buffer[layer][dst_tokens_gpu].reshape(
                -1, page_size, HEADS, HEAD_DIM
            ),
            ours_pool.v_buffer[layer][dst_pages_gpu],
        )


def _run_condition(page_size, tokens, pattern, warmup, repetitions):
    if tokens % page_size:
        raise ValueError(f"tokens={tokens} is not divisible by page_size={page_size}")
    pages = tokens // page_size
    capacity_pages, src_pages, host_pages, dst_pages = _page_sets(pages, pattern)
    static_pool, static_host = _make_static(capacity_pages, page_size)
    ours_pool, ours_host, arena = _make_ours(capacity_pages, page_size)
    try:
        _initialize_sources(static_pool, ours_pool, src_pages, page_size)
        _validate(
            static_pool,
            static_host,
            ours_pool,
            ours_host,
            src_pages,
            host_pages,
            dst_pages,
            page_size,
        )
        src_tokens = _expand_pages(src_pages, page_size).cuda()
        host_tokens = _expand_pages(host_pages, page_size)
        dst_tokens = _expand_pages(dst_pages, page_size).cuda()
        baseline_host_gpu = host_tokens.cuda()

        functions = {
            (
                "d2h",
                "baseline-static",
            ): lambda: static_host.backup_from_device_all_layer(
                static_pool, host_tokens, src_tokens, "kernel"
            ),
            (
                "d2h",
                "ours-unified-typed",
            ): lambda: ours_host.backup_from_device_all_layer(
                ours_pool, host_tokens, src_tokens, "kernel"
            ),
            ("h2d", "baseline-static"): lambda: [
                static_host.load_to_device_per_layer(
                    static_pool, baseline_host_gpu, dst_tokens, layer, "kernel"
                )
                for layer in range(LAYERS)
            ],
            ("h2d", "ours-unified-typed"): lambda: [
                ours_host.load_to_device_per_layer(
                    ours_pool, host_tokens, dst_tokens, layer, "kernel"
                )
                for layer in range(LAYERS)
            ],
        }
        payload = tokens * 2 * LAYERS * HEADS * HEAD_DIM * DTYPE.itemsize
        rows = []
        for (direction, variant), fn in functions.items():
            elapsed = _measure(fn, warmup, repetitions)
            calls = (
                LAYERS
                if direction == "h2d"
                else 1 if variant.startswith("ours") else ((pages + 63) // 64)
            )
            for repetition, ms in enumerate(elapsed):
                rows.append(
                    Measurement(
                        page_size=page_size,
                        tokens=tokens,
                        pages=pages,
                        pattern=pattern,
                        direction=direction,
                        variant=variant,
                        repetition=repetition,
                        elapsed_ms=ms,
                        payload_bytes=payload,
                        gib_per_s=payload / (ms * 1e-3) / (1024**3),
                        transfer_calls=calls,
                    )
                )
        return rows
    finally:
        static_host.destroy()
        arena.destroy()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--page-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--patterns", nargs="+", default=["contiguous", "fragmented"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/unified_typed_page_transfer"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for page_size in args.page_sizes:
        for tokens in args.tokens:
            for pattern in args.patterns:
                condition = _run_condition(
                    page_size, tokens, pattern, args.warmup, args.repetitions
                )
                rows.extend(condition)
                for direction in ("d2h", "h2d"):
                    baseline = statistics.median(
                        row.elapsed_ms
                        for row in condition
                        if row.direction == direction
                        and row.variant == "baseline-static"
                    )
                    ours = statistics.median(
                        row.elapsed_ms
                        for row in condition
                        if row.direction == direction
                        and row.variant == "ours-unified-typed"
                    )
                    print(
                        f"P={page_size:>2} tokens={tokens:>4} {pattern:>10} "
                        f"{direction.upper()} baseline={baseline:.4f} ms "
                        f"ours={ours:.4f} ms speedup={baseline / ours:.3f}x",
                        flush=True,
                    )

    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": str(DTYPE),
            "shape": {"layers": LAYERS, "heads": HEADS, "head_dim": HEAD_DIM},
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "elapsed_seconds": time.time() - started,
        },
        "measurements": [asdict(row) for row in rows],
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()
