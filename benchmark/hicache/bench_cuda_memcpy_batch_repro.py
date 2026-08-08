"""Reproduce large registered-host HiCache staged write-back failures.

Run one condition per process because an illegal CUDA access poisons the CUDA
context. The benchmark intentionally uses the production MHA device/host pools
and staged write-back dispatch while excluding the scheduler and Mamba paths.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


def expand_pages(
    start_page: int, num_pages: int, page_size: int, *, device: str
) -> torch.Tensor:
    pages = torch.arange(
        start_page, start_page + num_pages, dtype=torch.int64, device=device
    )
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (pages[:, None] * page_size + offsets).reshape(-1)


def environment_snapshot() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_memory_bytes": properties.total_memory,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = args.element_dim // args.heads
    if head_dim * args.heads != args.element_dim:
        raise ValueError("element_dim must be divisible by heads")
    source_end = (
        args.source_page_start + args.num_pages * args.source_page_sets
    ) * args.page_size
    if source_end > args.device_tokens:
        raise ValueError(
            f"source token end {source_end} exceeds device_tokens={args.device_tokens}"
        )

    device_pool = MHATokenToKVPool(
        size=args.device_tokens,
        page_size=args.page_size,
        head_num=args.heads,
        head_dim=head_dim,
        dtype=dtype,
        layer_num=args.layers,
        device=device,
        enable_memory_saver=False,
    )
    allocator_key = device_pool.device
    original_alloc = ALLOC_MEMORY_FUNCS[allocator_key]
    if args.allocator == "torch_pinned":
        ALLOC_MEMORY_FUNCS[allocator_key] = alloc_with_pin_memory
    try:
        host_pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=1.0,
            host_size=args.host_size_gb,
            page_size=args.page_size,
            layout="page_first",
            pin_memory=True,
            device="cpu",
            allocator_type="default",
        )
    finally:
        ALLOC_MEMORY_FUNCS[allocator_key] = original_alloc

    source_index_sets = [
        expand_pages(
            args.source_page_start + source_set * args.num_pages,
            args.num_pages,
            args.page_size,
            device=device,
        )
        for source_set in range(args.source_page_sets)
    ]
    host_indices = expand_pages(
        args.host_page_start,
        args.num_pages,
        args.page_size,
        device="cpu",
    )
    for source_set, source_indices in enumerate(source_index_sets):
        for layer_id in range(args.layers):
            device_pool.k_buffer[layer_id][source_indices] = (
                source_set * 1000 + layer_id + 1
            )
            device_pool.v_buffer[layer_id][source_indices] = (
                source_set * 1000 + layer_id + 101
            )
    host_pool.k_data_refs[0][host_indices] = -7
    host_pool.v_data_refs[0][host_indices] = -9
    torch.cuda.synchronize()

    k_page_bytes = args.page_size * args.layers * args.element_dim * dtype.itemsize
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if k_page_bytes >= 128 * 1024:
        expected_path = "cudaMemcpyBatchAsync"
    else:
        expected_path = "cudaMemcpyAsync fallback"
    payload = {
        "args": serialized_args,
        "environment": environment_snapshot(),
        "derived": {
            "head_dim": head_dim,
            "k_page_bytes": k_page_bytes,
            "batch_threshold_bytes": 128 * 1024,
            "expected_path": expected_path,
            "device_pool_bytes": sum(
                tensor.nbytes for tensor in device_pool.k_buffer + device_pool.v_buffer
            ),
            "host_pool_bytes": host_pool.kv_buffer.nbytes,
            "k_base_ptr": host_pool.k_buffer.data_ptr(),
            "v_base_ptr": host_pool.v_buffer.data_ptr(),
            "v_base_offset_bytes": (
                host_pool.v_buffer.data_ptr() - host_pool.k_buffer.data_ptr()
            ),
            "transfer_tokens_per_iteration": source_index_sets[0].numel(),
            "iterations": args.iterations,
            "source_page_sets": args.source_page_sets,
            "dedicated_stream": args.dedicated_stream,
            "synchronize_each": args.synchronize_each,
        },
        "result": {},
    }

    start = time.perf_counter()
    success = False
    try:
        transfer_stream = torch.cuda.Stream() if args.dedicated_stream else None
        if transfer_stream is not None:
            start_event = torch.cuda.Event()
            start_event.record()
            stream_context = torch.cuda.stream(transfer_stream)
        else:
            start_event = None
            stream_context = torch.cuda.stream(torch.cuda.current_stream())

        with stream_context:
            if start_event is not None:
                transfer_stream.wait_event(start_event)
            for iteration in range(args.iterations):
                source_indices = source_index_sets[iteration % args.source_page_sets]
                host_pool.backup_from_device_all_layer(
                    device_pool, host_indices, source_indices, "kernel"
                )
                if args.synchronize_each:
                    torch.cuda.current_stream().synchronize()
        if transfer_stream is not None:
            transfer_stream.synchronize()
        else:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        final_source_set = (args.iterations - 1) % args.source_page_sets
        for layer_id in (0, args.layers - 1):
            expected_k = torch.full(
                (host_indices.numel(), args.heads, head_dim),
                final_source_set * 1000 + layer_id + 1,
                dtype=dtype,
            )
            expected_v = torch.full_like(
                expected_k, final_source_set * 1000 + layer_id + 101
            )
            torch.testing.assert_close(
                host_pool.k_data_refs[layer_id][host_indices], expected_k
            )
            torch.testing.assert_close(
                host_pool.v_data_refs[layer_id][host_indices], expected_v
            )
        payload["result"] = {
            "status": "passed",
            "elapsed_seconds": elapsed,
            "data_verified": True,
        }
        success = True
    except BaseException as error:
        payload["result"] = {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - start,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(args.output, payload)
    if success:
        host_pool.destroy()
        torch.cuda.synchronize()
    return payload, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--host-size-gb", type=float, default=3.77)
    parser.add_argument("--device-tokens", type=int, default=20000)
    parser.add_argument("--num-pages", type=int, default=40)
    parser.add_argument("--source-page-start", type=int, default=257)
    parser.add_argument("--source-page-sets", type=int, default=1)
    parser.add_argument("--host-page-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--element-dim", type=int, default=512)
    parser.add_argument(
        "--allocator",
        choices=["registered_mmap", "torch_pinned"],
        default="registered_mmap",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dedicated-stream", action="store_true")
    parser.add_argument("--synchronize-each", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, success = run(args)
    print(json.dumps(payload["result"], indent=2))
    sys.exit(0 if success else 2)


if __name__ == "__main__":
    main()
