"""Microbenchmark for HiCache D2H/H2D copy performance.

Measures fine-grained timing for each stage of the HiCache copy path:
  - V2P translation (unified-memory only)
  - Gather/relayout (staging buffer for old, JIT kernel for new)
  - DMA transfer (D2H / H2D)
  - Total D2H / H2D time

Three versions are benchmarked:
  - static:      --enable-unified-memory off (layer-first, contiguous)
  - unified_old:  staging buffer relayout path
  - unified_new:  optimized direct JIT kernel (no staging)

Usage:
  python bench_hicache_microbench.py --model Qwen/Qwen3.5-32B --gpu cuda:0 \
      --version static --num-iters 100 --output results.json
"""

import argparse
import json
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="HiCache copy microbenchmark")
    parser.add_argument("--model", default="Qwen/Qwen3.5-32B", help="Model name (for config)")
    parser.add_argument("--gpu", default="cuda:0", help="GPU device")
    parser.add_argument("--version", required=True, choices=["static", "unified_old", "unified_new"])
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSON file")
    # Config overrides
    parser.add_argument("--num-slots", type=int, default=4096, help="Total KV slots")
    parser.add_argument("--layer-num", type=int, default=48, help="Number of layers")
    parser.add_argument("--head-num", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--dtype", default="bfloat16", help="KV cache dtype")
    parser.add_argument("--token-counts", default="1,16,64,256,1024,4096",
                        help="Comma-separated token counts to benchmark")
    return parser.parse_args()


def create_static_pool(num_slots, layer_num, head_num, head_dim, dtype, device):
    """Create a static-fraction (layer-first, contiguous) KV pool."""
    k_buffer = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, device=device)
                for _ in range(layer_num)]
    v_buffer = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, device=device)
                for _ in range(layer_num)]
    host_k = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, pin_memory=True)
              for _ in range(layer_num)]
    host_v = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, pin_memory=True)
              for _ in range(layer_num)]
    return k_buffer, v_buffer, host_k, host_v


def create_unified_pool(num_slots, layer_num, head_num, head_dim, dtype, device):
    """Create a unified-memory (envelope-major, strided) KV pool."""
    element_dim = head_num * head_dim
    k_row_bytes = head_num * head_dim
    v_row_bytes = head_num * head_dim
    entry_bytes = layer_num * (k_row_bytes + v_row_bytes)

    raw = torch.empty(num_slots * entry_bytes, dtype=dtype, device=device)

    k_shape = (num_slots, 1, head_num, head_dim)
    k_stride = (entry_bytes, element_dim, head_dim, 1)

    k_buffer = []
    v_buffer = []
    for layer in range(layer_num):
        k_base = layer * (k_row_bytes + v_row_bytes)
        v_base = k_base + k_row_bytes
        k_buffer.append(torch.as_strided(raw, size=k_shape, stride=k_stride, storage_offset=k_base))
        v_buffer.append(torch.as_strided(raw, size=k_shape, stride=k_stride, storage_offset=v_base))

    host_k = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, pin_memory=True)
              for _ in range(layer_num)]
    host_v = [torch.empty(num_slots, head_num, head_dim, dtype=dtype, pin_memory=True)
              for _ in range(layer_num)]

    # V2P table (identity for simplicity)
    v2p_table = torch.arange(num_slots, dtype=torch.long, device=device)

    return k_buffer, v_buffer, host_k, host_v, v2p_table


def cuda_event_time(fn, *args, **kwargs):
    """Measure GPU time of fn using CUDA events. Returns time in microseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start.record()
    fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000  # ms -> μs


def bench_static_d2h(k_buffer, v_buffer, host_k, host_v, indices, layer_num):
    """Benchmark D2H for static-fraction: .to('cpu') per layer."""
    def d2h():
        for layer_id in range(layer_num):
            k_cpu = k_buffer[layer_id][indices].to("cpu", non_blocking=True)
            v_cpu = v_buffer[layer_id][indices].to("cpu", non_blocking=True)
            host_k[layer_id][:len(indices)] = k_cpu
            host_v[layer_id][:len(indices)] = v_cpu
        torch.cuda.synchronize()

    return d2h


def bench_static_h2d(k_buffer, v_buffer, host_k, host_v, indices, layer_num, device):
    """Benchmark H2D for static-fraction: .to(device) per layer."""
    def h2d():
        for layer_id in range(layer_num):
            k_chunk = host_k[layer_id][:len(indices)].to(device, non_blocking=True)
            v_chunk = host_v[layer_id][:len(indices)].to(device, non_blocking=True)
            k_buffer[layer_id][indices] = k_chunk
            v_buffer[layer_id][indices] = v_chunk
        torch.cuda.synchronize()

    return h2d


def bench_unified_old_d2h(k_buffer, v_buffer, host_k, host_v, indices, layer_num,
                           v2p_table, staging_k, staging_v, device):
    """Benchmark D2H for unified-old: V2P + staging relayout + DMA."""
    n = len(indices)
    def d2h():
        # V2P
        phys = v2p_table[indices]
        # Staging relayout (gather strided -> contiguous)
        for layer_id in range(layer_num):
            # k_buffer[layer_id] shape: (num_slots, 1, head_num, head_dim)
            # After indexing: (n, 1, head_num, head_dim) -> squeeze(1) -> (n, head_num, head_dim)
            staging_k[:n] = k_buffer[layer_id][phys].squeeze(1)
            staging_v[:n] = v_buffer[layer_id][phys].squeeze(1)
        # DMA to host
        for layer_id in range(layer_num):
            host_k[layer_id][:n] = staging_k[:n].to("cpu", non_blocking=True)
            host_v[layer_id][:n] = staging_v[:n].to("cpu", non_blocking=True)
        torch.cuda.synchronize()

    return d2h



def bench_unified_new_d2h(k_buffer, v_buffer, host_k, host_v, indices, layer_num, v2p_table):
    """Benchmark D2H for unified-new: V2P + direct gather (no staging)."""
    n = len(indices)
    def d2h():
        # V2P
        phys = v2p_table[indices]
        # Direct gather per layer (no staging)
        for layer_id in range(layer_num):
            # k_buffer[layer_id] shape: (num_slots, 1, head_num, head_dim)
            # After indexing: (n, 1, head_num, head_dim) -> squeeze(1) -> (n, head_num, head_dim)
            k_cpu = k_buffer[layer_id][phys].squeeze(1).to("cpu", non_blocking=True)
            v_cpu = v_buffer[layer_id][phys].squeeze(1).to("cpu", non_blocking=True)
            host_k[layer_id][:n] = k_cpu
            host_v[layer_id][:n] = v_cpu
        torch.cuda.synchronize()

    return d2h



def bench_unified_old_h2d(k_buffer, v_buffer, host_k, host_v, indices, layer_num,
                          v2p_table, device):
    """Benchmark H2D for unified-old: index_select + index_copy_."""
    n = len(indices)
    def h2d():
        phys = v2p_table[indices]
        for layer_id in range(layer_num):
            # host_k shape: (num_slots, head_num, head_dim)
            # k_buffer[layer_id] shape: (num_slots, 1, head_num, head_dim)
            # Need to unsqueeze(1) to match destination dimensionality
            k_rows = host_k[layer_id][:n].to(device, non_blocking=True).unsqueeze(1)
            v_rows = host_v[layer_id][:n].to(device, non_blocking=True).unsqueeze(1)
            k_buffer[layer_id].index_copy_(0, phys, k_rows)
            v_buffer[layer_id].index_copy_(0, phys, v_rows)
        torch.cuda.synchronize()

    return h2d



def bench_unified_new_h2d(k_buffer, v_buffer, host_k, host_v, indices, layer_num,
                          v2p_table, device):
    """Benchmark H2D for unified-new: direct .to(device) + scatter."""
    n = len(indices)
    def h2d():
        phys = v2p_table[indices]
        for layer_id in range(layer_num):
            # host_k shape: (num_slots, head_num, head_dim)
            # k_buffer[layer_id] shape: (num_slots, 1, head_num, head_dim)
            # Need unsqueeze(1) to match destination shape
            k_chunk = host_k[layer_id][:n].to(device, non_blocking=True).unsqueeze(1)
            v_chunk = host_v[layer_id][:n].to(device, non_blocking=True).unsqueeze(1)
            k_buffer[layer_id][phys] = k_chunk
            v_buffer[layer_id][phys] = v_chunk
        torch.cuda.synchronize()

    return h2d



def main():
    args = parse_args()
    device = args.gpu
    dtype = getattr(torch, args.dtype)
    token_counts = [int(x) for x in args.token_counts.split(",")]

    num_slots = args.num_slots
    layer_num = args.layer_num
    head_num = args.head_num
    head_dim = args.head_dim
    element_dim = head_num * head_dim

    results = {
        "version": args.version,
        "model": args.model,
        "config": {
            "num_slots": num_slots,
            "layer_num": layer_num,
            "head_num": head_num,
            "head_dim": head_dim,
            "dtype": args.dtype,
        },
        "d2h": [],
        "h2d": [],
    }

    print(f"=== HiCache Microbenchmark: {args.version} ===")
    print(f"  Layers: {layer_num}, Heads: {head_num}, Dim: {head_dim}")
    print(f"  Slots: {num_slots}, Dtype: {args.dtype}")
    print()

    # Create pool based on version
    if args.version == "static":
        k_buf, v_buf, h_k, h_v = create_static_pool(num_slots, layer_num, head_num, head_dim, dtype, device)
        v2p = None
        staging_k = staging_v = None
    else:
        k_buf, v_buf, h_k, h_v, v2p = create_unified_pool(num_slots, layer_num, head_num, head_dim, dtype, device)
        if args.version == "unified_old":
            staging_k = torch.empty(num_slots, head_num, head_dim, dtype=dtype, device=device)
            staging_v = torch.empty(num_slots, head_num, head_dim, dtype=dtype, device=device)
        else:
            staging_k = staging_v = None

    # Fill with random data
    for layer_id in range(layer_num):
        k_buf[layer_id].fill_(1.0)
        v_buf[layer_id].fill_(2.0)

    for n_tokens in token_counts:
        n_tokens = min(n_tokens, num_slots)
        indices = torch.randperm(num_slots)[:n_tokens].to(device)

        print(f"  Tokens: {n_tokens}")

        # D2H benchmark
        if args.version == "static":
            d2h_fn = bench_static_d2h(k_buf, v_buf, h_k, h_v, indices, layer_num)
        elif args.version == "unified_old":
            d2h_fn = bench_unified_old_d2h(k_buf, v_buf, h_k, h_v, indices, layer_num,
                                           v2p, staging_k, staging_v, device)
        else:
            d2h_fn = bench_unified_new_d2h(k_buf, v_buf, h_k, h_v, indices, layer_num, v2p)

        d2h_time = cuda_event_time(d2h_fn)
        print(f"    D2H: {d2h_time:.1f} μs")

        results["d2h"].append({"tokens": n_tokens, "time_us": d2h_time})

        # H2D benchmark
        if args.version == "static":
            h2d_fn = bench_static_h2d(k_buf, v_buf, h_k, h_v, indices, layer_num, device)
        elif args.version == "unified_old":
            h2d_fn = bench_unified_old_h2d(k_buf, v_buf, h_k, h_v, indices, layer_num, v2p, device)
        else:
            h2d_fn = bench_unified_new_h2d(k_buf, v_buf, h_k, h_v, indices, layer_num, v2p, device)

        h2d_time = cuda_event_time(h2d_fn)
        print(f"    H2D: {h2d_time:.1f} μs")

        results["h2d"].append({"tokens": n_tokens, "time_us": h2d_time})

    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
