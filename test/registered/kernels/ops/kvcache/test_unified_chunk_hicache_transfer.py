"""CUDA correctness tests for the shared KV/Mamba typed-chunk L2 arena."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import transfer_hicache_pages
from sglang.srt.mem_cache.pool_host.unified_chunk import (
    UnifiedChunkMambaPoolHost,
    UnifiedChunkMHAPoolHost,
)
from sglang.srt.mem_cache.typed_chunk_host import (
    HostChunkOwner,
    SharedTypedChunkHostArena,
    build_shared_kv_envelope_view,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="typed-chunk transfer test requires CUDA"
)
requires_cuda_sleep = pytest.mark.skipif(
    not hasattr(torch.cuda, "_sleep"), reason="CUDA sleep is required for event tests"
)


def _expand_pages(page_ids: torch.Tensor, page_size: int) -> torch.Tensor:
    offsets = torch.arange(page_size, dtype=torch.int64, device=page_ids.device)
    return (page_ids[:, None] * page_size + offsets).reshape(-1)


@pytest.mark.parametrize("page_size", [1, 2, 4, 8, 16, 32])
def test_raw_page_copy_roundtrip(page_size):
    page_bytes = page_size * 256
    num_pages = 8
    arena = SharedTypedChunkHostArena(
        total_bytes=num_pages * page_bytes,
        kv_page_bytes=page_bytes,
        mamba_slot_bytes=page_bytes,
        host_device="cpu",
        accelerator_device="cuda",
        pin_memory=True,
        allocator_type="default",
    )
    try:
        gpu = (
            torch.arange(num_pages * page_bytes, dtype=torch.int64, device="cuda")
            .remainder_(251)
            .to(torch.uint8)
        )
        src_pages = torch.tensor([1, 4, 6], dtype=torch.int64, device="cuda")
        host_pages = torch.tensor([0, 2, 5], dtype=torch.int64, device="cuda")
        transfer_hicache_pages(
            arena.raw,
            host_pages,
            gpu,
            src_pages,
            page_stride_bytes=page_bytes,
        )
        torch.cuda.synchronize()
        for host_page, src_page in zip(host_pages.cpu(), src_pages.cpu()):
            torch.testing.assert_close(
                arena.raw[host_page * page_bytes : (host_page + 1) * page_bytes],
                gpu[src_page * page_bytes : (src_page + 1) * page_bytes].cpu(),
            )

        # The reverse direction copies just one aligned subrange, matching the
        # per-layer H2D path while leaving the rest of each L1 page untouched.
        dst_pages = torch.tensor([2, 3, 7], dtype=torch.int64, device="cuda")
        gpu[
            dst_pages[:, None] * page_bytes + torch.arange(page_bytes, device="cuda")
        ] = 0
        offset = page_bytes // 2
        transfer_hicache_pages(
            gpu,
            dst_pages,
            arena.raw,
            host_pages,
            page_stride_bytes=page_bytes,
            copy_offset_bytes=offset,
            copy_bytes=page_bytes - offset,
        )
        torch.cuda.synchronize()
        for dst_page, host_page in zip(dst_pages.cpu(), host_pages.cpu()):
            torch.testing.assert_close(
                gpu[dst_page * page_bytes + offset : (dst_page + 1) * page_bytes].cpu(),
                arena.raw[
                    host_page * page_bytes + offset : (host_page + 1) * page_bytes
                ],
            )
    finally:
        arena.destroy()


@requires_cuda_sleep
def test_cuda_event_blocks_same_type_kv_page_reuse():
    """A KV page cannot be recycled while its host bytes are in flight."""

    arena = SharedTypedChunkHostArena(
        total_bytes=2 * 1_024,
        kv_page_bytes=256,
        mamba_slot_bytes=1_024,
        host_device="cpu",
        accelerator_device="cuda",
        pin_memory=True,
        allocator_type="default",
    )
    try:
        pages = arena.alloc_kv(2)
        protected = arena.pin_chunks_for_transfer(arena.chunks.kv_chunks(pages[:1]))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            finish = torch.cuda.Event()
            finish.record()
        arena.release_chunks_after_event(protected, finish)

        arena.free_kv(pages[:1])
        replacement = arena.alloc_kv(1)
        assert replacement.tolist() == [4]
        assert not finish.query()

        finish.synchronize()
        after_event = arena.alloc_kv(1)
        assert after_event.tolist() == [0]
        arena.chunks.assert_consistent()
    finally:
        arena.destroy()


@requires_cuda_sleep
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shared_typed_chunk_kv_mamba_roundtrip(dtype):
    layers = 3
    device_slots = 16
    head_num = 1
    head_dim = 64  # one K/V row is 128 B: supported by the JIT kernel
    temporal_shape = (16, 128)
    conv_shape = (4, 128)

    kv_entry_bytes = 2 * layers * head_num * head_dim * dtype.itemsize
    mamba_entry_bytes = layers * (
        torch.tensor(temporal_shape).prod().item() * dtype.itemsize
        + torch.tensor(conv_shape).prod().item() * dtype.itemsize
    )
    chunks = 4
    chunk_bytes = (
        (mamba_entry_bytes + kv_entry_bytes - 1) // kv_entry_bytes
    ) * kv_entry_bytes

    arena = SharedTypedChunkHostArena(
        total_bytes=chunks * chunk_bytes,
        kv_page_bytes=kv_entry_bytes,
        mamba_slot_bytes=mamba_entry_bytes,
        host_device="cpu",
        accelerator_device="cuda",
        pin_memory=True,
        allocator_type="default",
    )
    try:
        # Device KV uses the same page-major envelope stride as unified memory.
        device_kv_raw = torch.zeros(
            device_slots * kv_entry_bytes, dtype=torch.uint8, device="cuda"
        )
        device_kv_envelope = build_shared_kv_envelope_view(
            device_kv_raw,
            num_pages=device_slots,
            page_size=1,
            layer_num=layers,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
        )
        # Match the production UnifiedMHATokenToKVPool layout exactly:
        # [num_pages, page_size=1, head_num, head_dim].
        k_layers = [device_kv_envelope[0, :, layer] for layer in range(layers)]
        v_layers = [device_kv_envelope[1, :, layer] for layer in range(layers)]
        unified_marker = SimpleNamespace(
            total_bytes=device_kv_raw.numel(), _raw=device_kv_raw
        )
        kv_device = SimpleNamespace(
            _unified_buffer=unified_marker,
            page_size=1,
            store_dtype=dtype,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=head_dim,
            layer_num=layers,
            start_layer=0,
            end_layer=layers,
            device="cuda",
            k_buffer=k_layers,
            v_buffer=v_layers,
            k_data_ptrs=torch.tensor(
                [x.data_ptr() for x in k_layers],
                dtype=torch.uint64,
                device="cuda",
            ),
            v_data_ptrs=torch.tensor(
                [x.data_ptr() for x in v_layers],
                dtype=torch.uint64,
                device="cuda",
            ),
        )

        # Device Mamba rows are deliberately strided to exercise unified gather.
        temporal_backing = torch.full(
            (device_slots, layers, 3, *temporal_shape),
            -77,
            dtype=dtype,
            device="cuda",
        )
        conv_backing = torch.full(
            (device_slots, layers, 3, *conv_shape),
            -77,
            dtype=dtype,
            device="cuda",
        )
        temporal = temporal_backing[:, :, 1].transpose(0, 1)
        conv = conv_backing[:, :, 1].transpose(0, 1)
        mamba_device = SimpleNamespace(
            _unified_buffer=unified_marker,
            num_mamba_layers=layers,
            size=device_slots,
            device="cuda",
            mamba_cache=SimpleNamespace(temporal=temporal, conv=[conv]),
        )

        kv_host = UnifiedChunkMHAPoolHost(kv_device, arena)
        mamba_host = UnifiedChunkMambaPoolHost(mamba_device, arena)

        kv_src = torch.tensor([1, 5, 10], dtype=torch.int64, device="cuda")
        mamba_src = torch.tensor([2], dtype=torch.int64, device="cuda")
        kv_host_indices = kv_host.alloc(len(kv_src))
        mamba_host_indices = mamba_host.alloc(len(mamba_src))
        assert kv_host_indices.tolist() == [0, 1, 2]
        assert mamba_host_indices.tolist() == [3]
        assert arena.chunks.owner(0) == HostChunkOwner.KV
        assert arena.chunks.owner(3) == HostChunkOwner.MAMBA

        for layer in range(layers):
            k_layers[layer].copy_(
                torch.arange(k_layers[layer].numel(), device="cuda", dtype=dtype)
                .reshape_as(k_layers[layer])
                .add_(layer * 1000)
            )
            v_layers[layer].copy_(k_layers[layer] + 100)
            temporal[layer].copy_(
                torch.arange(temporal[layer].numel(), device="cuda", dtype=dtype)
                .reshape_as(temporal[layer])
                .add_(layer * 2000)
            )
            conv[layer].copy_(
                torch.arange(conv[layer].numel(), device="cuda", dtype=dtype)
                .reshape_as(conv[layer])
                .add_(layer * 3000)
            )

        expected_k = [x[kv_src].clone() for x in k_layers]
        expected_v = [x[kv_src].clone() for x in v_layers]
        expected_temporal = [x[mamba_src].clone() for x in temporal]
        expected_conv = [x[mamba_src].clone() for x in conv]

        stream = torch.cuda.Stream()
        # Protect the Mamba destination chunk exactly as the controller does,
        # then put a long-running GPU op ahead of D2H.  The CPU must enqueue the
        # transfer without waiting, and freeing metadata must not make the chunk
        # retypable until the recorded event completes.
        protected_mamba_chunks = arena.pin_chunks_for_transfer(
            arena.chunks.mamba_chunks(mamba_host_indices)
        )
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            kv_host.backup_from_device_all_layer(
                kv_device, kv_host_indices, kv_src, "kernel"
            )
            mamba_host.backup_from_device_all_layer(
                mamba_device, mamba_host_indices, mamba_src, "kernel"
            )
            d2h_finish = torch.cuda.Event()
            d2h_finish.record()
        arena.release_chunks_after_event(protected_mamba_chunks, d2h_finish)
        mamba_host.free(mamba_host_indices)
        assert not d2h_finish.query(), "Mamba D2H unexpectedly synchronized the CPU"
        assert arena.chunks.owner(3) == HostChunkOwner.MAMBA

        d2h_finish.synchronize()
        # Polling the arena releases the deferred pin and makes chunk 3 the
        # highest free Mamba chunk again.  Allocation does not clear its data.
        mamba_host_indices = mamba_host.alloc(1)
        assert mamba_host_indices.tolist() == [3]

        for layer in range(layers):
            torch.testing.assert_close(
                kv_host.k_data_refs[layer][kv_host_indices], expected_k[layer].cpu()
            )
            torch.testing.assert_close(
                kv_host.v_data_refs[layer][kv_host_indices], expected_v[layer].cpu()
            )
            torch.testing.assert_close(
                mamba_host.temporal_buffer[mamba_host_indices, layer, 0],
                expected_temporal[layer].cpu(),
            )
            torch.testing.assert_close(
                mamba_host.conv_buffer[0][mamba_host_indices, layer, 0],
                expected_conv[layer].cpu(),
            )

        kv_dst = torch.tensor([3, 7, 12], dtype=torch.int64, device="cuda")
        mamba_dst = torch.tensor([9], dtype=torch.int64, device="cuda")
        for layer in range(layers):
            k_layers[layer][kv_dst] = -1
            v_layers[layer][kv_dst] = -2
            temporal[layer][mamba_dst] = -3
            conv[layer][mamba_dst] = -4

        with torch.cuda.stream(stream):
            for layer in range(layers):
                kv_host.load_to_device_per_layer(
                    kv_device,
                    kv_host_indices,
                    kv_dst,
                    layer,
                    "kernel",
                )
                mamba_host.load_to_device_per_layer(
                    mamba_device,
                    mamba_host_indices,
                    mamba_dst,
                    layer,
                    "kernel",
                )
        stream.synchronize()

        for layer in range(layers):
            torch.testing.assert_close(k_layers[layer][kv_dst], expected_k[layer])
            torch.testing.assert_close(v_layers[layer][kv_dst], expected_v[layer])
            torch.testing.assert_close(
                temporal[layer][mamba_dst], expected_temporal[layer]
            )
            torch.testing.assert_close(conv[layer][mamba_dst], expected_conv[layer])

        # Empty chunks are immediately available to the opposite type.
        kv_host.free(kv_host_indices)
        retyped = mamba_host.alloc(1)
        assert retyped.tolist() == [2]
        assert arena.chunks.owner(2) == HostChunkOwner.MAMBA

        # Exercise the reverse direction on the real adapter views as well.
        mamba_host.free(retyped)
        kv_after_mamba = kv_host.alloc(arena.chunks.kv_pages_per_chunk)
        assert kv_after_mamba.tolist() == list(range(arena.chunks.kv_pages_per_chunk))
        assert arena.chunks.owner(0) == HostChunkOwner.KV
        arena.chunks.assert_consistent()
    finally:
        arena.destroy()


@pytest.mark.parametrize("page_size", [2, 4, 8, 16, 32])
def test_shared_typed_chunk_multi_token_page_kv_roundtrip(page_size):
    """D2H/H2D preserves every token in a unified multi-token KV page."""

    dtype = torch.float16
    layers = 3
    device_pages = 10
    head_num = 1
    head_dim = 64
    token_bytes = 2 * layers * head_num * head_dim * dtype.itemsize
    kv_page_bytes = page_size * token_bytes
    arena = SharedTypedChunkHostArena(
        total_bytes=6 * kv_page_bytes,
        kv_page_bytes=kv_page_bytes,
        mamba_slot_bytes=2 * kv_page_bytes,
        host_device="cpu",
        accelerator_device="cuda",
        pin_memory=True,
        allocator_type="default",
    )
    try:
        device_raw = torch.zeros(
            device_pages * kv_page_bytes, dtype=torch.uint8, device="cuda"
        )
        envelope = build_shared_kv_envelope_view(
            device_raw,
            num_pages=device_pages,
            page_size=page_size,
            layer_num=layers,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
        )
        k_layers = [envelope[0, :, layer] for layer in range(layers)]
        v_layers = [envelope[1, :, layer] for layer in range(layers)]
        marker = SimpleNamespace(total_bytes=device_raw.numel(), _raw=device_raw)
        device_pool = SimpleNamespace(
            _unified_buffer=marker,
            page_size=page_size,
            store_dtype=dtype,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=head_dim,
            layer_num=layers,
            start_layer=0,
            end_layer=layers,
            device="cuda",
            k_buffer=k_layers,
            v_buffer=v_layers,
            k_data_ptrs=torch.tensor(
                [x.data_ptr() for x in k_layers], dtype=torch.uint64, device="cuda"
            ),
            v_data_ptrs=torch.tensor(
                [x.data_ptr() for x in v_layers], dtype=torch.uint64, device="cuda"
            ),
        )
        host = UnifiedChunkMHAPoolHost(device_pool, arena)

        for layer in range(layers):
            values = torch.arange(
                k_layers[layer].numel(), dtype=dtype, device="cuda"
            ).reshape_as(k_layers[layer])
            k_layers[layer].copy_(values + layer * 1000)
            v_layers[layer].copy_(values + layer * 1000 + 100)

        src_pages = torch.tensor([1, 4, 8], dtype=torch.int64, device="cuda")
        src_tokens = _expand_pages(src_pages, page_size)
        host_tokens = host.alloc(src_tokens.numel())
        expected_k = [layer[src_pages].clone() for layer in k_layers]
        expected_v = [layer[src_pages].clone() for layer in v_layers]

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            host.backup_from_device_all_layer(
                device_pool, host_tokens, src_tokens, "kernel"
            )
        stream.synchronize()
        host_pages = host_tokens.reshape(-1, page_size)[:, 0] // page_size
        for layer in range(layers):
            torch.testing.assert_close(
                host.k_data_refs[layer][host_pages], expected_k[layer].cpu()
            )
            torch.testing.assert_close(
                host.v_data_refs[layer][host_pages], expected_v[layer].cpu()
            )

        dst_pages = torch.tensor([2, 5, 9], dtype=torch.int64, device="cuda")
        dst_tokens = _expand_pages(dst_pages, page_size)
        for layer in range(layers):
            k_layers[layer][dst_pages] = -1
            v_layers[layer][dst_pages] = -2
        with torch.cuda.stream(stream):
            for layer in range(layers):
                host.load_to_device_per_layer(
                    device_pool, host_tokens, dst_tokens, layer, "kernel"
                )
        stream.synchronize()
        for layer in range(layers):
            torch.testing.assert_close(k_layers[layer][dst_pages], expected_k[layer])
            torch.testing.assert_close(v_layers[layer][dst_pages], expected_v[layer])
    finally:
        arena.destroy()
