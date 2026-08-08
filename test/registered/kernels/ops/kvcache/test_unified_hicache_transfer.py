"""CUDA correctness tests for HiCache transfers on unified strided MHA views."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.layout.page_major import build_page_major_mha_views
from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Unified HiCache transfer tests require CUDA."
)

DEVICE = "cuda"
SIZE = 12
NUM_LAYERS = 3
HEAD_NUM = 2
HEAD_DIM = 16
ENVELOPE_FIELDS = 4
DTYPES = [torch.float16, torch.bfloat16]


def _fill_view(view: torch.Tensor, offset: int) -> None:
    values = torch.arange(view.numel(), dtype=torch.int32, device=view.device)
    view.copy_(((values % 97) + offset).reshape(view.shape).to(view.dtype))


def _make_unified_device_pool(dtype: torch.dtype):
    # K and V occupy two fields inside a larger per-slot envelope. Selecting a
    # field leaves stride(0) equal to the complete shared entry, not one KV row.
    backing = torch.full(
        (SIZE, NUM_LAYERS, ENVELOPE_FIELDS, HEAD_NUM, HEAD_DIM),
        -77,
        dtype=dtype,
        device=DEVICE,
    )
    k_buffer = [backing[:, layer_id, 1] for layer_id in range(NUM_LAYERS)]
    v_buffer = [backing[:, layer_id, 3] for layer_id in range(NUM_LAYERS)]
    for layer_id in range(NUM_LAYERS):
        _fill_view(k_buffer[layer_id], 10 + layer_id * 20)
        _fill_view(v_buffer[layer_id], 110 + layer_id * 20)

    return SimpleNamespace(
        _unified_buffer=object(),
        _backing=backing,
        device=torch.device(DEVICE),
        size=SIZE - 1,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=NUM_LAYERS,
        dtype=dtype,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in k_buffer],
            dtype=torch.uint64,
            device=DEVICE,
        ),
        v_data_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in v_buffer],
            dtype=torch.uint64,
            device=DEVICE,
        ),
    )


def _make_host_pool(device_pool) -> MHATokenToKVPoolHost:
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.device_pool = device_pool
    host.layout = "page_first"
    host.page_size = 1
    host.page_num = SIZE
    host.size = SIZE
    host.layer_num = NUM_LAYERS
    host.head_num = HEAD_NUM
    host.head_dim = HEAD_DIM
    host.dtype = device_pool.dtype
    host.element_dim = HEAD_NUM * HEAD_DIM
    host.token_stride_size = host.element_dim * host.dtype.itemsize
    host.layout_dim = host.token_stride_size * NUM_LAYERS
    host.kv_buffer = (
        torch.zeros(
            (SIZE, NUM_LAYERS, HEAD_NUM, HEAD_DIM), dtype=host.dtype
        ).pin_memory(),
        torch.zeros(
            (SIZE, NUM_LAYERS, HEAD_NUM, HEAD_DIM), dtype=host.dtype
        ).pin_memory(),
    )
    k_transposed = host.k_buffer.transpose(0, 1)
    v_transposed = host.v_buffer.transpose(0, 1)
    host.k_data_refs = [k_transposed[i] for i in range(NUM_LAYERS)]
    host.v_data_refs = [v_transposed[i] for i in range(NUM_LAYERS)]
    host.k_data_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in host.k_data_refs],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.v_data_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in host.v_data_refs],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.can_use_write_back_jit = True
    host.staging_page_capacity = SIZE
    host.staging_token_capacity = SIZE
    host.staging_k_buffer = torch.empty(
        (SIZE, NUM_LAYERS, HEAD_NUM, HEAD_DIM),
        dtype=host.dtype,
        device=DEVICE,
    )
    host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
    return host


def _make_paged_unified_device_pool(dtype: torch.dtype, page_size: int):
    page_count = 10
    token_count = page_count * page_size
    row_elements = HEAD_NUM * HEAD_DIM
    raw = torch.empty(
        token_count * NUM_LAYERS * 2 * row_elements * dtype.itemsize,
        dtype=torch.uint8,
        device=DEVICE,
    )
    k_buffer, v_buffer = build_page_major_mha_views(
        raw,
        layer_num=NUM_LAYERS,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        store_dtype=dtype,
        page_size=page_size,
        num_pages=page_count,
    )
    for layer_id in range(NUM_LAYERS):
        _fill_view(k_buffer[layer_id], 10 + layer_id * 20)
        _fill_view(v_buffer[layer_id], 110 + layer_id * 20)

    return SimpleNamespace(
        _unified_buffer=object(),
        _backing=raw,
        _page_size=page_size,
        device=torch.device(DEVICE),
        size=token_count - page_size,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=NUM_LAYERS,
        dtype=dtype,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in k_buffer],
            dtype=torch.uint64,
            device=DEVICE,
        ),
        v_data_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in v_buffer],
            dtype=torch.uint64,
            device=DEVICE,
        ),
    )


def _make_paged_host_pool(device_pool, page_size: int) -> MHATokenToKVPoolHost:
    token_count = device_pool.k_buffer[0].shape[0] * page_size
    host = _make_host_pool(device_pool)
    host.page_size = page_size
    host.page_num = token_count // page_size
    host.size = token_count
    host.kv_buffer = (
        torch.zeros(
            (token_count, NUM_LAYERS, HEAD_NUM, HEAD_DIM), dtype=host.dtype
        ).pin_memory(),
        torch.zeros(
            (token_count, NUM_LAYERS, HEAD_NUM, HEAD_DIM), dtype=host.dtype
        ).pin_memory(),
    )
    k_transposed = host.k_buffer.transpose(0, 1)
    v_transposed = host.v_buffer.transpose(0, 1)
    host.k_data_refs = [k_transposed[i] for i in range(NUM_LAYERS)]
    host.v_data_refs = [v_transposed[i] for i in range(NUM_LAYERS)]
    host.k_data_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in host.k_data_refs],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.v_data_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in host.v_data_refs],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.staging_page_capacity = host.page_num
    host.staging_token_capacity = token_count
    host.staging_k_buffer = torch.empty(
        (token_count, NUM_LAYERS, HEAD_NUM, HEAD_DIM),
        dtype=host.dtype,
        device=DEVICE,
    )
    host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
    return host


def _page_token_indices(
    page_ids: list[int], page_size: int, *, device: str
) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                page_id * page_size,
                (page_id + 1) * page_size,
                dtype=torch.int64,
                device=device,
            )
            for page_id in page_ids
        ]
    )


def _gather_paged_rows(
    view: torch.Tensor, token_indices: torch.Tensor, page_size: int
) -> torch.Tensor:
    token_indices = token_indices.to(device=view.device, dtype=torch.long)
    return view[token_indices // page_size, token_indices % page_size]


@pytest.mark.parametrize("dtype", DTYPES)
def test_unified_mha_strided_backup_load_roundtrip(dtype):
    """Exercise real staged D2H and strided H2D paths with non-identity indices."""
    device_pool = _make_unified_device_pool(dtype)
    host = _make_host_pool(device_pool)
    source_indices = torch.tensor([2, 6, 10], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([1, 4, 8], dtype=torch.int64)
    restore_indices = torch.tensor([3, 7, 9], dtype=torch.int64, device=DEVICE)

    expected_k = [layer[source_indices].clone() for layer in device_pool.k_buffer]
    expected_v = [layer[source_indices].clone() for layer in device_pool.v_buffer]
    untouched_envelope = device_pool._backing[:, :, [0, 2]].clone()

    host.backup_from_device_all_layer(
        device_pool, host_indices, source_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            host.k_data_refs[layer_id][host_indices], expected_k[layer_id].cpu()
        )
        torch.testing.assert_close(
            host.v_data_refs[layer_id][host_indices], expected_v[layer_id].cpu()
        )
        device_pool.k_buffer[layer_id][restore_indices] = -3
        device_pool.v_buffer[layer_id][restore_indices] = -5
        host.load_to_device_per_layer(
            device_pool,
            host_indices,
            restore_indices,
            layer_id,
            io_backend="kernel",
        )

    torch.cuda.synchronize()
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            device_pool.k_buffer[layer_id][restore_indices], expected_k[layer_id]
        )
        torch.testing.assert_close(
            device_pool.v_buffer[layer_id][restore_indices], expected_v[layer_id]
        )
    torch.testing.assert_close(device_pool._backing[:, :, [0, 2]], untouched_envelope)


@pytest.mark.parametrize("page_size", [8, 32])
def test_unified_mha_paged_backup_load_roundtrip(page_size):
    """Exercise legacy L2 D2H/H2D against the real unified page envelope."""
    device_pool = _make_paged_unified_device_pool(torch.bfloat16, page_size)
    host = _make_paged_host_pool(device_pool, page_size)
    source_indices = _page_token_indices([1, 3, 5], page_size, device=DEVICE)
    host_indices = _page_token_indices([0, 2, 4], page_size, device="cpu")
    restore_indices = _page_token_indices([2, 6, 8], page_size, device=DEVICE)

    expected_k = [
        _gather_paged_rows(layer, source_indices, page_size).clone()
        for layer in device_pool.k_buffer
    ]
    expected_v = [
        _gather_paged_rows(layer, source_indices, page_size).clone()
        for layer in device_pool.v_buffer
    ]

    host.backup_from_device_all_layer(
        device_pool, host_indices, source_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            host.k_data_refs[layer_id][host_indices], expected_k[layer_id].cpu()
        )
        torch.testing.assert_close(
            host.v_data_refs[layer_id][host_indices], expected_v[layer_id].cpu()
        )
        restore_pages = restore_indices // page_size
        restore_offsets = restore_indices % page_size
        device_pool.k_buffer[layer_id][restore_pages, restore_offsets] = -3
        device_pool.v_buffer[layer_id][restore_pages, restore_offsets] = -5
        host.load_to_device_per_layer(
            device_pool,
            host_indices,
            restore_indices,
            layer_id,
            io_backend="kernel",
        )

    torch.cuda.synchronize()
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            _gather_paged_rows(
                device_pool.k_buffer[layer_id], restore_indices, page_size
            ),
            expected_k[layer_id],
        )
        torch.testing.assert_close(
            _gather_paged_rows(
                device_pool.v_buffer[layer_id], restore_indices, page_size
            ),
            expected_v[layer_id],
        )


class _CudaMarkerPool:
    """Minimal physical-row store used to observe compaction ordering."""

    def __init__(self, size: int):
        self.buffer = torch.full((size,), -1, dtype=torch.int64, device=DEVICE)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        self.buffer[dst_loc] = self.buffer[src_loc].clone()


def _make_cuda_allocator(*, lazy_compaction: bool):
    full = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=8,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction="down",
    )
    pool = UnifiedKVPool(
        total_bytes=full.entry_bytes() * 64 + mamba.entry_bytes() * 16,
        sub_pool_specs=[full, mamba],
        device=DEVICE,
        enable_memory_saver=False,
    )
    marker_pool = _CudaMarkerPool(pool.max_slots("full"))
    allocator = MultiEndedAllocator(
        kvcache=marker_pool,
        unified_buffer=pool,
        sub_pool_name="full",
        device=DEVICE,
        is_id_owner=True,
        lazy_compaction=lazy_compaction,
    )
    return allocator, marker_pool


@pytest.mark.skipif(
    not hasattr(torch.cuda, "_sleep"),
    reason="CUDA sleep is required for race coverage.",
)
def test_external_transfer_event_orders_eager_relocation():
    """A delayed external reader must observe the pre-compaction physical row.

    Freeing the first slot moves the last survivor into that hole. The transfer
    intentionally reads the survivor after a delay; allocator event fencing must
    order the move after that read without calling ``Event.synchronize()`` in the
    controller.
    """
    allocator, marker_pool = _make_cuda_allocator(lazy_compaction=False)
    virtual = allocator.alloc(4)
    physical = allocator.virtual_to_physical[virtual]
    marker_pool.buffer[physical] = virtual
    survivor_virtual = virtual[-1].clone()
    survivor_physical = physical[-1].clone()
    captured = torch.full((), -1, dtype=torch.int64, device=DEVICE)

    transfer_stream = torch.cuda.Stream()
    layout_ready = torch.cuda.Event()
    transfer_done = torch.cuda.Event()
    layout_ready.record()
    with torch.cuda.stream(transfer_stream):
        transfer_stream.wait_event(layout_ready)
        torch.cuda._sleep(20_000_000)
        captured.copy_(marker_pool.buffer[survivor_physical])
        transfer_done.record()

    allocator.register_external_transfer_event(transfer_done)
    assert not transfer_done.query()
    allocator.free(virtual[:1])
    torch.cuda.synchronize()

    assert captured.item() == survivor_virtual.item()
    relocated = allocator.virtual_to_physical[survivor_virtual]
    assert relocated.item() == physical[0].item()
    assert marker_pool.buffer[relocated].item() == survivor_virtual.item()


@pytest.mark.skipif(
    not hasattr(torch.cuda, "_sleep"),
    reason="CUDA sleep is required for race coverage.",
)
def test_external_transfer_event_orders_lazy_hole_reuse():
    """A delayed D2H-like reader must finish before a freed row is reused."""
    allocator, marker_pool = _make_cuda_allocator(lazy_compaction=True)
    virtual = allocator.alloc(4)
    physical = allocator.virtual_to_physical[virtual]
    marker_pool.buffer[physical] = virtual
    freed_virtual = virtual[0].clone()
    freed_physical = physical[0].clone()
    captured = torch.full((), -1, dtype=torch.int64, device=DEVICE)

    transfer_stream = torch.cuda.Stream()
    layout_ready = torch.cuda.Event()
    transfer_done = torch.cuda.Event()
    layout_ready.record()
    with torch.cuda.stream(transfer_stream):
        transfer_stream.wait_event(layout_ready)
        torch.cuda._sleep(20_000_000)
        captured.copy_(marker_pool.buffer[freed_physical])
        transfer_done.record()

    allocator.register_external_transfer_event(transfer_done)
    assert not transfer_done.query()
    allocator.free(virtual[:1])
    replacement = allocator.alloc(1)
    replacement_physical = allocator.virtual_to_physical[replacement]
    marker_pool.buffer[replacement_physical] = 9999
    torch.cuda.synchronize()

    assert captured.item() == freed_virtual.item()
    assert replacement_physical.item() == freed_physical.item()
    assert marker_pool.buffer[replacement_physical].item() == 9999
