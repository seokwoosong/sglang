"""Unit tests for the Mamba JIT transfer kernel.

Verifies kernel backup (D2H) and load (H2D) correctness for
``MambaPoolHost`` via the ``io_backend='kernel'`` path, across both
supported layouts and multiple index scenarios.
"""

import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba transfer kernel tests require CUDA."
)

DEVICE = "cuda"
NUM_LAYERS = 3
SIZE = 16
TEMPORAL_SHAPE = (16, 128)  # 16*128*2 = 4096 bytes (fp16), 16-byte aligned
CONV_SHAPE = (4, 128)  # 4*128*2 = 1024 bytes (fp16), 16-byte aligned
DTYPES = [torch.float16, torch.bfloat16]
LAYOUTS = ["page_first", "page_first_direct"]


def test_registered_mmap_large_state_roundtrip():
    """Exercise registered L2 with Qwen3.5-sized strided unified rows."""
    from sglang.kernels.ops.mamba.transfer_mamba import (
        transfer_kv_mamba_lf_pf,
        transfer_kv_mamba_pf_lf,
    )

    num_layers = 18
    num_slots = 2
    state_shape = (16, 128, 128)  # 1 MiB per fp32 layer/slot
    slot_size = 19_537_920
    item_size = int(torch.tensor(state_shape).prod()) * torch.float32.itemsize
    slot_elems = slot_size // torch.float32.itemsize
    item_elems = item_size // torch.float32.itemsize
    source_backing = torch.empty(
        num_slots * slot_elems, dtype=torch.float32, device=DEVICE
    )
    source = torch.as_strided(
        source_backing,
        size=(num_layers, num_slots, *state_shape),
        stride=(item_elems, slot_elems, 16_384, 128, 1),
    )
    for layer_id in range(num_layers):
        source[layer_id].fill_(layer_id + 1)
    host = alloc_with_host_register(
        (num_slots, num_layers, 1, *state_shape),
        dtype=torch.float32,
        device="cpu",
        pin_memory=True,
        allocator=HostTensorAllocator(),
    )
    host.zero_()
    source_ptrs = torch.tensor(
        [source[layer_id].data_ptr() for layer_id in range(num_layers)],
        dtype=torch.uint64,
        device=DEVICE,
    )
    source_indices = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([0], dtype=torch.int64, device=DEVICE)

    try:
        transfer_kv_mamba_lf_pf(
            source_ptrs,
            host,
            source_indices,
            host_indices,
            item_size,
            item_size * num_layers,
            num_layers,
            slot_size,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            host[0, :, 0, 0, 0, 0],
            torch.arange(1, num_layers + 1, dtype=torch.float32),
        )

        # Match the unified L1 envelope: each Mamba row is contiguous, but
        # consecutive slots are separated by the complete Mamba slot stride.
        restored_backing = torch.zeros(
            num_slots * slot_elems, dtype=torch.float32, device=DEVICE
        )
        restored = torch.as_strided(
            restored_backing,
            (num_slots, *state_shape),
            (slot_elems, 16_384, 128, 1),
        )
        transfer_kv_mamba_pf_lf(
            host,
            restored,
            host_indices,
            source_indices,
            7,
            item_size,
            item_size * num_layers,
            slot_size,
        )
        torch.cuda.synchronize()
        assert restored[1, 0, 0, 0].item() == 8.0
    finally:
        _cuda_host_unregister(host)


def make_device_pool(dtype, device=DEVICE):
    """Create a minimal mock device pool that MambaPoolHost can use."""
    temporal = torch.zeros(
        (NUM_LAYERS, SIZE) + TEMPORAL_SHAPE, dtype=dtype, device=device
    )
    conv = [torch.zeros((NUM_LAYERS, SIZE) + CONV_SHAPE, dtype=dtype, device=device)]

    mamba_cache = SimpleNamespace(temporal=temporal, conv=conv)
    return SimpleNamespace(
        mamba_cache=mamba_cache,
        size=SIZE,
        device=device,
    )


def make_unified_device_pool(dtype, device=DEVICE):
    """Build Mamba views whose slot rows are separated by a shared envelope."""
    temporal_backing = torch.full(
        (SIZE, NUM_LAYERS, 3) + TEMPORAL_SHAPE,
        -77,
        dtype=dtype,
        device=device,
    )
    conv_backing = torch.full(
        (SIZE, NUM_LAYERS, 3) + CONV_SHAPE,
        -77,
        dtype=dtype,
        device=device,
    )
    temporal = temporal_backing[:, :, 1].transpose(0, 1)
    conv = [conv_backing[:, :, 1].transpose(0, 1)]
    return SimpleNamespace(
        _unified_buffer=object(),
        _temporal_backing=temporal_backing,
        _conv_backing=[conv_backing],
        mamba_cache=SimpleNamespace(temporal=temporal, conv=conv),
        size=SIZE,
        device=device,
    )


def bind_device_pool(host, device_pool):
    host.device_pool = device_pool
    host.temporal_device_ptrs = torch.tensor(
        [device_pool.mamba_cache.temporal[i].data_ptr() for i in range(NUM_LAYERS)],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.conv_device_ptrs = [
        torch.tensor(
            [conv_state[i].data_ptr() for i in range(NUM_LAYERS)],
            dtype=torch.uint64,
            device=DEVICE,
        )
        for conv_state in device_pool.mamba_cache.conv
    ]
    # The production constructor creates bounded staging after binding the
    # device pool. This fixture swaps in a strided unified pool later, so repeat
    # that initialization instead of leaving the old contiguous-pool buffers.
    host._init_write_back_staging_buffers()


def make_host_pool(dtype, layout):
    """Create a MambaPoolHost bypassing __init__, manually setting attributes.

    NOTE: If MambaPoolHost adds/renames attributes accessed by
    backup_from_device_all_layer or load_to_device_per_layer, this mock
    must be updated to match. See assert_host_mock_complete() below.
    """
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.layout = layout
    host.page_size = 1
    host.page_num = SIZE
    host.size = SIZE
    host.pin_memory = True
    host.device = "cpu"
    host.num_mamba_layers = NUM_LAYERS
    host.conv_state_shapes = [CONV_SHAPE]
    host.temporal_state_shape = TEMPORAL_SHAPE
    host.temporal_state_elem_size = int(torch.prod(torch.tensor(TEMPORAL_SHAPE)).item())
    host.conv_state_elem_sizes = [int(torch.prod(torch.tensor(CONV_SHAPE)).item())]
    host.conv_dtype = dtype
    host.temporal_dtype = dtype
    host.dtype = dtype
    host.size_per_token = host.get_size_per_token()

    # Allocate host buffers (page_first layout)
    temporal_dims = (SIZE, NUM_LAYERS, 1) + TEMPORAL_SHAPE
    host.temporal_buffer = torch.zeros(temporal_dims, dtype=dtype).pin_memory()

    host.conv_buffer = []
    conv_dims = (SIZE, NUM_LAYERS, 1) + CONV_SHAPE
    host.conv_buffer.append(torch.zeros(conv_dims, dtype=dtype).pin_memory())

    # Unified backup no longer needs GPU staging buffers.
    host.temporal_staging_buffer = None
    host.conv_staging_buffers = [None]
    host.can_use_write_back_jit = True

    # Device pointers (needed for backup kernel path)
    device_pool = make_device_pool(dtype)
    host.device_pool = device_pool
    host.temporal_device_ptrs = torch.tensor(
        [device_pool.mamba_cache.temporal[i].data_ptr() for i in range(NUM_LAYERS)],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.conv_device_ptrs = [
        torch.tensor(
            [conv_state[i].data_ptr() for i in range(NUM_LAYERS)],
            dtype=torch.uint64,
            device=DEVICE,
        )
        for conv_state in device_pool.mamba_cache.conv
    ]

    host.lock = threading.RLock()
    host.clear()
    return host


def assert_host_mock_complete(host):
    """Sanity check: ensure mock covers attributes used by backup/load paths."""
    required = [
        "layout",
        "page_size",
        "page_num",
        "size",
        "pin_memory",
        "device",
        "num_mamba_layers",
        "conv_state_shapes",
        "temporal_state_shape",
        "temporal_state_elem_size",
        "conv_state_elem_sizes",
        "conv_dtype",
        "temporal_dtype",
        "dtype",
        "size_per_token",
        "temporal_buffer",
        "conv_buffer",
        "temporal_staging_buffer",
        "conv_staging_buffers",
        "can_use_write_back_jit",
        "device_pool",
        "temporal_device_ptrs",
        "conv_device_ptrs",
        "lock",
    ]
    missing = [attr for attr in required if not hasattr(host, attr)]
    assert not missing, f"Mock MambaPoolHost missing attributes: {missing}"


def fill_device_data(device_pool, dtype):
    """Fill device temporal and conv states with deterministic data."""
    for layer_id in range(NUM_LAYERS):
        offset = layer_id * 1000
        data = torch.arange(
            device_pool.mamba_cache.temporal[layer_id].numel(),
            device=DEVICE,
            dtype=dtype,
        )
        device_pool.mamba_cache.temporal[layer_id].copy_(
            (data + offset).view_as(device_pool.mamba_cache.temporal[layer_id])
        )
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            conv_data = torch.arange(
                device_pool.mamba_cache.conv[conv_idx][layer_id].numel(),
                device=DEVICE,
                dtype=dtype,
            )
            device_pool.mamba_cache.conv[conv_idx][layer_id].copy_(
                (conv_data + offset + conv_idx * 500).view_as(
                    device_pool.mamba_cache.conv[conv_idx][layer_id]
                )
            )


def assert_host_matches_device(host, device_pool, host_indices, device_indices):
    """Verify host backup data matches device source data."""
    for layer_id in range(NUM_LAYERS):
        # Temporal
        host_temporal = host.temporal_buffer[host_indices, layer_id, 0].cpu()
        dev_temporal = device_pool.mamba_cache.temporal[layer_id][device_indices].cpu()
        torch.testing.assert_close(host_temporal, dev_temporal)

        # Conv
        for conv_idx in range(len(host.conv_buffer)):
            host_conv = host.conv_buffer[conv_idx][host_indices, layer_id, 0].cpu()
            dev_conv = device_pool.mamba_cache.conv[conv_idx][layer_id][
                device_indices
            ].cpu()
            torch.testing.assert_close(host_conv, dev_conv)


def assert_device_matches_host(host, device_pool, host_indices, device_indices):
    """Verify device load data matches host source data."""
    for layer_id in range(NUM_LAYERS):
        # Temporal
        host_temporal = host.temporal_buffer[host_indices, layer_id, 0].to(DEVICE)
        dev_temporal = device_pool.mamba_cache.temporal[layer_id][device_indices]
        torch.testing.assert_close(dev_temporal, host_temporal)

        # Conv
        for conv_idx in range(len(host.conv_buffer)):
            host_conv = host.conv_buffer[conv_idx][host_indices, layer_id, 0].to(DEVICE)
            dev_conv = device_pool.mamba_cache.conv[conv_idx][layer_id][device_indices]
            torch.testing.assert_close(dev_conv, host_conv)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_backup_load_roundtrip(dtype, layout):
    """Test D2H backup + H2D load roundtrip with io_backend='kernel'."""
    host = make_host_pool(dtype, layout)
    assert_host_mock_complete(host)
    device_pool = host.device_pool

    # Fill device with known data
    fill_device_data(device_pool, dtype)

    # Use a few indices for the test
    device_indices = torch.tensor([1, 5, 10], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    load_indices = torch.tensor([3, 7, 12], dtype=torch.int64, device=DEVICE)

    # --- Backup: device -> host (kernel) ---
    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    # --- Clear device buffers ---
    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    # --- Load: host -> device (kernel), per layer ---
    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool,
            host_indices,
            load_indices,
            layer_id,
            io_backend="kernel",
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)

    # Verify non-target positions remain zero (catch kernel writing wrong indices)
    all_indices = set(range(SIZE))
    target_set = set(load_indices.tolist())
    untouched = sorted(all_indices - target_set)
    if untouched:
        untouched_t = torch.tensor(untouched, dtype=torch.int64, device=DEVICE)
        for layer_id in range(NUM_LAYERS):
            assert (
                device_pool.mamba_cache.temporal[layer_id][untouched_t].abs().max() == 0
            )
            for conv_idx in range(len(device_pool.mamba_cache.conv)):
                assert (
                    device_pool.mamba_cache.conv[conv_idx][layer_id][untouched_t]
                    .abs()
                    .max()
                    == 0
                )


@pytest.mark.parametrize("dtype", DTYPES)
def test_unified_mamba_strided_backup_load_roundtrip(dtype):
    """Exercise direct D2H and H2D on unified envelope strides."""
    host = make_host_pool(dtype, "page_first")
    device_pool = make_unified_device_pool(dtype)
    bind_device_pool(host, device_pool)
    assert host.temporal_staging_buffer is None
    assert host.conv_staging_buffers[0] is None
    fill_device_data(device_pool, dtype)

    source_indices = torch.tensor([1, 5, 10], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([0, 3, 7], dtype=torch.int64)
    restore_indices = torch.tensor([2, 8, 12], dtype=torch.int64, device=DEVICE)
    expected_temporal = [
        device_pool.mamba_cache.temporal[layer_id][source_indices].clone()
        for layer_id in range(NUM_LAYERS)
    ]
    expected_conv = [
        device_pool.mamba_cache.conv[0][layer_id][source_indices].clone()
        for layer_id in range(NUM_LAYERS)
    ]
    untouched_temporal_envelope = device_pool._temporal_backing[:, :, [0, 2]].clone()
    untouched_conv_envelope = device_pool._conv_backing[0][:, :, [0, 2]].clone()

    host.backup_from_device_all_layer(
        device_pool, host_indices, source_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, source_indices)

    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id][restore_indices] = -3
        device_pool.mamba_cache.conv[0][layer_id][restore_indices] = -5
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
            device_pool.mamba_cache.temporal[layer_id][restore_indices],
            expected_temporal[layer_id],
        )
        torch.testing.assert_close(
            device_pool.mamba_cache.conv[0][layer_id][restore_indices],
            expected_conv[layer_id],
        )
    torch.testing.assert_close(
        device_pool._temporal_backing[:, :, [0, 2]], untouched_temporal_envelope
    )
    torch.testing.assert_close(
        device_pool._conv_backing[0][:, :, [0, 2]], untouched_conv_envelope
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_empty_indices(dtype, layout):
    """Test that empty indices are handled gracefully (no crash)."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    empty_device = torch.tensor([], dtype=torch.int64, device=DEVICE)
    empty_host = torch.tensor([], dtype=torch.int64)

    host.backup_from_device_all_layer(
        device_pool, empty_host, empty_device, io_backend="kernel"
    )
    torch.cuda.synchronize()
    # Host buffers should remain all zeros
    assert host.temporal_buffer.abs().max() == 0

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, empty_host, empty_device, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_single_item(dtype, layout):
    """Test single item backup + load."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    device_indices = torch.tensor([7], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([3], dtype=torch.int64)
    load_indices = torch.tensor([9], dtype=torch.int64, device=DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_full_indices(dtype, layout):
    """Test full-size backup + load (all SIZE items)."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    device_indices = torch.arange(SIZE, dtype=torch.int64, device=DEVICE)
    host_indices = torch.arange(SIZE, dtype=torch.int64)
    load_indices = torch.arange(SIZE, dtype=torch.int64, device=DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
