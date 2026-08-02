"""CUDA correctness tests for HiCache transfers on unified strided MHA views."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
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
