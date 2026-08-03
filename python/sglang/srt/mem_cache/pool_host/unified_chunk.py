"""Host-pool adapters for unified KV/Mamba typed chunks.

These adapters expose the existing tensor-index HiCache API while both logical
pools alias one pinned byte arena.  ``TypedChunkHostAllocator`` is the ownership
authority; the overlapping tensor views never allocate memory themselves.

The first implementation intentionally targets the unified Mamba configuration
qualified by this branch: CUDA/HIP kernel I/O, page-first host layout,
``page_size == 1``, and symmetric MHA K/V head dimensions.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
    transfer_hicache_all_layer_staged_lf_pf,
)
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.typed_chunk_host import (
    SharedTypedChunkHostArena,
    build_shared_kv_envelope_view,
    build_shared_mamba_envelope_views,
)


logger = logging.getLogger(__name__)


class UnifiedChunkMHAPoolHost(MHATokenToKVPoolHost):
    """MHA host view and KV-page suballocator over a typed-chunk arena."""

    def __init__(self, device_pool, arena: SharedTypedChunkHostArena):
        if getattr(device_pool, "page_size", 1) != 1:
            raise ValueError("typed-chunk HiCache currently requires page_size == 1")
        v_head_dim = getattr(device_pool, "v_head_dim", device_pool.head_dim)
        if v_head_dim != device_pool.head_dim:
            raise ValueError(
                "typed-chunk HiCache currently requires symmetric K/V head dims; "
                f"head_dim={device_pool.head_dim}, v_head_dim={v_head_dim}"
            )

        self.shared_arena = arena
        self.device_pool = device_pool
        self.page_size = 1
        self.layout = "page_first"
        self.pin_memory = arena.pin_memory
        self.device = arena.host_device
        self.allocator = arena.allocator
        self.dtype = device_pool.store_dtype
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.v_head_dim = v_head_dim
        self.layer_num = device_pool.layer_num
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.element_dim = self.head_num * self.head_dim
        self.token_stride_size = self.element_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        self.size_per_token = 2 * self.layer_num * self.token_stride_size
        self.size = arena.chunks.kv_capacity
        self.allocation_units_per_chunk = arena.chunks.kv_pages_per_chunk
        self.page_num = self.size
        self.kv_entry_bytes = self.size_per_token
        if self.kv_entry_bytes != arena.chunks.kv_page_bytes:
            raise ValueError(
                "typed-chunk KV page byte mismatch: "
                f"pool={self.kv_entry_bytes}, allocator={arena.chunks.kv_page_bytes}"
            )

        self.kv_buffer = build_shared_kv_envelope_view(
            arena.raw,
            num_tokens=self.size,
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_dim=self.head_dim,
            dtype=self.dtype,
        )
        self.k_data_refs = [self.k_buffer[:, layer] for layer in range(self.layer_num)]
        self.v_data_refs = [self.v_buffer[:, layer] for layer in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [view.data_ptr() for view in self.k_data_refs],
            dtype=torch.uint64,
            device=device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [view.data_ptr() for view in self.v_data_refs],
            dtype=torch.uint64,
            device=device_pool.device,
        )
        self.can_use_jit = can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )
        if not self.can_use_jit:
            raise RuntimeError(
                "typed-chunk HiCache requires the stride-aware HiCache JIT kernel"
            )
        # Keep tree-owned host indices on CPU.  D2H converts the small KV index
        # tensor to the accelerator; Mamba needs CPU indices to enqueue final
        # pinned-host DMA without a device-to-host synchronization.
        self.can_use_write_back_jit = can_use_write_back_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )
        if not self.can_use_write_back_jit:
            raise RuntimeError(
                "typed-chunk HiCache requires the staged write-back JIT kernel"
            )
        self.requires_host_indices_cpu = True
        self.staging_page_capacity = min(self.page_num, 64)
        self.staging_token_capacity = self.staging_page_capacity
        self.staging_k_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                self.head_num,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=device_pool.device,
        )
        self.staging_v_buffer = torch.empty_like(self.staging_k_buffer)
        # Unified/PageMajor device KV rows retain an explicit page axis even
        # when page_size == 1: [page, 1, head, dim].  Derive the staging row
        # from the real destination instead of assuming the legacy 3-D layout.
        self.load_staging_k_buffer = torch.empty(
            (self.staging_token_capacity, *device_pool.k_buffer[0].shape[1:]),
            dtype=self.dtype,
            device=device_pool.device,
        )
        self.load_staging_v_buffer = torch.empty(
            (self.staging_token_capacity, *device_pool.v_buffer[0].shape[1:]),
            dtype=self.dtype,
            device=device_pool.device,
        )
        self.fd = arena.fd
        self.lock = threading.RLock()

    def clear(self):
        self.shared_arena.clear()

    def destroy(self):
        self.shared_arena.destroy()

    def get_hybrid_pool_buffer(self):
        return self.shared_arena.get_hybrid_pool_buffer()

    def available_size(self):
        return self.shared_arena.available_kv_pages()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.shared_arena.alloc_kv(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.shared_arena.free_kv(indices)

    def transfer_chunk_ids(self, indices: torch.Tensor) -> list[int]:
        return self.shared_arena.chunks.kv_chunks(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel":
            raise ValueError(
                "typed-chunk HiCache currently supports io_backend='kernel' only"
            )
        if host_indices.numel() == 0:
            return
        host_indices_cpu = host_indices.to(device="cpu", dtype=torch.int64)
        transfer_hicache_all_layer_staged_lf_pf(
            k_ptr_src=device_pool.k_data_ptrs,
            v_ptr_src=device_pool.v_data_ptrs,
            src_indices=device_indices,
            dst_indices=host_indices_cpu,
            staging_k=self.staging_k_buffer,
            staging_v=self.staging_v_buffer,
            dst_k=self.k_buffer,
            dst_v=self.v_buffer,
            page_size=1,
            src_stride_bytes=(
                device_pool.k_buffer[0].stride(0)
                * device_pool.k_buffer[0].dtype.itemsize
            ),
            element_size=self.element_dim * self.dtype.itemsize,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend != "kernel":
            raise ValueError(
                "typed-chunk HiCache currently supports io_backend='kernel' only"
            )
        if host_indices.numel() == 0:
            return
        host_indices_cpu = host_indices.to(device="cpu", dtype=torch.int64)
        for begin in range(0, len(host_indices_cpu), self.staging_token_capacity):
            end = min(begin + self.staging_token_capacity, len(host_indices_cpu))
            count = end - begin
            for staging_index, host_index in enumerate(
                host_indices_cpu[begin:end].tolist()
            ):
                self.load_staging_k_buffer[staging_index].view(-1).copy_(
                    self.k_data_refs[layer_id][host_index].view(-1),
                    non_blocking=True,
                )
                self.load_staging_v_buffer[staging_index].view(-1).copy_(
                    self.v_data_refs[layer_id][host_index].view(-1),
                    non_blocking=True,
                )
            dst = device_indices[begin:end]
            device_pool.k_buffer[layer_id].index_copy_(
                0, dst, self.load_staging_k_buffer[:count]
            )
            device_pool.v_buffer[layer_id].index_copy_(
                0, dst, self.load_staging_v_buffer[:count]
            )

    def get_page_buffer_meta(self, indices):
        # One KV token is one contiguous [L0 K,V ... Ln K,V] envelope.
        ptrs = [
            self.shared_arena.raw.data_ptr() + int(index) * self.kv_entry_bytes
            for index in indices[:: self.page_size].tolist()
        ]
        return ptrs, [self.kv_entry_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return (
            self.shared_arena.raw.data_ptr() % page_size_bytes == 0
            and self.kv_entry_bytes % page_size_bytes == 0
        )


class UnifiedChunkMambaPoolHost(MambaPoolHost):
    """One-Mamba-slot-per-chunk host adapter."""

    def __init__(self, device_pool, arena: SharedTypedChunkHostArena):
        self.shared_arena = arena
        self.device_pool = device_pool
        self.page_size = 1
        self.layout = "page_first"
        self.pin_memory = arena.pin_memory
        self.device = arena.host_device
        self.allocator = arena.allocator
        self.num_mamba_layers = device_pool.num_mamba_layers
        self.conv_state_shapes = [
            tuple(conv_state.shape[2:]) for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = tuple(device_pool.mamba_cache.temporal.shape[2:])
        self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
        self.conv_state_elem_sizes = [
            int(np.prod(shape)) for shape in self.conv_state_shapes
        ]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()
        if self.size_per_token != arena.chunks.mamba_slot_bytes:
            raise ValueError(
                "typed-chunk Mamba slot byte mismatch: "
                f"pool={self.size_per_token}, "
                f"allocator={arena.chunks.mamba_slot_bytes}"
            )
        self.size = arena.chunks.mamba_capacity
        self.allocation_units_per_chunk = 1
        self.page_num = self.size
        self.temporal_buffer, self.conv_buffer = build_shared_mamba_envelope_views(
            arena.raw,
            num_chunks=arena.chunks.num_chunks,
            chunk_bytes=arena.chunks.chunk_bytes,
            layer_num=self.num_mamba_layers,
            temporal_shape=self.temporal_state_shape,
            temporal_dtype=self.temporal_dtype,
            conv_shapes=tuple(self.conv_state_shapes),
            conv_dtype=self.conv_dtype,
        )
        self.temporal_device_ptrs = torch.tensor(
            [
                device_pool.mamba_cache.temporal[layer].data_ptr()
                for layer in range(self.num_mamba_layers)
            ],
            dtype=torch.uint64,
            device=device_pool.device,
        )
        self.conv_device_ptrs = [
            torch.tensor(
                [
                    conv_state[layer].data_ptr()
                    for layer in range(self.num_mamba_layers)
                ],
                dtype=torch.uint64,
                device=device_pool.device,
            )
            for conv_state in device_pool.mamba_cache.conv
        ]
        self._init_write_back_staging_buffers()
        self.requires_host_indices_cpu = True
        self.fd = arena.fd
        self.lock = threading.RLock()

    def clear(self):
        self.shared_arena.clear()

    def destroy(self):
        # The anchor KV pool owns arena teardown.
        pass

    def get_hybrid_pool_buffer(self):
        # The anchor registers the same raw arena exactly once.
        return []

    def available_size(self):
        return self.shared_arena.available_mamba_slots()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.shared_arena.alloc_mamba(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.shared_arena.free_mamba(indices)

    def transfer_chunk_ids(self, indices: torch.Tensor) -> list[int]:
        return self.shared_arena.chunks.mamba_chunks(indices)

    def get_page_buffer_meta(self, indices):
        indices_cpu = indices.to(device="cpu", dtype=torch.int64).tolist()
        ptr_list = []
        sizes = []
        components = []
        if self.temporal_state_elem_size > 0:
            components.append(self.temporal_buffer)
        components.extend(self.conv_buffer)
        for index in indices_cpu:
            for component in components:
                ptr_list.append(
                    component.data_ptr()
                    + index * component.stride(0) * component.element_size()
                )
                sizes.append(component[0].numel() * component.element_size())
        return ptr_list, sizes

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        components = [self.temporal_buffer, *self.conv_buffer]
        return all(
            component.data_ptr() % page_size_bytes == 0
            and component.stride(0) * component.element_size() % page_size_bytes == 0
            for component in components
            if component.numel() > 0
        )


def build_unified_chunk_host_pools(
    *,
    kv_pool,
    mamba_pool,
    hicache_ratio: float,
    hicache_size_gb: float,
    allocator_type: str,
    pin_memory: bool = True,
) -> tuple[UnifiedChunkMHAPoolHost, UnifiedChunkMambaPoolHost]:
    """Create the two adapters over one total host-memory budget."""

    unified_buffer = getattr(kv_pool, "_unified_buffer", None)
    if (
        unified_buffer is None
        or getattr(mamba_pool, "_unified_buffer", None) is not unified_buffer
    ):
        raise ValueError("typed-chunk host pools require one shared unified GPU buffer")
    if getattr(kv_pool, "page_size", 1) != 1:
        raise ValueError("typed-chunk HiCache currently requires page_size == 1")

    head_dim = kv_pool.head_dim
    v_head_dim = getattr(kv_pool, "v_head_dim", head_dim)
    if head_dim != v_head_dim:
        raise ValueError("typed-chunk HiCache does not yet support asymmetric K/V")
    kv_page_bytes = (
        2
        * kv_pool.layer_num
        * kv_pool.head_num
        * head_dim
        * kv_pool.store_dtype.itemsize
    )
    conv_bytes = sum(
        int(np.prod(tensor.shape[2:])) * tensor.dtype.itemsize
        for tensor in mamba_pool.mamba_cache.conv
    )
    temporal = mamba_pool.mamba_cache.temporal
    temporal_bytes = int(np.prod(temporal.shape[2:])) * temporal.dtype.itemsize
    mamba_slot_bytes = mamba_pool.num_mamba_layers * (conv_bytes + temporal_bytes)

    total_bytes = (
        int(hicache_size_gb * 1e9)
        if hicache_size_gb > 0
        else int(unified_buffer.total_bytes * hicache_ratio)
    )
    arena = SharedTypedChunkHostArena(
        total_bytes=total_bytes,
        kv_page_bytes=kv_page_bytes,
        mamba_slot_bytes=mamba_slot_bytes,
        host_device="cpu",
        accelerator_device=kv_pool.device,
        pin_memory=pin_memory,
        allocator_type=allocator_type,
    )
    return (
        UnifiedChunkMHAPoolHost(kv_pool, arena),
        UnifiedChunkMambaPoolHost(mamba_pool, arena),
    )
