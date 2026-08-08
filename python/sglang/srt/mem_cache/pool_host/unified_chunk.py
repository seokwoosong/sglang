"""Host-pool adapters for unified KV/Mamba typed chunks.

These adapters expose the existing tensor-index HiCache API while both logical
pools alias one pinned byte arena.  ``TypedChunkHostAllocator`` is the ownership
authority; the overlapping tensor views never allocate memory themselves.

The adapters target unified Mamba with CUDA/HIP kernel I/O, a page-first host
layout, and symmetric MHA K/V head dimensions.  A typed L2 KV page has exactly
the same byte envelope as one unified L1 KV page for every supported page size.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_page_copy_kernel,
    transfer_hicache_pages,
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
        v_head_dim = getattr(device_pool, "v_head_dim", device_pool.head_dim)
        if v_head_dim != device_pool.head_dim:
            raise ValueError(
                "typed-chunk HiCache currently requires symmetric K/V head dims; "
                f"head_dim={device_pool.head_dim}, v_head_dim={v_head_dim}"
            )

        self.shared_arena = arena
        self.device_pool = device_pool
        self.page_size = int(getattr(device_pool, "page_size", 1))
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
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
        self.page_num = arena.chunks.kv_capacity
        self.size = self.page_num * self.page_size
        self.allocation_units_per_chunk = (
            arena.chunks.kv_pages_per_chunk * self.page_size
        )
        self.kv_page_bytes = self.size_per_token * self.page_size
        if self.kv_page_bytes != arena.chunks.kv_page_bytes:
            raise ValueError(
                "typed-chunk KV page byte mismatch: "
                f"pool={self.kv_page_bytes}, allocator={arena.chunks.kv_page_bytes}"
            )

        self.kv_buffer = build_shared_kv_envelope_view(
            arena.raw,
            num_pages=self.page_num,
            page_size=self.page_size,
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
        self.layer_page_bytes = (
            2 * self.page_size * self.element_dim * self.dtype.itemsize
        )
        self.can_use_jit = can_use_hicache_page_copy_kernel(
            page_stride_bytes=self.kv_page_bytes,
            copy_bytes=self.layer_page_bytes,
        )
        if not self.can_use_jit:
            raise RuntimeError("typed-chunk HiCache requires the raw page-copy kernel")
        # Keep tree-owned host indices on CPU.  D2H converts the small KV index
        # tensor to the accelerator; Mamba needs CPU indices to enqueue final
        # pinned-host DMA without a device-to-host synchronization.
        self.can_use_write_back_jit = True
        self.requires_host_indices_cpu = True
        self.staging_page_capacity = self.page_num
        self.staging_token_capacity = self.size
        self._load_host_pages_device = None
        self._load_device_pages = None
        self.fd = arena.fd
        self.lock = threading.RLock()

    def clear(self):
        self.shared_arena.clear()

    def destroy(self):
        self.shared_arena.destroy()

    def get_hybrid_pool_buffer(self):
        return self.shared_arena.get_hybrid_pool_buffer()

    def available_size(self):
        return self.shared_arena.available_kv_pages() * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size:
            raise ValueError(
                f"KV allocation must be a multiple of page_size={self.page_size}, "
                f"got {need_size}"
            )
        pages = self.shared_arena.alloc_kv(need_size // self.page_size)
        if pages is None:
            return None
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        return (pages[:, None] * self.page_size + offsets).reshape(-1)

    def free(self, indices: torch.Tensor) -> int:
        pages = self._host_page_ids(indices)
        self.shared_arena.free_kv(pages)
        return int(indices.numel())

    def transfer_chunk_ids(self, indices: torch.Tensor) -> list[int]:
        return self.shared_arena.chunks.kv_chunks(self._host_page_ids(indices))

    def _host_page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if indices_cpu.numel() % self.page_size:
            raise ValueError(
                f"KV indices must contain whole pages of {self.page_size} tokens"
            )
        if indices_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        groups = indices_cpu.reshape(-1, self.page_size)
        starts = groups[:, 0]
        expected = starts[:, None] + torch.arange(self.page_size, dtype=torch.int64)
        if torch.any(starts % self.page_size) or not torch.equal(groups, expected):
            raise ValueError("KV host indices must be aligned, contiguous full pages")
        return starts // self.page_size

    def _device_page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.page_size:
            raise ValueError(
                f"KV device indices must contain whole pages of {self.page_size} tokens"
            )
        return indices.reshape(-1, self.page_size)[:, 0] // self.page_size

    @staticmethod
    def _device_raw(device_pool) -> torch.Tensor:
        unified = getattr(device_pool, "_unified_buffer", None)
        raw = getattr(unified, "_raw", None)
        if raw is None:
            raise RuntimeError("typed-chunk HiCache requires unified L1 raw storage")
        return raw

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel":
            raise ValueError(
                "typed-chunk HiCache currently supports io_backend='kernel' only"
            )
        if host_indices.numel() == 0:
            return
        host_pages = self._host_page_ids(host_indices)
        device_pages = self._device_page_ids(device_indices)
        device_raw = self._device_raw(device_pool)
        host_pages_device = host_pages.to(device=device_raw.device, non_blocking=True)
        transfer_hicache_pages(
            self.shared_arena.raw,
            host_pages_device,
            device_raw,
            device_pages,
            page_stride_bytes=self.kv_page_bytes,
        )
        host_pages_device.record_stream(torch.cuda.current_stream())
        device_pages.record_stream(torch.cuda.current_stream())

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
        device_raw = self._device_raw(device_pool)
        if layer_id == 0:
            host_pages = self._host_page_ids(host_indices)
            self._load_host_pages_device = host_pages.to(
                device=device_raw.device, non_blocking=True
            )
            self._load_device_pages = self._device_page_ids(device_indices)
        if self._load_host_pages_device is None or self._load_device_pages is None:
            raise RuntimeError("typed-chunk H2D must begin with layer 0")
        layer_offset = layer_id * self.layer_page_bytes
        transfer_hicache_pages(
            device_raw,
            self._load_device_pages,
            self.shared_arena.raw,
            self._load_host_pages_device,
            page_stride_bytes=self.kv_page_bytes,
            copy_offset_bytes=layer_offset,
            copy_bytes=self.layer_page_bytes,
        )
        stream = torch.cuda.current_stream()
        self._load_host_pages_device.record_stream(stream)
        self._load_device_pages.record_stream(stream)

    def get_page_buffer_meta(self, indices):
        pages = self._host_page_ids(indices)
        ptrs = [
            self.shared_arena.raw.data_ptr() + int(page) * self.kv_page_bytes
            for page in pages.tolist()
        ]
        return ptrs, [self.kv_page_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return (
            self.shared_arena.raw.data_ptr() % page_size_bytes == 0
            and self.kv_page_bytes % page_size_bytes == 0
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

    @staticmethod
    def _device_raw(device_pool) -> torch.Tensor:
        unified = getattr(device_pool, "_unified_buffer", None)
        raw = getattr(unified, "_raw", None)
        if raw is None:
            raise RuntimeError("typed-chunk HiCache requires unified L1 raw storage")
        return raw

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if io_backend != "kernel":
            raise ValueError(
                "typed-chunk HiCache currently supports io_backend='kernel' only"
            )
        if host_indices.numel() == 0:
            return
        device_raw = self._device_raw(device_pool)
        host_slots = host_indices.to(device=device_raw.device, non_blocking=True)
        if device_indices.device != device_raw.device:
            device_indices = device_indices.to(device_raw.device, non_blocking=True)
        transfer_hicache_pages(
            self.shared_arena.raw,
            host_slots,
            device_raw,
            device_indices,
            dst_page_stride_bytes=self.shared_arena.chunks.chunk_bytes,
            src_page_stride_bytes=self.size_per_token,
            copy_bytes=self.size_per_token,
        )
        stream = torch.cuda.current_stream()
        host_slots.record_stream(stream)
        device_indices.record_stream(stream)

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
    head_dim = kv_pool.head_dim
    v_head_dim = getattr(kv_pool, "v_head_dim", head_dim)
    if head_dim != v_head_dim:
        raise ValueError("typed-chunk HiCache does not yet support asymmetric K/V")
    kv_page_bytes = (
        getattr(kv_pool, "page_size", 1)
        * 2
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
