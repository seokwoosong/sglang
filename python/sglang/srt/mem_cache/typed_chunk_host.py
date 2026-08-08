"""Typed-chunk ownership for a shared HiCache host-memory budget.

The allocator deliberately owns *metadata only*.  A caller allocates one raw
host arena of :attr:`usable_bytes` and builds KV/Mamba views over that arena.
Chunk ownership is what makes those overlapping views safe:

* one Mamba state occupies exactly one chunk;
* one KV chunk contains an integral number of KV pages;
* an empty, unpinned chunk may be retyped without moving host data.

Keeping the allocator independent of CUDA and tensor layouts makes its safety
invariants testable on CPU before it is wired into the transfer path.
"""

from __future__ import annotations

import enum
import heapq
import logging
import threading
from collections import defaultdict
from typing import Any, Iterable, Optional

import psutil
import torch

from sglang.srt.mem_cache.hicache_trace import (
    hicache_trace_object_id,
    trace_enabled,
    trace_hicache_event,
)
from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()


class HostChunkOwner(enum.IntEnum):
    FREE = 0
    KV = 1
    MAMBA = 2


class TypedChunkHostAllocator:
    """Allocate KV pages and one-slot Mamba chunks from one byte budget.

    ``chunk_bytes`` is rounded up to an integral number of KV pages.  This
    preserves the ``1 chunk == 1 Mamba slot`` policy while ensuring that a KV
    chunk has no unusable tail.  Padding, if any, belongs only to the Mamba
    representation and is bounded by one KV page.

    Host indices intentionally remain plain int64 tensors:

    * KV index = ``chunk_id * kv_pages_per_chunk + page_offset``
    * Mamba index = ``chunk_id``

    This lets the existing radix/cache-controller metadata continue to carry
    tensors while physical host layouts migrate to the shared arena.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        kv_page_bytes: int,
        mamba_slot_bytes: int,
    ) -> None:
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        if kv_page_bytes <= 0:
            raise ValueError(f"kv_page_bytes must be positive, got {kv_page_bytes}")
        if mamba_slot_bytes <= 0:
            raise ValueError(
                f"mamba_slot_bytes must be positive, got {mamba_slot_bytes}"
            )

        self.total_bytes = int(total_bytes)
        self.kv_page_bytes = int(kv_page_bytes)
        self.mamba_slot_bytes = int(mamba_slot_bytes)
        self.kv_pages_per_chunk = max(
            1,
            (self.mamba_slot_bytes + self.kv_page_bytes - 1) // self.kv_page_bytes,
        )
        self.chunk_bytes = self.kv_pages_per_chunk * self.kv_page_bytes
        self.num_chunks = self.total_bytes // self.chunk_bytes
        if self.num_chunks <= 0:
            raise ValueError(
                "HiCache host budget cannot hold one typed chunk: "
                f"total_bytes={self.total_bytes}, chunk_bytes={self.chunk_bytes}, "
                f"kv_page_bytes={self.kv_page_bytes}, "
                f"mamba_slot_bytes={self.mamba_slot_bytes}"
            )
        self.usable_bytes = self.num_chunks * self.chunk_bytes
        self.unused_budget_bytes = self.total_bytes - self.usable_bytes
        self.mamba_padding_bytes = self.chunk_bytes - self.mamba_slot_bytes
        self.kv_capacity = self.num_chunks * self.kv_pages_per_chunk
        self.mamba_capacity = self.num_chunks

        self._lock = threading.RLock()
        self.clear()
        trace_hicache_event(
            "l2_arena_initialized",
            total_bytes=self.total_bytes,
            usable_bytes=self.usable_bytes,
            unused_budget_bytes=self.unused_budget_bytes,
            chunk_bytes=self.chunk_bytes,
            num_chunks=self.num_chunks,
            kv_page_bytes=self.kv_page_bytes,
            kv_pages_per_chunk=self.kv_pages_per_chunk,
            mamba_slot_bytes=self.mamba_slot_bytes,
            mamba_padding_bytes=self.mamba_padding_bytes,
        )

    def _chunk_trace_state(self, chunk_id: int) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "owner": self._owners[chunk_id].name,
            "kv_used_offsets": sorted(self._kv_used.get(chunk_id, ())),
            "pin_count": self._chunk_pin_count[chunk_id],
            "pending_release": self._pending_release[chunk_id],
        }

    def _trace_chunks(
        self, event: str, chunk_ids: Iterable[int], **fields: Any
    ) -> None:
        if not trace_enabled():
            return
        normalized = sorted(set(int(chunk_id) for chunk_id in chunk_ids))
        trace_hicache_event(
            event,
            chunks=[self._chunk_trace_state(chunk_id) for chunk_id in normalized],
            free_chunks=len(self._free_chunk_set),
            **fields,
        )

    def _set_owner(self, chunk_id: int, owner: HostChunkOwner, *, reason: str) -> None:
        previous = self._owners[chunk_id]
        self._owners[chunk_id] = owner
        if previous != owner:
            trace_hicache_event(
                "l2_chunk_owner_changed",
                chunk_id=chunk_id,
                previous_owner=previous.name,
                owner=owner.name,
                reason=reason,
            )

    def clear(self) -> None:
        with self._lock:
            self._owners = [HostChunkOwner.FREE] * self.num_chunks
            self._free_chunks_low = list(range(self.num_chunks))
            self._free_chunks_high = [-chunk_id for chunk_id in range(self.num_chunks)]
            heapq.heapify(self._free_chunks_low)
            heapq.heapify(self._free_chunks_high)
            self._free_chunk_set = set(range(self.num_chunks))
            self._kv_used: dict[int, set[int]] = {}
            self._chunk_pin_count = [0] * self.num_chunks
            self._pending_release = [False] * self.num_chunks
            trace_hicache_event(
                "l2_arena_cleared",
                num_chunks=self.num_chunks,
                free_chunks=self.num_chunks,
            )

    def _rebuild_free_heaps_if_needed(self) -> None:
        # Each heap keeps stale entries popped by the opposite end.  Rebuild
        # periodically so repeated allocate/free cycles cannot grow metadata
        # without bound.
        max_heap_entries = max(self.num_chunks * 2, 16)
        if len(self._free_chunks_low) > max_heap_entries:
            self._free_chunks_low = list(self._free_chunk_set)
            heapq.heapify(self._free_chunks_low)
        if len(self._free_chunks_high) > max_heap_entries:
            self._free_chunks_high = [-chunk_id for chunk_id in self._free_chunk_set]
            heapq.heapify(self._free_chunks_high)

    def _pop_free_chunk(self, *, high: bool = False) -> Optional[int]:
        self._rebuild_free_heaps_if_needed()
        heap = self._free_chunks_high if high else self._free_chunks_low
        while heap:
            value = heapq.heappop(heap)
            chunk_id = -value if high else value
            if chunk_id in self._free_chunk_set:
                self._free_chunk_set.remove(chunk_id)
                assert self._owners[chunk_id] == HostChunkOwner.FREE
                return chunk_id
        return None

    def _release_chunk(self, chunk_id: int) -> None:
        assert self._chunk_pin_count[chunk_id] == 0
        self._set_owner(chunk_id, HostChunkOwner.FREE, reason="empty_unpinned")
        self._pending_release[chunk_id] = False
        if chunk_id not in self._free_chunk_set:
            self._free_chunk_set.add(chunk_id)
            heapq.heappush(self._free_chunks_low, chunk_id)
            heapq.heappush(self._free_chunks_high, -chunk_id)

    def available_kv_pages(self) -> int:
        with self._lock:
            partial = sum(
                self.kv_pages_per_chunk - len(used)
                for chunk_id, used in self._kv_used.items()
                if self._chunk_pin_count[chunk_id] == 0
            )
            return partial + len(self._free_chunk_set) * self.kv_pages_per_chunk

    def available_mamba_slots(self) -> int:
        with self._lock:
            return len(self._free_chunk_set)

    def alloc_kv(self, num_pages: int) -> Optional[torch.Tensor]:
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages == 0:
            return torch.empty(0, dtype=torch.int64)

        with self._lock:
            if num_pages > self.available_kv_pages():
                return None

            result: list[int] = []
            remaining = num_pages

            # Fill already-typed chunks first.  Sorting keeps allocation and
            # tests deterministic without imposing ordering on cache callers.
            for chunk_id in sorted(self._kv_used):
                used = self._kv_used[chunk_id]
                # A transfer pin protects the byte ranges, not merely the
                # KV/Mamba type tag. Reusing a just-freed KV offset in a pinned
                # chunk would overwrite an in-flight H2D source or D2H target.
                if (
                    self._chunk_pin_count[chunk_id] > 0
                    or len(used) == self.kv_pages_per_chunk
                ):
                    continue
                for offset in range(self.kv_pages_per_chunk):
                    if offset in used:
                        continue
                    used.add(offset)
                    result.append(chunk_id * self.kv_pages_per_chunk + offset)
                    remaining -= 1
                    if remaining == 0:
                        self._trace_chunks(
                            "l2_kv_allocated",
                            (index // self.kv_pages_per_chunk for index in result),
                            requested_pages=num_pages,
                            host_indices=result,
                        )
                        return torch.tensor(result, dtype=torch.int64)

            while remaining:
                chunk_id = self._pop_free_chunk()
                assert chunk_id is not None  # guarded by available_kv_pages()
                self._set_owner(chunk_id, HostChunkOwner.KV, reason="kv_allocation")
                used: set[int] = set()
                self._kv_used[chunk_id] = used
                take = min(remaining, self.kv_pages_per_chunk)
                for offset in range(take):
                    used.add(offset)
                    result.append(chunk_id * self.kv_pages_per_chunk + offset)
                remaining -= take

            self._trace_chunks(
                "l2_kv_allocated",
                (index // self.kv_pages_per_chunk for index in result),
                requested_pages=num_pages,
                host_indices=result,
            )
            return torch.tensor(result, dtype=torch.int64)

    def free_kv(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if indices_cpu.numel() == 0:
            return 0

        values = indices_cpu.tolist()
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate KV host indices in free: {values}")

        by_chunk: dict[int, list[int]] = defaultdict(list)
        for index in values:
            if not 0 <= index < self.kv_capacity:
                raise AssertionError(
                    f"KV host index {index} is outside [0, {self.kv_capacity})"
                )
            chunk_id, offset = divmod(index, self.kv_pages_per_chunk)
            by_chunk[chunk_id].append(offset)

        with self._lock:
            for chunk_id, offsets in by_chunk.items():
                if self._owners[chunk_id] != HostChunkOwner.KV:
                    raise AssertionError(
                        f"KV free targets chunk {chunk_id} owned by "
                        f"{self._owners[chunk_id].name}"
                    )
                if self._pending_release[chunk_id] or chunk_id not in self._kv_used:
                    raise AssertionError(
                        f"KV chunk {chunk_id} was already freed while transfer-pinned"
                    )
                used = self._kv_used[chunk_id]
                missing = [offset for offset in offsets if offset not in used]
                if missing:
                    raise AssertionError(
                        f"double-free or unallocated KV offsets in chunk "
                        f"{chunk_id}: {missing}"
                    )

            for chunk_id, offsets in by_chunk.items():
                used = self._kv_used[chunk_id]
                used.difference_update(offsets)
                if not used:
                    del self._kv_used[chunk_id]
                    if self._chunk_pin_count[chunk_id]:
                        self._pending_release[chunk_id] = True
                    else:
                        self._release_chunk(chunk_id)
            self._trace_chunks(
                "l2_kv_freed",
                by_chunk,
                freed_pages=len(values),
                host_indices=values,
            )
            return len(values)

    def alloc_mamba(self, num_slots: int) -> Optional[torch.Tensor]:
        if num_slots < 0:
            raise ValueError(f"num_slots must be non-negative, got {num_slots}")
        if num_slots == 0:
            return torch.empty(0, dtype=torch.int64)

        with self._lock:
            if num_slots > len(self._free_chunk_set):
                return None
            result = []
            for _ in range(num_slots):
                chunk_id = self._pop_free_chunk(high=True)
                assert chunk_id is not None
                self._set_owner(
                    chunk_id, HostChunkOwner.MAMBA, reason="mamba_allocation"
                )
                result.append(chunk_id)
            self._trace_chunks(
                "l2_mamba_allocated",
                result,
                requested_slots=num_slots,
                host_indices=result,
            )
            return torch.tensor(result, dtype=torch.int64)

    def free_mamba(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if indices_cpu.numel() == 0:
            return 0
        values = indices_cpu.tolist()
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate Mamba host indices in free: {values}")

        with self._lock:
            for chunk_id in values:
                if not 0 <= chunk_id < self.num_chunks:
                    raise AssertionError(
                        f"Mamba chunk {chunk_id} is outside [0, {self.num_chunks})"
                    )
                if self._owners[chunk_id] != HostChunkOwner.MAMBA:
                    raise AssertionError(
                        f"Mamba free targets chunk {chunk_id} owned by "
                        f"{self._owners[chunk_id].name}"
                    )
                if self._pending_release[chunk_id]:
                    raise AssertionError(
                        f"Mamba chunk {chunk_id} was already freed while transfer-pinned"
                    )
            for chunk_id in values:
                if self._chunk_pin_count[chunk_id]:
                    self._pending_release[chunk_id] = True
                else:
                    self._release_chunk(chunk_id)
            self._trace_chunks(
                "l2_mamba_freed",
                values,
                freed_slots=len(values),
                host_indices=values,
            )
            return len(values)

    def _normalize_chunks(self, chunk_ids: Iterable[int]) -> list[int]:
        values = [int(x) for x in chunk_ids]
        unique = sorted(set(values))
        for chunk_id in unique:
            if not 0 <= chunk_id < self.num_chunks:
                raise AssertionError(
                    f"chunk {chunk_id} is outside [0, {self.num_chunks})"
                )
            if self._owners[chunk_id] == HostChunkOwner.FREE:
                raise AssertionError(f"cannot pin free chunk {chunk_id}")
            if self._pending_release[chunk_id]:
                raise AssertionError(f"cannot pin already-freed chunk {chunk_id}")
        return unique

    def kv_chunks(self, indices: torch.Tensor) -> list[int]:
        values = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        return sorted(
            {int(index) // self.kv_pages_per_chunk for index in values.tolist()}
        )

    def mamba_chunks(self, indices: torch.Tensor) -> list[int]:
        values = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        return sorted(set(int(index) for index in values.tolist()))

    def pin_chunks(self, chunk_ids: Iterable[int]) -> None:
        with self._lock:
            normalized = self._normalize_chunks(chunk_ids)
            for chunk_id in normalized:
                self._chunk_pin_count[chunk_id] += 1
            self._trace_chunks("l2_chunks_pinned", normalized)

    def unpin_chunks(self, chunk_ids: Iterable[int]) -> None:
        with self._lock:
            normalized = sorted(set(int(x) for x in chunk_ids))
            for chunk_id in normalized:
                if not 0 <= chunk_id < self.num_chunks:
                    raise AssertionError(
                        f"chunk {chunk_id} is outside [0, {self.num_chunks})"
                    )
                if self._chunk_pin_count[chunk_id] <= 0:
                    raise AssertionError(f"unbalanced unpin for chunk {chunk_id}")
                self._chunk_pin_count[chunk_id] -= 1
                if (
                    self._chunk_pin_count[chunk_id] == 0
                    and self._pending_release[chunk_id]
                ):
                    self._release_chunk(chunk_id)
            self._trace_chunks("l2_chunks_unpinned", normalized)

    def owner(self, chunk_id: int) -> HostChunkOwner:
        with self._lock:
            return self._owners[chunk_id]

    def owner_counts(self) -> dict[HostChunkOwner, int]:
        with self._lock:
            return {owner: self._owners.count(owner) for owner in HostChunkOwner}

    def assert_consistent(self) -> None:
        with self._lock:
            free_from_owner = {
                i
                for i, owner in enumerate(self._owners)
                if owner == HostChunkOwner.FREE
            }
            assert free_from_owner == self._free_chunk_set
            assert set(self._kv_used) == {
                i
                for i, owner in enumerate(self._owners)
                if owner == HostChunkOwner.KV and not self._pending_release[i]
            }
            for chunk_id, used in self._kv_used.items():
                assert used
                assert all(0 <= offset < self.kv_pages_per_chunk for offset in used)
                assert not self._pending_release[chunk_id]
            for chunk_id, pending in enumerate(self._pending_release):
                if pending:
                    assert self._chunk_pin_count[chunk_id] > 0
                    assert self._owners[chunk_id] != HostChunkOwner.FREE


class SharedTypedChunkHostArena:
    """One pinned byte allocation shared by KV and Mamba host views.

    The arena is intentionally layout-agnostic.  Pool adapters build
    overlapping strided views over :attr:`raw`; :attr:`chunks` is the single
    authority that prevents different types from owning the same bytes.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        kv_page_bytes: int,
        mamba_slot_bytes: int,
        host_device: str,
        accelerator_device: str,
        pin_memory: bool,
        allocator_type: str,
    ) -> None:
        self.chunks = TypedChunkHostAllocator(
            total_bytes=total_bytes,
            kv_page_bytes=kv_page_bytes,
            mamba_slot_bytes=mamba_slot_bytes,
        )
        available_bytes = (
            psutil.virtual_memory().available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        )
        if self.chunks.usable_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory for shared typed-chunk HiCache: "
                f"requesting {self.chunks.usable_bytes / 1e9:.2f} GB but only "
                f"{available_bytes / 1e9:.2f} GB is available after reserve"
            )

        self.host_device = host_device
        self.accelerator_device = accelerator_device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        alloc_func = ALLOC_MEMORY_FUNCS[accelerator_device]
        self.raw = alloc_func(
            (self.chunks.usable_bytes,),
            dtype=torch.uint8,
            device=host_device,
            pin_memory=pin_memory,
            allocator=self.allocator,
        )
        self.fd = getattr(self.allocator, "fd", None)
        self._destroyed = False
        self._transfer_event_lock = threading.RLock()
        self._pending_transfer_events: list[tuple[Any, tuple[int, ...]]] = []
        logger.info(
            "Allocated shared typed-chunk HiCache arena: requested=%.3f GB, "
            "usable=%.3f GB, chunks=%d, chunk=%.3f MB, KV-pages/chunk=%d, "
            "Mamba-padding/chunk=%.3f MB",
            total_bytes / 1e9,
            self.chunks.usable_bytes / 1e9,
            self.chunks.num_chunks,
            self.chunks.chunk_bytes / 1e6,
            self.chunks.kv_pages_per_chunk,
            self.chunks.mamba_padding_bytes / 1e6,
        )

    def _drain_transfer_events(self, *, synchronize: bool = False) -> None:
        """Release chunks whose asynchronous transfer has completed.

        A host chunk can be freed by radix metadata immediately after a D2H/H2D
        operation is enqueued.  The allocator must nevertheless keep the chunk
        typed (and therefore non-reusable) until the CUDA event fires.  Normal
        allocation/free calls poll events without blocking; clear/destroy use
        the synchronous path because they invalidate the whole arena.
        """

        with self._transfer_event_lock:
            pending = self._pending_transfer_events
            self._pending_transfer_events = []
            still_pending = []
            completed = []
            for event, chunk_ids in pending:
                if synchronize:
                    event.synchronize()
                    completed.append((event, chunk_ids))
                elif event.query():
                    completed.append((event, chunk_ids))
                else:
                    still_pending.append((event, chunk_ids))
            self._pending_transfer_events.extend(still_pending)
            for event, chunk_ids in completed:
                self.chunks.unpin_chunks(chunk_ids)
                trace_hicache_event(
                    "l2_transfer_pin_released",
                    cuda_event_id=hicache_trace_object_id(event),
                    chunk_ids=chunk_ids,
                    synchronized=synchronize,
                )

    def pin_chunks_for_transfer(self, chunk_ids: Iterable[int]) -> tuple[int, ...]:
        """Pin allocated chunks before enqueueing an asynchronous transfer."""

        normalized = tuple(sorted(set(int(x) for x in chunk_ids)))
        if not normalized:
            return normalized
        self._drain_transfer_events()
        self.chunks.pin_chunks(normalized)
        return normalized

    def release_chunks_after_event(self, chunk_ids: Iterable[int], event: Any) -> None:
        """Defer a matching unpin until ``event`` reports completion."""

        normalized = tuple(sorted(set(int(x) for x in chunk_ids)))
        if not normalized:
            return
        with self._transfer_event_lock:
            self._pending_transfer_events.append((event, normalized))
        trace_hicache_event(
            "l2_transfer_pin_armed",
            cuda_event_id=hicache_trace_object_id(event),
            chunk_ids=normalized,
        )

    def cancel_transfer_pin(self, chunk_ids: Iterable[int]) -> None:
        """Undo a pre-enqueue pin when scheduling the transfer raises."""

        normalized = tuple(sorted(set(int(x) for x in chunk_ids)))
        if normalized:
            self.chunks.unpin_chunks(normalized)

    def available_kv_pages(self) -> int:
        self._drain_transfer_events()
        return self.chunks.available_kv_pages()

    def available_mamba_slots(self) -> int:
        self._drain_transfer_events()
        return self.chunks.available_mamba_slots()

    def alloc_kv(self, num_pages: int) -> Optional[torch.Tensor]:
        self._drain_transfer_events()
        return self.chunks.alloc_kv(num_pages)

    def alloc_mamba(self, num_slots: int) -> Optional[torch.Tensor]:
        self._drain_transfer_events()
        return self.chunks.alloc_mamba(num_slots)

    def free_kv(self, indices: torch.Tensor) -> int:
        self._drain_transfer_events()
        return self.chunks.free_kv(indices)

    def free_mamba(self, indices: torch.Tensor) -> int:
        self._drain_transfer_events()
        return self.chunks.free_mamba(indices)

    def clear(self) -> None:
        self._drain_transfer_events(synchronize=True)
        self.chunks.clear()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._drain_transfer_events(synchronize=True)
        if self.pin_memory and (_is_cuda or _is_hip) and self.raw is not None:
            _cuda_host_unregister(self.raw)
        self.raw = None

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return [self.raw] if self.raw is not None else []


def build_shared_kv_envelope_view(
    raw: torch.Tensor,
    *,
    num_pages: int,
    page_size: int,
    layer_num: int,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[K/V, page, layer, token, head, dim]`` over KV pages.

    Each L2 KV page is byte-compatible with the corresponding unified L1 page:
    ``[L0 K tokens][L0 V tokens] ... [Ln K tokens][Ln V tokens]``.  Keeping
    the same envelope avoids a token-first relayout when ``page_size > 1``.
    """

    if num_pages < 0:
        raise ValueError(f"num_pages must be non-negative, got {num_pages}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    row_elems = head_num * head_dim
    page_elems = 2 * layer_num * page_size * row_elems
    needed_bytes = num_pages * page_elems * dtype.itemsize
    if needed_bytes > raw.numel() * raw.element_size():
        raise ValueError(
            f"KV envelope needs {needed_bytes} bytes but arena has "
            f"{raw.numel() * raw.element_size()}"
        )
    if (raw.data_ptr() % dtype.itemsize) != 0:
        raise ValueError("raw host arena is not aligned for KV dtype")
    typed = raw.view(dtype)
    return torch.as_strided(
        typed,
        size=(2, num_pages, layer_num, page_size, head_num, head_dim),
        stride=(
            page_size * row_elems,
            page_elems,
            2 * page_size * row_elems,
            row_elems,
            head_dim,
            1,
        ),
    )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    acc = 1
    for value in reversed(shape):
        strides.append(acc)
        acc *= int(value)
    return tuple(reversed(strides))


def build_shared_mamba_envelope_views(
    raw: torch.Tensor,
    *,
    num_chunks: int,
    chunk_bytes: int,
    layer_num: int,
    temporal_shape: tuple[int, ...],
    temporal_dtype: torch.dtype,
    conv_shapes: tuple[tuple[int, ...], ...],
    conv_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build page-first Mamba component views, one slot per chunk.

    One chunk uses the same payload order as unified L1: every conv component
    followed by ``temporal``. Each component keeps all layers contiguous, so
    the complete Mamba payload can be copied as one raw byte range.
    """

    raw_bytes = raw.numel() * raw.element_size()
    if num_chunks * chunk_bytes > raw_bytes:
        raise ValueError("Mamba envelope exceeds shared host arena")

    offset_bytes = 0

    def build(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        nonlocal offset_bytes
        inner_elems = 1
        for value in shape:
            inner_elems *= int(value)
        row_bytes = inner_elems * dtype.itemsize
        component_bytes = layer_num * row_bytes
        if offset_bytes + component_bytes > chunk_bytes:
            raise ValueError(
                "Mamba components exceed typed chunk: "
                f"offset={offset_bytes}, component={component_bytes}, "
                f"chunk={chunk_bytes}"
            )
        if offset_bytes % dtype.itemsize or chunk_bytes % dtype.itemsize:
            raise ValueError(
                f"Mamba chunk/offset is not aligned for {dtype}: "
                f"chunk={chunk_bytes}, offset={offset_bytes}"
            )
        typed = raw.view(dtype)
        view = torch.as_strided(
            typed,
            size=(num_chunks, layer_num, 1, *shape),
            stride=(
                chunk_bytes // dtype.itemsize,
                inner_elems,
                inner_elems,
                *_contiguous_strides(shape),
            ),
            storage_offset=offset_bytes // dtype.itemsize,
        )
        offset_bytes += component_bytes
        return view

    conv = [build(shape, conv_dtype) for shape in conv_shapes]
    temporal = build(temporal_shape, temporal_dtype)
    return temporal, conv
