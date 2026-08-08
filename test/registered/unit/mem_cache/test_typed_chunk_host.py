import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.typed_chunk_host import (
    HostChunkOwner,
    SharedTypedChunkHostArena,
    TypedChunkHostAllocator,
    build_shared_kv_envelope_view,
    build_shared_mamba_envelope_views,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTypedChunkHostAllocator(CustomTestCase):
    def test_layout_rounds_chunk_to_integral_kv_pages(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=10_000, kv_page_bytes=128, mamba_slot_bytes=300
        )
        self.assertEqual(alloc.kv_pages_per_chunk, 3)
        self.assertEqual(alloc.chunk_bytes, 384)
        self.assertEqual(alloc.mamba_padding_bytes, 84)
        self.assertEqual(alloc.num_chunks, 26)
        self.assertEqual(alloc.usable_bytes, 9_984)
        self.assertEqual(alloc.unused_budget_bytes, 16)

    def test_kv_suballocation_and_empty_chunk_retyping(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=4 * 400, kv_page_bytes=100, mamba_slot_bytes=350
        )
        kv = alloc.alloc_kv(5)
        self.assertEqual(kv.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(alloc.owner(0), HostChunkOwner.KV)
        self.assertEqual(alloc.owner(1), HostChunkOwner.KV)

        # A partially occupied KV chunk cannot be retyped.
        self.assertEqual(alloc.free_kv(torch.tensor([0, 1, 2, 3])), 4)
        self.assertEqual(alloc.owner(0), HostChunkOwner.FREE)
        mamba = alloc.alloc_mamba(3)
        self.assertEqual(mamba.tolist(), [3, 2, 0])
        self.assertEqual(alloc.owner(0), HostChunkOwner.MAMBA)
        self.assertEqual(alloc.owner(1), HostChunkOwner.KV)
        alloc.assert_consistent()

    def test_mamba_uses_exactly_one_chunk_per_slot(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=8 * 1_024, kv_page_bytes=256, mamba_slot_bytes=1_000
        )
        slots = alloc.alloc_mamba(3)
        self.assertEqual(slots.tolist(), [7, 6, 5])
        self.assertEqual(
            alloc.owner_counts()[HostChunkOwner.MAMBA],
            3,
        )
        self.assertEqual(alloc.free_mamba(slots[1:2]), 1)
        kv = alloc.alloc_kv(4)
        self.assertEqual(kv.tolist(), [0, 1, 2, 3])
        self.assertEqual(alloc.owner(0), HostChunkOwner.KV)
        self.assertEqual(alloc.owner(6), HostChunkOwner.FREE)
        alloc.assert_consistent()

    def test_kv_and_mamba_allocate_from_opposite_ends(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=6 * 512, kv_page_bytes=256, mamba_slot_bytes=512
        )

        kv = alloc.alloc_kv(3)
        mamba = alloc.alloc_mamba(2)

        self.assertEqual(kv.tolist(), [0, 1, 2])
        self.assertEqual(mamba.tolist(), [5, 4])
        self.assertEqual(
            [alloc.owner(i) for i in range(6)],
            [
                HostChunkOwner.KV,
                HostChunkOwner.KV,
                HostChunkOwner.FREE,
                HostChunkOwner.FREE,
                HostChunkOwner.MAMBA,
                HostChunkOwner.MAMBA,
            ],
        )
        alloc.assert_consistent()

    def test_pinned_empty_chunk_is_not_retyped_until_unpin(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=2_048, kv_page_bytes=256, mamba_slot_bytes=1_024
        )
        slot = alloc.alloc_mamba(1)
        chunk_id = int(slot.item())
        alloc.pin_chunks(alloc.mamba_chunks(slot))
        alloc.free_mamba(slot)
        with self.assertRaisesRegex(AssertionError, "already freed"):
            alloc.free_mamba(slot)
        self.assertEqual(alloc.owner(chunk_id), HostChunkOwner.MAMBA)
        self.assertEqual(alloc.available_mamba_slots(), 1)
        alloc.assert_consistent()

        alloc.unpin_chunks([chunk_id])
        self.assertEqual(alloc.owner(chunk_id), HostChunkOwner.FREE)
        self.assertEqual(alloc.available_mamba_slots(), 2)
        alloc.assert_consistent()

    def test_pinned_kv_page_is_not_reused_by_same_type_until_unpin(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=2_048, kv_page_bytes=256, mamba_slot_bytes=1_024
        )
        first_chunk_pages = alloc.alloc_kv(2)
        self.assertEqual(first_chunk_pages.tolist(), [0, 1])
        alloc.pin_chunks(alloc.kv_chunks(first_chunk_pages[:1]))
        alloc.free_kv(first_chunk_pages[:1])
        with self.assertRaisesRegex(AssertionError, "double-free|already freed"):
            alloc.free_kv(first_chunk_pages[:1])

        # Page 0 is free in metadata, but its bytes may still be used by the
        # asynchronous transfer. Allocation must use the other chunk.
        while_pinned = alloc.alloc_kv(1)
        self.assertEqual(while_pinned.tolist(), [4])
        self.assertEqual(alloc.available_kv_pages(), 3)

        alloc.unpin_chunks([0])
        after_event = alloc.alloc_kv(1)
        self.assertEqual(after_event.tolist(), [0])
        alloc.assert_consistent()

    def test_double_free_and_cross_type_free_are_rejected(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=4_096, kv_page_bytes=256, mamba_slot_bytes=1_024
        )
        kv = alloc.alloc_kv(1)
        with self.assertRaisesRegex(AssertionError, "owned by KV"):
            alloc.free_mamba(torch.tensor([0]))
        alloc.free_kv(kv)
        with self.assertRaisesRegex(AssertionError, "owned by FREE"):
            alloc.free_kv(kv)

    def test_failed_allocation_is_transactional(self):
        alloc = TypedChunkHostAllocator(
            total_bytes=2_048, kv_page_bytes=256, mamba_slot_bytes=1_024
        )
        self.assertIsNone(alloc.alloc_mamba(3))
        self.assertEqual(alloc.owner_counts()[HostChunkOwner.FREE], 2)
        self.assertIsNone(alloc.alloc_kv(9))
        self.assertEqual(alloc.owner_counts()[HostChunkOwner.FREE], 2)
        alloc.assert_consistent()

    def test_overlapping_kv_and_mamba_views_follow_chunk_ownership(self):
        # KV entry = 2(K/V) * 2 layers * 2 elements * fp16 = 16 bytes.
        # Mamba entry = temporal(2*2*fp32=16) + conv(2*4*fp16=16) = 32 bytes.
        # Therefore one 32-byte chunk holds exactly two KV entries.
        raw = torch.zeros(3 * 32, dtype=torch.uint8)
        kv = build_shared_kv_envelope_view(
            raw,
            num_pages=6,
            page_size=1,
            layer_num=2,
            head_num=1,
            head_dim=2,
            dtype=torch.float16,
        )
        temporal, conv = build_shared_mamba_envelope_views(
            raw,
            num_chunks=3,
            chunk_bytes=32,
            layer_num=2,
            temporal_shape=(1, 2),
            temporal_dtype=torch.float32,
            conv_shapes=((1, 4),),
            conv_dtype=torch.float16,
        )

        # Chunk 0 as KV: its two logical tokens occupy bytes [0, 32).
        kv[:, 0].fill_(1)
        kv[:, 1].fill_(2)
        self.assertTrue(torch.all(kv[:, 0] == 1))
        self.assertTrue(torch.all(kv[:, 1] == 2))
        self.assertTrue(torch.any(raw[:16] != 0))
        self.assertTrue(torch.any(raw[16:32] != 0))

        # Chunk 1 as Mamba: the one slot occupies bytes [32, 64).
        temporal[1].fill_(3)
        conv[0][1].fill_(4)
        self.assertTrue(torch.all(temporal[1] == 3))
        self.assertTrue(torch.all(conv[0][1] == 4))
        self.assertTrue(torch.any(raw[32:64] != 0))
        self.assertTrue(torch.all(raw[64:] == 0))

        # The views alias by design: KV tokens 2/3 cover Mamba chunk 1.
        self.assertTrue(torch.any(kv[:, 2] != 0))
        self.assertTrue(torch.any(kv[:, 3] != 0))

    def test_multi_token_kv_pages_match_unified_l1_byte_order(self):
        """Typed L2 pages must be byte-compatible with unified L1 pages."""

        page_size = 2
        num_pages = 2
        layer_num = 2
        # One scalar per row makes the expected physical order unambiguous:
        # [L0 K T0,T1][L0 V T0,T1][L1 K T0,T1][L1 V T0,T1].
        raw = torch.zeros(
            num_pages * page_size * layer_num * 2 * 2,
            dtype=torch.uint8,
        )
        kv = build_shared_kv_envelope_view(
            raw,
            num_pages=num_pages,
            page_size=page_size,
            layer_num=layer_num,
            head_num=1,
            head_dim=1,
            dtype=torch.float16,
        )

        for page in range(num_pages):
            for layer in range(layer_num):
                for token in range(page_size):
                    kv[0, page, layer, token] = 100 * page + 10 * layer + token + 1
                    kv[1, page, layer, token] = 100 * page + 10 * layer + token + 5

        expected = torch.tensor(
            [1, 2, 5, 6, 11, 12, 15, 16, 101, 102, 105, 106, 111, 112, 115, 116],
            dtype=torch.float16,
        )
        torch.testing.assert_close(raw.view(torch.float16), expected)

    def test_mamba_padding_is_not_exposed_by_component_views(self):
        raw = torch.full((64,), 0x7F, dtype=torch.uint8)
        temporal, conv = build_shared_mamba_envelope_views(
            raw,
            num_chunks=2,
            chunk_bytes=32,
            layer_num=1,
            temporal_shape=(1, 2),
            temporal_dtype=torch.float32,
            conv_shapes=((1, 4),),
            conv_dtype=torch.float16,
        )
        # Only 16 of each 32-byte chunk is state; the rest is KV-alignment padding.
        temporal[0].zero_()
        conv[0][0].zero_()
        self.assertTrue(torch.all(raw[:16] == 0))
        self.assertTrue(torch.all(raw[16:32] == 0x7F))
        self.assertTrue(torch.all(raw[32:] == 0x7F))

    def test_pool_adapters_share_one_arena_and_retype_empty_chunks(self):
        from sglang.srt.mem_cache.pool_host.unified_chunk import (
            UnifiedChunkMambaPoolHost,
            UnifiedChunkMHAPoolHost,
        )

        arena = SharedTypedChunkHostArena(
            total_bytes=4 * 32,
            kv_page_bytes=16,
            mamba_slot_bytes=32,
            host_device="cpu",
            accelerator_device="cpu",
            pin_memory=False,
            allocator_type="default",
        )
        kv_pool = SimpleNamespace(
            page_size=1,
            store_dtype=torch.float16,
            head_num=1,
            head_dim=2,
            v_head_dim=2,
            layer_num=2,
            start_layer=0,
            end_layer=2,
            device="cpu",
            k_buffer=[torch.zeros((4, 1, 1, 2), dtype=torch.float16) for _ in range(2)],
            v_buffer=[torch.zeros((4, 1, 1, 2), dtype=torch.float16) for _ in range(2)],
        )
        temporal = torch.zeros((2, 4, 1, 2), dtype=torch.float32)
        conv = torch.zeros((2, 4, 1, 4), dtype=torch.float16)
        mamba_pool = SimpleNamespace(
            num_mamba_layers=2,
            device="cpu",
            mamba_cache=SimpleNamespace(temporal=temporal, conv=[conv]),
        )

        with (
            mock.patch(
                "sglang.srt.mem_cache.pool_host.unified_chunk.can_use_hicache_page_copy_kernel",
                return_value=True,
            ),
            mock.patch("torch.empty", wraps=torch.empty),
        ):
            kv_host = UnifiedChunkMHAPoolHost(kv_pool, arena)
        mamba_host = UnifiedChunkMambaPoolHost(mamba_pool, arena)

        kv_indices = kv_host.alloc(2)
        mamba_indices = mamba_host.alloc(1)
        self.assertEqual(kv_indices.tolist(), [0, 1])
        self.assertEqual(mamba_indices.tolist(), [3])
        kv_host.kv_buffer[:, kv_indices] = 7
        mamba_host.temporal_buffer[mamba_indices] = 11
        mamba_host.conv_buffer[0][mamba_indices] = 13

        self.assertEqual(arena.chunks.owner(0), HostChunkOwner.KV)
        self.assertEqual(arena.chunks.owner(3), HostChunkOwner.MAMBA)
        self.assertTrue(torch.all(kv_host.kv_buffer[:, kv_indices] == 7))
        self.assertTrue(torch.all(mamba_host.temporal_buffer[mamba_indices] == 11))
        self.assertTrue(torch.all(mamba_host.conv_buffer[0][mamba_indices] == 13))

        # Freeing both KV pages empties chunk 0, which Mamba can immediately own.
        kv_host.free(kv_indices)
        next_mamba = mamba_host.alloc(3)
        self.assertEqual(next_mamba.tolist(), [2, 1, 0])
        self.assertEqual(arena.chunks.owner(0), HostChunkOwner.MAMBA)
        arena.chunks.assert_consistent()
        arena.destroy()

    def test_multi_token_page_adapter_uses_token_index_api(self):
        """HiCache metadata stays token-indexed while L2 allocates full pages."""

        from sglang.srt.mem_cache.pool_host.unified_chunk import UnifiedChunkMHAPoolHost

        page_size = 4
        layers = 2
        token_bytes = 2 * layers * 1 * 2 * torch.float16.itemsize
        arena = SharedTypedChunkHostArena(
            total_bytes=4 * page_size * token_bytes,
            kv_page_bytes=page_size * token_bytes,
            mamba_slot_bytes=2 * page_size * token_bytes,
            host_device="cpu",
            accelerator_device="cpu",
            pin_memory=False,
            allocator_type="default",
        )
        kv_pool = SimpleNamespace(
            page_size=page_size,
            store_dtype=torch.float16,
            head_num=1,
            head_dim=2,
            v_head_dim=2,
            layer_num=layers,
            start_layer=0,
            end_layer=layers,
            device="cpu",
            k_buffer=[
                torch.zeros((4, page_size, 1, 2), dtype=torch.float16)
                for _ in range(layers)
            ],
            v_buffer=[
                torch.zeros((4, page_size, 1, 2), dtype=torch.float16)
                for _ in range(layers)
            ],
        )

        with (
            mock.patch(
                "sglang.srt.mem_cache.pool_host.unified_chunk.can_use_hicache_page_copy_kernel",
                return_value=True,
            ),
        ):
            host = UnifiedChunkMHAPoolHost(kv_pool, arena)

        self.assertEqual(host.available_size(), 16)
        indices = host.alloc(2 * page_size)
        self.assertEqual(indices.tolist(), list(range(2 * page_size)))
        self.assertEqual(host.transfer_chunk_ids(indices), [0])
        with self.assertRaisesRegex(ValueError, "multiple of page_size"):
            host.alloc(page_size + 1)
        self.assertEqual(host.free(indices), 2 * page_size)
        arena.chunks.assert_consistent()
        arena.destroy()

    def test_arena_defers_retyping_until_transfer_event_completes(self):
        class FakeEvent:
            def __init__(self):
                self.completed = False
                self.synchronize_count = 0

            def query(self):
                return self.completed

            def synchronize(self):
                self.synchronize_count += 1
                self.completed = True

        arena = SharedTypedChunkHostArena(
            total_bytes=2_048,
            kv_page_bytes=256,
            mamba_slot_bytes=1_024,
            host_device="cpu",
            accelerator_device="cpu",
            pin_memory=False,
            allocator_type="default",
        )
        kv = arena.alloc_kv(4)
        chunks = arena.chunks.kv_chunks(kv)
        pinned = arena.pin_chunks_for_transfer(chunks)
        event = FakeEvent()
        arena.release_chunks_after_event(pinned, event)

        arena.free_kv(kv)
        self.assertEqual(arena.chunks.owner(0), HostChunkOwner.KV)
        self.assertEqual(arena.available_mamba_slots(), 1)
        arena.chunks.assert_consistent()

        event.completed = True
        self.assertEqual(arena.available_mamba_slots(), 2)
        self.assertEqual(arena.chunks.owner(0), HostChunkOwner.FREE)
        arena.chunks.assert_consistent()
        arena.destroy()

    def test_host_pool_group_reclaims_fragmented_opposite_type_chunks(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

        arena = SharedTypedChunkHostArena(
            total_bytes=3 * 1_024,
            kv_page_bytes=256,
            mamba_slot_bytes=1_024,
            host_device="cpu",
            accelerator_device="cpu",
            pin_memory=False,
            allocator_type="default",
        )

        class FakePool:
            layout = "page_first"
            page_size = 1
            device = "cpu"
            size = 12
            logical_size = 12
            can_use_write_back_jit = True

            def __init__(self, kind):
                self.shared_arena = arena
                self.kind = kind
                self.allocation_units_per_chunk = 4 if kind == "kv" else 1

            def available_size(self):
                if self.kind == "kv":
                    return arena.available_kv_pages()
                return arena.available_mamba_slots()

            def alloc(self, size):
                if self.kind == "kv":
                    return arena.alloc_kv(size)
                return arena.alloc_mamba(size)

            def free(self, indices):
                if self.kind == "kv":
                    return arena.free_kv(indices)
                return arena.free_mamba(indices)

        kv_pool = FakePool("kv")
        mamba_pool = FakePool("mamba")
        self.assertEqual(kv_pool.alloc(12).numel(), 12)
        # Spread LRU frees across chunks so one eviction callback is not enough
        # to produce an empty/retypable chunk.
        kv_evict_order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        kv_evict_calls = []

        def evict_kv(size):
            values = kv_evict_order[:size]
            del kv_evict_order[:size]
            if values:
                kv_pool.free(torch.tensor(values))
            kv_evict_calls.append(values)
            return len(values)

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=kv_pool,
                device_pool=None,
                layer_mapper=lambda _: None,
                is_primary_index_anchor=True,
                host_evict_fn=evict_kv,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=mamba_pool,
                device_pool=None,
                layer_mapper=lambda _: None,
            ),
        ]
        group = HostPoolGroup(entries)
        self.assertIsNone(mamba_pool.alloc(1))
        self.assertTrue(group.reclaim_shared_capacity(PoolName.MAMBA, 1))
        self.assertGreater(len(kv_evict_calls), 1)
        mamba_slot = mamba_pool.alloc(1)
        self.assertIsNotNone(mamba_slot)
        self.assertEqual(arena.chunks.owner(int(mamba_slot[0])), HostChunkOwner.MAMBA)

        # Remove the remaining owners, then verify the reverse Mamba -> KV path.
        remaining_kv = torch.tensor(kv_evict_order)
        if remaining_kv.numel():
            kv_pool.free(remaining_kv)
        mamba_pool.free(mamba_slot)
        all_mamba = mamba_pool.alloc(3)
        self.assertIsNone(kv_pool.alloc(4))

        def evict_mamba(size):
            victims = all_mamba[:size]
            if victims.numel():
                mamba_pool.free(victims)
            return len(victims)

        entries[1].host_evict_fn = evict_mamba
        self.assertTrue(group.reclaim_shared_capacity(PoolName.KV, 4))
        self.assertIsNotNone(kv_pool.alloc(4))
        arena.chunks.assert_consistent()
        arena.destroy()


if __name__ == "__main__":
    unittest.main()
