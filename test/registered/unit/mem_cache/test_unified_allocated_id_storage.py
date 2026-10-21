"""Retained allocation IDs must not pin retired free-list storage."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedAllocator,
    MultiEndedAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Cache:
    def __init__(self, size, device):
        self.values = torch.zeros(size, dtype=torch.int64, device=device)

    def move_physical_pages(self, dst_pages, src_pages, page_size):
        offsets = torch.arange(page_size, device=dst_pages.device)
        dst_loc = (dst_pages[:, None] * page_size + offsets).flatten()
        src_loc = (src_pages[:, None] * page_size + offsets).flatten()
        self.values[dst_loc] = self.values[src_loc].clone()


class TestAllocatedIDStorage(CustomTestCase):
    device = "cpu"

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def make_allocator(self, page_size, lazy, direction="up"):
        layout = [("full", "up"), ("peer", "down")]
        if direction == "float":
            layout.insert(1, ("middle", "float"))
        specs = [
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=8,
                store_dtype=torch.float32,
                grow_direction=grow,
            )
            for name, grow in layout
        ]
        pool = UnifiedKVPool(
            total_bytes=16384,
            sub_pool_specs=specs,
            device=self.device,
            enable_memory_saver=False,
            page_size=page_size,
        )
        allocators = [
            (
                FloatMultiEndedAllocator
                if spec.grow_direction == "float"
                else MultiEndedAllocator
            )(
                kvcache=_Cache(pool.max_slots(spec.name), self.device),
                unified_buffer=pool,
                sub_pool_name=spec.name,
                device=self.device,
                is_id_owner=True,
                lazy_compaction=lazy and spec.grow_direction != "float",
                page_size=page_size,
            )
            for spec in specs
        ]
        for low, high in zip(allocators, allocators[1:]):
            low.bind_high_peer(high)
            high.bind_low_peer(low)
        return allocators[{"up": 0, "down": -1, "float": 1}[direction]]

    def test_retained_ids_have_bounded_storage_through_free_churn(self):
        for page_size, lazy in [(1, True), (1, False), (4, True)]:
            with self.subTest(page_size=page_size, lazy=lazy):
                alloc = self.make_allocator(page_size, lazy)
                initial = alloc.free_virtual_ids.numel()
                live = []
                for _ in range(24):
                    ids = alloc.alloc(page_size)
                    self.assertIsNotNone(ids)
                    live.append(ids)
                    temporary = alloc.alloc(page_size)
                    self.assertIsNotNone(temporary)
                    alloc.free(temporary)
                    self.assertEqual(
                        alloc.free_virtual_ids.numel() + len(live), initial
                    )
                storage = {
                    ids.untyped_storage().data_ptr(): ids.untyped_storage().nbytes()
                    for ids in live
                }
                self.assertEqual(
                    sum(storage.values()),
                    sum(ids.numel() * ids.element_size() for ids in live),
                )
                all_ids = torch.cat(live)
                self.assertEqual(all_ids.unique().numel(), all_ids.numel())
                self.assertTrue(
                    torch.all(alloc.virtual_to_physical[all_ids // page_size] >= 0)
                )
                for ids in live:
                    alloc.free(ids)
                alloc.flush_for_allocation()
                self.assertEqual(alloc.free_virtual_ids.unique().numel(), initial)
                self.assertEqual(alloc.allocated_count(), 0)

    def test_owned_ids_survive_hole_reuse_and_compaction(self):
        for direction in ["up", "down", "float"]:
            with self.subTest(direction=direction):
                alloc = self.make_allocator(1, True, direction)
                initial = alloc.free_virtual_ids.numel()
                blocks = [alloc.alloc(4) for _ in range(3)]
                snapshots = [v.clone() for v in blocks]
                for v in blocks:
                    alloc._kvcache.values[alloc.virtual_to_physical[v]] = v + 1000
                middle = next(
                    v
                    for v in blocks
                    if int(alloc.virtual_to_physical[v].min())
                    > int(
                        torch.cat([alloc.virtual_to_physical[b] for b in blocks]).min()
                    )
                    and int(alloc.virtual_to_physical[v].max())
                    < int(
                        torch.cat([alloc.virtual_to_physical[b] for b in blocks]).max()
                    )
                )
                holes = alloc.virtual_to_physical[middle].clone()
                alloc.free(middle)
                self.assertEqual(alloc._free_phys_pages.numel(), 4)
                replacement = alloc.alloc(4)
                self.assertTrue(
                    torch.equal(
                        alloc.virtual_to_physical[replacement].sort().values,
                        holes.sort().values,
                    )
                )
                self.assertEqual(replacement.untyped_storage().nbytes(), 32)
                alloc._kvcache.values[alloc.virtual_to_physical[replacement]] = (
                    replacement + 1000
                )
                live = [b for b in blocks if b is not middle] + [replacement]
                alloc.free(live.pop(0))
                alloc.flush_for_allocation()
                for v in live:
                    physical = alloc.virtual_to_physical[v]
                    self.assertTrue(torch.equal(alloc.physical_to_virtual[physical], v))
                    self.assertTrue(
                        torch.equal(alloc._kvcache.values[physical], v + 1000)
                    )
                for v, saved in zip(blocks, snapshots):
                    self.assertTrue(torch.equal(v, saved))
                for v in live:
                    alloc.free(v)
                alloc.flush_for_allocation()
                self.assertEqual(alloc.free_virtual_ids.unique().numel(), initial)

    def test_output_allocation_failure_preserves_owned_state(self):
        alloc = self.make_allocator(1, True)
        live = alloc.alloc(3)
        free = alloc.free_virtual_ids.clone()
        v2p = alloc.virtual_to_physical.clone()
        p2v = alloc.physical_to_virtual.clone()
        watermark = alloc.watermark_physical
        count = alloc.allocated_count()
        with patch(
            "sglang.srt.mem_cache.allocator.unified_sub_pool.torch.empty_like",
            side_effect=torch.OutOfMemoryError("owned output allocation failed"),
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                alloc.alloc(1)
        self.assertTrue(torch.equal(alloc.free_virtual_ids, free))
        self.assertTrue(torch.equal(alloc.virtual_to_physical, v2p))
        self.assertTrue(torch.equal(alloc.physical_to_virtual, p2v))
        self.assertEqual(alloc.watermark_physical, watermark)
        self.assertEqual(alloc.allocated_count(), count)
        self.assertTrue(torch.equal(alloc.physical_to_virtual[v2p[live]], live))

    def test_empty_shortfall_and_non_owner_contract(self):
        alloc = self.make_allocator(1, True)
        live = alloc.alloc(3)
        saved_live = live.clone()
        alloc._kvcache.values[alloc.virtual_to_physical[live]] = live + 1000
        free = alloc.free_virtual_ids.clone()
        mapping = alloc.virtual_to_physical.clone()
        inverse = alloc.physical_to_virtual.clone()
        watermark = alloc.watermark_physical
        count = alloc.allocated_count()
        self.assertEqual(alloc.alloc(0).numel(), 0)
        self.assertIsNone(alloc.alloc(alloc.size * 2))
        # Force only the physical-reservation boundary to fail, checking rollback.
        with patch.object(alloc, "_extend_watermark", return_value=False):
            self.assertIsNone(alloc.alloc(1))
        self.assertTrue(torch.equal(alloc.free_virtual_ids, free))
        self.assertTrue(torch.equal(alloc.virtual_to_physical, mapping))
        self.assertTrue(torch.equal(alloc.physical_to_virtual, inverse))
        self.assertEqual(alloc.watermark_physical, watermark)
        self.assertEqual(alloc.allocated_count(), count)
        self.assertTrue(torch.equal(live, saved_live))
        self.assertTrue(
            torch.equal(
                alloc._kvcache.values[alloc.virtual_to_physical[live]], live + 1000
            )
        )
        alloc.is_id_owner = False
        v = free[:3].clone()
        alloc.alloc_with_virtual(v)
        self.assertTrue(
            torch.equal(alloc.physical_to_virtual[alloc.virtual_to_physical[v]], v)
        )
        self.assertTrue(torch.equal(alloc.free_virtual_ids, free))
        with self.assertRaises(AssertionError):
            alloc.alloc(1)


if __name__ == "__main__":
    unittest.main()
