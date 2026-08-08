# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU integrity checks for row-aware unified HiCache transfer fences."""

import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")


class _FakeKVCache:
    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64, device="cuda")

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestUnifiedRowTransferFence(unittest.TestCase):
    def _make_allocator(self, pool_name: str):
        full = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        mamba = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((4, 3),),
            conv_dtype=torch.float32,
            temporal_state_shape=(2, 2, 2),
            temporal_dtype=torch.float32,
            grow_direction="down",
        )
        pool = UnifiedKVPool(
            total_bytes=full.entry_bytes() * 64 + mamba.entry_bytes() * 16,
            sub_pool_specs=[full, mamba],
            device="cuda",
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_allocator = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            lazy_compaction=True,
        )
        mamba_allocator = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device="cuda",
            is_id_owner=True,
            lazy_compaction=True,
        )
        full_allocator.bind_peer(mamba_allocator)
        mamba_allocator.bind_peer(full_allocator)
        if pool_name == "kv":
            return full_allocator, full_kv
        if pool_name == "mamba":
            return mamba_allocator, mamba_kv
        raise ValueError(f"unknown pool: {pool_name}")

    def _make_compaction_case(self, pool_name: str):
        allocator, kv = self._make_allocator(pool_name)
        values = allocator.alloc(6)
        physical = allocator.virtual_to_physical[values]
        kv.buf[physical] = values

        # Full-attention grows upward, so freeing its first allocation makes the
        # last allocation the compaction source. Mamba grows downward, so the
        # mirror image is required: free the last allocation and move the first.
        if allocator.grow_direction == "up":
            free_virtual = values[:1]
            source_virtual = values[-1:]
        else:
            free_virtual = values[-1:]
            source_virtual = values[:1]
        return allocator, kv, values, physical, free_virtual, source_virtual

    @staticmethod
    def _start_delayed_read(source: torch.Tensor):
        if not hasattr(torch.cuda, "_sleep"):
            raise unittest.SkipTest("torch.cuda._sleep is unavailable")
        snapshot = torch.empty_like(source, device="cpu", pin_memory=True)
        transfer_stream = torch.cuda.Stream()
        finish_event = torch.cuda.Event()
        transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(transfer_stream):
            # Widen the overlap window without making correctness depend on GPU
            # clock rate. The event may still finish before the host observes it
            # on a fast or otherwise lightly loaded device; both outcomes are
            # valid and are handled by the tests below.
            torch.cuda._sleep(100_000_000)
            snapshot.copy_(source, non_blocking=True)
            finish_event.record()
        return snapshot, finish_event

    def test_unrelated_compaction_overlaps_d2h_without_corruption(self):
        for pool_name in ("kv", "mamba"):
            with self.subTest(pool=pool_name):
                allocator, kv, _, physical, free_virtual, _ = (
                    self._make_compaction_case(pool_name)
                )
                protected_physical = physical[2:3].clone()
                protected_row = int(protected_physical.item())
                source = kv.buf[protected_row : protected_row + 1]
                expected = source.detach().cpu()
                allocator.free(free_virtual)
                snapshot, finish_event = self._start_delayed_read(source)
                allocator.register_external_transfer(finish_event, protected_physical)

                self.assertEqual(allocator.flush_opportunistic(), 1)
                finish_event.synchronize()
                torch.testing.assert_close(snapshot, expected)

    def test_overlapping_compaction_waits_for_d2h(self):
        for pool_name in ("kv", "mamba"):
            with self.subTest(pool=pool_name):
                allocator, kv, _, _, free_virtual, source_virtual = (
                    self._make_compaction_case(pool_name)
                )
                protected_physical = allocator.virtual_to_physical[
                    source_virtual
                ].clone()
                protected_row = int(protected_physical.item())
                source = kv.buf[protected_row : protected_row + 1]
                expected = source.detach().cpu()
                allocator.free(free_virtual)
                snapshot, finish_event = self._start_delayed_read(source)
                allocator.register_external_transfer(finish_event, protected_physical)

                moves = allocator.flush_opportunistic()
                self.assertIn(moves, (0, 1))
                # A device-to-host free-list snapshot may itself let the transfer
                # finish on some GPU/stream configurations. Compaction may only
                # proceed in that case; otherwise the row fence defers it.
                if moves == 1:
                    self.assertTrue(finish_event.query())
                finish_event.synchronize()
                if moves == 0:
                    self.assertEqual(allocator.flush_opportunistic(), 1)
                torch.testing.assert_close(snapshot, expected)

    def test_overlapping_compaction_waits_for_h2d_and_preserves_loaded_row(self):
        for pool_name in ("kv", "mamba"):
            with self.subTest(pool=pool_name):
                allocator, kv, _, _, free_virtual, source_virtual = (
                    self._make_compaction_case(pool_name)
                )
                protected_physical = allocator.virtual_to_physical[
                    source_virtual
                ].clone()
                protected_row = int(protected_physical.item())
                loaded_value = torch.tensor(
                    [777_000 + int(source_virtual.item())],
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                allocator.free(free_virtual)
                transfer_stream = torch.cuda.Stream()
                finish_event = torch.cuda.Event()
                transfer_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(transfer_stream):
                    # As with D2H above, this only widens the overlap window; the
                    # assertions do not require a device-specific duration.
                    torch.cuda._sleep(100_000_000)
                    kv.buf[protected_row : protected_row + 1].copy_(
                        loaded_value, non_blocking=True
                    )
                    finish_event.record()

                allocator.register_external_transfer(finish_event, protected_physical)

                moves = allocator.flush_opportunistic()
                self.assertIn(moves, (0, 1))
                if moves == 1:
                    self.assertTrue(finish_event.query())
                finish_event.synchronize()
                if moves == 0:
                    self.assertEqual(allocator.flush_opportunistic(), 1)
                relocated_physical = allocator.virtual_to_physical[source_virtual]
                self.assertEqual(kv.buf[relocated_physical].item(), loaded_value.item())


if __name__ == "__main__":
    unittest.main()
