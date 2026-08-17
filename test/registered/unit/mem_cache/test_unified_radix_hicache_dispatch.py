import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocation import alloc_req_slots
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import (
    hybrid_cache_controller,
    hybrid_pool_assembler,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    StackBuildResult,
    StackStrategy,
    _apply_stack_result,
    _DeepSeekV4Strategy,
    _DsaStrategy,
    _MambaStrategy,
    _MiniMaxSparseStrategy,
    _PlainKvStrategy,
    _select_strategy,
    _SwaStrategy,
    register_stack_strategy,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_kvcache(cls):
    return MagicMock(spec=cls)


FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA


class TestUnifiedRadixHiCacheDispatch(unittest.TestCase):
    def test_strategy_registry_ordering(self):
        order = [type(s) for s in _STRATEGIES]
        # DeepSeekV4 inherits from SWAKVPool, so it must resolve before _SwaStrategy.
        self.assertLess(order.index(_DeepSeekV4Strategy), order.index(_SwaStrategy))
        self.assertLess(
            order.index(_MiniMaxSparseStrategy), order.index(_PlainKvStrategy)
        )
        self.assertEqual(order[-1], _PlainKvStrategy)

    def test_deepseek_v4_full_swa(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = _mock_kvcache(DeepSeekV4TokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _DeepSeekV4Strategy)

    def test_mamba(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = _mock_kvcache(HybridLinearKVPool)
        strategy = _select_strategy(kvcache, {FULL, MAMBA})
        self.assertIsInstance(strategy, _MambaStrategy)

    def test_swa(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = _mock_kvcache(SWAKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _SwaStrategy)

    def test_dsa(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = _mock_kvcache(DSATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _DsaStrategy)

    def test_minimax_sparse(self):
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        kvcache = _mock_kvcache(MiniMaxSparseKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _MiniMaxSparseStrategy)

    def test_minimax_sparse_build_registers_indexer_sidecar(self):
        strategy = _MiniMaxSparseStrategy()
        host_pool_group = MagicMock()
        kv_host_pool = object()
        host_pool_group.get_pool.return_value = kv_host_pool
        cache_controller = MagicMock()
        cache = MagicMock(page_size=4)
        kvcache = MagicMock()
        kvcache.index_k_pool = object()
        kvcache.main_pool.layer_num = 8
        params = MagicMock()
        params.tp_cache_group = None
        params.pp_rank = 0
        params.pp_size = 1
        server_args = MagicMock()

        with patch.object(
            hybrid_pool_assembler,
            "build_minimax_sparse_hicache_stack",
            return_value=(host_pool_group, cache_controller),
        ) as build_stack:
            result = strategy.build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=server_args,
                load_cache_event=object(),
            )

        build_stack.assert_called_once()
        self.assertIs(build_stack.call_args.kwargs["sparse_pool"], kvcache)
        self.assertIs(result.host_pool_group, host_pool_group)
        self.assertIs(result.cache_controller, cache_controller)
        self.assertIs(result.component_host_pools[FULL], kv_host_pool)
        self.assertEqual(result.pools_desc, "KV + INDEXER(k-only)")
        self.assertEqual(result.transfer_layer_num, 8)
        self.assertEqual(len(result.sidecars), 1)
        self.assertEqual(result.sidecars[0].pool_name, PoolName.INDEXER)
        self.assertEqual(result.sidecars[0].indices_from_pool, PoolName.KV)

    def test_plain_kv_fallback(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kvcache = _mock_kvcache(MHATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_mla_routes_to_plain(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        kvcache = _mock_kvcache(MLATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_unknown_combo_raises(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        for cls in (SWAKVPool, DeepSeekV4TokenToKVPool):
            kvcache = _mock_kvcache(cls)
            with self.assertRaises(AssertionError) as cm:
                _select_strategy(kvcache, {FULL})
            self.assertIn("No matching HiCache strategy", str(cm.exception))

    def test_register_custom_strategy_takes_precedence(self):
        class _CustomStrategy(StackStrategy):
            def matches(self, kvcache, components):
                return components == {FULL}

            def build(self, **_):
                raise NotImplementedError

        custom = _CustomStrategy()
        original = list(hybrid_pool_assembler._STRATEGIES)
        try:
            register_stack_strategy(custom)
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            kvcache = _mock_kvcache(MHATokenToKVPool)
            self.assertIs(_select_strategy(kvcache, {FULL}), custom)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = original


class TestApplyStackResult(unittest.TestCase):
    @staticmethod
    def _fake_cache(component_types):
        cache = MagicMock()
        cache.components = {ct: MagicMock() for ct in component_types}
        return cache

    def test_wires_components_sidecars_and_counters(self):
        full_host, swa_host, mamba_host = MagicMock(), MagicMock(), MagicMock()
        cache = self._fake_cache([FULL, SWA, MAMBA])
        kvcache = MagicMock()
        params = MagicMock()
        controller = MagicMock()
        sidecar = SidecarPoolSpec(
            pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV
        )
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=controller,
            component_host_pools={FULL: full_host, SWA: swa_host, MAMBA: mamba_host},
            sidecars=[sidecar],
            register_req_to_token_counter=True,
            transfer_layer_num=8,
            pools_desc="KV + SWA + MAMBA",
        )

        _apply_stack_result(cache, kvcache, params, result)

        self.assertIs(cache.host_pool_group, result.host_pool_group)
        self.assertIs(cache.cache_controller, controller)
        self.assertIs(cache.full_kv_pool_host, full_host)
        self.assertIs(cache.swa_kv_pool_host, swa_host)
        self.assertIs(cache.mamba_pool_host, mamba_host)
        self.assertIs(cache.components[FULL]._full_kv_pool_host, full_host)
        self.assertIs(cache.components[SWA]._swa_kv_pool_host, swa_host)
        self.assertIs(cache.components[MAMBA]._mamba_pool_host, mamba_host)
        cache.register_sidecar_pool.assert_called_once_with(sidecar)
        kvcache.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )
        params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )

    def test_skips_req_to_token_counter_when_flag_false(self):
        cache = self._fake_cache([FULL])
        kvcache = MagicMock()
        params = MagicMock()
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=MagicMock(),
            component_host_pools={FULL: MagicMock()},
            sidecars=[],
            register_req_to_token_counter=False,
            transfer_layer_num=1,
            pools_desc="KV",
        )

        _apply_stack_result(cache, kvcache, params, result)

        kvcache.register_layer_transfer_counter.assert_called_once()
        params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        cache.register_sidecar_pool.assert_not_called()


class TestHybridTransferIndexTranslation(unittest.TestCase):
    def test_unified_ready_reaps_synchronized_load_immediately(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = MagicMock(
            synchronize_unified_transfers=True,
            start_loading=MagicMock(return_value=2),
        )
        cache.loading_check = MagicMock()

        self.assertEqual(cache.ready_to_load_host_cache(), 2)
        cache.loading_check.assert_called_once_with()

        cache.loading_check.reset_mock()
        cache.cache_controller.synchronize_unified_transfers = False
        cache.ready_to_load_host_cache()
        cache.loading_check.assert_not_called()

        cache.cache_controller.synchronize_unified_transfers = True
        cache.cache_controller.start_loading.return_value = -1
        cache.ready_to_load_host_cache()
        cache.loading_check.assert_not_called()

    def test_unified_transfer_is_finished_before_scheduler_resumes(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        event = MagicMock()
        controller.mem_pool_device_allocator = MagicMock()

        controller.synchronize_unified_transfers = True
        controller.is_unified_memory = True
        controller._finish_transfer_before_scheduler(event)
        event.synchronize.assert_called_once_with()
        controller.mem_pool_device_allocator.register_external_transfer_event.assert_not_called()

        event.reset_mock()
        controller.synchronize_unified_transfers = False
        controller._finish_transfer_before_scheduler(event)
        event.synchronize.assert_not_called()
        controller.mem_pool_device_allocator.register_external_transfer_event.assert_called_once_with(
            event
        )

        controller.mem_pool_device_allocator.reset_mock()
        controller.is_unified_memory = False
        controller._finish_transfer_before_scheduler(event)
        controller.mem_pool_device_allocator.register_external_transfer_event.assert_not_called()

    def test_unified_transfer_fences_each_physical_pool_independently(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.synchronize_unified_transfers = False
        controller.is_unified_memory = True
        controller.mem_pool_device_allocator = MagicMock()

        full_fence = MagicMock()
        mamba_fence = MagicMock()
        full_entry = SimpleNamespace(device_transfer_fence_fn=full_fence)
        mamba_entry = SimpleNamespace(device_transfer_fence_fn=mamba_fence)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=full_entry,
            entry_map={PoolName.KV: full_entry, PoolName.MAMBA: mamba_entry},
        )
        finish_event = MagicMock()
        full_physical = torch.tensor([3, 7], dtype=torch.int64)
        mamba_physical = torch.tensor([11], dtype=torch.int64)

        controller._finish_transfer_before_scheduler(
            finish_event,
            full_physical,
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=mamba_physical,
                )
            ],
        )

        full_fence.assert_called_once_with(finish_event, full_physical)
        mamba_fence.assert_called_once_with(finish_event, mamba_physical)
        controller.mem_pool_device_allocator.register_external_transfer_event.assert_not_called()

    def test_unified_transfer_uses_global_fence_if_one_pool_lacks_row_api(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.synchronize_unified_transfers = False
        controller.is_unified_memory = True
        controller.mem_pool_device_allocator = MagicMock()

        full_fence = MagicMock()
        full_entry = SimpleNamespace(device_transfer_fence_fn=full_fence)
        mamba_entry = SimpleNamespace(device_transfer_fence_fn=None)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=full_entry,
            entry_map={PoolName.KV: full_entry, PoolName.MAMBA: mamba_entry},
        )
        finish_event = MagicMock()

        controller._finish_transfer_before_scheduler(
            finish_event,
            torch.tensor([3], dtype=torch.int64),
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=torch.tensor([11], dtype=torch.int64),
                )
            ],
        )

        full_fence.assert_not_called()
        controller.mem_pool_device_allocator.register_external_transfer_event.assert_called_once_with(
            finish_event
        )

    def test_rejected_load_can_be_synchronized_before_destination_free(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        first_ack, second_ack = MagicMock(), MagicMock()
        cache.cache_controller = MagicMock(ack_load_queue=[first_ack, second_ack])
        cache.loading_check = MagicMock()

        cache.synchronize_pending_loads()

        first_ack.finish_event.synchronize.assert_called_once_with()
        second_ack.finish_event.synchronize.assert_called_once_with()
        cache.loading_check.assert_called_once_with()

    def test_translates_execution_indices_without_mutating_virtual_ids(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.move_indices = lambda host, device: (host, device)

        full_translate = MagicMock(side_effect=lambda x: x + 100)
        mamba_translate = MagicMock(side_effect=lambda x: x + 200)
        controller.mem_pool_device_allocator = SimpleNamespace(
            translate_kv_indices_for_transfer=full_translate
        )
        full_entry = MagicMock(
            is_primary_index_anchor=True,
            device_index_translate_fn=None,
        )
        mamba_entry = MagicMock(device_index_translate_fn=mamba_translate)
        controller.mem_pool_host = MagicMock(
            anchor_entry=full_entry,
            entry_map={
                PoolName.KV: full_entry,
                PoolName.MAMBA: mamba_entry,
            },
        )

        full_virtual = torch.tensor([3, 7], dtype=torch.int64)
        mamba_virtual = torch.tensor([11], dtype=torch.int64)
        operation = CacheOperation(
            host_indices=torch.tensor([0, 1], dtype=torch.int64),
            device_indices=full_virtual,
            node_id=1,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.tensor([2], dtype=torch.int64),
                    device_indices=mamba_virtual,
                    nodes_to_load=[12],
                )
            ],
        )

        _, full_physical, transfers = controller.move_hybrid_indices(operation)

        torch.testing.assert_close(full_physical, torch.tensor([103, 107]))
        torch.testing.assert_close(transfers[0].device_indices, torch.tensor([211]))
        self.assertEqual(transfers[0].nodes_to_load, [12])
        # The operation remains tree/allocator-owned virtual state.
        torch.testing.assert_close(operation.device_indices, full_virtual)
        torch.testing.assert_close(
            operation.pool_transfers[0].device_indices, mamba_virtual
        )
        full_translate.assert_called_once_with(full_virtual)
        mamba_translate.assert_called_once_with(mamba_virtual)

    def test_identity_when_pool_has_no_translator(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.move_indices = lambda host, device: (host, device)
        controller.mem_pool_device_allocator = SimpleNamespace(
            translate_kv_indices_for_transfer=lambda indices: indices
        )
        anchor_entry = MagicMock(
            is_primary_index_anchor=True,
            device_index_translate_fn=None,
        )
        controller.mem_pool_host = MagicMock(
            anchor_entry=anchor_entry,
            entry_map={PoolName.KV: anchor_entry},
        )
        virtual = torch.tensor([2, 4], dtype=torch.int64)
        operation = CacheOperation(
            host_indices=torch.tensor([0, 1], dtype=torch.int64),
            device_indices=virtual,
            node_id=1,
        )

        _, execution_indices, transfers = controller.move_hybrid_indices(operation)

        self.assertIs(execution_indices, virtual)
        self.assertIsNone(transfers)

    def test_rejects_translated_indices_outside_physical_pool(self):
        entry = SimpleNamespace(name=PoolName.KV, device_pool=SimpleNamespace(size=7))

        HybridCacheController._validate_translated_indices(
            entry, torch.tensor([0, 7], dtype=torch.int64)
        )
        for invalid in (
            torch.tensor([-1], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "outside the physical device pool"
            ):
                HybridCacheController._validate_translated_indices(entry, invalid)

    def test_accepts_complete_static_padding_page(self):
        entry = SimpleNamespace(
            name=PoolName.KV,
            device_pool=SimpleNamespace(size=120000, page_size=8),
        )

        HybridCacheController._validate_translated_indices(
            entry, torch.tensor([120000, 120007], dtype=torch.int64)
        )
        with self.assertRaisesRegex(RuntimeError, "outside the physical device pool"):
            HybridCacheController._validate_translated_indices(
                entry, torch.tensor([120008], dtype=torch.int64)
            )

    def test_start_writing_translates_before_backup_and_waits_before_ack(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        virtual = torch.tensor([3, 5], dtype=torch.int64)
        physical = torch.tensor([103, 105], dtype=torch.int64)
        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        operation = CacheOperation(host_indices, virtual, node_id=7)
        controller.write_queue = [operation]
        controller.io_backend = "kernel"
        controller.mem_pool_host = MagicMock(
            layout="page_first", can_use_write_back_jit=True
        )
        controller.mem_pool_device = MagicMock()
        controller.has_draft = False
        controller.write_stream = MagicMock()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = MagicMock()

        calls = MagicMock()
        controller.translate_hybrid_device_indices = calls.translate
        calls.translate.return_value = (physical, None)
        controller.mem_pool_host.backup_from_device_all_layer = calls.backup
        controller._finish_transfer_before_scheduler = calls.wait_for_finish
        start_event, ack_start_event, finish_event = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        finish_event.record = calls.record_finish

        with (
            patch.object(
                hybrid_cache_controller,
                "make_timing_event_pair",
                return_value=(ack_start_event, finish_event, False),
            ),
            patch.object(
                hybrid_cache_controller.device_module, "Event", return_value=start_event
            ),
            patch.object(
                hybrid_cache_controller.device_module,
                "stream",
                return_value=nullcontext(),
            ),
        ):
            controller.start_writing()

        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["translate", "backup", "record_finish", "wait_for_finish"],
        )
        calls.translate.assert_called_once_with(operation)
        calls.backup.assert_called_once_with(
            controller.mem_pool_device,
            host_indices,
            physical,
            "kernel",
            pool_transfers=None,
        )
        calls.wait_for_finish.assert_called_once_with(finish_event, physical, None)
        self.assertEqual(len(controller.ack_write_queue), 1)

    def test_start_loading_translates_before_load_and_waits_before_ack(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        virtual = torch.tensor([4], dtype=torch.int64)
        physical = torch.tensor([204], dtype=torch.int64)
        host_indices = torch.tensor([2], dtype=torch.int64)
        operation = CacheOperation(host_indices, virtual, node_id=9)
        controller.load_queue = [operation]
        controller.mem_pool_device = MagicMock()
        controller.mem_pool_host = MagicMock()
        controller.io_backend = "kernel"
        controller.has_draft = False
        controller.has_mtp_draft = False
        controller.load_stream = MagicMock()
        controller.layer_num = 2
        controller.ack_load_queue = []
        controller._record_transfer_indices_on_stream = MagicMock()

        producer_event = MagicMock()
        controller.layer_done_counter = MagicMock(
            events=[producer_event], update_producer=MagicMock(return_value=0)
        )
        calls = MagicMock()
        controller.move_hybrid_indices = calls.translate
        calls.translate.return_value = (host_indices, physical, None)
        controller.mem_pool_host.load_to_device_per_layer = calls.load_layer
        controller._finish_transfer_before_scheduler = calls.wait_for_finish
        ack_start_event, ack_finish_event = MagicMock(), MagicMock()
        ack_finish_event.record = calls.record_finish

        with (
            patch.object(
                hybrid_cache_controller,
                "make_timing_event_pair",
                return_value=(ack_start_event, ack_finish_event, False),
            ),
            patch.object(
                hybrid_cache_controller.device_module,
                "stream",
                return_value=nullcontext(),
            ),
        ):
            producer_id = controller.start_loading()

        self.assertEqual(producer_id, 0)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            [
                "translate",
                "load_layer",
                "load_layer",
                "record_finish",
                "wait_for_finish",
            ],
        )
        for layer_id in range(2):
            self.assertEqual(
                calls.load_layer.call_args_list[layer_id].args[1:5],
                (host_indices, physical, layer_id, "kernel"),
            )
        calls.wait_for_finish.assert_called_once_with(ack_finish_event, physical, None)
        self.assertEqual(len(controller.ack_load_queue), 1)

    def test_failed_write_drains_stream_before_canceling_host_chunk_pins(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        host_indices = torch.tensor([0], dtype=torch.int64)
        device_indices = torch.tensor([3], dtype=torch.int64)
        controller.write_queue = [
            CacheOperation(host_indices, device_indices, node_id=1)
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = MagicMock(
            layout="page_first", can_use_write_back_jit=True
        )
        controller.mem_pool_device = MagicMock()
        controller.has_draft = False
        controller.write_stream = MagicMock()
        controller.ack_write_queue = []
        controller.translate_hybrid_device_indices = MagicMock(
            return_value=(device_indices, None)
        )
        guards = [(MagicMock(), (1,))]
        controller.mem_pool_host.pin_transfer_chunks.return_value = guards
        controller.mem_pool_host.backup_from_device_all_layer.side_effect = (
            RuntimeError("injected D2H failure")
        )

        with (
            patch.object(
                hybrid_cache_controller,
                "make_timing_event_pair",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch.object(
                hybrid_cache_controller.device_module,
                "Event",
                return_value=MagicMock(),
            ),
            patch.object(
                hybrid_cache_controller.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            self.assertRaisesRegex(RuntimeError, "injected D2H failure"),
        ):
            controller.start_writing()

        controller.write_stream.synchronize.assert_called_once_with()
        controller.mem_pool_host.cancel_transfer_chunk_pins.assert_called_once_with(
            guards
        )
        controller.mem_pool_host.release_transfer_chunks_after_event.assert_not_called()
        self.assertEqual(controller.ack_write_queue, [])

    def test_failed_load_drains_stream_before_canceling_host_chunk_pins(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        host_indices = torch.tensor([2], dtype=torch.int64)
        device_indices = torch.tensor([4], dtype=torch.int64)
        controller.load_queue = [
            CacheOperation(host_indices, device_indices, node_id=1)
        ]
        controller.mem_pool_device = MagicMock()
        controller.mem_pool_host = MagicMock()
        controller.io_backend = "kernel"
        controller.has_draft = False
        controller.load_stream = MagicMock()
        controller.layer_num = 2
        controller.ack_load_queue = []
        controller.move_hybrid_indices = MagicMock(
            return_value=(host_indices, device_indices, None)
        )
        producer_event = MagicMock()
        controller.layer_done_counter = MagicMock(
            events=[producer_event], update_producer=MagicMock(return_value=0)
        )
        guards = [(MagicMock(), (2,))]
        controller.mem_pool_host.pin_transfer_chunks.return_value = guards
        controller.mem_pool_host.load_to_device_per_layer.side_effect = RuntimeError(
            "injected H2D failure"
        )

        with (
            patch.object(
                hybrid_cache_controller,
                "make_timing_event_pair",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch.object(
                hybrid_cache_controller.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            self.assertRaisesRegex(RuntimeError, "injected H2D failure"),
        ):
            controller.start_loading()

        controller.load_stream.synchronize.assert_called_once_with()
        controller.mem_pool_host.cancel_transfer_chunk_pins.assert_called_once_with(
            guards
        )
        controller.mem_pool_host.release_transfer_chunks_after_event.assert_not_called()
        self.assertEqual(controller.ack_load_queue, [])


class TestUnifiedMambaCrossPoolEviction(unittest.TestCase):
    @staticmethod
    def _req(*, active=False, tracking=False):
        return SimpleNamespace(
            req_pool_idx=None,
            mamba_pool_idx=object() if active else None,
            mamba_ping_pong_track_buffer=object() if tracking else None,
        )

    def test_request_slot_preallocation_reuses_mamba_before_full(self):
        req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        req_pool.enable_mamba_extra_buffer = True
        req_pool.enable_mamba_extra_buffer_lazy = False
        req_pool.mamba_ping_pong_track_buffer_size = 2
        req_pool.mamba_allocator = MagicMock()
        req_pool.mamba_allocator.schedulable_available_size.side_effect = [1, 4]
        req_pool.mamba_allocator.available_size.side_effect = [3, 6]
        req_pool.alloc = MagicMock(return_value=[3, 4])

        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True
        tree_cache.mamba_evictable_size.return_value = 3
        tree_cache.token_to_kv_pool_allocator.full_tokens_for_mamba_slots.side_effect = (
            lambda slots: 7 * slots
        )

        self.assertEqual(
            alloc_req_slots(req_pool, [self._req(), self._req()], tree_cache), [3, 4]
        )
        first, second = [call.args[0] for call in tree_cache.evict.call_args_list]
        self.assertEqual((first.num_tokens, first.mamba_num), (0, 3))
        # The three reclaimed rows resolve ID pressure and create physical holes;
        # only the two-row physical residual needs Full shared bytes.
        self.assertEqual((second.num_tokens, second.mamba_num), (14, 0))

    def test_preallocated_active_states_are_not_charged_again(self):
        req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        req_pool.enable_mamba_extra_buffer = True
        req_pool.enable_mamba_extra_buffer_lazy = False
        req_pool.mamba_ping_pong_track_buffer_size = 2
        req_pool.mamba_allocator = MagicMock()
        req_pool.mamba_allocator.schedulable_available_size.side_effect = [1, 4]
        req_pool.mamba_allocator.available_size.side_effect = [1, 4]
        req_pool.alloc = MagicMock(return_value=[3, 4])
        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True
        tree_cache.mamba_evictable_size.return_value = 3
        tree_cache.token_to_kv_pool_allocator.full_tokens_for_mamba_slots.return_value = 0

        alloc_req_slots(
            req_pool,
            [self._req(active=True), self._req(active=True)],
            tree_cache,
        )

        params = tree_cache.evict.call_args_list[0].args[0]
        self.assertEqual((params.num_tokens, params.mamba_num), (0, 3))

    def test_mamba_slot_reuses_evictable_mamba_state_first(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost.return_value = 1590
        component.cache.mamba_evictable_size.return_value = 1
        component.cache.req_to_token_pool.mamba_allocator.available_size.return_value = 0

        params = component._mamba_slot_eviction_params()

        self.assertEqual(params.num_tokens, 0)
        self.assertEqual(params.mamba_num, 1)

    def test_mamba_slot_uses_full_bytes_when_no_mamba_is_evictable(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost.return_value = 1590
        component.cache.mamba_evictable_size.return_value = 0
        component.cache.req_to_token_pool.mamba_allocator.available_size.return_value = 1

        params = component._mamba_slot_eviction_params()

        self.assertEqual(params.num_tokens, 1590)
        self.assertEqual(params.mamba_num, 0)

    def test_transfer_fenced_mamba_eviction_falls_back_to_full_gap(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        component.cache.mamba_evictable_size.return_value = 1
        component.cache.req_to_token_pool.mamba_allocator.available_size.return_value = 0
        component.cache.token_to_kv_pool_allocator.full_tokens_for_mamba_slots.return_value = 1590
        component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost.return_value = 1590
        allocated = torch.tensor([9])
        component.cache.req_to_token_pool.mamba_allocator.alloc.side_effect = [
            None,
            None,
            None,
            allocated,
        ]

        self.assertIs(component._alloc_mamba_slot(), allocated)

        first, second = [call.args[0] for call in component.cache.evict.call_args_list]
        self.assertEqual((first.num_tokens, first.mamba_num), (0, 1))
        self.assertEqual((second.num_tokens, second.mamba_num), (1590, 0))
        component.cache.writing_check.assert_called_once_with(write_back=True)

    def test_full_id_exhaustion_retries_mamba_after_write_ack(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        component.cache.mamba_evictable_size.return_value = 0
        component.cache.req_to_token_pool.mamba_allocator.available_size.return_value = 0
        component.cache.token_to_kv_pool_allocator.full_tokens_for_mamba_slots.return_value = 1590
        component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost.return_value = 1590
        allocated = torch.tensor([9])
        component.cache.req_to_token_pool.mamba_allocator.alloc.side_effect = [
            None,
            None,
            None,
            allocated,
        ]

        self.assertIs(component._alloc_mamba_slot(), allocated)

        first, second = [call.args[0] for call in component.cache.evict.call_args_list]
        self.assertEqual((first.num_tokens, first.mamba_num), (0, 1))
        self.assertEqual((second.num_tokens, second.mamba_num), (1590, 0))
        component.cache.writing_check.assert_called_once_with(write_back=True)

    def test_mamba_slot_flushes_batch_deferred_full_eviction_before_retry(self):
        component = MambaComponent.__new__(MambaComponent)
        allocated = torch.tensor([9])

        class _Allocator:
            def __init__(self):
                self.flushed = False

            def mamba_slot_full_token_cost(self):
                return 1590

            def flush_deferred_frees(self):
                self.flushed = True

        token_allocator = _Allocator()
        mamba_allocator = MagicMock()
        mamba_allocator.alloc.side_effect = lambda _: (
            allocated if token_allocator.flushed else None
        )
        component.cache = MagicMock()
        component.cache.token_to_kv_pool_allocator = token_allocator
        component.cache.req_to_token_pool.mamba_allocator = mamba_allocator
        mamba_allocator.available_size.return_value = 1
        component.cache.mamba_evictable_size.return_value = 0

        self.assertIs(component._alloc_mamba_slot(), allocated)
        component.cache.evict.assert_called_once()
        self.assertTrue(token_allocator.flushed)

    def test_static_pool_does_not_evict_full_tokens(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        del component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost

        params = component._mamba_slot_eviction_params()

        self.assertEqual(params.num_tokens, 0)
        self.assertEqual(params.mamba_num, 1)


class TestUnifiedMHAHostLoad(unittest.TestCase):
    def test_load_honors_strided_unified_destination(self):
        host_pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host_pool.page_size = 1
        host_pool.layout = "page_first"
        host_pool.k_data_refs = [torch.arange(12).view(3, 2, 2)]
        host_pool.v_data_refs = [torch.arange(12, 24).view(3, 2, 2)]

        # Selecting one field from each shared entry models unified memory's
        # non-contiguous token-row stride.
        backing_k = torch.full((4, 3, 2, 2), -1, dtype=torch.long)
        backing_v = torch.full((4, 3, 2, 2), -1, dtype=torch.long)
        device_pool = MagicMock()
        device_pool._unified_buffer = object()
        device_pool.device = torch.device("cpu")
        device_pool.k_buffer = [backing_k[:, 1]]
        device_pool.v_buffer = [backing_v[:, 1]]

        host_pool.load_to_device_per_layer(
            device_pool=device_pool,
            host_indices=torch.tensor([2, 0]),
            device_indices=torch.tensor([1, 3]),
            layer_id=0,
            io_backend="kernel",
        )

        torch.testing.assert_close(
            device_pool.k_buffer[0][1], host_pool.k_data_refs[0][2]
        )
        torch.testing.assert_close(
            device_pool.k_buffer[0][3], host_pool.k_data_refs[0][0]
        )
        torch.testing.assert_close(
            device_pool.v_buffer[0][1], host_pool.v_data_refs[0][2]
        )

    def test_mamba_load_honors_strided_unified_destination(self):
        src = torch.arange(3 * 2 * 1 * 4).view(3, 2, 1, 4)
        backing = torch.full((4, 3, 4), -1, dtype=torch.long)
        dst = backing[:, 1]

        MambaPoolHost._copy_tensor_pf_lf(
            src=src,
            dst=dst,
            src_indices=torch.tensor([2, 0]),
            dst_indices=torch.tensor([1, 3]),
            layer_id=1,
            io_backend="kernel",
        )

        torch.testing.assert_close(dst[1], src[2, 1, 0])
        torch.testing.assert_close(dst[3], src[0, 1, 0])
        self.assertTrue(torch.all(backing[:, 0] == -1))
        self.assertTrue(torch.all(backing[:, 2] == -1))


class TestUnifiedHiCacheServerArgs(unittest.TestCase):
    def test_write_back_is_accepted(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            hicache_write_policy="write_back",
            page_size=1,
        )

        args._handle_unified_memory_pool()

    def test_l3_storage_backend_is_rejected(self):
        from sglang.srt.server_args import ServerArgs

        with self.assertRaisesRegex(AssertionError, "only the L2 host tier"):
            ServerArgs(
                model_path="dummy",
                enable_unified_memory=True,
                enable_hierarchical_cache=True,
                hicache_io_backend="kernel",
                hicache_write_policy="write_through",
                hicache_storage_backend="file",
                page_size=1,
            )._handle_unified_memory_pool()

    def test_unsupported_transfer_combinations_are_rejected(self):
        from sglang.srt.server_args import ServerArgs

        invalid_cases = (
            ({"hicache_io_backend": "direct"}, "hicache-io-backend kernel"),
            (
                {"hicache_write_policy": "write_through_selective"},
                "write_through or write_back",
            ),
        )
        for overrides, message in invalid_cases:
            kwargs = {
                "model_path": "dummy",
                "enable_unified_memory": True,
                "enable_hierarchical_cache": True,
                "hicache_io_backend": "kernel",
                "hicache_write_policy": "write_through",
                "page_size": 1,
                **overrides,
            }
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(AssertionError, message):
                    ServerArgs(**kwargs)._handle_unified_memory_pool()

    def test_multi_token_pages_are_supported(self):
        from sglang.srt.server_args import ServerArgs

        for page_size in (1, 2, 4, 8, 16, 32):
            with self.subTest(page_size=page_size):
                ServerArgs(
                    model_path="dummy",
                    enable_unified_memory=True,
                    enable_hierarchical_cache=True,
                    hicache_io_backend="kernel",
                    hicache_write_policy="write_through",
                    page_size=page_size,
                )._handle_unified_memory_pool()

    def test_pd_disaggregation_with_hicache_is_rejected(self):
        from sglang.srt.server_args import ServerArgs

        with self.assertRaisesRegex(AssertionError, "HiCache.*PD disaggregation"):
            ServerArgs(
                model_path="dummy",
                enable_unified_memory=True,
                enable_hierarchical_cache=True,
                hicache_io_backend="kernel",
                hicache_write_policy="write_back",
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
            )._handle_unified_memory_pool()

    def test_pd_disaggregation_remains_supported_without_hicache(self):
        from sglang.srt.server_args import ServerArgs

        ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
        )._handle_unified_memory_pool()

    def test_dspark_with_hicache_is_rejected(self):
        from sglang.srt.server_args import ServerArgs

        with self.assertRaisesRegex(AssertionError, "HiCache.*speculative decoding"):
            ServerArgs(
                model_path="dummy",
                enable_unified_memory=True,
                enable_hierarchical_cache=True,
                hicache_io_backend="kernel",
                hicache_write_policy="write_back",
                speculative_algorithm="DSPARK",
                speculative_eagle_topk=1,
                attention_backend="triton",
            )._handle_unified_memory_pool()

    def test_dspark_remains_supported_without_hicache(self):
        from sglang.srt.server_args import ServerArgs

        ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            speculative_algorithm="DSPARK",
            speculative_eagle_topk=1,
            attention_backend="triton",
        )._handle_unified_memory_pool()


if __name__ == "__main__":
    unittest.main()
