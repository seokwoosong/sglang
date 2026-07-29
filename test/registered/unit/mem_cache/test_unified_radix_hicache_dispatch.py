import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
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
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
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
    def test_unified_transfer_is_finished_before_scheduler_resumes(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        event = MagicMock()

        controller.synchronize_unified_transfers = True
        controller._finish_transfer_before_scheduler(event)
        event.synchronize.assert_called_once_with()

        event.reset_mock()
        controller.synchronize_unified_transfers = False
        controller._finish_transfer_before_scheduler(event)
        event.synchronize.assert_not_called()

    def test_translates_execution_indices_without_mutating_virtual_ids(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.move_indices = lambda host, device: (host, device)

        full_translate = MagicMock(side_effect=lambda x: x + 100)
        mamba_translate = MagicMock(side_effect=lambda x: x + 200)
        full_entry = MagicMock(
            is_primary_index_anchor=True,
            device_index_translate_fn=full_translate,
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
                )
            ],
        )

        _, full_physical, transfers = controller.move_hybrid_indices(operation)

        torch.testing.assert_close(full_physical, torch.tensor([103, 107]))
        torch.testing.assert_close(transfers[0].device_indices, torch.tensor([211]))
        # The operation remains tree/allocator-owned virtual state.
        torch.testing.assert_close(operation.device_indices, full_virtual)
        torch.testing.assert_close(
            operation.pool_transfers[0].device_indices, mamba_virtual
        )

    def test_identity_when_pool_has_no_translator(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.move_indices = lambda host, device: (host, device)
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


class TestUnifiedMambaCrossPoolEviction(unittest.TestCase):
    def test_requests_full_token_equivalent_for_unified_pool(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = MagicMock()
        component.cache.token_to_kv_pool_allocator.mamba_slot_full_token_cost.return_value = (
            1590
        )

        params = component._mamba_slot_eviction_params()

        self.assertEqual(params.num_tokens, 1590)
        self.assertEqual(params.mamba_num, 1)

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
            num_layers=2,
            io_backend="kernel",
        )

        torch.testing.assert_close(dst[1], src[2, 1, 0])
        torch.testing.assert_close(dst[3], src[0, 1, 0])


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


if __name__ == "__main__":
    unittest.main()
