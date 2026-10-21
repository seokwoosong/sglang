"""Request-table overhead must fit alongside the profiled KV allocation."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor import pool_configurator as pc
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, stage="base-a", runner_config="cpu")

_GIB = 1 << 30


def _pp_budget_worker(rank, directory, cell_sizes, memory_fractions):
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + directory + "/init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    collectives = []
    all_gather = torch.distributed.all_gather
    all_reduce = torch.distributed.all_reduce

    def traced_all_reduce(*args, **kwargs):
        collectives.append(str(kwargs.get("op", torch.distributed.ReduceOp.SUM)))
        return all_reduce(*args, **kwargs)

    def traced_all_gather(*args, **kwargs):
        collectives.append("all_gather")
        return all_gather(*args, **kwargs)

    try:
        test = TestRequestTableBudget()
        is_dsv4 = cell_sizes[rank] is None
        pool = test.dsv4_pool() if is_dsv4 else None
        configurator = test.make_configurator(
            context=262144 if None in cell_sizes else 1048576
        )
        selected = []
        original_budget = kcc.KVCacheConfigurator._budget_with_request_tables

        def capture_budget(self, available):
            value = original_budget(self, available)
            selected.append(value)
            return value

        with (
            patch.object(
                kcc.KVCacheConfigurator, "_budget_with_request_tables", capture_budget
            ),
            get_parallel().override(pp_size=2),
            patch.object(
                torch.distributed, "all_gather", side_effect=traced_all_gather
            ),
            patch.object(
                torch.distributed, "all_reduce", side_effect=traced_all_reduce
            ),
        ):
            try:
                config, table = test.resolve(
                    configurator,
                    cell_size=cell_sizes[rank],
                    pool=pool,
                    page_size=128 if None in cell_sizes else 1,
                    enable_unified_memory=None in cell_sizes,
                    mem_fraction_static=memory_fractions[rank],
                )
            except ValueError as error:
                Path(directory, f"{rank}.json").write_text(
                    json.dumps(dict(error=str(error), collectives=collectives))
                )
                return
        Path(directory, f"{rank}.json").write_text(
            json.dumps(
                dict(
                    selected=selected[-1],
                    tokens=config.max_total_num_tokens,
                    requests=config.max_running_requests,
                    table=table,
                    kv=(
                        config.max_total_num_tokens * cell_sizes[rank]
                        if not is_dsv4
                        else int(
                            config.max_total_num_tokens * pool.bytes_per_full_token
                            + pool._fixed_swa_bytes(config.max_running_requests)
                            + pool._fixed_c4_state_bytes(config.max_running_requests)
                        )
                    ),
                    available=int(32 * memory_fractions[rank] * _GIB),
                    collectives=collectives,
                )
            )
        )
    finally:
        torch.distributed.destroy_process_group()


class TestRequestTableBudget(CustomTestCase):
    def make_configurator(self, context=1048576, existing=None):
        c = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        for key, value in dict(
            device="cpu",
            gpu_id=0,
            attn_dp_size=1,
            pp_size=1,
            model_config=NS(context_len=context, is_multimodal=False),
            mambaish_config=None,
            post_capture_kv_active=False,
            req_to_token_pool=existing,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        ).items():
            setattr(c, key, value)
        return c

    def resolve(self, c, *, cell_size=2048, pool=None, **overrides):
        if pool is None:
            pool = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
            pool._fixed_overhead_bytes = 0
            pool._cell_size = cell_size
            pool._zero_kv_max_tokens = (
                torch.iinfo(torch.int64).max
                if cell_size == 0
                else c.model_config.context_len
            )
        args = dict(
            mem_fraction_static=0.85,
            max_running_requests=None,
            max_total_tokens=None,
            page_size=1,
            enable_unified_memory=False,
        )
        args.update(overrides)
        with (
            get_context().override_server_args(**args),
            get_exec().graph.override(
                cuda_graph_config=NS(
                    prefill=NS(full_prefill_max_req=None, backend=None)
                )
            ),
            get_parallel().override(world_group=NS(world_size=1, cpu_group=None)),
            patch.object(kcc, "get_available_gpu_memory", return_value=32),
            patch.object(pc, "create_memory_pool_configurator", return_value=pool),
        ):
            result = c._resolve_memory_pool_config(32)
            table_bytes = (
                (
                    result.max_running_requests
                    + (overrides.get("disaggregation_decode_extra_slots") or 0)
                    + 1
                )
                * (c.model_config.context_len + 4)
                * 4
            )
        return result, table_bytes

    def test_long_context_fits_static_budget(self):
        c = self.make_configurator()
        config, table = self.resolve(c)
        self.assertLessEqual(
            config.max_total_num_tokens * 2048 + table, int(32 * 0.85 * _GIB)
        )
        self.assertGreater(config.max_total_num_tokens, 0)

    def test_explicit_request_count_above_default_upper_bound(self):
        c = self.make_configurator(context=65536)
        config, table = self.resolve(c, max_running_requests=8192)
        self.assertEqual(config.max_running_requests, 8192)
        self.assertLessEqual(
            config.max_total_num_tokens * 2048 + table, int(32 * 0.85 * _GIB)
        )

    def test_token_cap_does_not_truncate_table_width(self):
        c = self.make_configurator()
        config, table = self.resolve(c, max_total_tokens=8192)
        self.assertEqual(config.max_total_num_tokens, 8192)
        self.assertEqual(table, (config.max_running_requests + 1) * (1048576 + 4) * 4)

    def test_shared_request_table_is_not_reserved_twice(self):
        c = self.make_configurator(existing=object())
        config, _ = self.resolve(c)
        self.assertEqual(config.max_total_num_tokens, int(32 * 0.85 * _GIB) // 2048)

    def test_pd_extra_slots_are_charged(self):
        c = self.make_configurator(context=65536)
        config, table = self.resolve(
            c, disaggregation_mode="decode", disaggregation_decode_extra_slots=1024
        )
        self.assertEqual(
            table, (config.max_running_requests + 1024 + 1) * (65536 + 4) * 4
        )
        self.assertLessEqual(
            config.max_total_num_tokens * 2048 + table, int(32 * 0.85 * _GIB)
        )

    def test_upper_bound_larger_than_budget_can_still_fit(self):
        c = self.make_configurator()
        config, table = self.resolve(c, mem_fraction_static=0.375)
        self.assertGreater(config.max_total_num_tokens, 1_500_000)
        self.assertLessEqual(config.max_total_num_tokens * 2048 + table, 12 * _GIB)

    def test_reservation_does_not_strand_gigabytes(self):
        c = self.make_configurator()
        config, table = self.resolve(c)
        slack = int(32 * 0.85 * _GIB) - config.max_total_num_tokens * 2048 - table
        self.assertGreaterEqual(slack, 0)
        self.assertLess(slack, (1048576 + 4) * 4 + 4096)

    def test_compact_draft_reserves_a_second_full_context_table(self):
        c = self.make_configurator(context=65536)
        c.spec_algorithm = SpeculativeAlgorithm.DFLASH
        with get_context().override_server_args(
            page_size=1, speculative_draft_window_size=128
        ):
            self.assertEqual(c._req_to_token_bytes(7), 2 * 8 * (65536 + 4) * 4)
        with get_context().override_server_args(
            page_size=1, speculative_draft_window_size=None
        ):
            self.assertEqual(c._req_to_token_bytes(7), 8 * (65536 + 4) * 4)

    def test_dspark_window_keeps_the_shared_table_budget(self):
        c = self.make_configurator(context=65536)
        c.spec_algorithm = SpeculativeAlgorithm.DSPARK
        with get_context().override_server_args(
            page_size=1, speculative_draft_window_size=128
        ):
            self.assertEqual(c._req_to_token_bytes(7), 8 * (65536 + 4) * 4)
        with_window, _ = self.resolve(c, speculative_draft_window_size=128)
        without_window, _ = self.resolve(c, speculative_draft_window_size=None)
        self.assertEqual(with_window, without_window)

    def test_compact_draft_real_factory_tables_have_independent_storage(self):
        target = self.make_configurator(context=128)
        draft = self.make_configurator(context=128)
        target.spec_algorithm = draft.spec_algorithm = SpeculativeAlgorithm.DFLASH
        draft.is_draft_worker = True
        with get_context().override_server_args(
            page_size=16,
            speculative_algorithm="DFLASH",
            speculative_num_draft_tokens=8,
            speculative_draft_window_size=32,
        ):
            tables = [
                c._build_req_to_token_pool(max_num_reqs=7).req_to_token
                for c in (target, draft)
            ]
            self.assertEqual(tables[0].shape, tables[1].shape)
            self.assertNotEqual(tables[0].data_ptr(), tables[1].data_ptr())
            self.assertEqual(
                target._req_to_token_bytes(7),
                sum(t.untyped_storage().nbytes() for t in tables),
            )
        with get_context().override_server_args(
            page_size=16,
            speculative_algorithm="DFLASH",
            speculative_num_draft_tokens=8,
            speculative_draft_window_size=None,
        ):
            self.assertEqual(
                target._req_to_token_bytes(7), tables[0].untyped_storage().nbytes()
            )

    def test_compact_draft_tables_and_kv_share_one_budget(self):
        c = self.make_configurator()
        c.spec_algorithm = SpeculativeAlgorithm.DFLASH
        config, one_table = self.resolve(c, speculative_draft_window_size=128)
        self.assertLessEqual(
            config.max_total_num_tokens * 2048 + 2 * one_table,
            int(32 * 0.85 * _GIB),
        )
        self.assertGreater(config.max_total_num_tokens, 0)

    def resolve_pp(self, cell_sizes, memory_fractions):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _pp_budget_worker,
                args=(directory, cell_sizes, memory_fractions),
                nprocs=2,
                join=True,
            )
            return [
                json.loads(Path(directory, f"{rank}.json").read_text())
                for rank in range(2)
            ]

    def assert_pp_consensus(self, rows):
        for key in ("tokens", "requests", "table", "collectives"):
            self.assertEqual(rows[0][key], rows[1][key])
        self.assertTrue(rows[0]["collectives"])
        for row in rows:
            self.assertGreater(row["tokens"], 0)
            self.assertLessEqual(row["kv"] + row["table"], row["available"])

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_pp_stage_without_kv_uses_common_capacity(self):
        rows = self.resolve_pp((0, 2048), (0.375, 0.375))
        self.assert_pp_consensus(rows)
        self.assertGreater(rows[1]["tokens"], 1_500_000)
        self.assertEqual(rows[1]["requests"], 2048)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_pp_heterogeneous_budgets_fit_each_rank(self):
        rows = self.resolve_pp((2048, 4096), (0.375, 0.625))
        self.assert_pp_consensus(rows)
        self.assertNotEqual(rows[0]["available"], rows[1]["available"])
        self.assertNotEqual(rows[0]["kv"], rows[1]["kv"])

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_pp_one_rank_insufficient_terminates_together(self):
        rows = self.resolve_pp((2048, 4096), (0.375, 0.0001))
        self.assertEqual(rows[0]["error"], rows[1]["error"])
        self.assertIn("No KV capacity", rows[0]["error"])
        self.assertEqual(rows[0]["collectives"], rows[1]["collectives"])
        self.assertLess(len(rows[0]["collectives"]), 200)

    def test_only_final_request_limit_warning_is_emitted(self):
        c = self.make_configurator(context=65536)
        with (
            patch.object(
                kcc.KVCacheConfigurator,
                "_profile_available_bytes",
                return_value=8 << 20,
            ),
            self.assertLogs(kcc.logger, level="WARNING") as logs,
        ):
            config, _ = self.resolve(c, max_running_requests=8192)
        self.assertEqual(config.max_running_requests, 30)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("requested 8192 to 30", logs.output[0])

    def test_repeated_resolution_is_stable(self):
        c = self.make_configurator(context=65536)
        first, _ = self.resolve(c, max_running_requests=37, page_size=16)
        second, _ = self.resolve(c, max_running_requests=37, page_size=16)
        self.assertEqual(first, second)
        self.assertEqual(first.max_total_num_tokens % 16, 0)

    def test_factory_table_matches_estimate_with_pd_extra_slots(self):
        c = self.make_configurator(context=128)
        with get_context().override_server_args(
            page_size=16,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=11,
        ):
            pool = c._build_decode_req_pool(
                max_num_reqs=7,
                extra_max_context_len=kcc.get_req_to_token_extra_context_len(),
                pre_alloc_size=11,
            )
            self.assertEqual(pool.req_to_token.dtype, torch.int32)
            self.assertEqual(pool.req_to_token.device.type, "cpu")
            self.assertEqual(
                c._req_to_token_bytes(7), pool.req_to_token.untyped_storage().nbytes()
            )

    @staticmethod
    def dsv4_pool():
        pool = pc.DSV4PoolConfigurator.__new__(pc.DSV4PoolConfigurator)
        fields = dict(
            attn_head_dim=192,
            indexer_head_dim=128,
            num_layers_total=1,
            stage_owner_layers={4: [0]},
            ring_sizes={4: 8, 128: 128},
            _swa_ring_size=128,
            _spec_infl=1.0,
            context_len=262144,
            bytes_per_full_token=576.0,
            requested_max_running_requests_per_worker=None,
            swa_ratio=0.1,
            sliding_window_size=4096,
            page_size=128,
            c4_shrink_factor=1,
            online_c128_mtp_max_draft_tokens=0,
            disaggregation_mode=None,
            disaggregation_decode_extra_slots=0,
            _unified=True,
            operator_swa_ratio=None,
            swa_cap_tokens=None,
            swa_prefix_tails=0,
            request_window_bytes=0,
            bytes_per_swa_token=0.0,
            _unified_fp8=False,
            _dspark_draft_on_bf16=False,
            _unified_row_bytes=384,
        )
        for key, value in fields.items():
            setattr(pool, key, value)
        return pool

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_pp_dsv4_heterogeneous_transition_union(self):
        fractions = (3022354488 / (32 * _GIB), 3025336408 / (32 * _GIB))
        rows = self.resolve_pp((None, None), fractions)
        self.assert_pp_consensus(rows)
        self.assert_dsv4_pp_oracle(rows, (None, None))
        self.assertIn("all_gather", rows[0]["collectives"])
        self.assertLess(len(rows[0]["collectives"]), 10000)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_pp_dsv4_duplicate_and_one_rank_only_boundaries(self):
        for cells in ((None, None), (None, 0)):
            with self.subTest(cells=cells):
                rows = self.resolve_pp(cells, (3022354488 / (32 * _GIB),) * 2)
                self.assert_pp_consensus(rows)
                self.assert_dsv4_pp_oracle(rows, cells)

    def assert_dsv4_pp_oracle(self, rows, cells):
        from sglang.srt.environ import envs

        available = [row["available"] for row in rows]
        steps = (max(available) + 4095) // 4096
        table_min = 2049 * (262144 + 4) * 4
        upper = min((a - table_min + 4096) * steps // a + 2 for a in available)
        lower = min(
            max(0, row["selected"] - (4 << 20)) * steps // a
            for row, a in zip(rows, available)
        )
        best = None
        pool = self.dsv4_pool()
        with envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(False):
            for step in range(lower, min(upper, steps) + 1):
                budgets = [a * step // steps // 4096 * 4096 for a in available]
                tokens = min(
                    pool.calculate_pool_sizes(b, 128).max_total_num_tokens
                    if cell is None
                    else torch.iinfo(torch.int64).max
                    for b, cell in zip(budgets, cells)
                )
                self.assertGreaterEqual(tokens, 4096)
                reqs = min(
                    max(min(int(tokens / 262144 * 512), 4096), 2048), tokens // 2
                )
                table = (reqs + 1) * (262144 + 4) * 4
                if all(b + table <= a for b, a in zip(budgets, available)):
                    best = budgets
        self.assertEqual([row["selected"] for row in rows], best)

    def test_dsv4_nonmonotone_boundary_matches_exhaustive_oracle(self):
        from sglang.srt.environ import envs

        pool = self.dsv4_pool()
        c = self.make_configurator(context=pool.context_len)
        available = 3022354488

        def cost(budget):
            config = c.config_from_budget(budget, sync_pp=False)
            requests = c.resolve_max_num_reqs(config.max_total_num_tokens)
            # Independent table shape, not the reservation helper.
            return budget + (requests + 1) * (262144 + 4) * 4

        with (
            get_context().override_server_args(
                page_size=128,
                max_total_tokens=None,
                max_running_requests=None,
                enable_unified_memory=True,
            ),
            get_parallel().override(pp_size=1),
            envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(False),
            patch.object(pc, "create_memory_pool_configurator", return_value=pool),
        ):
            self.assertGreater(cost(871165952), available)
            self.assertLessEqual(cost(871170048), available)
            # All budgets above lower have >4096 tokens, hence >=2048 requests.
            # The minimum table bounds the highest possible safe byte budget.
            lower = 867000320
            upper = (available - 2049 * (262144 + 4) * 4) // 4096 * 4096
            budgets = range(lower, upper + 1, 4096)
            self.assertTrue(
                all(
                    c.config_from_budget(b, sync_pp=False).max_total_num_tokens >= 4096
                    for b in budgets
                )
            )
            expected = max(b for b in budgets if cost(b) <= available)
            selected = c._budget_with_request_tables(available)
            self.assertEqual(selected, expected)
            self.assertEqual(selected, 871202816)
            config = c.config_from_budget(selected, sync_pp=False)
            config.max_running_requests = c.resolve_max_num_reqs(
                config.max_total_num_tokens
            )
            config = pool.finalize_with_max_running_requests(config)
            reqs = config.max_running_requests
            actual = (
                config.max_total_num_tokens * 576
                + pool._fixed_swa_bytes(reqs)
                + pool._fixed_c4_state_bytes(reqs)
                + (reqs + 1) * (262144 + 4) * 4
            )
            self.assertLessEqual(actual, available)

    def test_impossible_budget_has_actionable_error(self):
        c = self.make_configurator()
        with self.assertRaisesRegex(ValueError, "No KV capacity"):
            self.resolve(c, mem_fraction_static=0.0001)

    def test_final_budget_with_factory_allocated_table(self):
        for page in (1, 16):
            for requests in (None, 7):
                for cap in (None, 8192):
                    with self.subTest(page=page, requests=requests, cap=cap):
                        c = self.make_configurator(context=128)
                        config, _ = self.resolve(
                            c,
                            page_size=page,
                            max_running_requests=requests,
                            max_total_tokens=cap,
                        )
                        with get_context().override_server_args(
                            page_size=page, disaggregation_mode="null"
                        ):
                            table = c._build_req_to_token_pool(
                                max_num_reqs=config.max_running_requests
                            ).req_to_token
                        self.assertEqual(table.dtype, torch.int32)
                        self.assertEqual(table.device.type, "cpu")
                        self.assertLessEqual(
                            config.max_total_num_tokens * 2048
                            + table.untyped_storage().nbytes(),
                            int(32 * 0.85 * _GIB),
                        )

    def test_post_capture_does_not_charge_resident_table_again(self):
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            CudaGraphConfig,
            PhaseConfig,
        )
        from sglang.srt.model_executor.model_runner_components import kv_pool_runtime

        c = self.make_configurator(context=128, existing=object())
        pool = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        pool._cell_size = 2048
        pool._fixed_overhead_bytes = 0
        pool._zero_kv_max_tokens = 128
        backing = Mock(post_capture_backed_bytes=2 * _GIB, dtype="bfloat16")
        runner = NS(
            token_to_kv_pool=backing,
            device="cpu",
            gpu_id=0,
            pre_model_load_memory=32,
            mem_fraction_static=0.875,
            max_running_requests=None,
            model_config=c.model_config,
            sampling_prewarm_result=NS(sampling_headroom_bytes=0),
            canary_manager=None,
            kv_cache_configurator=c,
            max_total_num_tokens=20_000_000,
            token_to_kv_pool_allocator=Mock(),
            req_to_token_pool=Mock(),
        )
        with (
            get_context().override_server_args(
                page_size=1,
                max_total_tokens=None,
                disaggregation_mode="null",
                cuda_graph_config=CudaGraphConfig(
                    decode=PhaseConfig(backend=Backend.DISABLED)
                ),
            ),
            get_parallel().override(
                pp_size=1, world_group=NS(world_size=1, cpu_group=None)
            ),
            patch.object(pc, "create_memory_pool_configurator", return_value=pool),
            patch.object(kv_pool_runtime.torch.cuda, "synchronize"),
            patch.object(kv_pool_runtime, "get_available_gpu_memory", return_value=20),
            patch.object(kv_pool_runtime, "mambaish_config", return_value=None),
            patch.object(
                kv_pool_runtime, "graph_pool_borrow_enabled", return_value=True
            ),
            patch.object(kv_pool_runtime, "mm_runtime_reservation_gb", return_value=0),
            patch.object(
                kcc.KVCacheConfigurator,
                "_budget_with_request_tables",
                side_effect=AssertionError("resident table charged again"),
            ),
        ):
            first = kv_pool_runtime.compute_post_capture_kv_resize(runner)
            second = kv_pool_runtime.compute_post_capture_kv_resize(runner)
        self.assertEqual(first, second)
        self.assertEqual(first.max_total_num_tokens, (20 - 4 + 2) * _GIB // 2048)
        self.assertEqual(backing.finalize_backing.call_count, 2)

    def test_estimate_matches_real_table_allocation(self):
        c = self.make_configurator(context=128)
        with get_context().override_server_args(
            disaggregation_mode="null", speculative_num_draft_tokens=None, page_size=1
        ):
            pool = ReqToTokenPool(7, 132, "cpu", False)
            self.assertEqual(
                c._req_to_token_bytes(7),
                pool.req_to_token.numel() * pool.req_to_token.element_size(),
            )


if __name__ == "__main__":
    unittest.main()
