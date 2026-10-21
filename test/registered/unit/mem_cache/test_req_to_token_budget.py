"""Request-table overhead must fit alongside the profiled KV allocation."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor import pool_configurator as pc
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, stage="base-a", runner_config="cpu")

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
    all_reduce = torch.distributed.all_reduce

    def traced_all_reduce(*args, **kwargs):
        collectives.append(str(kwargs.get("op", torch.distributed.ReduceOp.SUM)))
        return all_reduce(*args, **kwargs)

    try:
        test = TestRequestTableBudget()
        configurator = test.make_configurator()
        with (
            get_parallel().override(pp_size=2),
            patch.object(
                torch.distributed, "all_reduce", side_effect=traced_all_reduce
            ),
        ):
            try:
                config, table = test.resolve(
                    configurator,
                    cell_size=cell_sizes[rank],
                    mem_fraction_static=memory_fractions[rank],
                )
            except ValueError as error:
                Path(directory, f"{rank}.json").write_text(json.dumps(dict(error=str(error), collectives=collectives)))
                return
        Path(directory, f"{rank}.json").write_text(
            json.dumps(
                dict(
                    tokens=config.max_total_num_tokens,
                    requests=config.max_running_requests,
                    table=table,
                    kv=config.max_total_num_tokens * cell_sizes[rank],
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

    def resolve(self, c, *, cell_size=2048, **overrides):
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
            get_exec().graph.override(cuda_graph_config=NS(prefill=NS(full_prefill_max_req=None, backend=None))),
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

    def test_repeated_resolution_is_stable(self):
        c = self.make_configurator(context=65536)
        first, _ = self.resolve(c, max_running_requests=37, page_size=16)
        second, _ = self.resolve(c, max_running_requests=37, page_size=16)
        self.assertEqual(first, second)
        self.assertEqual(first.max_total_num_tokens % 16, 0)

    def test_factory_table_matches_estimate_with_pd_extra_slots(self):
        c = self.make_configurator(context=128)
        with get_context().override_server_args(page_size=16, disaggregation_mode="decode", disaggregation_decode_extra_slots=11):
            pool = c._build_decode_req_pool(max_num_reqs=7, extra_max_context_len=kcc.get_req_to_token_extra_context_len(), pre_alloc_size=11)
            self.assertEqual(pool.req_to_token.dtype, torch.int32)
            self.assertEqual(pool.req_to_token.device.type, "cpu")
            self.assertEqual(c._req_to_token_bytes(7), pool.req_to_token.untyped_storage().nbytes())

    def test_impossible_budget_has_actionable_error(self):
        c = self.make_configurator()
        with self.assertRaisesRegex(ValueError, "No KV capacity"):
            self.resolve(c, mem_fraction_static=0.0001)

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
