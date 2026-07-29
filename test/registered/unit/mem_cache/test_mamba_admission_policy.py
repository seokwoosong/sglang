"""CPU-only tests for unified-cache Mamba state admission policies."""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
from sglang.srt.mem_cache.mamba_admission_policy import (
    MambaAdmissionStatsTracker,
    MambaCheckpointKind,
)
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaAdmissionPolicy(unittest.TestCase):
    def test_server_arg_default_and_cli(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            ServerArgs(model_path="dummy").mamba_state_admission_policy,
            "default",
        )
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--mamba-state-admission-policy",
                "marconi",
            ]
        )
        self.assertEqual(args.mamba_state_admission_policy, "marconi")

    def test_marconi_requires_unified_memory(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy",
            mamba_state_admission_policy="marconi",
        )
        with self.assertRaisesRegex(ValueError, "requires --enable-unified-memory"):
            args._handle_unified_memory_pool()

    def test_component_skips_intermediate_without_donating_snapshot(self):
        component = object.__new__(MambaComponent)
        component.cache = SimpleNamespace(enable_mamba_extra_buffer=True)
        component.mamba_state_admission_policy = "marconi"
        component.mamba_admission_stats = MambaAdmissionStatsTracker("marconi")
        req = SimpleNamespace(
            rid="intermediate",
            mamba_last_track_seqlen=64,
            mamba_branching_seqlen=None,
        )
        params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req, params, token_ids_len=96, is_finished=False
        )

        self.assertEqual(cache_len, 64)
        self.assertIsNone(params.mamba_value)
        self.assertTrue(params.skip_radix_insert)
        self.assertEqual(
            params.mamba_admission_kind,
            MambaCheckpointKind.INTERMEDIATE.value,
        )

        component.cleanup_after_caching_req(
            req,
            is_finished=False,
            insert_result=None,
            insert_params=params,
        )
        self.assertEqual(req.mamba_last_track_seqlen, 64)

    def test_consumed_branch_marker_is_cleared(self):
        component = object.__new__(MambaComponent)
        component.mamba_state_admission_policy = "marconi"
        component.mamba_admission_stats = MambaAdmissionStatsTracker("marconi")
        req = SimpleNamespace(
            rid="branch",
            mamba_last_track_seqlen=128,
            mamba_branching_seqlen=128,
        )
        params = InsertParams(mamba_admission_kind=MambaCheckpointKind.BRANCH.value)

        component.cleanup_after_caching_req(
            req,
            is_finished=False,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=params,
        )

        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertIsNone(req.mamba_last_track_seqlen)

    def test_default_does_not_change_branch_marker_cleanup(self):
        component = object.__new__(MambaComponent)
        component.mamba_state_admission_policy = "default"
        component.mamba_admission_stats = MambaAdmissionStatsTracker("default")
        req = SimpleNamespace(
            rid="default-branch",
            mamba_last_track_seqlen=128,
            mamba_branching_seqlen=128,
        )
        params = InsertParams(mamba_admission_kind=MambaCheckpointKind.BRANCH.value)

        component.cleanup_after_caching_req(
            req,
            is_finished=False,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=params,
        )

        self.assertEqual(req.mamba_branching_seqlen, 128)
        self.assertIsNone(req.mamba_last_track_seqlen)


if __name__ == "__main__":
    unittest.main()
