"""Dependency-free unit tests for the Mamba admission decision layer."""

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "mamba_admission_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mamba_admission_policy_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


class TestMambaAdmissionPolicyCore(unittest.TestCase):
    def test_default_persists_intermediate_checkpoint(self):
        decision = POLICY.decide_mamba_state_admission(
            policy="default",
            is_finished=False,
            checkpoint_seqlen=64,
            branching_seqlen=None,
        )
        self.assertEqual(decision.kind, POLICY.MambaCheckpointKind.INTERMEDIATE)
        self.assertTrue(decision.persist)

    def test_marconi_skips_intermediate_checkpoint(self):
        decision = POLICY.decide_mamba_state_admission(
            policy="marconi",
            is_finished=False,
            checkpoint_seqlen=64,
            branching_seqlen=None,
        )
        self.assertEqual(decision.kind, POLICY.MambaCheckpointKind.INTERMEDIATE)
        self.assertFalse(decision.persist)

    def test_marconi_persists_only_matching_branch_and_final(self):
        nonmatching = POLICY.decide_mamba_state_admission(
            policy="marconi",
            is_finished=False,
            checkpoint_seqlen=64,
            branching_seqlen=128,
        )
        branch = POLICY.decide_mamba_state_admission(
            policy="marconi",
            is_finished=False,
            checkpoint_seqlen=128,
            branching_seqlen=128,
        )
        final = POLICY.decide_mamba_state_admission(
            policy="marconi",
            is_finished=True,
            checkpoint_seqlen=256,
            branching_seqlen=128,
        )
        self.assertFalse(nonmatching.persist)
        self.assertEqual(branch.kind, POLICY.MambaCheckpointKind.BRANCH)
        self.assertTrue(branch.persist)
        self.assertEqual(final.kind, POLICY.MambaCheckpointKind.FINAL)
        self.assertTrue(final.persist)

    def test_marconi_has_at_most_two_persistent_candidates_per_request(self):
        decisions = [
            POLICY.decide_mamba_state_admission(
                policy="marconi",
                is_finished=False,
                checkpoint_seqlen=seqlen,
                branching_seqlen=128,
            )
            for seqlen in (64, 128, 192, 256)
        ]
        decisions.append(
            POLICY.decide_mamba_state_admission(
                policy="marconi",
                is_finished=True,
                checkpoint_seqlen=320,
                branching_seqlen=128,
            )
        )

        persistent = [decision for decision in decisions if decision.persist]
        self.assertEqual(
            [decision.kind for decision in persistent],
            [
                POLICY.MambaCheckpointKind.BRANCH,
                POLICY.MambaCheckpointKind.FINAL,
            ],
        )

    def test_stats_make_admission_difference_objective(self):
        default = POLICY.MambaAdmissionStatsTracker("default")
        marconi = POLICY.MambaAdmissionStatsTracker("marconi")
        for tracker, policy in ((default, "default"), (marconi, "marconi")):
            decision = POLICY.decide_mamba_state_admission(
                policy=policy,
                is_finished=False,
                checkpoint_seqlen=64,
                branching_seqlen=None,
            )
            tracker.note_candidate("r", decision)
            if decision.persist:
                tracker.note_result(
                    "r",
                    decision.kind,
                    inserted=True,
                    duplicate=False,
                )

        default_stats = default.finish("r")
        marconi_stats = marconi.finish("r")
        self.assertEqual(default_stats["intermediate_admitted"], 1)
        self.assertEqual(marconi_stats["intermediate_admitted"], 0)
        self.assertEqual(marconi_stats["intermediate_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
