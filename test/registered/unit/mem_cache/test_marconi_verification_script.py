"""Dependency-free tests for the end-to-end result validator."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "verify_marconi_mamba_admission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "marconi_verification_script_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def stats(policy, **overrides):
    record = {
        "policy": policy,
        "rid": f"{policy}-request",
        "branch_candidates": 0,
        "final_candidates": 1,
        "intermediate_candidates": 1,
        "branch_admitted": 0,
        "final_admitted": 1,
        "intermediate_admitted": 0,
        "duplicate_candidates": 0,
        "intermediate_skipped": 0,
    }
    record.update(overrides)
    return record


class TestMarconiVerificationScript(unittest.TestCase):
    def test_valid_comparison_passes(self):
        results = {
            "default": {
                "responses": {"request": "same"},
                "stats": [stats("default", intermediate_admitted=1)],
            },
            "marconi": {
                "responses": {"request": "same"},
                "stats": [
                    stats(
                        "marconi",
                        branch_candidates=1,
                        branch_admitted=1,
                        intermediate_skipped=1,
                    )
                ],
            },
        }
        self.assertEqual(SCRIPT.validate(results), [])

    def test_intermediate_admission_fails(self):
        results = {
            "default": {
                "responses": {"request": "same"},
                "stats": [stats("default", intermediate_admitted=1)],
            },
            "marconi": {
                "responses": {"request": "same"},
                "stats": [
                    stats(
                        "marconi",
                        branch_candidates=1,
                        intermediate_admitted=1,
                        intermediate_skipped=1,
                    )
                ],
            },
        }
        failures = SCRIPT.validate(results)
        self.assertIn("Marconi admitted an intermediate checkpoint", failures)


if __name__ == "__main__":
    unittest.main()
