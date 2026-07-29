"""Unit tests for opt-in unified HiCache test artifacts."""

import json
import tempfile
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_hicache_artifacts import (
    ARTIFACT_DIR_ENV,
    UnifiedHiCacheArtifactRecorder,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnifiedHiCacheArtifactRecorder(CustomTestCase):
    def test_disabled_by_default_does_not_create_artifacts(self):
        with patch.dict("os.environ", {}, clear=True):
            recorder = UnifiedHiCacheArtifactRecorder()
            recorder.update(value=1)
            recorder.save_metrics("stage", "metric 1\n")
            recorder.pass_test()
            recorder.close()

        self.assertFalse(recorder.enabled)
        self.assertIsNone(recorder.run_dir)

    def test_enabled_records_success_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {ARTIFACT_DIR_ENV: temp_dir}, clear=True):
                recorder = UnifiedHiCacheArtifactRecorder()
                recorder.update(configuration={"overlap": True})
                recorder.save_metrics("01_pressure", "metric 7\n")
                recorder.pass_test()
                run_dir = recorder.run_dir
                recorder.close()

            self.assertIsNotNone(run_dir)
            report = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(report["result"], "PASS")
            self.assertTrue(report["configuration"]["overlap"])
            self.assertEqual(
                (run_dir / "metrics" / "01_pressure.prom").read_text(),
                "metric 7\n",
            )
            self.assertTrue((run_dir / "environment.json").is_file())
            self.assertTrue((run_dir / "server.log").is_file())

    def test_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = UnifiedHiCacheArtifactRecorder(temp_dir)
            try:
                raise RuntimeError("expected failure")
            except RuntimeError as error:
                recorder.fail_test(error)
            run_dir = recorder.run_dir
            recorder.close()

            report = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(report["result"], "FAIL")
            self.assertEqual(report["error"]["type"], "RuntimeError")
            self.assertIn("expected failure", report["error"]["traceback"])


if __name__ == "__main__":
    unittest.main()
