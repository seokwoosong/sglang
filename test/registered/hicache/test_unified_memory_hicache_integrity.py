"""End-to-end integrity test for unified memory with L2/L3 HiCache.

The test deliberately exceeds both the GPU and host-cache capacities, then
proves which tier served a replay using Prometheus counters.  It also compares
greedy output token IDs and log probabilities before and after restoration.
"""

import hashlib
import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.unified_hicache_artifacts import UnifiedHiCacheArtifactRecorder

register_cuda_ci(est_time=360, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedMemoryHiCacheIntegrity(CustomTestCase):
    # Keep the CI-sized default, while allowing the same integrity proof to be
    # rerun against a larger local checkpoint without copying the test.
    model = os.getenv(
        "SGLANG_UNIFIED_HICACHE_TEST_MODEL", "Qwen/Qwen3.5-0.8B"
    )
    pressure_requests = int(
        os.getenv("SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS", "60")
    )
    mem_fraction_static = os.getenv(
        "SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC", "0.075"
    )
    write_policy = "write_through"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.artifacts = UnifiedHiCacheArtifactRecorder()
        cls.storage_dir = tempfile.mkdtemp(prefix="unified-hicache-integrity-")
        cls.server_args = [
            "--enable-unified-memory",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "1",
            "--hicache-write-policy",
            cls.write_policy,
            "--hicache-io-backend",
            "kernel",
            "--hicache-storage-backend",
            "file",
            "--page-size",
            "1",
            "--attention-backend",
            "triton",
            "--linear-attn-backend",
            "triton",
            "--mamba-backend",
            "triton",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--mem-fraction-static",
            cls.mem_fraction_static,
            "--disable-cuda-graph",
            "--enable-metrics",
            "--log-level",
            "error",
        ]
        cls.artifacts.update(
            model=cls.model,
            configuration={
                "server_args": cls.server_args,
                "overlap_schedule": True,
                "write_policy": cls.write_policy,
                "mamba_radix_cache_strategy": "extra_buffer",
                "pressure_requests": cls.pressure_requests,
                "mem_fraction_static": cls.mem_fraction_static,
            },
        )
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env={"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir},
                other_args=cls.server_args,
                return_stdout_stderr=cls.artifacts.subprocess_output,
            )
        except BaseException as error:
            cls.artifacts.fail_test(error)
            cls.artifacts.close()
            shutil.rmtree(cls.storage_dir, ignore_errors=True)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)
        cls.artifacts.close()

    @staticmethod
    def _prompt(index):
        digest = hashlib.sha256(str(index).encode()).hexdigest()
        payload = "payload_" + hashlib.md5(str(index).encode()).hexdigest()
        return f"{digest} " + (payload + " ") * 90

    def _generate(self, text):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        logprobs = body["meta_info"]["output_token_logprobs"]
        return {
            "output_ids": [int(x) for x in body["output_ids"]],
            "logprobs": [float(x[0]) for x in logprobs],
            "logprob_token_ids": [int(x[1]) for x in logprobs],
        }

    def _metrics_text(self):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _metric_from_text(text, name):
        total = 0.0
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
        return total

    def _metric(self, name):
        return self._metric_from_text(self._metrics_text(), name)

    def _snapshot_metrics(self, stage):
        text = self._metrics_text()
        self.artifacts.save_metrics(stage, text)
        return {
            name: self._metric_from_text(text, name)
            for name in (
                "sglang:evicted_tokens_total",
                "sglang:backuped_tokens_total",
                "sglang:load_back_tokens_total",
                "sglang:prefetched_tokens_total",
            )
        }

    def _drain_scheduler(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"metric-drain-{time.monotonic_ns()}",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
            timeout=120,
        )
        response.raise_for_status()

    def _assert_same_result(self, expected, actual, logprob_tolerance):
        self.assertEqual(actual["output_ids"], expected["output_ids"])
        self.assertEqual(actual["logprob_token_ids"], expected["logprob_token_ids"])
        self.assertEqual(len(actual["logprobs"]), len(expected["logprobs"]))
        max_difference = 0.0
        for lhs, rhs in zip(actual["logprobs"], expected["logprobs"]):
            self.assertTrue(math.isfinite(lhs) and math.isfinite(rhs))
            self.assertAlmostEqual(lhs, rhs, delta=logprob_tolerance)
            max_difference = max(max_difference, abs(lhs - rhs))
        return max_difference

    def _storage_manifest(self):
        summary = {
            "file_count": 0,
            "total_size_bytes": 0,
            "empty_file_count": 0,
            "mamba_sidecar_count": 0,
            "samples": [],
        }
        for path in sorted(Path(self.storage_dir).iterdir()):
            if path.is_file():
                size_bytes = path.stat().st_size
                is_mamba_sidecar = ".mamba" in path.name.lower()
                summary["file_count"] += 1
                summary["total_size_bytes"] += size_bytes
                summary["empty_file_count"] += size_bytes == 0
                summary["mamba_sidecar_count"] += is_mamba_sidecar
                if len(summary["samples"]) < 100:
                    summary["samples"].append(
                        {
                            "name": path.name,
                            "size_bytes": size_bytes,
                            "is_mamba_sidecar": is_mamba_sidecar,
                        }
                    )
        self.artifacts.write_json("storage_manifest.json", summary)
        return summary

    def test_l2_and_l3_restore_integrity(self):
        try:
            initial = self._snapshot_metrics("00_server_ready")
            results = {}
            for index in range(self.pressure_requests):
                results[index] = self._generate(self._prompt(index))

            pressure = self._snapshot_metrics("01_after_pressure")
            for name, value in pressure.items():
                self.assertGreaterEqual(value, initial[name], f"{name} decreased")
            self.assertGreater(
                pressure["sglang:evicted_tokens_total"],
                initial["sglang:evicted_tokens_total"],
                "pressure did not evict any L1 tokens",
            )
            self.assertGreater(
                pressure["sglang:backuped_tokens_total"],
                initial["sglang:backuped_tokens_total"],
                "pressure did not back up any tokens to L3",
            )

            manifest = self._storage_manifest()
            self.assertGreater(
                manifest["file_count"], 0, "file backend did not create L3 pages"
            )
            self.assertEqual(
                manifest["empty_file_count"],
                0,
                "file backend created an empty L3 page",
            )
            mamba_sidecars = manifest["mamba_sidecar_count"]
            self.assertGreater(mamba_sidecars, 0, "no Mamba sidecar pages were created")

            # Discover an L2-only entry from observed counters rather than assuming
            # a fixed split in the dynamically shared unified pool.
            l2_hit = None
            for index in reversed(range(self.pressure_requests)):
                before = self._snapshot_metrics("02_before_l2_candidate")
                replay = self._generate(self._prompt(index))
                self._drain_scheduler()
                after = self._snapshot_metrics("03_after_l2_candidate")
                if (
                    after["sglang:load_back_tokens_total"]
                    > before["sglang:load_back_tokens_total"]
                    and after["sglang:prefetched_tokens_total"]
                    == before["sglang:prefetched_tokens_total"]
                ):
                    l2_hit = (index, replay, before, after)
                    break
            self.assertIsNotNone(l2_hit, "could not identify an observed L2-only hit")
            l2_logprob_difference = self._assert_same_result(
                results[l2_hit[0]], l2_hit[1], logprob_tolerance=1e-1
            )

            # Discover an L3 entry by requiring the storage-prefetch counter itself
            # to increase.
            l3_hit = None
            for index in range(self.pressure_requests):
                before = self._snapshot_metrics("04_before_l3_candidate")
                replay = self._generate(self._prompt(index))
                self._drain_scheduler()
                after = self._snapshot_metrics("05_after_l3_candidate")
                if (
                    after["sglang:prefetched_tokens_total"]
                    > before["sglang:prefetched_tokens_total"]
                ):
                    l3_hit = (index, replay, before, after)
                    break
            self.assertIsNotNone(l3_hit, "could not identify an observed L3 hit")
            l3_logprob_difference = self._assert_same_result(
                results[l3_hit[0]], l3_hit[1], logprob_tolerance=1e-1
            )
            health_status = requests.get(f"{self.base_url}/health").status_code
            self.assertEqual(health_status, 200)

            self.artifacts.update(
                pressure_metrics=pressure,
                storage={
                    "file_count": manifest["file_count"],
                    "total_size_bytes": manifest["total_size_bytes"],
                    "mamba_sidecar_count": mamba_sidecars,
                },
                l2_restore={
                    "request_index": l2_hit[0],
                    "load_back_tokens_delta": (
                        l2_hit[3]["sglang:load_back_tokens_total"]
                        - l2_hit[2]["sglang:load_back_tokens_total"]
                    ),
                    "prefetched_tokens_delta": (
                        l2_hit[3]["sglang:prefetched_tokens_total"]
                        - l2_hit[2]["sglang:prefetched_tokens_total"]
                    ),
                    "output_token_ids_match": True,
                    "max_logprob_difference": l2_logprob_difference,
                },
                l3_restore={
                    "request_index": l3_hit[0],
                    "prefetched_tokens_delta": (
                        l3_hit[3]["sglang:prefetched_tokens_total"]
                        - l3_hit[2]["sglang:prefetched_tokens_total"]
                    ),
                    "output_token_ids_match": True,
                    "max_logprob_difference": l3_logprob_difference,
                },
                health_status=health_status,
            )
            self.artifacts.pass_test()
        except BaseException as error:
            self.artifacts.fail_test(error)
            raise


class TestUnifiedMemoryHiCacheWriteBack(TestUnifiedMemoryHiCacheIntegrity):
    """Run the same tier and output-integrity proof through write-back."""

    write_policy = "write_back"


if __name__ == "__main__":
    unittest.main()
