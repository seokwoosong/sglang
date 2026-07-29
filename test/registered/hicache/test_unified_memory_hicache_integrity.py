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

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedMemoryHiCacheIntegrity(CustomTestCase):
    model = "Qwen/Qwen3.5-0.8B"
    pressure_requests = 60

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="unified-hicache-integrity-")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir},
            other_args=[
                "--enable-unified-memory",
                "--enable-hierarchical-cache",
                "--hicache-size",
                "1",
                "--hicache-write-policy",
                "write_through",
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
                "--mem-fraction-static",
                "0.075",
                "--disable-cuda-graph",
                "--enable-metrics",
                "--log-level",
                "error",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

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

    def _metric(self, name):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        response.raise_for_status()
        total = 0.0
        for line in response.text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
        return total

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
        for lhs, rhs in zip(actual["logprobs"], expected["logprobs"]):
            self.assertTrue(math.isfinite(lhs) and math.isfinite(rhs))
            self.assertAlmostEqual(lhs, rhs, delta=logprob_tolerance)

    def test_l2_and_l3_restore_integrity(self):
        results = {}
        for index in range(self.pressure_requests):
            results[index] = self._generate(self._prompt(index))

        self.assertGreater(
            self._metric("sglang:evicted_tokens_total"),
            0,
            "pressure did not evict any L1 tokens",
        )
        self.assertGreater(
            self._metric("sglang:backuped_tokens_total"),
            0,
            "pressure did not back up any tokens to L3",
        )
        filenames = os.listdir(self.storage_dir)
        self.assertTrue(filenames, "file backend did not create L3 pages")
        self.assertTrue(
            any(".mamba" in name.lower() for name in filenames),
            "file backend did not create Mamba sidecar pages",
        )

        # Discover an L2-only entry from observed counters rather than assuming
        # a fixed split in the dynamically shared unified pool.
        l2_hit = None
        for index in reversed(range(self.pressure_requests)):
            load_before = self._metric("sglang:load_back_tokens_total")
            prefetch_before = self._metric("sglang:prefetched_tokens_total")
            replay = self._generate(self._prompt(index))
            self._drain_scheduler()
            load_after = self._metric("sglang:load_back_tokens_total")
            prefetch_after = self._metric("sglang:prefetched_tokens_total")
            if load_after > load_before and prefetch_after == prefetch_before:
                l2_hit = (index, replay)
                break
        self.assertIsNotNone(l2_hit, "could not identify an observed L2-only hit")
        self._assert_same_result(results[l2_hit[0]], l2_hit[1], logprob_tolerance=1e-1)

        # Discover an L3 entry by requiring the storage-prefetch counter itself
        # to increase.
        l3_hit = None
        for index in range(self.pressure_requests):
            prefetch_before = self._metric("sglang:prefetched_tokens_total")
            replay = self._generate(self._prompt(index))
            self._drain_scheduler()
            prefetch_after = self._metric("sglang:prefetched_tokens_total")
            if prefetch_after > prefetch_before:
                l3_hit = (index, replay)
                break
        self.assertIsNotNone(l3_hit, "could not identify an observed L3 hit")
        self._assert_same_result(results[l3_hit[0]], l3_hit[1], logprob_tolerance=1e-1)
        self.assertEqual(requests.get(f"{self.base_url}/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
