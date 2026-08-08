"""L2-only serving coverage for HiCache with the unified memory pool."""

import math
import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


class TestUnifiedMemoryHiCacheL2(CustomTestCase):
    """Force L1 eviction with fixed logical capacities, then restore from L2."""

    model = "Qwen/Qwen3.5-0.8B"
    prompt_tokens = 768
    pressure_requests = 5

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-unified-memory",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "kernel",
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
                # These two limits define the shared pool byte budget. Do not
                # derive pressure from --mem-fraction-static or physical DRAM.
                "--max-total-tokens",
                "1024",
                "--max-mamba-cache-size",
                "16",
                "--max-running-requests",
                "1",
                "--chunked-prefill-size",
                "256",
                "--context-length",
                "2048",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--enable-metrics",
                "--log-level",
                "error",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @classmethod
    def _input_ids(cls, index):
        # Distinct first-level radix branches avoid accidental shared-prefix
        # hits and make the pressure count deterministic.
        return [1000 + index] * cls.prompt_tokens

    def _generate(self, index):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": self._input_ids(index),
                "sampling_params": {"temperature": 0, "max_new_tokens": 2},
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        logprobs = body["meta_info"]["output_token_logprobs"]
        return {
            "output_ids": [int(token_id) for token_id in body["output_ids"]],
            "logprob_token_ids": [int(item[1]) for item in logprobs],
            "logprobs": [float(item[0]) for item in logprobs],
            "cached_tokens": int(body["meta_info"].get("cached_tokens", 0)),
        }

    def _metric(self, name):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        )

    def _wait_for_metric_above(self, name, baseline, timeout=5):
        deadline = time.monotonic() + timeout
        value = self._metric(name)
        while value <= baseline and time.monotonic() < deadline:
            time.sleep(0.05)
            value = self._metric(name)
        return value

    def _assert_same_result(self, expected, actual):
        self.assertEqual(actual["output_ids"], expected["output_ids"])
        self.assertEqual(actual["logprob_token_ids"], expected["logprob_token_ids"])
        self.assertEqual(len(actual["logprobs"]), len(expected["logprobs"]))
        for lhs, rhs in zip(actual["logprobs"], expected["logprobs"]):
            self.assertTrue(math.isfinite(lhs) and math.isfinite(rhs))
            self.assertAlmostEqual(lhs, rhs, delta=1e-2)

    def test_fixed_capacity_eviction_and_l2_restore(self):
        server_info = requests.get(f"{self.base_url}/server_info", timeout=10).json()
        self.assertEqual(server_info["max_total_tokens"], 1024)
        self.assertEqual(server_info["max_mamba_cache_size"], 16)
        self.assertIsNone(server_info["hicache_storage_backend"])

        evicted_before = self._metric("sglang:evicted_tokens_total")
        baselines = {
            index: self._generate(index) for index in range(self.pressure_requests)
        }
        evicted_after = self._wait_for_metric_above(
            "sglang:evicted_tokens_total", evicted_before
        )
        self.assertGreater(
            evicted_after,
            evicted_before,
            "fixed logical capacity did not force L1 eviction",
        )

        # Oldest-first finds a host-only prefix without assuming an LRU split.
        # A counter increase proves this was an L2 restore, not a device hit.
        restored = None
        for index in range(self.pressure_requests):
            load_back_before = self._metric("sglang:load_back_tokens_total")
            replay = self._generate(index)
            load_back_after = self._wait_for_metric_above(
                "sglang:load_back_tokens_total", load_back_before, timeout=1
            )
            if load_back_after > load_back_before:
                restored = (index, replay)
                break

        self.assertIsNotNone(restored, "no L2-resident prefix was restored")
        self.assertGreater(restored[1]["cached_tokens"], 0)
        self._assert_same_result(baselines[restored[0]], restored[1])
        self.assertEqual(
            requests.get(f"{self.base_url}/health", timeout=10).status_code, 200
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
