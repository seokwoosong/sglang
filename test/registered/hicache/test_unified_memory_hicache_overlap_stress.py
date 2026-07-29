"""Concurrent overlap and repeated-tier-churn test for unified HiCache."""

import hashlib
import math
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class TestUnifiedMemoryHiCacheOverlapStress(CustomTestCase):
    model = "Qwen/Qwen3.5-0.8B"
    prompt_count = 60
    churn_rounds = 3
    concurrent_workers = 8

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.artifacts = UnifiedHiCacheArtifactRecorder()
        cls.storage_dir = tempfile.mkdtemp(prefix="unified-hicache-overlap-")
        cls.server_args = [
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
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--mem-fraction-static",
            "0.075",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--log-level",
            "error",
        ]
        cls.artifacts.update(
            test="overlap_stress",
            model=cls.model,
            configuration={
                "server_args": cls.server_args,
                "overlap_schedule": True,
                "mamba_radix_cache_strategy": "extra_buffer",
                "prompt_count": cls.prompt_count,
                "churn_rounds": cls.churn_rounds,
                "concurrent_workers": cls.concurrent_workers,
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
        identity = hashlib.sha256(f"stress-{index}".encode()).hexdigest()
        shared_group = hashlib.md5(f"group-{index % 6}".encode()).hexdigest()
        unique_payload = hashlib.md5(f"payload-{index}".encode()).hexdigest()
        return (
            (f"shared_{shared_group} " * 45)
            + f"{identity} "
            + (f"payload_{unique_payload} " * 90)
        )

    def _generate(self, index, max_new_tokens=8):
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self._prompt(index),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
            },
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        logprobs = body["meta_info"]["output_token_logprobs"]
        return {
            "index": index,
            "latency_seconds": time.monotonic() - started,
            "output_ids": [int(value) for value in body["output_ids"]],
            "logprobs": [float(value[0]) for value in logprobs],
            "logprob_token_ids": [int(value[1]) for value in logprobs],
        }

    def _metrics(self, stage):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        response.raise_for_status()
        self.artifacts.save_metrics(stage, response.text)
        values = {}
        for name in (
            "sglang:evicted_tokens_total",
            "sglang:backuped_tokens_total",
            "sglang:load_back_tokens_total",
            "sglang:prefetched_tokens_total",
        ):
            values[name] = sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith(name + "{") or line.startswith(name + " ")
            )
        return values

    def _assert_matches(self, baseline, replay):
        self.assertEqual(replay["output_ids"], baseline["output_ids"])
        self.assertEqual(replay["logprob_token_ids"], baseline["logprob_token_ids"])
        self.assertEqual(len(replay["logprobs"]), len(baseline["logprobs"]))
        max_difference = 0.0
        for actual, expected in zip(replay["logprobs"], baseline["logprobs"]):
            self.assertTrue(math.isfinite(actual) and math.isfinite(expected))
            difference = abs(actual - expected)
            self.assertLessEqual(difference, 1e-1)
            max_difference = max(max_difference, difference)
        return max_difference

    def _run_concurrent_round(self, order):
        results = {}
        with ThreadPoolExecutor(max_workers=self.concurrent_workers) as executor:
            futures = {
                executor.submit(self._generate, index, 1): index for index in order
            }
            for future in as_completed(futures):
                result = future.result()
                results[result["index"]] = result
        self.assertEqual(len(results), len(order))
        return results

    def test_concurrent_overlap_and_repeated_tier_churn(self):
        try:
            initial_metrics = self._metrics("00_server_ready")

            # Establish deterministic references while filling all cache tiers.
            baselines = {
                index: self._generate(index) for index in range(self.prompt_count)
            }
            after_baseline = self._metrics("01_after_baseline")
            self.assertGreater(
                after_baseline["sglang:evicted_tokens_total"],
                initial_metrics["sglang:evicted_tokens_total"],
            )
            self.assertGreater(
                after_baseline["sglang:backuped_tokens_total"],
                initial_metrics["sglang:backuped_tokens_total"],
            )

            round_reports = []
            max_logprob_difference = 0.0
            verified_probe_requests = 0
            previous_metrics = after_baseline
            for round_index in range(self.churn_rounds):
                # Alternating traversal makes old and recent prefixes compete and
                # repeatedly pushes restored entries back through L1/L2/L3.
                order = list(range(self.prompt_count))
                if round_index % 2 == 0:
                    order.reverse()
                else:
                    order = order[::2] + order[1::2]

                results = self._run_concurrent_round(order)
                self.assertTrue(
                    all(len(result["output_ids"]) == 1 for result in results.values())
                )

                # Compare sequential probes after concurrent churn. This removes
                # batch-shape numerical variation from the oracle while still
                # detecting state corrupted by overlapping restore/eviction.
                probe_indices = [
                    (round_index * 20 + offset * 5) % self.prompt_count
                    for offset in range(12)
                ]
                for index in probe_indices:
                    replay = self._generate(index)
                    max_logprob_difference = max(
                        max_logprob_difference,
                        self._assert_matches(baselines[index], replay),
                    )
                    verified_probe_requests += 1

                health_status = requests.get(
                    f"{self.base_url}/health", timeout=10
                ).status_code
                self.assertEqual(health_status, 200)
                current_metrics = self._metrics(
                    f"{round_index + 2:02d}_after_churn_round_{round_index + 1}"
                )
                for name, value in current_metrics.items():
                    self.assertGreaterEqual(
                        value,
                        previous_metrics[name],
                        f"{name} decreased in churn round {round_index + 1}",
                    )
                round_reports.append(
                    {
                        "round": round_index + 1,
                        "request_count": len(results),
                        "probe_indices": probe_indices,
                        "max_latency_seconds": max(
                            result["latency_seconds"] for result in results.values()
                        ),
                        "metrics": current_metrics,
                    }
                )
                previous_metrics = current_metrics

            self.assertGreater(
                previous_metrics["sglang:load_back_tokens_total"],
                after_baseline["sglang:load_back_tokens_total"],
                "concurrent churn did not restore any L2 entries",
            )
            self.assertGreater(
                previous_metrics["sglang:prefetched_tokens_total"],
                after_baseline["sglang:prefetched_tokens_total"],
                "concurrent churn did not restore any L3 entries",
            )
            self.assertGreater(
                previous_metrics["sglang:evicted_tokens_total"],
                after_baseline["sglang:evicted_tokens_total"],
                "concurrent churn did not cause repeated L1 eviction",
            )
            self.assertTrue(
                any(
                    ".mamba" in path.name.lower() and path.stat().st_size > 0
                    for path in Path(self.storage_dir).iterdir()
                    if path.is_file()
                ),
                "L3 storage has no Mamba sidecar",
            )

            self.artifacts.update(
                initial_metrics=initial_metrics,
                after_baseline_metrics=after_baseline,
                final_metrics=previous_metrics,
                churn_rounds=round_reports,
                total_replay_requests=self.prompt_count * self.churn_rounds,
                verified_probe_requests=verified_probe_requests,
                max_logprob_difference=max_logprob_difference,
                health_status=200,
            )
            self.artifacts.pass_test()
        except BaseException as error:
            self.artifacts.fail_test(error)
            raise


if __name__ == "__main__":
    unittest.main()
