"""Concurrent overlap and repeated-tier-churn test for unified HiCache."""

import hashlib
import math
import os
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
    # Environment overrides make the exact same overlap oracle reusable for a
    # larger checkpoint in manual qualification runs. CI keeps these defaults.
    model = os.getenv("SGLANG_UNIFIED_HICACHE_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
    prompt_count = int(os.getenv("SGLANG_UNIFIED_HICACHE_PROMPT_COUNT", "60"))
    churn_rounds = int(os.getenv("SGLANG_UNIFIED_HICACHE_CHURN_ROUNDS", "3"))
    concurrent_workers = int(
        os.getenv("SGLANG_UNIFIED_HICACHE_CONCURRENT_WORKERS", "8")
    )
    max_running_requests = int(
        os.getenv("SGLANG_UNIFIED_HICACHE_MAX_RUNNING_REQUESTS", "8")
    )
    mem_fraction_static = os.getenv(
        "SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC", "0.10"
    )
    hicache_size = os.getenv("SGLANG_UNIFIED_HICACHE_SIZE", "1")
    hicache_ratio = os.getenv("SGLANG_UNIFIED_HICACHE_RATIO", "2.0")
    tp_size = int(os.getenv("SGLANG_UNIFIED_HICACHE_TP_SIZE", "1"))
    enable_cuda_graph = os.getenv(
        "SGLANG_UNIFIED_HICACHE_ENABLE_CUDA_GRAPH", "0"
    ).lower() in ("1", "true", "yes")

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.artifacts = UnifiedHiCacheArtifactRecorder()
        cls.storage_dir = tempfile.mkdtemp(prefix="unified-hicache-overlap-")
        cls.server_args = [
            "--enable-unified-memory",
            "--enable-hierarchical-cache",
            "--hicache-size",
            cls.hicache_size,
            "--hicache-ratio",
            cls.hicache_ratio,
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-storage-backend",
            "file",
            "--page-size",
            "1",
            "--tp-size",
            str(cls.tp_size),
            "--attention-backend",
            "triton",
            "--linear-attn-backend",
            "triton",
            "--mamba-backend",
            "triton",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            # Keep eight clients contending while reserving transient shared-pool
            # headroom. At mem_fraction_static=0.075 this model exposes only 24
            # physical Mamba slots: eight requests consume all of them with active
            # + two ping-pong states, leaving no slot for HiCache tree/load state.
            # 0.10 remains small enough for the 60 prompts to force tier churn.
            "--max-running-requests",
            str(cls.max_running_requests),
            "--mamba-full-memory-ratio",
            "1.5",
            "--mem-fraction-static",
            cls.mem_fraction_static,
            "--enable-metrics",
            "--log-level",
            "error",
        ]
        if not cls.enable_cuda_graph:
            cls.server_args.extend(
                [
                    "--cuda-graph-backend-decode",
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                ]
            )
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
                "max_running_requests": cls.max_running_requests,
                "mem_fraction_static": cls.mem_fraction_static,
                "hicache_size": cls.hicache_size,
                "hicache_ratio": cls.hicache_ratio,
                "tp_size": cls.tp_size,
                "cuda_graph": cls.enable_cuda_graph,
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
                "top_logprobs_num": 5,
            },
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        logprobs = body["meta_info"]["output_token_logprobs"]
        top_logprobs = body["meta_info"].get("output_top_logprobs") or []
        return {
            "index": index,
            "latency_seconds": time.monotonic() - started,
            "output_ids": [int(value) for value in body["output_ids"]],
            "logprobs": [float(value[0]) for value in logprobs],
            "logprob_token_ids": [int(value[1]) for value in logprobs],
            "top_logprobs": [
                [
                    {"logprob": float(candidate[0]), "token_id": int(candidate[1])}
                    for candidate in position
                ]
                for position in top_logprobs
            ],
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
        baseline_ids = baseline["output_ids"]
        replay_ids = replay["output_ids"]
        self.assertEqual(len(replay_ids), len(baseline_ids))
        self.assertEqual(len(replay["logprobs"]), len(baseline["logprobs"]))
        self.assertEqual(baseline["logprob_token_ids"], baseline_ids)
        self.assertEqual(replay["logprob_token_ids"], replay_ids)
        mismatch_positions = [
            index
            for index, (actual, expected) in enumerate(zip(replay_ids, baseline_ids))
            if actual != expected
        ]
        first_mismatch = mismatch_positions[0] if mismatch_positions else None
        compare_length = len(baseline_ids) if first_mismatch is None else first_mismatch
        self.assertEqual(replay_ids[:compare_length], baseline_ids[:compare_length])
        self.assertEqual(
            replay["logprob_token_ids"][:compare_length],
            baseline["logprob_token_ids"][:compare_length],
        )
        max_difference = 0.0
        for actual, expected in zip(
            replay["logprobs"][:compare_length],
            baseline["logprobs"][:compare_length],
        ):
            self.assertTrue(math.isfinite(actual) and math.isfinite(expected))
            difference = abs(actual - expected)
            self.assertLessEqual(difference, 1e-1)
            max_difference = max(max_difference, difference)
        if first_mismatch is None:
            self.assertEqual(replay["logprob_token_ids"], baseline["logprob_token_ids"])
            return max_difference, False

        # As in the integrity test, only a bounded argmax tie is accepted.
        # All preceding autoregressive state remains an exact token match.
        self.assertGreater(len(baseline["top_logprobs"]), first_mismatch)
        self.assertGreater(len(replay["top_logprobs"]), first_mismatch)
        expected_top = {
            item["token_id"]: item["logprob"]
            for item in baseline["top_logprobs"][first_mismatch]
        }
        actual_top = {
            item["token_id"]: item["logprob"]
            for item in replay["top_logprobs"][first_mismatch]
        }
        expected_token = baseline_ids[first_mismatch]
        actual_token = replay_ids[first_mismatch]
        self.assertIn(actual_token, expected_top)
        self.assertIn(expected_token, actual_top)
        self.assertLessEqual(
            abs(expected_top[expected_token] - expected_top[actual_token]), 1.5e-1
        )
        self.assertLessEqual(
            abs(actual_top[expected_token] - actual_top[actual_token]), 1.5e-1
        )
        return max_difference, True

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
            near_tie_probe_divergences = 0
            concurrent_token_mismatches = 0
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
                # Concurrent batching can change a near-tied bf16 greedy token, so
                # a sequential token ID is not a valid exact oracle here. Validate
                # each live response internally and record batch-shape divergence;
                # the sequential probes below remain the strict cache-integrity
                # oracle after every churn round.
                for index, result in results.items():
                    self.assertEqual(result["logprob_token_ids"], result["output_ids"])
                    self.assertEqual(len(result["logprobs"]), 1)
                    self.assertTrue(math.isfinite(result["logprobs"][0]))
                    concurrent_token_mismatches += int(
                        result["output_ids"] != baselines[index]["output_ids"][:1]
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
                    probe_difference, near_tie = self._assert_matches(
                        baselines[index], replay
                    )
                    max_logprob_difference = max(
                        max_logprob_difference,
                        probe_difference,
                    )
                    near_tie_probe_divergences += int(near_tie)
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
                concurrent_token_mismatches=concurrent_token_mismatches,
                near_tie_probe_divergences=near_tie_probe_divergences,
                max_logprob_difference=max_logprob_difference,
                health_status=200,
            )
            self.artifacts.pass_test()
        except BaseException as error:
            self.artifacts.fail_test(error)
            raise


if __name__ == "__main__":
    unittest.main()
