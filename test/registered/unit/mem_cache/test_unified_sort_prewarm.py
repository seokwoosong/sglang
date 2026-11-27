import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc.band import _prewarm_free_list_sort
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnifiedSortPrewarm(CustomTestCase):
    def test_rocm_prewarms_small_and_large_int64_sorts(self):
        """ROCm must initialize both sort regimes before the first request."""
        probes = []
        sorted_probes = []

        def fake_arange(start, end, step, *, dtype, device):
            probe = (start, end, step, dtype, device)
            probes.append(probe)
            return probe

        with (
            patch.object(torch.version, "hip", "7.15"),
            patch("torch.arange", side_effect=fake_arange),
            patch("torch.sort", side_effect=lambda probe: sorted_probes.append(probe)),
            patch("torch.cuda.synchronize") as synchronize,
        ):
            _prewarm_free_list_sort("cuda")

        self.assertEqual([probe[0] + 1 for probe in sorted_probes], [1024, 8192])
        for probe in sorted_probes:
            self.assertEqual(probe[1:3], (-1, -1))
            self.assertEqual(probe[3], torch.int64)
            self.assertEqual(probe[4], "cuda")
        synchronize.assert_called_once_with()

    def test_non_rocm_does_not_prewarm_sort(self):
        with (
            patch.object(torch.version, "hip", None),
            patch("torch.sort") as sort,
            patch("torch.cuda.synchronize") as synchronize,
        ):
            _prewarm_free_list_sort("cuda")

        sort.assert_not_called()
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
