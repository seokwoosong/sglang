import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc.band import _prewarm_bind_kernels
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnifiedBindPrewarm(CustomTestCase):
    def test_non_rocm_does_not_prewarm_bind_kernels(self):
        """Non-ROCm must leave request-path bind kernels untouched."""
        with (
            patch.object(torch.version, "hip", None),
            patch("sglang.srt.mem_cache.kv.alloc.band.free_unbind_inplace") as free,
            patch("sglang.srt.mem_cache.kv.alloc.band.alloc_bind_inplace") as alloc,
            patch("sglang.srt.mem_cache.kv.alloc.band.bind_inplace") as bind,
            patch("sglang.srt.mem_cache.kv.alloc.band.bind_zero_pages") as zero,
            patch(
                "sglang.srt.mem_cache.kv.alloc.band._prewarm_large_arena_bind_zero"
            ) as zero_large,
            patch("sglang.srt.mem_cache.kv.alloc.band.translate_token_ids") as translate,
            patch("torch.cuda.synchronize") as synchronize,
        ):
            _prewarm_bind_kernels("cpu", page_bytes=12288)

        for call in (free, alloc, bind, zero, zero_large, translate, synchronize):
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
