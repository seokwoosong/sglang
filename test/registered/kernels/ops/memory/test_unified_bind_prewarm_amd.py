import unittest

import torch
import triton
from triton.runtime.jit import MockTensor

from sglang.kernels.ops.kvcache.zero_pages import _bind_zero_pages_kernel
from sglang.srt.mem_cache.kv.alloc.band import _prewarm_bind_kernels
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="base-b", runner_config="1-gpu-small-amd-mi35x")


def _variant_keys(kernel):
    return {
        key
        for entry in kernel.device_caches.values()
        for key in entry[0]
    }


class _LargeMockTensor(MockTensor):
    @staticmethod
    def ptr_range():
        return 1 << 31


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "ROCm required"
)
class TestUnifiedBindPrewarmAMD(CustomTestCase):
    def test_bind_zero_prewarm_covers_arena_pointer_ranges(self):
        """A large arena must not compile a new zero-bind variant on first handout."""
        _bind_zero_pages_kernel.device_caches.clear()
        page_bytes = 12288
        _prewarm_bind_kernels("cuda", page_bytes=page_bytes)
        warmed = _variant_keys(_bind_zero_pages_kernel)
        self.assertEqual(len(warmed), 4)

        n = 2
        raw = torch.empty(n * page_bytes, dtype=torch.uint8, device="cuda")
        virtual = torch.arange(n, dtype=torch.int64, device="cuda")
        v2p = torch.empty(n, dtype=torch.int64, device="cuda")
        p2v = torch.empty(n + 1, dtype=torch.int64, device="cuda")
        for words in (raw.view(torch.int64), _LargeMockTensor(torch.int64)):
            for has_pages in (False, True):
                pages = virtual if has_pages else None
                out = (
                    torch.empty(n, dtype=torch.int64, device="cuda")
                    if pages is None
                    else pages
                )
                _bind_zero_pages_kernel.warmup(
                    words,
                    virtual,
                    out,
                    out,
                    v2p,
                    p2v,
                    0,
                    page_bytes // 8,
                    HAS_PAGES=has_pages,
                    BLOCK=2048,
                    grid=(n, triton.cdiv(page_bytes // 8, 2048)),
                )
        self.assertEqual(_variant_keys(_bind_zero_pages_kernel) - warmed, set())


if __name__ == "__main__":
    unittest.main()
