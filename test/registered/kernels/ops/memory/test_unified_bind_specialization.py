import unittest

import torch

from sglang.kernels.ops.memory.virtual_slot import (
    alloc_bind_inplace,
    alloc_bind_inplace_kernel,
    bind_inplace,
    bind_inplace_kernel,
    free_unbind_inplace,
    free_unbind_inplace_kernel,
    translate_token_ids,
    translate_token_ids_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _variant_count(kernel):
    return sum(len(entry[0]) for entry in kernel.device_caches.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestUnifiedBindSpecialization(CustomTestCase):
    def test_free_shares_one_variant_across_request_sizes(self):
        """A larger free must not trigger a second request-size compilation."""
        free_unbind_inplace_kernel.device_caches.clear()
        for n in (2, 4096):
            virtual = torch.arange(n, dtype=torch.int64, device="cuda")
            v2p = torch.arange(n, dtype=torch.int64, device="cuda")
            p2v = torch.arange(n, dtype=torch.int64, device="cuda")
            freed = free_unbind_inplace(virtual, v2p, p2v)
            self.assertTrue(torch.equal(freed, virtual))
            self.assertTrue(bool((v2p == -1).all()))
            self.assertTrue(bool((p2v == -1).all()))
        self.assertEqual(_variant_count(free_unbind_inplace_kernel), 1)

    def test_alloc_shares_one_variant_across_request_sizes(self):
        alloc_bind_inplace_kernel.device_caches.clear()
        for n in (2, 4096):
            virtual = torch.arange(n, dtype=torch.int64, device="cuda")
            v2p = torch.full((n,), -1, dtype=torch.int64, device="cuda")
            p2v = torch.full((n + 1,), -1, dtype=torch.int64, device="cuda")
            physical = alloc_bind_inplace(virtual, v2p, p2v, 1)
            expected = torch.arange(1, n + 1, dtype=torch.int64, device="cuda")
            self.assertTrue(torch.equal(physical, expected))
            self.assertTrue(torch.equal(v2p, physical))
            self.assertTrue(torch.equal(p2v[1:], virtual))
        self.assertEqual(_variant_count(alloc_bind_inplace_kernel), 1)

    def test_translate_shares_one_variant_across_request_sizes(self):
        translate_token_ids_kernel.device_caches.clear()
        for n in (2, 4096):
            virtual = torch.arange(n, dtype=torch.int64, device="cuda")
            v2p = torch.arange(n, dtype=torch.int64, device="cuda")
            physical = translate_token_ids(loc=virtual, v2p=v2p, page_size=1)
            self.assertTrue(torch.equal(physical, virtual))
        self.assertEqual(_variant_count(translate_token_ids_kernel), 1)

    def test_bind_shares_one_variant_across_request_sizes(self):
        """Physical-page binds must not compile N==1 and general-N variants."""
        bind_inplace_kernel.device_caches.clear()
        for n in (1, 2, 4096):
            virtual = torch.arange(n, dtype=torch.int64, device="cuda")
            physical = torch.arange(n, dtype=torch.int64, device="cuda")
            v2p = torch.empty(n, dtype=torch.int64, device="cuda")
            p2v = torch.empty(n, dtype=torch.int64, device="cuda")
            bind_inplace(virtual, physical, v2p, p2v)
            self.assertTrue(torch.equal(v2p, physical))
            self.assertTrue(torch.equal(p2v, virtual))
        self.assertEqual(_variant_count(bind_inplace_kernel), 1)


if __name__ == "__main__":
    unittest.main()
