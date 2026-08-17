import unittest

import numpy as np
import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Triton kernel unit test for KV indices creation
register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def _run_test(self, batch, max_batch, max_context_len):
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device=get_device()
        ).reshape((max_batch, max_context_len))
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )
        paged_kernel_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )

        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=get_device())
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # ref
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
        kv_indices_ref = torch.cat(
            [
                req_to_token[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
                for i in range(batch)
            ],
            dim=0,
        ).contiguous()

        # triton
        kv_indices_triton = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device=get_device()
        )
        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices_triton,
            req_to_token.size(1),
        )

        # Check
        self.assertTrue(torch.equal(kv_indices_ref, kv_indices_triton))

    def test_create_kvindices(self):
        BATCH = [1, 37, 1786]
        MAX_BATCH = 4096
        MAX_CONTEXT_LEN = 4096
        for batch in BATCH:
            self._run_test(batch, MAX_BATCH, MAX_CONTEXT_LEN)

    def test_create_kvindices_with_fused_v2p_translation(self):
        page_size = 8
        page_multiplier = 3
        max_context_len = 32
        req_to_token = torch.tensor(
            [
                list(range(0, max_context_len)),
                list(range(max_context_len, 2 * max_context_len)),
            ],
            dtype=torch.int64,
            device=get_device(),
        )
        req_pool_indices = torch.tensor([1, 0], dtype=torch.int32, device=get_device())
        seq_lens = torch.tensor([19, 13], dtype=torch.int32, device=get_device())
        kv_indptr = torch.tensor([0, 19, 32], dtype=torch.int32, device=get_device())
        # Deliberately non-identity so the fused gather cannot pass accidentally.
        v2p = torch.tensor(
            [7, 3, 6, 2, 5, 1, 4, 0], dtype=torch.int64, device=get_device()
        )
        got = torch.empty(32, dtype=torch.int64, device=get_device())

        create_flashinfer_kv_indices_triton[(2,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            got,
            req_to_token.stride(0),
            v2p,
            PAGE_SIZE=page_size,
            PAGE_MULT=page_multiplier,
        )

        virtual = torch.cat((req_to_token[1, :19], req_to_token[0, :13]))
        expected = (
            v2p[virtual // page_size] * (page_size * page_multiplier)
            + virtual % page_size
        ).clamp_min(0)
        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
