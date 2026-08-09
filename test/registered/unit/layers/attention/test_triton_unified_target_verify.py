# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unified-memory location translation in eager Triton TARGET_VERIFY metadata."""

import types
import unittest

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TargetVerifyMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_target_verify():
        return True


class TestTritonUnifiedTargetVerify(unittest.TestCase):
    def _make_backend(self, translator):
        backend = object.__new__(TritonAttnBackend)
        backend.device = "cpu"
        backend.max_context_len = 32
        backend.num_draft_tokens = 4
        backend.sliding_window_size = None
        backend.use_sliding_window_kv_pool = False
        backend.window_kv_indptr = torch.zeros(2, dtype=torch.int32)
        backend.mask_indptr = torch.zeros(2, dtype=torch.int32)
        backend._translate_kv_loc = translator

        def _fill(_self, bs, seq_lens, req_pool_indices, kv_indices):
            del seq_lens, req_pool_indices
            self.assertEqual(bs, 1)
            kv_indices.copy_(torch.tensor([7, 8, 9], dtype=torch.int64))
            return torch.tensor([0, 3], dtype=torch.int32)

        backend._fill_kv_indptr_and_indices = types.MethodType(_fill, backend)
        return backend

    @staticmethod
    def _make_forward_batch():
        return types.SimpleNamespace(
            batch_size=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            forward_mode=_TargetVerifyMode(),
            spec_info=types.SimpleNamespace(
                draft_token_num=4,
                custom_mask=torch.ones(28, dtype=torch.bool),
            ),
            seq_lens_sum=3,
            seq_lens=torch.tensor([3], dtype=torch.int32),
            out_cache_loc=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        )

    def test_unified_translates_target_verify_read_and_write_locations(self):
        backend = self._make_backend(lambda loc: loc + 100)
        backend.init_forward_metadata(self._make_forward_batch())

        torch.testing.assert_close(
            backend.forward_metadata.kv_indices,
            torch.tensor([107, 108, 109], dtype=torch.int64),
        )
        torch.testing.assert_close(
            backend.forward_metadata.out_cache_loc_full_physical,
            torch.tensor([120, 121, 122, 123], dtype=torch.int64),
        )

    def test_static_keeps_target_verify_locations_unchanged(self):
        backend = self._make_backend(None)
        backend.init_forward_metadata(self._make_forward_batch())

        torch.testing.assert_close(
            backend.forward_metadata.kv_indices,
            torch.tensor([7, 8, 9], dtype=torch.int64),
        )
        self.assertIsNone(backend.forward_metadata.out_cache_loc_full_physical)


if __name__ == "__main__":
    unittest.main()
