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
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    TritonMultiStepDraftBackend,
)
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

    def test_graph_replay_uses_verify_length_instead_of_stale_batch_length(self):
        backend = self._make_backend(lambda loc: loc + 100)
        backend.cuda_graph_kv_indices = torch.tensor([7, 8, 9, 0], dtype=torch.int64)
        backend.cuda_graph_out_cache_loc_full_physical = torch.zeros(
            4, dtype=torch.int64
        )
        backend.kv_indptr = torch.tensor([0, 3], dtype=torch.int32)
        forward_batch = self._make_forward_batch()
        forward_batch.seq_lens_sum = 1
        forward_batch.spec_info.seq_lens_sum = 3
        forward_batch.num_padding = 0

        backend._translate_cuda_graph_shared_pool_locs(forward_batch, bs=1)

        torch.testing.assert_close(
            backend.cuda_graph_kv_indices,
            torch.tensor([107, 108, 109, 0], dtype=torch.int64),
        )


class _FakeDraftIndicesKernel:
    def __getitem__(self, _grid):
        def launch(*args):
            args[3].fill_(7)

        return launch


class TestTritonUnifiedMultiStepDraft(unittest.TestCase):
    def test_graph_replay_translates_generated_indices_in_place(self):
        backend = object.__new__(TritonMultiStepDraftBackend)
        backend.speculative_num_steps = 3
        backend.topk = 1
        backend.max_context_len = 8
        backend.pool_len = 8
        backend.page_size = 1
        backend.kv_indptr = torch.zeros((3, 2), dtype=torch.int32)
        backend.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((1, 8), dtype=torch.int32)
        )

        translated_ptrs = []

        def translate(loc, *, out=None):
            self.assertIs(out, loc)
            translated_ptrs.append(loc.data_ptr())
            out.copy_(loc + 100)
            return out

        backend.attn_backends = [
            types.SimpleNamespace(_translate_kv_loc=translate),
            types.SimpleNamespace(_translate_kv_loc=translate),
        ]
        kv_indices = torch.zeros((3, 8), dtype=torch.int64)
        forward_batch = types.SimpleNamespace(
            batch_size=1,
            seq_lens_sum=2,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            positions=torch.tensor([1], dtype=torch.int64),
        )

        with patch(
            "sglang.srt.layers.attention.triton_backend.generate_draft_decode_kv_indices",
            _FakeDraftIndicesKernel(),
        ):
            backend.common_template(forward_batch, kv_indices, call_fn=None)

        self.assertEqual(len(translated_ptrs), 2)
        torch.testing.assert_close(kv_indices[0, :3], torch.full((3,), 107))
        torch.testing.assert_close(kv_indices[1, :4], torch.full((4,), 107))


if __name__ == "__main__":
    unittest.main()
