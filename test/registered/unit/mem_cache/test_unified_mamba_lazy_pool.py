"""Unified factories must preserve lazy checkpoint allocation and cleanup."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    init_unified_mamba_pools,
    init_unified_mamba_swa_pools,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestUnifiedMambaLazyPool(CustomTestCase):
    def _build(self, *, tri_pool, lazy, device):
        params = SimpleNamespace(
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(2, 2, 2)),
            dtype=SimpleNamespace(conv=torch.float32, temporal=torch.float32),
            layers=[0, 1],
        )
        kwargs = dict(
            device=device,
            kv_cache_dtype=torch.float32,
            head_num=2,
            head_dim=4,
            page_size=1,
            start_layer=0,
            end_layer=2,
            full_attention_layer_ids=[0],
            mamba_layer_ids=[0, 1],
            mamba2_cache_params=params,
            max_mamba_cache_size=16,
            model_context_len=16,
            extra_max_context_len=4,
            max_num_reqs=4,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=lazy,
            disable_overlap_schedule=False,
            need_sort=False,
            lazy_compaction=True,
        )
        if tri_pool:
            return init_unified_mamba_swa_pools(
                **kwargs,
                v_head_dim=4,
                swa_head_num=2,
                swa_head_dim=4,
                swa_v_head_dim=4,
                swa_attention_layer_ids=[1],
                full_max_total_num_tokens=64,
                swa_max_total_num_tokens=32,
            )
        return init_unified_mamba_pools(
            **kwargs,
            is_draft_worker=False,
            use_mla_backend=False,
            max_total_num_tokens=64,
            speculative_num_draft_tokens=None,
        )

    def _check_lifecycle(self, *, device):
        for tri_pool in (False, True):
            for lazy in (False, True):
                with self.subTest(device=device, tri_pool=tri_pool, lazy=lazy):
                    bundle = self._build(tri_pool=tri_pool, lazy=lazy, device=device)
                    pool = bundle.req_to_token_pool
                    allocator = pool.mamba_allocator
                    self.assertEqual(pool.enable_mamba_extra_buffer_lazy, lazy)
                    initial_available = allocator.available_size()
                    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=1))
                    req.kv.mamba_pool_idx = allocator.alloc(1)[0]
                    pool._alloc_ping_pong_buffer(req)
                    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1] = (
                        req.kv.mamba_ping_pong_track_buffer
                    )
                    self.assertEqual(
                        initial_available - allocator.available_size(), 2 if lazy else 3
                    )
                    self.assertEqual(
                        int((req.kv.mamba_ping_pong_track_buffer == -1).sum()),
                        1 if lazy else 0,
                    )
                    if lazy:
                        # A decode boundary promotes the second checkpoint and
                        # leaves a -1 sentinel where the old checkpoint lived.
                        pool.set_mamba_ping_pong_slot(req, 1, allocator.alloc(1)[0])
                        allocator.free(req.kv.mamba_ping_pong_track_buffer[:1])
                        pool.set_mamba_ping_pong_slot(req, 0, -1)
                    retained = req.kv.mamba_ping_pong_track_buffer[1:2].clone()
                    pool.free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=1)
                    self.assertEqual(allocator.available_size(), initial_available - 1)
                    self.assertTrue(bool((allocator.translate(retained) >= 0).all()))
                    self.assertEqual(
                        int((allocator.virtual_to_physical[1:] >= 0).sum()), 1
                    )
                    allocator.free(retained)
                    self.assertEqual(allocator.available_size(), initial_available)
                    self.assertEqual(
                        int((allocator.virtual_to_physical[1:] >= 0).sum()), 0
                    )
                    self.assertIsNone(req.kv.mamba_pool_idx)
                    self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)

    def test_cpu_lifecycle(self):
        self._check_lifecycle(device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_lifecycle(self):
        self._check_lifecycle(device="cuda")


if __name__ == "__main__":
    unittest.main()
