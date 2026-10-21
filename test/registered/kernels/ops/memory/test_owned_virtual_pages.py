"""Fused binding must return owned IDs without changing either mapping."""

import unittest

import torch

from sglang.kernels.ops.memory.virtual_slot import alloc_bind_inplace, bind_inplace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestOwnedVirtualPages(CustomTestCase):
    def test_binding_without_owned_output(self):
        for contiguous_physical in [False, True]:
            with self.subTest(contiguous_physical=contiguous_physical):
                virtual = torch.tensor([7, 3, 9], dtype=torch.int64, device="cuda")
                v2p = torch.full((16,), -1, dtype=torch.int64, device="cuda")
                p2v = torch.full_like(v2p, -1)
                if contiguous_physical:
                    physical = alloc_bind_inplace(virtual, v2p, p2v, 1)
                else:
                    physical = torch.tensor([4, 1, 6], dtype=torch.int64, device="cuda")
                    bind_inplace(virtual, physical, v2p, p2v)
                self.assertTrue(torch.equal(v2p[virtual], physical))
                self.assertTrue(torch.equal(p2v[physical], virtual))
                self.assertEqual(v2p[0].item(), -1)
                self.assertEqual(p2v[0].item(), -1)

    def test_owned_ids_and_mappings_after_graph_replay(self):
        for n in [0, 1, 127, 128, 129, 1025]:
            for contiguous_physical in [False, True]:
                with self.subTest(n=n, contiguous_physical=contiguous_physical):
                    free = torch.arange(1, n + 257, device="cuda", dtype=torch.int64)
                    virtual = free[17 : 17 + n]
                    expected = virtual.clone()
                    backing = torch.full(
                        (n + 2,), -99, dtype=torch.int64, device="cuda"
                    )
                    owned = backing[1:-1]
                    physical = torch.arange(
                        1, n + 1, device="cuda", dtype=torch.int64
                    ).flip(0)
                    v2p = torch.full(
                        (2 * n + 258,), -1, device="cuda", dtype=torch.int64
                    )
                    p2v = torch.full_like(v2p, -1)

                    def run():
                        if contiguous_physical:
                            return alloc_bind_inplace(
                                virtual, v2p, p2v, 1, out_virtual_pages=owned
                            )
                        bind_inplace(
                            virtual, physical, v2p, p2v, out_virtual_pages=owned
                        )
                        return physical

                    run()  # Compile before capture.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        mapped = run()
                    owned.fill_(-1)
                    v2p.fill_(-1)
                    p2v.fill_(-1)
                    graph.replay()
                    self.assertTrue(torch.equal(owned, expected))
                    self.assertTrue(torch.equal(v2p[owned], mapped))
                    self.assertTrue(torch.equal(p2v[mapped], owned))
                    self.assertEqual(backing[0].item(), -99)
                    self.assertEqual(backing[-1].item(), -99)
                    # Replays must read updated inputs, not captured ID values.
                    virtual.add_(n + 1)
                    expected = virtual.clone()
                    owned.fill_(-1)
                    v2p.fill_(-1)
                    p2v.fill_(-1)
                    graph.replay()
                    self.assertTrue(torch.equal(owned, expected))
                    self.assertTrue(torch.equal(v2p[owned], mapped))
                    self.assertTrue(torch.equal(p2v[mapped], owned))
                    self.assertEqual(backing[0].item(), -99)
                    self.assertEqual(backing[-1].item(), -99)
                    free.fill_(-1)
                    self.assertTrue(torch.equal(owned, expected))
                    self.assertEqual(v2p[0].item(), -1)
                    self.assertEqual(p2v[0].item(), -1)


if __name__ == "__main__":
    unittest.main()
