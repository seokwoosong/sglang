from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd-mi35x")

import unittest

import torch

try:
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
        fused_mamba_state_scatter_with_mask,
    )

    _FUSED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    fused_mamba_state_scatter_with_mask = None
    _FUSED_IMPORT_ERROR = e


def _ref_scatter(dst, src, dst_indices, src_indices, step_indices):
    """Reference implementation using PyTorch advanced indexing."""
    # dst: [L, C, E]
    # src: [L, S, D, E]
    dst[:, dst_indices] = src[:, src_indices, step_indices].to(dst.dtype, copy=False)


def _ref_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    step_indices_raw,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Reference implementation using PyTorch advanced indexing for correctness verification."""
    total_requests = step_indices_raw.shape[0]
    intermediate_state_indices = torch.arange(
        total_requests, dtype=torch.int32, device=step_indices_raw.device
    )

    valid_mask = step_indices_raw >= 0
    dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)
    src_state_indices = intermediate_state_indices[valid_mask].to(torch.int64)
    last_steps = step_indices_raw[valid_mask].to(torch.int64)

    # Only scatter if there are valid indices (but don't early return -
    # mamba_track_indices processing is independent)
    if dst_state_indices.numel() > 0:
        _ref_scatter(
            ssm_states,
            intermediate_ssm,
            dst_state_indices,
            src_state_indices,
            last_steps,
        )
        _ref_scatter(
            conv_states,
            intermediate_conv,
            dst_state_indices,
            src_state_indices,
            last_steps,
        )

    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
        track_mask = mamba_steps_to_track >= 0
        if not track_mask.any():
            return
        dst_track_indices = mamba_track_indices[track_mask].to(torch.int64)
        src_track_indices = intermediate_state_indices[track_mask].to(torch.int64)
        track_steps = mamba_steps_to_track[track_mask].to(torch.int64)

        _ref_scatter(
            ssm_states,
            intermediate_ssm,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )
        _ref_scatter(
            conv_states,
            intermediate_conv,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )


def _fused_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    step_indices_raw,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Matches the fully fused logic that avoids index_select and nonzero calls."""
    # Use fully fused kernel that handles masking internally
    fused_mamba_state_scatter_with_mask(
        ssm_states,
        intermediate_ssm,
        state_indices_tensor,
        step_indices_raw,
    )
    fused_mamba_state_scatter_with_mask(
        conv_states,
        intermediate_conv,
        state_indices_tensor,
        step_indices_raw,
    )

    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_ssm,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv,
            mamba_track_indices,
            mamba_steps_to_track,
        )


class TestMambaStateScatterCorrectness(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_fused_matches_reference(self):
        """Test that fused_mamba_state_scatter_with_mask matches the reference."""
        if fused_mamba_state_scatter_with_mask is None:
            self.skipTest(
                f"fused_mamba_state_scatter_with_mask import failed: {_FUSED_IMPORT_ERROR}"
            )

        torch.manual_seed(42)
        device = torch.device("cuda")

        # Keep sizes moderate so this test is quick.
        L = 8
        B = 32
        C = 49
        D = 5
        ssm_elems = 1024
        conv_elems = 512

        ssm_states0 = torch.randn(
            (L, C, ssm_elems), device=device, dtype=torch.bfloat16
        )
        conv_states0 = torch.randn(
            (L, C, conv_elems), device=device, dtype=torch.bfloat16
        )
        intermediate_ssm = torch.randn(
            (L, B, D, ssm_elems), device=device, dtype=torch.bfloat16
        )
        intermediate_conv = torch.randn(
            (L, B, D, conv_elems), device=device, dtype=torch.bfloat16
        )

        # unique cache lines (no duplicates) to avoid nondeterministic write order
        state_indices_tensor = torch.randperm(C, device=device, dtype=torch.int64)[
            :B
        ].to(torch.int32)

        step_indices_raw = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        # set ~10% invalid
        invalid = torch.rand((B,), device=device) < 0.1
        step_indices_raw[invalid] = -1

        # Optional track update
        mamba_track_indices = torch.randperm(C, device=device, dtype=torch.int64)[:B]
        mamba_steps_to_track = torch.randint(
            0, D, (B,), device=device, dtype=torch.int64
        )
        track_invalid = torch.rand((B,), device=device) < 0.7
        mamba_steps_to_track[track_invalid] = -1

        ssm_ref = ssm_states0.clone()
        conv_ref = conv_states0.clone()
        ssm_fused = ssm_states0.clone()
        conv_fused = conv_states0.clone()

        _ref_update_like(
            ssm_ref,
            intermediate_ssm,
            conv_ref,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            step_indices_raw=step_indices_raw,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )
        _fused_update_like(
            ssm_fused,
            intermediate_ssm,
            conv_fused,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            step_indices_raw=step_indices_raw,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )

        torch.testing.assert_close(ssm_fused, ssm_ref)
        torch.testing.assert_close(conv_fused, conv_ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_fused_scatter_supports_unified_envelope_strides(self):
        """Speculative verify must commit into page-first unified Mamba views.

        Their layer and cache-slot axes are not contiguous: each physical slot
        contains every Mamba layer's conv and SSM state in one envelope.  The
        payload axes within a state row remain compact, which is the contract
        supported by the scatter kernels.
        """
        if fused_mamba_state_scatter_with_mask is None:
            self.skipTest(
                f"fused_mamba_state_scatter_with_mask import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_layers, cache_size = 3, 7
        batch_size, draft_steps = 3, 4
        temporal_shape = (2, 3)
        conv_shape = (4, 3)
        temporal_elems = 6
        conv_elems = 12
        envelope_elems = 64

        raw = torch.full(
            (cache_size * envelope_elems,), -99, device=device, dtype=dtype
        )
        conv_dst = raw.as_strided(
            (num_layers, cache_size, *conv_shape),
            (conv_elems, envelope_elems, conv_shape[1], 1),
        )
        temporal_dst = raw.as_strided(
            (num_layers, cache_size, *temporal_shape),
            (temporal_elems, envelope_elems, temporal_shape[1], 1),
            storage_offset=num_layers * conv_elems,
        )
        conv_dst.zero_()
        temporal_dst.zero_()
        self.assertFalse(conv_dst.is_contiguous())
        self.assertFalse(temporal_dst.is_contiguous())

        temporal_src = (
            torch.arange(
                num_layers * batch_size * draft_steps * temporal_elems,
                device=device,
                dtype=torch.float32,
            )
            .reshape(num_layers, batch_size, draft_steps, *temporal_shape)
            .to(dtype)
        )
        conv_src = (
            torch.arange(
                num_layers * batch_size * draft_steps * conv_elems,
                device=device,
                dtype=torch.float32,
            )
            .reshape(num_layers, batch_size, draft_steps, *conv_shape)
            .to(dtype)
            + 1000
        )

        state_indices = torch.tensor([1, 5, 3], device=device, dtype=torch.int32)
        accepted_steps = torch.tensor([2, -1, 0], device=device, dtype=torch.int32)
        track_indices = torch.tensor([6, 4, 0], device=device, dtype=torch.int32)
        track_steps = torch.tensor([1, 3, -1], device=device, dtype=torch.int32)

        temporal_ref = torch.zeros(
            temporal_dst.shape, device=device, dtype=temporal_dst.dtype
        )
        conv_ref = torch.zeros(conv_dst.shape, device=device, dtype=conv_dst.dtype)
        for request_idx in range(batch_size):
            for dst_indices, step_indices in (
                (state_indices, accepted_steps),
                (track_indices, track_steps),
            ):
                step = int(step_indices[request_idx].item())
                if step < 0:
                    continue
                dst_idx = int(dst_indices[request_idx].item())
                temporal_ref[:, dst_idx] = temporal_src[:, request_idx, step]
                conv_ref[:, dst_idx] = conv_src[:, request_idx, step]

        for dst_indices, step_indices in (
            (state_indices, accepted_steps),
            (track_indices, track_steps),
        ):
            fused_mamba_state_scatter_with_mask(
                temporal_dst, temporal_src, dst_indices, step_indices
            )
            fused_conv_window_scatter_with_mask(
                conv_dst, conv_src, dst_indices, step_indices
            )

        torch.testing.assert_close(temporal_dst.contiguous(), temporal_ref)
        torch.testing.assert_close(conv_dst.contiguous(), conv_ref)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
