import pytest
import torch

from sglang.kernels.ops.moe.inkling_moe import (
    BLOCK_SIZE_M,
    SMALL_M_BLOCK_SIZE_M,
    compute_grouped_gemm_metadata,
    grouped_gemm_triton,
)
from sglang.srt.utils.device import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=12, stage="base-a-jit-kernel-unit", runner_config="amd")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx942_supported(), reason="gfx942 required"
)
@pytest.mark.parametrize("block_size_m", [SMALL_M_BLOCK_SIZE_M, BLOCK_SIZE_M])
def test_grouped_gemm_inkling_boot_shape(block_size_m: int):
    """Inkling BF16 boot tiles must fit MI300X LDS and match an fp32 reference."""
    torch.manual_seed(0)
    e, m, k, n = 8, 6, 1024, 8192
    sorted_ids = torch.arange(m, dtype=torch.int32, device="cuda") % e
    a = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    b = (torch.randn(e, n, k, device="cuda") * 0.02).to(torch.bfloat16)

    meta = compute_grouped_gemm_metadata(sorted_ids, e, block_size_m=block_size_m)
    got = grouped_gemm_triton(a, b, e, *meta, block_size_m=block_size_m)
    expected = torch.stack(
        [a[i].float() @ b[expert].float().T for i, expert in enumerate(sorted_ids)]
    )
    torch.testing.assert_close(got.float(), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
