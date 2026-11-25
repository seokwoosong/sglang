import sys

import pytest

from sglang.srt.layers.quantization import unquant
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_gfx942_deterministic_mode_skips_aiter(monkeypatch):
    """gfx942 deterministic mode must avoid AITER MoE's atomic stage-2 reduction."""
    monkeypatch.setattr(unquant, "_is_hip", True)
    monkeypatch.setattr(unquant, "_is_gfx95", False)
    monkeypatch.setattr(unquant, "_skip_aiter_unquant", False)

    with get_context().override_server_args(
        enable_deterministic_inference=True,
        bf16_gemm_backend="auto",
    ):
        unquant.initialize_bf16_gemm_config()

    assert unquant._skip_aiter_unquant


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
