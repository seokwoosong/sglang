import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization import unquant
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnquantAiterDeterminism(CustomTestCase):
    def test_architecture_and_mode_gate(self):
        for gfx942, gfx95, deterministic, expected in (
            (True, False, True, True),
            (False, True, True, True),
            (True, False, False, False),
            (False, False, True, False),
        ):
            with (
                self.subTest(gfx942=gfx942, gfx95=gfx95, deterministic=deterministic),
                patch.multiple(
                    unquant,
                    _is_hip=True,
                    _is_gfx942=gfx942,
                    _is_gfx95=gfx95,
                    _skip_aiter_unquant=False,
                ),
                get_context().override_server_args(
                    enable_deterministic_inference=deterministic,
                    bf16_gemm_backend="auto",
                ),
            ):
                unquant.initialize_bf16_gemm_config()
                self.assertIs(unquant._skip_aiter_unquant, expected)

    def test_explicit_aiter_rejected_under_gfx942_deterministic_mode(self):
        with (
            patch.multiple(
                unquant,
                _is_hip=True,
                _is_gfx942=True,
                _is_gfx95=False,
                _skip_aiter_unquant=False,
                _use_aiter=True,
            ),
            patch.object(
                unquant,
                "get_moe_runner_backend",
                return_value=unquant.MoeRunnerBackend.AITER,
            ),
            get_context().override_server_args(
                enable_deterministic_inference=True,
                bf16_gemm_backend="auto",
            ),
        ):
            unquant.initialize_bf16_gemm_config()
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                unquant.UnquantizedFusedMoEMethod()


if __name__ == "__main__":
    unittest.main()
