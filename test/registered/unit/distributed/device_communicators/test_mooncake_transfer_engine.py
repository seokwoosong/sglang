"""Regression tests for Mooncake engine platform selection."""

import os
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import mooncake_transfer_engine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeTransferEngine(CustomTestCase):
    def test_rocm_disables_hip_ipc_for_multi_buffer_writes(self):
        """ROCm must bypass HIP IPC, whose sub-allocation relocation corrupts
        multi-buffer PD writes; non-ROCm construction must not change the env."""
        for hip, expected in ((True, "1"), (False, None)):
            with self.subTest(hip=hip):
                fake_package = types.ModuleType("mooncake")
                fake_package.__path__ = []
                fake_engine = types.ModuleType("mooncake.engine")

                class FakeTransferEngine:
                    def get_rpc_port(self):
                        return 12345

                fake_engine.TransferEngine = FakeTransferEngine
                fake_package.engine = fake_engine
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.dict(
                        sys.modules,
                        {"mooncake": fake_package, "mooncake.engine": fake_engine},
                    ),
                    patch.object(
                        mooncake_transfer_engine,
                        "is_hip",
                        return_value=hip,
                        create=True,
                    ),
                    patch.object(
                        mooncake_transfer_engine.MooncakeTransferEngine,
                        "initialize",
                        return_value=None,
                    ),
                ):
                    mooncake_transfer_engine.MooncakeTransferEngine(
                        hostname="127.0.0.1", gpu_id=0
                    )
                    self.assertEqual(os.environ.get("MC_DISABLE_HIP"), expected)


if __name__ == "__main__":
    unittest.main()
