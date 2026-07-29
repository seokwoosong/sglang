"""Opt-in artifact recording for unified-memory HiCache integrity tests."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

ARTIFACT_DIR_ENV = "SGLANG_UNIFIED_HICACHE_TEST_ARTIFACT_DIR"


def _command_output(command: list[str]) -> Optional[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class UnifiedHiCacheArtifactRecorder:
    """Write reproducibility evidence only when ``ARTIFACT_DIR_ENV`` is set."""

    def __init__(self, root: Optional[str] = None):
        root = root if root is not None else os.getenv(ARTIFACT_DIR_ENV)
        self.enabled = bool(root)
        self.run_dir: Optional[Path] = None
        self.server_log: Optional[TextIO] = None
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "result": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if not self.enabled:
            return

        commit = _command_output(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_dir = Path(root).expanduser().resolve() / f"{timestamp}_{commit}"
        (self.run_dir / "metrics").mkdir(parents=True)
        self.server_log = (self.run_dir / "server.log").open(
            "w", encoding="utf-8", buffering=1
        )
        self.write_json(
            "environment.json",
            {
                "commit": commit,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "gpu": _command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ]
                ),
            },
        )
        self._flush_report()

    @property
    def subprocess_output(self) -> Optional[tuple[TextIO, TextIO]]:
        if self.server_log is None:
            return None
        return (self.server_log, self.server_log)

    def write_json(self, relative_path: str, payload: Any) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def save_metrics(self, stage: str, prometheus_text: str) -> None:
        if self.run_dir is None:
            return
        (self.run_dir / "metrics" / f"{stage}.prom").write_text(
            prometheus_text, encoding="utf-8"
        )

    def update(self, **values: Any) -> None:
        self.report.update(values)
        self._flush_report()

    def pass_test(self) -> None:
        self.update(
            result="PASS",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def fail_test(self, error: BaseException) -> None:
        self.update(
            result="FAIL",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            },
        )

    def close(self) -> None:
        if self.report["result"] == "RUNNING":
            self.update(
                result="INCOMPLETE",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        if self.server_log is not None:
            self.server_log.flush()
            self.server_log.close()
            self.server_log = None

    def _flush_report(self) -> None:
        self.write_json("result.json", self.report)
