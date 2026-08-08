"""Opt-in aggregate profiler for hybrid/static and unified L1 memory paths.

The normal production path pays only a cached ``None`` check.  Experiments set
``SGLANG_MEMORY_BREAKDOWN_PROFILE_DIR`` and receive one periodically refreshed
JSON snapshot per process.  Aggregates are intentionally used instead of an
event log: allocator/translation calls are hot, and the profiler must not turn
the workload into an I/O benchmark.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_singleton_lock = threading.Lock()
_singleton: Optional[MemoryBreakdownProfiler] = None
_singleton_initialized = False


def _metric_key(category: str, pool: str, operation: str) -> str:
    return f"{category}|{pool}|{operation}"


def _empty_metric(category: str, pool: str, operation: str) -> dict[str, Any]:
    return {
        "category": category,
        "pool": pool,
        "operation": operation,
        "calls": 0,
        "errors": 0,
        "cpu_time_ns": 0,
        "rows": 0,
        "bytes": 0,
    }


class MemoryBreakdownProfiler:
    def __init__(
        self,
        output_dir: str,
        *,
        flush_interval_s: float,
        enable_cuda_timing: bool,
    ) -> None:
        if flush_interval_s <= 0:
            raise ValueError(
                "SGLANG_MEMORY_BREAKDOWN_PROFILE_INTERVAL must be positive"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"memory_profile.{os.getpid()}.json"
        self.flush_interval_s = flush_interval_s
        self.enable_cuda_timing = enable_cuda_timing and torch.cuda.is_available()
        self.started_wall_time_ns = time.time_ns()
        self.started_monotonic_ns = time.perf_counter_ns()

        self._lock = threading.RLock()
        self._metrics: dict[str, dict[str, Any]] = {}
        self._cuda_metrics: dict[str, dict[str, Any]] = {}
        self._samples: dict[str, dict[str, Any]] = {}
        self._layouts: dict[str, dict[str, Any]] = {}
        self._pending_cuda: list[
            tuple[torch.cuda.Event, torch.cuda.Event, str, str, str, int, int]
        ] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="memory-breakdown-profiler",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)
        logger.info("Memory breakdown profiler enabled: %s", self.output_path)

    def _get_metric(
        self,
        collection: dict[str, dict[str, Any]],
        category: str,
        pool: str,
        operation: str,
    ) -> dict[str, Any]:
        key = _metric_key(category, pool, operation)
        metric = collection.get(key)
        if metric is None:
            metric = _empty_metric(category, pool, operation)
            collection[key] = metric
        return metric

    def record_cpu(
        self,
        category: str,
        pool: str,
        operation: str,
        elapsed_ns: int,
        *,
        rows: int = 0,
        num_bytes: int = 0,
        error: bool = False,
    ) -> None:
        with self._lock:
            metric = self._get_metric(self._metrics, category, pool, operation)
            metric["calls"] += 1
            metric["errors"] += int(error)
            metric["cpu_time_ns"] += max(0, int(elapsed_ns))
            metric["rows"] += max(0, int(rows))
            metric["bytes"] += max(0, int(num_bytes))

    def increment(
        self,
        category: str,
        pool: str,
        operation: str,
        *,
        calls: int = 1,
        rows: int = 0,
        num_bytes: int = 0,
    ) -> None:
        with self._lock:
            metric = self._get_metric(self._metrics, category, pool, operation)
            metric["calls"] += max(0, int(calls))
            metric["rows"] += max(0, int(rows))
            metric["bytes"] += max(0, int(num_bytes))

    def record_sample(
        self,
        category: str,
        pool: str,
        operation: str,
        value: int,
    ) -> None:
        key = _metric_key(category, pool, operation)
        value = int(value)
        with self._lock:
            sample = self._samples.get(key)
            if sample is None:
                sample = {
                    "category": category,
                    "pool": pool,
                    "operation": operation,
                    "count": 0,
                    "sum": 0,
                    "min": value,
                    "max": value,
                    "histogram": {},
                }
                self._samples[key] = sample
            sample["count"] += 1
            sample["sum"] += value
            sample["min"] = min(sample["min"], value)
            sample["max"] = max(sample["max"], value)
            histogram = sample["histogram"]
            bucket = str(value) if value <= 64 else "65+"
            histogram[bucket] = histogram.get(bucket, 0) + 1

    def record_layout(
        self,
        pool: str,
        layout_kind: str,
        metadata: dict[str, Any],
    ) -> None:
        with self._lock:
            self._layouts[f"{pool}|{layout_kind}"] = {
                "pool": pool,
                "layout_kind": layout_kind,
                **metadata,
            }

    def start_cuda_interval(self) -> Optional[torch.cuda.Event]:
        if not self.enable_cuda_timing:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def finish_cuda_interval(
        self,
        start: Optional[torch.cuda.Event],
        category: str,
        pool: str,
        operation: str,
        *,
        rows: int = 0,
        num_bytes: int = 0,
    ) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        with self._lock:
            self._pending_cuda.append(
                (start, end, category, pool, operation, int(rows), int(num_bytes))
            )

    def _drain_cuda(self) -> None:
        if not self._pending_cuda:
            return
        remaining = []
        for item in self._pending_cuda:
            start, end, category, pool, operation, rows, num_bytes = item
            try:
                ready = end.query()
            except Exception:  # pragma: no cover - defensive during CUDA teardown
                ready = False
            if not ready:
                remaining.append(item)
                continue
            elapsed_ns = int(start.elapsed_time(end) * 1_000_000)
            metric = self._get_metric(self._cuda_metrics, category, pool, operation)
            metric["calls"] += 1
            metric["cpu_time_ns"] += max(0, elapsed_ns)
            metric["rows"] += max(0, rows)
            metric["bytes"] += max(0, num_bytes)
        self._pending_cuda = remaining

    def _snapshot_locked(self) -> dict[str, Any]:
        self._drain_cuda()
        return {
            "schema_version": _SCHEMA_VERSION,
            "pid": os.getpid(),
            "started_wall_time_ns": self.started_wall_time_ns,
            "updated_wall_time_ns": time.time_ns(),
            "elapsed_seconds": (time.perf_counter_ns() - self.started_monotonic_ns)
            / 1e9,
            "cuda_timing_enabled": self.enable_cuda_timing,
            "pending_cuda_intervals": len(self._pending_cuda),
            "metrics": sorted(
                (dict(value) for value in self._metrics.values()),
                key=lambda item: (
                    item["category"],
                    item["pool"],
                    item["operation"],
                ),
            ),
            "cuda_metrics": sorted(
                (dict(value) for value in self._cuda_metrics.values()),
                key=lambda item: (
                    item["category"],
                    item["pool"],
                    item["operation"],
                ),
            ),
            "samples": sorted(
                (dict(value) for value in self._samples.values()),
                key=lambda item: (
                    item["category"],
                    item["pool"],
                    item["operation"],
                ),
            ),
            "layouts": sorted(
                (dict(value) for value in self._layouts.values()),
                key=lambda item: (item["pool"], item["layout_kind"]),
            ),
        }

    def flush(self) -> None:
        with self._lock:
            snapshot = self._snapshot_locked()
        temporary = self.output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output_path)

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_interval_s):
            try:
                self.flush()
            except Exception:  # pragma: no cover - profiler must not kill serving
                logger.exception("Failed to flush memory breakdown profile")

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.flush_interval_s * 2))
        try:
            self.flush()
        except Exception:  # pragma: no cover - defensive during interpreter exit
            pass


def get_memory_breakdown_profiler() -> Optional[MemoryBreakdownProfiler]:
    global _singleton, _singleton_initialized
    if _singleton_initialized:
        return _singleton
    with _singleton_lock:
        if _singleton_initialized:
            return _singleton
        output_dir = envs.SGLANG_MEMORY_BREAKDOWN_PROFILE_DIR.get()
        if output_dir:
            _singleton = MemoryBreakdownProfiler(
                output_dir,
                flush_interval_s=envs.SGLANG_MEMORY_BREAKDOWN_PROFILE_INTERVAL.get(),
                enable_cuda_timing=(
                    envs.SGLANG_MEMORY_BREAKDOWN_PROFILE_CUDA_TIMING.get()
                ),
            )
        _singleton_initialized = True
        return _singleton


@contextmanager
def profile_cpu_scope(
    profiler: Optional[MemoryBreakdownProfiler],
    category: str,
    pool: str,
    operation: str,
    *,
    rows: int = 0,
    num_bytes: int = 0,
) -> Iterator[None]:
    if profiler is None:
        yield
        return
    started_ns = time.perf_counter_ns()
    failed = False
    try:
        yield
    except Exception:
        failed = True
        raise
    finally:
        profiler.record_cpu(
            category,
            pool,
            operation,
            time.perf_counter_ns() - started_ns,
            rows=rows,
            num_bytes=num_bytes,
            error=failed,
        )


def record_mamba_layout(
    profiler: Optional[MemoryBreakdownProfiler],
    *,
    pool: str,
    layout_kind: str,
    conv: list[torch.Tensor],
    temporal: torch.Tensor,
) -> None:
    if profiler is None:
        return

    def tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
        inner_elements = 1
        for size in tensor.shape[2:]:
            inner_elements *= int(size)
        row_bytes = inner_elements * tensor.element_size()
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "row_bytes": row_bytes,
            "layer_stride_bytes": int(tensor.stride(0) * tensor.element_size()),
            "slot_stride_bytes": int(tensor.stride(1) * tensor.element_size()),
            "slot_stride_amplification": (
                float(tensor.stride(1) * tensor.element_size()) / row_bytes
                if row_bytes
                else 0.0
            ),
        }

    profiler.record_layout(
        pool,
        layout_kind,
        {
            "num_layers": int(temporal.shape[0]),
            "num_slots": int(temporal.shape[1]),
            "temporal": tensor_layout(temporal),
            "conv": [tensor_layout(tensor) for tensor in conv],
        },
    )


def _reset_memory_breakdown_profiler_for_testing() -> None:
    global _singleton, _singleton_initialized
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
        _singleton = None
        _singleton_initialized = False
