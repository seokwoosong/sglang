"""Opt-in structured tracing for unified-memory HiCache debugging.

The trace is deliberately disabled unless ``SGLANG_HICACHE_TRACE_PATH`` is
set. Call sites normally pass CPU/Python metadata only. The unified allocator's
lazy-free hook deliberately materializes exact physical hole pages when tracing
is enabled because this recorder is for internal lifecycle validation, not
performance measurement; tracing disabled retains the asynchronous hot path.

Each process owns an independent monotonically ordered JSONL stream.  The
configured path may contain ``{pid}``; this is useful for TP/multi-process
runs.  File I/O and JSON encoding happen on a daemon thread so event producers
only copy a small Python dictionary into a queue.
"""

from __future__ import annotations

import atexit
import enum
import itertools
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

TRACE_SCHEMA_VERSION = 1
_TRACE_ENV = "SGLANG_HICACHE_TRACE_PATH"


def _json_value(value: Any) -> Any:
    """Normalize trace metadata without importing or inspecting CUDA tensors."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = value
        if isinstance(value, (set, frozenset)):
            items = sorted(value)
        return [_json_value(item) for item in items]
    # Trace hooks should never send tensors.  Keeping the fallback harmless is
    # preferable to turning optional observability into a serving failure.
    return repr(value)


class _FlushRequest:
    def __init__(self) -> None:
        self.done = threading.Event()


class HiCacheTraceRecorder:
    def __init__(self, path_template: str) -> None:
        self.pid = os.getpid()
        self.path = Path(path_template.format(pid=self.pid)).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.SimpleQueue[dict[str, Any] | _FlushRequest | None] = (
            queue.SimpleQueue()
        )
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_main,
            name="hicache-trace-writer",
            daemon=True,
        )
        self._thread.start()
        self.emit(
            "trace_started",
            path=str(self.path),
            schema_version=TRACE_SCHEMA_VERSION,
        )

    def _writer_main(self) -> None:
        # Line buffering makes already-consumed events survive an abrupt server
        # termination while keeping all disk I/O off the scheduler thread.
        with self.path.open("a", encoding="utf-8", buffering=1) as output:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if isinstance(item, _FlushRequest):
                    output.flush()
                    item.done.set()
                    continue
                output.write(json.dumps(item, separators=(",", ":")) + "\n")

    def emit(self, event: str, **fields: Any) -> None:
        normalized_fields = {key: _json_value(value) for key, value in fields.items()}
        with self._sequence_lock:
            if self._closed:
                return
            self._sequence += 1
            sequence = self._sequence
            record = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "seq": sequence,
                "ts_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "pid": self.pid,
                "thread": threading.current_thread().name,
                "event": event,
            }
            record.update(normalized_fields)
            # Enqueue while holding the sequence lock: queue order is then also
            # sequence order when multiple scheduler/helper threads emit.
            self._queue.put(record)

    def flush(self, timeout: float = 5.0) -> bool:
        request = _FlushRequest()
        with self._sequence_lock:
            if self._closed:
                return True
            self._queue.put(request)
        return request.done.wait(timeout)

    def close(self) -> None:
        request = _FlushRequest()
        with self._sequence_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(request)
        request.done.wait(5.0)
        self._queue.put(None)
        self._thread.join(timeout=1.0)


_recorder: Optional[HiCacheTraceRecorder] = None
_recorder_pid: Optional[int] = None
_recorder_lock = threading.Lock()
_object_id_sequence = itertools.count(1)
_object_id_lock = threading.Lock()
_OBJECT_ID_ATTRIBUTE = "_sglang_hicache_trace_id"


def trace_enabled() -> bool:
    return bool(os.getenv(_TRACE_ENV, ""))


def hicache_trace_object_id(value: Any) -> int | str:
    """Return a process-unique stable ID for a traced runtime object.

    Python may reuse ``id()`` as soon as a CUDA Event is collected.  HiCache
    transfers are short-lived enough for that to happen repeatedly in one
    trace, which can make replay merge unrelated D2H/H2D lifecycles.  A weak
    ID stored on the object keeps it stable for exactly that lifetime without
    retaining the CUDA object after its real owners release it.  Disabled
    tracing keeps the old zero-allocation fast path.
    """

    if not trace_enabled():
        return id(value)
    with _object_id_lock:
        object_id = getattr(value, _OBJECT_ID_ATTRIBUTE, None)
        if object_id is None:
            object_id = f"{os.getpid()}:{next(_object_id_sequence)}"
            try:
                setattr(value, _OBJECT_ID_ATTRIBUTE, object_id)
            except (AttributeError, TypeError):
                # Trace metadata must never make an otherwise valid transfer
                # fail. Runtime CUDA Event implementations used by SGLang have
                # a ``__dict__``; this fallback is for third-party event shims.
                return f"{os.getpid()}:py-{id(value)}"
        return object_id


def get_hicache_trace_recorder() -> Optional[HiCacheTraceRecorder]:
    global _recorder, _recorder_pid

    path = os.getenv(_TRACE_ENV, "")
    if not path:
        return None
    pid = os.getpid()
    if _recorder is not None and _recorder_pid == pid:
        return _recorder
    with _recorder_lock:
        if _recorder is None or _recorder_pid != pid:
            _recorder = HiCacheTraceRecorder(path)
            _recorder_pid = pid
        return _recorder


def trace_hicache_event(event: str, **fields: Any) -> None:
    recorder = get_hicache_trace_recorder()
    if recorder is not None:
        recorder.emit(event, **fields)


def flush_hicache_trace(timeout: float = 5.0) -> bool:
    recorder = get_hicache_trace_recorder()
    return recorder is None or recorder.flush(timeout)


def _close_hicache_trace() -> None:
    if _recorder is not None:
        _recorder.close()


def reset_hicache_trace_for_test() -> None:
    """Close and forget the process recorder; intended for isolated unit tests."""

    global _recorder, _recorder_pid
    with _recorder_lock:
        if _recorder is not None:
            _recorder.close()
        _recorder = None
        _recorder_pid = None


atexit.register(_close_hicache_trace)
