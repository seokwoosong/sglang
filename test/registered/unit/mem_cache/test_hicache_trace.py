"""CPU tests for structured unified-memory HiCache tracing and replay."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_trace import (
    flush_hicache_trace,
    hicache_trace_object_id,
    reset_hicache_trace_for_test,
    trace_hicache_event,
)
from sglang.srt.mem_cache.hicache_trace_replay import (
    ReplayState,
    _byte_bar,
    _row_fence_cases,
    _select_frame_indices,
    _visual_frame,
    coverage_report,
    load_events,
    render_state,
    write_html,
)
from sglang.srt.mem_cache.typed_chunk_host import TypedChunkHostAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheTrace(unittest.TestCase):
    def tearDown(self):
        reset_hicache_trace_for_test()

    def _record(self, callback):
        with tempfile.TemporaryDirectory() as directory:
            path_template = str(Path(directory) / "trace-{pid}.jsonl")
            with patch.dict(os.environ, {"SGLANG_HICACHE_TRACE_PATH": path_template}):
                reset_hicache_trace_for_test()
                callback()
                self.assertTrue(flush_hicache_trace())
                path = Path(path_template.format(pid=os.getpid()))
                return load_events([path])

    def test_recorder_preserves_order_and_normalizes_sets(self):
        events = self._record(
            lambda: (
                trace_hicache_event("first", pages={3, 1, 2}),
                trace_hicache_event("second", value="done"),
            )
        )
        self.assertEqual(
            [event["event"] for event in events], ["trace_started", "first", "second"]
        )
        self.assertEqual(events[1]["pages"], [1, 2, 3])
        self.assertLess(events[1]["seq"], events[2]["seq"])

    def test_runtime_object_ids_are_stable_and_never_reused(self):
        class Event:
            pass

        with patch.dict(os.environ, {"SGLANG_HICACHE_TRACE_PATH": "/tmp/enabled"}):
            first = Event()
            second = Event()
            first_id = hicache_trace_object_id(first)
            self.assertEqual(hicache_trace_object_id(first), first_id)
            self.assertNotEqual(hicache_trace_object_id(second), first_id)

    def test_multithreaded_file_order_matches_sequence_order(self):
        def exercise():
            def emit(worker):
                for item in range(25):
                    trace_hicache_event("parallel", worker=worker, item=item)

            threads = [
                threading.Thread(target=emit, args=(worker,)) for worker in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        events = self._record(exercise)
        sequences = [event["seq"] for event in events]
        self.assertEqual(sequences, list(range(1, len(events) + 1)))

    def test_typed_chunks_emit_owner_usage_and_pin_lifecycle(self):
        def exercise():
            allocator = TypedChunkHostAllocator(
                total_bytes=256,
                kv_page_bytes=16,
                mamba_slot_bytes=48,
            )
            kv = allocator.alloc_kv(2)
            mamba = allocator.alloc_mamba(1)
            allocator.pin_chunks(allocator.kv_chunks(kv))
            allocator.free_kv(kv)
            allocator.unpin_chunks([0])
            allocator.free_mamba(mamba)

        events = self._record(exercise)
        kinds = {event["event"] for event in events}
        self.assertTrue(
            {
                "l2_arena_initialized",
                "l2_kv_allocated",
                "l2_mamba_allocated",
                "l2_chunks_pinned",
                "l2_kv_freed",
                "l2_chunks_unpinned",
            }.issubset(kinds)
        )
        owner_changes = [
            event for event in events if event["event"] == "l2_chunk_owner_changed"
        ]
        self.assertTrue(
            any(
                event["previous_owner"] == "FREE" and event["owner"] == "KV"
                for event in owner_changes
            )
        )
        self.assertTrue(
            any(
                event["previous_owner"] == "KV" and event["owner"] == "FREE"
                for event in owner_changes
            )
        )

    def test_replay_renders_layout_and_validates_fence_intersection(self):
        events = [
            {
                "seq": 1,
                "event": "l1_allocator_initialized",
                "pool": "full",
                "grow_direction": "up",
                "total_bytes": 1024,
                "entry_bytes_per_page": 16,
                "page_size": 1,
                "num_pages": 64,
                "min_page_index": 1,
                "watermark_page": 20,
                "byte_low_frontier": 16,
                "byte_high_frontier": 320,
                "live_pages": 18,
                "free_hole_pages": 1,
                "pending_reuse_pages": 0,
            },
            {
                "seq": 2,
                "event": "l1_allocator_initialized",
                "pool": "mamba",
                "grow_direction": "down",
                "total_bytes": 1024,
                "entry_bytes_per_page": 64,
                "page_size": 1,
                "num_pages": 16,
                "min_page_index": 1,
                "watermark_page": 12,
                "byte_low_frontier": 832,
                "byte_high_frontier": 1024,
                "live_pages": 3,
                "free_hole_pages": 0,
                "pending_reuse_pages": 0,
            },
            {
                "seq": 3,
                "event": "l1_transfer_fence_registered",
                "pool": "full",
                "hazard_id": 7,
            },
            {
                "seq": 4,
                "event": "l1_transfer_fence_checked",
                "pool": "full",
                "hazard_id": 7,
                "protected_pages": [19],
                "touched_pages": [2, 19],
                "intersection_pages": [19],
                "blocking": True,
            },
            {
                "seq": 5,
                "event": "l1_compaction_decision",
                "pool": "full",
                "decision": "deferred_row_fence",
                "touched_pages": [2, 19],
                "destination_pages": [2],
                "source_pages": [19],
            },
        ]
        state = ReplayState()
        for event in events:
            state.apply(event)
        frame = render_state(state, len(events) - 1, len(events))
        self.assertIn("L1 unified GPU byte arena", frame)
        self.assertIn("deferred_row_fence", frame)
        visual = _visual_frame(state, len(events) - 1, len(events))
        self.assertEqual(visual["l1"]["frontier_gap_bytes"], 512)
        self.assertEqual(visual["l1"]["frontier_gap_pct"], 50)
        self.assertEqual(
            {
                item["pool"]: item["watermark_pct"]
                for item in visual["l1"]["allocators"]
            },
            {"full": 31.25, "mamba": 81.25},
        )
        self.assertFalse(state.errors)

    def test_l2_chunk_fill_tracks_kv_usage_and_full_mamba_state(self):
        state = ReplayState(
            l2_config={"kv_pages_per_chunk": 10},
            chunks={
                0: {
                    "owner": "KV",
                    "kv_used_offsets": [0, 1, 2],
                },
                1: {"owner": "MAMBA"},
                2: {"owner": "FREE"},
            },
        )
        chunks = _visual_frame(state, 0, 1)["l2"]["chunks"]
        self.assertEqual([chunk["fill_pct"] for chunk in chunks], [30, 100, 0])

    def test_mamba_lru_risk_maps_to_l1_rows_and_l2_chunks(self):
        state = ReplayState(
            allocators={
                "mamba": {
                    "grow_direction": "up",
                    "total_bytes": 100,
                    "entry_bytes_per_page": 10,
                    "min_page_index": 1,
                    "watermark_page": 5,
                    "byte_low_frontier": 10,
                    "byte_high_frontier": 50,
                    "live_pages": 3,
                    "free_hole_pages": 0,
                    "pending_reuse_pages": 0,
                }
            },
            l2_config={"kv_pages_per_chunk": 10},
            chunks={7: {"owner": "MAMBA"}},
        )
        state.apply(
            {
                "seq": 1,
                "event": "mamba_lru_state",
                "reason": "device_eviction_requested",
                "device": [
                    {
                        "node_id": 11,
                        "indices": [2],
                        "virtual_indices": [9],
                        "index_space": "physical",
                        "eviction_rank": 1,
                        "eviction_score": 100,
                        "protected": False,
                        "lock_ref": 0,
                    },
                    {
                        "node_id": 12,
                        "indices": [3],
                        "virtual_indices": [3],
                        "index_space": "identity",
                        "eviction_rank": None,
                        "eviction_score": 0,
                        "protected": True,
                        "lock_ref": 1,
                    },
                    {
                        "node_id": 13,
                        "indices": [99],
                        "eviction_rank": 3,
                        "eviction_score": 25,
                        "protected": False,
                        "lock_ref": 0,
                    },
                ],
                "host": [
                    {
                        "node_id": 21,
                        "indices": [7],
                        "eviction_rank": 1,
                        "eviction_score": 100,
                        "protected": False,
                        "lock_ref": 0,
                    }
                ],
            }
        )
        frame = _visual_frame(state, 0, 1)
        self.assertEqual(
            [span["eviction_score"] for span in frame["l1"]["mamba_lru_spans"]],
            [100, 0],
        )
        self.assertEqual(frame["l1"]["mamba_lru_spans"][0]["page"], 2)
        self.assertTrue(frame["l1"]["mamba_lru_spans"][1]["protected"])
        self.assertEqual(frame["l2"]["chunks"][0]["mamba_lru"]["node_id"], 21)
        self.assertEqual(frame["l2"]["chunks"][0]["mamba_lru"]["eviction_score"], 100)

    def test_l2_mamba_chunk_explains_why_host_lru_score_is_absent(self):
        state = ReplayState(
            l2_config={"kv_pages_per_chunk": 10},
            chunks={5: {"owner": "MAMBA"}},
        )
        state.apply(
            {
                "seq": 1,
                "event": "l2_backup_requested",
                "node_id": 36,
                "full_tokens": 64,
                "component_transfers": {"MAMBA": ["MAMBA"]},
            }
        )
        state.apply(
            {
                "seq": 2,
                "event": "l2_backup_metadata_committed",
                "node_id": 36,
                "mamba_host_chunks": [5],
            }
        )
        state.apply(
            {
                "seq": 3,
                "event": "mamba_lru_state",
                "reason": "match",
                "device": [
                    {
                        "node_id": 36,
                        "indices": [2],
                        "index_space": "physical",
                        "protected": True,
                        "lock_ref": 1,
                        "eviction_rank": None,
                        "eviction_score": 0,
                    }
                ],
                "host": [],
            }
        )
        chunk = _visual_frame(state, 0, 1)["l2"]["chunks"][0]
        self.assertEqual(chunk["mamba_node_id"], 36)
        self.assertEqual(chunk["mamba_status"], "l1_locked_backup")
        self.assertIsNone(chunk["mamba_lru"])

        state.apply(
            {
                "seq": 4,
                "event": "l1_node_evicted",
                "node_id": 36,
                "evicted_by_component": {"MAMBA": 1},
            }
        )
        state.apply(
            {
                "seq": 5,
                "event": "mamba_lru_state",
                "reason": "device_eviction_completed",
                "device": [],
                "host": [],
            }
        )
        chunk = _visual_frame(state, 0, 1)["l2"]["chunks"][0]
        self.assertEqual(chunk["mamba_status"], "l2_lru_pending")

        state.apply(
            {
                "seq": 6,
                "event": "mamba_lru_state",
                "reason": "device_eviction_completed",
                "device": [],
                "host": [
                    {
                        "node_id": 36,
                        "indices": [5],
                        "protected": False,
                        "lock_ref": 0,
                        "eviction_rank": 1,
                        "eviction_score": 100,
                    }
                ],
            }
        )
        chunk = _visual_frame(state, 0, 1)["l2"]["chunks"][0]
        self.assertEqual(chunk["mamba_status"], "host_lru")
        self.assertEqual(chunk["mamba_lru"]["eviction_score"], 100)

    def test_l1_bar_uses_pool_identity_not_growth_direction_for_glyphs(self):
        state = ReplayState(
            allocators={
                "full": {
                    "grow_direction": "down",
                    "total_bytes": 100,
                    "entry_bytes_per_page": 1,
                    "min_page_index": 5,
                    "byte_low_frontier": 80,
                    "byte_high_frontier": 100,
                },
                "mamba": {
                    "grow_direction": "up",
                    "total_bytes": 100,
                    "entry_bytes_per_page": 5,
                    "min_page_index": 1,
                    "byte_low_frontier": 5,
                    "byte_high_frontier": 30,
                },
            }
        )
        bar = _byte_bar(state, width=20)
        self.assertLess(bar.index("M"), bar.index("K"))

    def test_compaction_markers_do_not_replace_base_ownership(self):
        state = ReplayState()
        state.apply(
            {
                "seq": 1,
                "event": "l1_allocator_initialized",
                "pool": "full",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 9,
                "byte_low_frontier": 10,
                "byte_high_frontier": 90,
                "live_pages": 7,
                "free_hole_pages": 1,
                "pending_reuse_pages": 0,
            }
        )
        state.apply(
            {
                "seq": 2,
                "event": "l1_compaction_decision",
                "pool": "full",
                "decision": "allowed",
                "free_pages": [3],
                "source_pages": [8],
                "destination_pages": [3],
                "touched_pages": [3, 8],
            }
        )
        frame = _visual_frame(state, 1, 2)
        roles_by_page = {}
        for marker in frame["l1"]["markers"]:
            roles_by_page.setdefault(marker["page"], set()).add(marker["role"])
        self.assertEqual(roles_by_page[3], {"H", "D"})
        self.assertEqual(roles_by_page[8], {"S"})
        bar = _byte_bar(state, width=20)
        self.assertIn("K", bar)
        self.assertFalse(set("HSDF") & set(bar))

    def test_compaction_completion_tracks_new_source_hole(self):
        state = ReplayState()
        state.apply(
            {
                "seq": 1,
                "event": "l1_allocator_initialized",
                "pool": "mamba",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 5,
                "byte_low_frontier": 10,
                "byte_high_frontier": 50,
                "live_pages": 1,
                "free_hole_pages": 3,
                "pending_reuse_pages": 0,
            }
        )
        state.apply(
            {
                "seq": 2,
                "event": "l1_compaction_decision",
                "pool": "mamba",
                "urgent": False,
                "decision": "allowed",
                "free_pages": [1, 3, 4],
                "boundary_pages": [3, 4],
                "source_pages": [4],
                "destination_pages": [1],
            }
        )
        preview = _visual_frame(state, 1, 3)
        self.assertEqual(
            preview["l1"]["latest_plans"][0]["invalid_preview_sources"], [4]
        )
        self.assertFalse(preview["l1"]["latest_plans"][0]["urgent"])

        state.apply(
            {
                "seq": 3,
                "event": "l1_compaction_completed",
                "pool": "mamba",
                "urgent": False,
                "source_pages": [2],
                "destination_pages": [1],
                "moves": 1,
                "watermark_before": 5,
                "watermark_after": 3,
            }
        )
        completed = _visual_frame(state, 2, 3)
        allocator = completed["l1"]["allocators"][0]
        self.assertEqual(allocator["watermark_page"], 3)
        self.assertEqual(allocator["end_pct"], 30)
        self.assertEqual(state.allocators["mamba"]["byte_high_frontier"], 30)
        self.assertEqual(
            completed["l1"]["exact_holes"],
            [{"pool": "mamba", "pages": [2]}],
        )
        self.assertEqual(
            completed["l1"]["latest_plans"][0]["invalid_preview_sources"], []
        )

    def test_large_contiguous_row_sets_are_compacted_for_html(self):
        holes = list(range(1, 301))
        sources = list(range(500, 300, -1))
        destinations = list(range(1, 201))
        state = ReplayState(
            allocators={
                "mamba": {
                    "grow_direction": "up",
                    "total_bytes": 10_000,
                    "entry_bytes_per_page": 10,
                    "min_page_index": 1,
                    "watermark_page": 600,
                    "byte_low_frontier": 10,
                    "byte_high_frontier": 6_000,
                    "live_pages": 299,
                    "free_hole_pages": len(holes),
                    "pending_reuse_pages": 0,
                }
            },
            known_holes={"mamba": holes},
            last_compaction={
                "mamba": {
                    "seq": 2,
                    "event": "l1_compaction_decision",
                    "pool": "mamba",
                    "decision": "allowed",
                    "free_pages": holes,
                    "source_pages": sources,
                    "destination_pages": destinations,
                }
            },
            last_event={
                "seq": 2,
                "event": "l1_compaction_decision",
                "pool": "mamba",
                "source_pages": sources,
                "destination_pages": destinations,
            },
        )

        frame = _visual_frame(state, 0, 1)
        hole_markers = [
            marker for marker in frame["l1"]["markers"] if marker["role"] == "H"
        ]
        self.assertEqual(len(hole_markers), 1)
        self.assertEqual(
            (hole_markers[0]["page"], hole_markers[0]["page_end"]),
            (1, 300),
        )
        exact = frame["l1"]["exact_holes"][0]
        self.assertEqual(exact["total"], 300)
        self.assertEqual(len(exact["pages"]), 128)
        plan = frame["l1"]["latest_plans"][0]
        self.assertEqual(plan["holes_total"], 300)
        self.assertEqual(len(plan["holes"]), 128)
        self.assertEqual(plan["moves_total"], 200)
        self.assertEqual(len(plan["moves"]), 128)

    def test_lazy_free_frame_exposes_hole_before_exact_position(self):
        state = ReplayState()
        state.apply(
            {
                "seq": 1,
                "event": "l1_allocator_state",
                "reason": "lazy_free",
                "pool": "mamba",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 5,
                "byte_low_frontier": 10,
                "byte_high_frontier": 50,
                "live_pages": 3,
                "free_hole_pages": 1,
                "holes_before": 0,
                "pending_reuse_pages": 0,
            }
        )
        notice = _visual_frame(state, 0, 1)["l1"]["hole_notice"]
        self.assertEqual(notice["new_holes"], 1)
        self.assertFalse(notice["position_known"])
        self.assertEqual(notice["region_start_pct"], 10)
        self.assertEqual(notice["region_end_pct"], 50)

        state.apply(
            {
                "seq": 2,
                "event": "l1_transfer_fence_registered",
                "pool": "mamba",
                "hazard_id": 1,
            }
        )
        pending = _visual_frame(state, 1, 3)["l1"]
        self.assertEqual(pending["hole_notices"][0]["total_holes"], 1)
        state.apply(
            {
                "seq": 3,
                "event": "l1_allocator_state",
                "reason": "allocation",
                "pool": "mamba",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 5,
                "byte_low_frontier": 10,
                "byte_high_frontier": 50,
                "live_pages": 4,
                "free_hole_pages": 0,
                "holes_before": 1,
                "pending_reuse_pages": 0,
            }
        )
        resolved = _visual_frame(state, 2, 3)["l1"]
        self.assertEqual(resolved["hole_notices"], [])
        self.assertEqual(
            resolved["hole_resolution"],
            {
                "kind": "reused_by_allocation",
                "pool": "mamba",
                "count": 1,
                "pages": [],
            },
        )

    def test_exact_traced_hole_is_marked_until_reused(self):
        state = ReplayState()
        state.apply(
            {
                "seq": 1,
                "event": "l1_allocator_state",
                "reason": "lazy_free",
                "pool": "mamba",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 5,
                "byte_low_frontier": 10,
                "byte_high_frontier": 50,
                "live_pages": 3,
                "free_hole_pages": 1,
                "holes_before": 0,
                "pending_reuse_pages": 0,
                "freed_physical_pages": [2],
            }
        )
        traced = _visual_frame(state, 0, 2)["l1"]
        self.assertEqual(traced["hole_notices"], [])
        self.assertEqual(traced["exact_holes"], [{"pool": "mamba", "pages": [2]}])
        hole_marker = next(
            marker
            for marker in traced["markers"]
            if marker["role"] == "H" and marker["page"] == 2
        )
        self.assertEqual(hole_marker["pct"], 20)
        self.assertEqual(hole_marker["width_pct"], 10)
        state.apply(
            {
                "seq": 2,
                "event": "l1_allocator_state",
                "reason": "allocation",
                "pool": "mamba",
                "grow_direction": "up",
                "total_bytes": 100,
                "entry_bytes_per_page": 10,
                "min_page_index": 1,
                "watermark_page": 5,
                "byte_low_frontier": 10,
                "byte_high_frontier": 50,
                "live_pages": 4,
                "free_hole_pages": 0,
                "holes_before": 1,
                "pending_reuse_pages": 0,
            }
        )
        reused = _visual_frame(state, 1, 2)["l1"]
        self.assertEqual(reused["exact_holes"], [])
        self.assertEqual(reused["hole_resolution"]["pages"], [2])

    def test_watermark_pressure_renders_blocked_attempt(self):
        state = ReplayState(
            allocators={
                "full": {
                    "grow_direction": "down",
                    "total_bytes": 100,
                    "entry_bytes_per_page": 1,
                    "min_page_index": 5,
                    "watermark_page": 79,
                    "byte_low_frontier": 80,
                    "byte_high_frontier": 100,
                    "live_pages": 20,
                    "free_hole_pages": 0,
                    "pending_reuse_pages": 0,
                },
                "mamba": {
                    "grow_direction": "up",
                    "total_bytes": 100,
                    "entry_bytes_per_page": 5,
                    "min_page_index": 1,
                    "watermark_page": 12,
                    "byte_low_frontier": 5,
                    "byte_high_frontier": 60,
                    "live_pages": 11,
                    "free_hole_pages": 0,
                    "pending_reuse_pages": 0,
                },
            }
        )
        state.apply(
            {
                "seq": 1,
                "event": "l1_watermark_pressure",
                "pool": "mamba",
                "reason": "peer_frontier_collision_prevented",
                "requested_pages": 5,
                "attempted_watermark_page": 17,
                "attempted_frontier_bytes": 85,
                "peer_pool": "full",
                "peer_frontier_bytes": 80,
                "would_cross_bytes": 5,
                "current_gap_bytes": 20,
            }
        )
        pressure = _visual_frame(state, 0, 1)["l1"]["pressure"]
        self.assertEqual(pressure["attempted_pct"], 85)
        self.assertEqual(pressure["would_cross_bytes"], 5)

    def test_html_player_is_standalone(self):
        events = [{"seq": 1, "event": "trace_started", "pid": 1}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.html"
            write_html(events, output)
            document = output.read_text(encoding="utf-8")
            self.assertIn("HiCache 동작 재생", document)
            self.assertIn('type="range"', document)
            self.assertIn('id="arena-root"', document)
            self.assertNotIn("화면은 두 겹입니다", document)
            self.assertNotIn("H ? ×", document)
            self.assertNotIn("노란 사선 영역", document)
            self.assertNotIn("H 실제 폭의 빈 공간", document)
            self.assertNotIn("H 빈 row", document)
            self.assertNotIn("D/H", document)
            self.assertIn("filter(role=>role!=='H')", document)
            self.assertIn('class="chunk-fill', document)
            self.assertIn("KV는 token 사용률만큼", document)
            self.assertIn("Mamba LRU 퇴출 위험도", document)
            self.assertIn("100이 다음 eviction 후보", document)
            self.assertIn("L1 LOCK · 백업본", document)
            self.assertIn("L2 전용 · 순위 갱신 중", document)
            self.assertIn("LOCK은 lock_ref", document)
            self.assertIn("원본 event/debug 텍스트", document)

    def test_compact_html_keeps_lifecycle_and_final_state(self):
        allocator = {
            "event": "l1_allocator_state",
            "pool": "mamba",
            "reason": "allocation",
            "grow_direction": "up",
            "total_bytes": 100,
            "entry_bytes_per_page": 10,
            "min_page_index": 1,
            "watermark_page": 2,
            "byte_low_frontier": 10,
            "byte_high_frontier": 20,
            "live_pages": 1,
            "free_hole_pages": 0,
            "pending_reuse_pages": 0,
        }
        events = [
            {"seq": 1, **allocator},
            {"seq": 2, **allocator},
            {"seq": 3, "event": "d2h_transfer_queued", "transfer_id": "x"},
            {"seq": 4, **allocator},
        ]
        self.assertEqual(
            _select_frame_indices(events, compact_repeated_allocator_states=True),
            [0, 2, 3],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "compact.html"
            write_html(
                events,
                output,
                compact_repeated_allocator_states=True,
            )
            document = output.read_text(encoding="utf-8")
            self.assertIn("중간 반복 event", document)
            self.assertIn('"source_event_index": 3', document)

    def test_row_fence_html_summarizes_intersection_and_jumps_to_decision(self):
        events = [
            {
                "seq": 1,
                "event": "l1_transfer_fence_registered",
                "hazard_id": 7,
                "pool": "full",
            },
            {
                "seq": 2,
                "event": "l1_transfer_fence_checked",
                "hazard_id": 7,
                "pool": "full",
                "protected_pages": [8],
                "touched_pages": [3, 8],
                "intersection_pages": [8],
                "blocking": True,
            },
            {
                "seq": 3,
                "event": "l1_compaction_decision",
                "pool": "full",
                "decision": "deferred_row_fence",
                "source_pages": [8],
                "destination_pages": [3],
            },
        ]
        cases = _row_fence_cases(events)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["step"], 2)
        self.assertEqual(cases[0]["intersection_pages"], [8])
        self.assertTrue(cases[0]["blocking"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "row-fence.html"
            write_html(events, output)
            document = output.read_text(encoding="utf-8")
            self.assertIn("Row fence가 판단하는 한 가지 질문", document)
            self.assertIn("서로 독립된 테스트", document)
            self.assertIn('"intersection_pages": [8]', document)

    def test_replay_accepts_stable_string_transfer_ids(self):
        events = [
            {
                "seq": 1,
                "event": "h2d_transfer_queued",
                "transfer_id": "7:1",
                "node_ids": [42],
            },
            {"seq": 2, "event": "h2d_transfer_enqueued", "transfer_id": "7:1"},
            {
                "seq": 3,
                "event": "h2d_transfer_completed",
                "transfer_id": "7:1",
                "node_ids": [42],
            },
        ]
        _, errors = coverage_report(events)
        self.assertEqual(errors, [])
        state = ReplayState()
        for event in events:
            state.apply(event)
        self.assertEqual(state.nodes[42]["FULL"], "l1_resident")
        self.assertEqual(state.nodes[42]["MAMBA"], "l1_resident")

    def test_backup_node_state_distinguishes_request_from_copy(self):
        events = [
            {
                "seq": 1,
                "event": "l2_backup_requested",
                "node_id": 18,
                "full_tokens": 64,
                "component_transfers": {"MAMBA": ["MAMBA"]},
            },
            {
                "seq": 2,
                "event": "d2h_transfer_queued",
                "transfer_id": "7:2",
                "node_ids": [18],
            },
            {
                "seq": 3,
                "event": "d2h_transfer_enqueued",
                "transfer_id": "7:2",
                "node_ids": [18],
            },
            {
                "seq": 4,
                "event": "d2h_transfer_completed",
                "transfer_id": "7:2",
                "node_ids": [18],
            },
            {"seq": 5, "event": "l2_backup_ready", "node_ids": [18]},
        ]
        expected_states = ["requested", "queued", "copying", "copied", "ready"]
        state = ReplayState()
        for event, expected in zip(events, expected_states):
            state.apply(event)
            self.assertEqual(state.nodes[18]["l2"], expected)


if __name__ == "__main__":
    unittest.main()
