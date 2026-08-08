"""Replay and validate JSONL produced by :mod:`hicache_trace`.

This module has no SGLang runtime dependencies, so a trace collected on a GPU
cluster can be copied to a laptop and rendered as text or a standalone HTML
step player.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ReplayState:
    allocators: dict[str, dict[str, Any]] = field(default_factory=dict)
    l2_config: dict[str, Any] = field(default_factory=dict)
    chunks: dict[int, dict[str, Any]] = field(default_factory=dict)
    transfers: dict[int | str, dict[str, Any]] = field(default_factory=dict)
    fences: dict[int, dict[str, Any]] = field(default_factory=dict)
    nodes: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_compaction: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved_holes: dict[str, int] = field(default_factory=dict)
    known_holes: dict[str, list[int]] = field(default_factory=dict)
    mamba_lru: dict[str, Any] = field(default_factory=dict)
    mamba_chunk_nodes: dict[int, int] = field(default_factory=dict)
    last_hole_resolution: dict[str, Any] | None = None
    last_event: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def _node(self, node_id: int) -> dict[str, Any]:
        return self.nodes.setdefault(
            node_id,
            {
                "FULL": "unknown",
                "MAMBA": "unknown",
                "l2": "none",
                "_last_seq": -1,
            },
        )

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("event", "unknown")
        self.counts[kind] += 1
        self.last_event = event
        self.last_hole_resolution = None

        if kind in ("l1_allocator_initialized", "l1_allocator_state"):
            pool = event["pool"]
            if kind == "l1_allocator_initialized":
                # Multiple isolated allocator tests may append to one trace file.
                # A fresh allocator instance must not inherit any transient row
                # markers or compaction plans from a previous test instance.
                self.fences.clear()
                self.last_compaction.clear()
                self.unresolved_holes.clear()
                self.known_holes.clear()
                self.mamba_lru.clear()
            unresolved_before = self.unresolved_holes.get(pool, 0)
            known_before = list(self.known_holes.get(pool, []))
            tracked_holes_before = unresolved_before + len(known_before)
            self.allocators[pool] = dict(event)
            if kind == "l1_allocator_state" and event.get("reason") == "lazy_free":
                exact_pages = event.get("freed_physical_pages")
                if exact_pages is None:
                    self.unresolved_holes[pool] = int(event.get("free_hole_pages", 0))
                    self.known_holes.pop(pool, None)
                else:
                    merged_pages = sorted(
                        set(known_before).union(int(page) for page in exact_pages)
                    )
                    self.known_holes[pool] = merged_pages
                    unresolved_count = max(
                        0, int(event.get("free_hole_pages", 0)) - len(merged_pages)
                    )
                    if unresolved_count:
                        self.unresolved_holes[pool] = unresolved_count
                    else:
                        self.unresolved_holes.pop(pool, None)
                    self.last_hole_resolution = {
                        "kind": "recorded_on_free",
                        "pool": pool,
                        "count": len(exact_pages),
                        "pages": [int(page) for page in exact_pages],
                    }
            elif (
                kind == "l1_allocator_state"
                and event.get("reason") == "allocation"
                and tracked_holes_before > int(event.get("free_hole_pages", 0))
            ):
                unresolved_after = int(event.get("free_hole_pages", 0))
                self.last_hole_resolution = {
                    "kind": "reused_by_allocation",
                    "pool": pool,
                    "count": tracked_holes_before - unresolved_after,
                    "pages": known_before if unresolved_after == 0 else [],
                }
                if unresolved_after:
                    self.unresolved_holes[pool] = unresolved_after
                    self.known_holes.pop(pool, None)
                else:
                    self.unresolved_holes.pop(pool, None)
                    self.known_holes.pop(pool, None)
            elif int(event.get("free_hole_pages", 0)) == 0:
                self.unresolved_holes.pop(pool, None)
                self.known_holes.pop(pool, None)
        elif kind == "mamba_lru_state":
            self.mamba_lru = {
                "reason": event.get("reason"),
                "device": [dict(entry) for entry in event.get("device") or []],
                "host": [dict(entry) for entry in event.get("host") or []],
                "seq": event.get("seq"),
            }
        elif kind == "l2_arena_initialized":
            self.l2_config = dict(event)
            self.mamba_chunk_nodes.clear()
            for chunk_id in range(int(event["num_chunks"])):
                self.chunks.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "owner": "FREE",
                        "kv_used_offsets": [],
                        "pin_count": 0,
                        "pending_release": False,
                    },
                )
        elif kind == "l2_chunk_owner_changed":
            chunk_id = int(event["chunk_id"])
            chunk = self.chunks.setdefault(chunk_id, {"chunk_id": chunk_id})
            previous = chunk.get("owner", "FREE")
            expected = event.get("previous_owner")
            if expected is not None and previous != expected:
                self.errors.append(
                    f"seq {event['seq']}: chunk {chunk_id} owner was {previous}, "
                    f"trace expected {expected}"
                )
            chunk["owner"] = event["owner"]
            if event["owner"] == "FREE":
                chunk["kv_used_offsets"] = []
                self.mamba_chunk_nodes.pop(chunk_id, None)
        elif kind.startswith("l2_") and event.get("chunks"):
            for update in event["chunks"]:
                self.chunks[int(update["chunk_id"])] = dict(update)
        elif kind.endswith("transfer_queued"):
            transfer_id = event["transfer_id"]
            if transfer_id in self.transfers:
                self.errors.append(
                    f"seq {event['seq']}: duplicate queued transfer {transfer_id}"
                )
            self.transfers[transfer_id] = {
                **event,
                "direction": kind.split("_", 1)[0].upper(),
                "status": "queued",
            }
            if kind == "d2h_transfer_queued":
                for node_id in event.get("node_ids", []):
                    node = self._node(int(node_id))
                    node["l2"] = "queued"
                    node["_last_seq"] = event["seq"]
        elif kind.endswith("transfer_enqueued"):
            transfer_id = event["transfer_id"]
            transfer = self.transfers.get(transfer_id)
            if transfer is None:
                self.errors.append(
                    f"seq {event['seq']}: transfer {transfer_id} enqueued "
                    "without a queued event"
                )
                transfer = self.transfers.setdefault(transfer_id, {})
            elif transfer.get("status") != "queued":
                self.errors.append(
                    f"seq {event['seq']}: transfer {transfer_id} enqueued from "
                    f"state {transfer.get('status')}"
                )
            transfer.update({**event, "status": "in_flight"})
            if kind == "d2h_transfer_enqueued":
                for node_id in event.get("node_ids", []):
                    node = self._node(int(node_id))
                    node["l2"] = "copying"
                    node["_last_seq"] = event["seq"]
        elif kind.endswith("transfer_completed"):
            transfer_id = event["transfer_id"]
            transfer = self.transfers.get(transfer_id)
            if transfer is None:
                self.errors.append(
                    f"seq {event['seq']}: transfer {transfer_id} completed "
                    "without a queued event"
                )
                transfer = self.transfers.setdefault(transfer_id, {})
            elif transfer.get("status") != "in_flight":
                self.errors.append(
                    f"seq {event['seq']}: transfer {transfer_id} completed from "
                    f"state {transfer.get('status')}"
                )
            transfer.update({**event, "status": "complete"})
            if kind == "h2d_transfer_completed":
                for node_id in event.get("node_ids", []):
                    node = self._node(int(node_id))
                    node["FULL"] = "l1_resident"
                    node["MAMBA"] = "l1_resident"
                    node["_last_seq"] = event["seq"]
            elif kind == "d2h_transfer_completed":
                for node_id in event.get("node_ids", []):
                    node = self._node(int(node_id))
                    node["l2"] = "copied"
                    node["_last_seq"] = event["seq"]
        elif kind == "l1_transfer_fence_registered":
            self.fences[int(event["hazard_id"])] = {
                **event,
                "status": "active",
                "protected_pages": [],
            }
        elif kind == "l1_transfer_fence_checked":
            hazard_id = int(event["hazard_id"])
            fence = self.fences.setdefault(hazard_id, {})
            fence.update(event)
            intersection = set(event.get("intersection_pages") or [])
            if bool(intersection) != bool(event.get("blocking")):
                self.errors.append(
                    f"seq {event['seq']}: fence {hazard_id} blocking flag does "
                    "not match protected/touched intersection"
                )
        elif kind == "l1_transfer_fence_released":
            hazard_id = int(event["hazard_id"])
            self.fences.setdefault(hazard_id, {}).update(
                {**event, "status": "released"}
            )
        elif kind in ("l1_compaction_decision", "l1_compaction_completed"):
            pool = event["pool"]
            previous = self.last_compaction.setdefault(pool, {})
            previous.update(event)
            if kind == "l1_compaction_decision":
                unresolved_before = self.unresolved_holes.pop(pool, 0)
                self.known_holes[pool] = [
                    int(page) for page in event.get("free_pages") or []
                ]
                if unresolved_before:
                    self.last_hole_resolution = {
                        "kind": "materialized_by_compaction",
                        "pool": pool,
                        "count": unresolved_before,
                        "pages": list(self.known_holes[pool]),
                    }
                checks = [
                    fence
                    for fence in self.fences.values()
                    if fence.get("pool") == pool
                    and fence.get("status") == "active"
                    and fence.get("blocking")
                ]
                if event.get("decision") == "deferred_row_fence" and not checks:
                    self.errors.append(
                        f"seq {event['seq']}: row-fence defer has no blocking fence"
                    )
            else:
                if event.get("urgent"):
                    # An urgent pass fully packs the band and reclaims every
                    # remaining hole beyond the new frontier.
                    self.known_holes.pop(pool, None)
                else:
                    # Boundary holes were absorbed before the move,
                    # destinations are occupied now, and each moved source is
                    # the new exact hole until the next maintenance pass
                    # absorbs or reuses it.
                    remaining = set(self.known_holes.get(pool, []))
                    remaining.difference_update(
                        int(page) for page in previous.get("boundary_pages") or []
                    )
                    remaining.difference_update(
                        int(page) for page in event.get("destination_pages") or []
                    )
                    remaining.update(
                        int(page) for page in event.get("source_pages") or []
                    )
                    if remaining:
                        self.known_holes[pool] = sorted(remaining)
                    else:
                        self.known_holes.pop(pool, None)
                self.unresolved_holes.pop(pool, None)
                # The detailed completion event is emitted immediately before
                # the matching l1_allocator_state snapshot. Apply its new
                # frontier now so the completion frame does not briefly paint
                # reclaimed boundary rows as live memory.
                allocator = self.allocators.get(pool)
                watermark_after = event.get("watermark_after")
                if allocator is not None and watermark_after is not None:
                    watermark_after = int(watermark_after)
                    allocator["watermark_page"] = watermark_after
                    frontier_bytes = watermark_after * int(
                        allocator["entry_bytes_per_page"]
                    )
                    frontier_key = (
                        "byte_high_frontier"
                        if allocator["grow_direction"] == "up"
                        else "byte_low_frontier"
                    )
                    allocator[frontier_key] = frontier_bytes
                    allocator["free_hole_pages"] = len(self.known_holes.get(pool, []))
        elif kind == "l2_backup_requested":
            node = self._node(int(event["node_id"]))
            if int(event.get("full_tokens", 0)):
                node["FULL"] = "l1_resident"
            for component in event.get("component_transfers") or {}:
                node[component] = "l1_resident"
            node["l2"] = "requested"
            node["_last_seq"] = event["seq"]
        elif kind == "l2_backup_metadata_committed":
            node = self._node(int(event["node_id"]))
            for chunk_id in event.get("mamba_host_chunks") or []:
                self.mamba_chunk_nodes[int(chunk_id)] = int(event["node_id"])
            if node["l2"] in ("none", "requested"):
                node["l2"] = "metadata_committed"
            node["_last_seq"] = event["seq"]
        elif kind == "l2_backup_ready":
            for node_id in event.get("node_ids", []):
                node = self._node(int(node_id))
                node["l2"] = "ready"
                node["_last_seq"] = event["seq"]
        elif kind == "l1_node_evicted":
            node = self._node(int(event["node_id"]))
            evicted = event.get("evicted_by_component") or {
                event.get("component", "FULL"): 1
            }
            for component, count in evicted.items():
                if count:
                    node[component] = "l2_only"
            node["_last_seq"] = event["seq"]
        elif kind == "loadback_metadata_committed":
            node = self._node(int(event["node_id"]))
            node["FULL"] = "loading"
            node["MAMBA"] = "loading"
            node["_last_seq"] = event["seq"]


def load_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return sorted(events, key=lambda item: (item.get("wall_time_ns", 0), item["seq"]))


def _byte_bar(state: ReplayState, width: int = 72) -> str:
    if not state.allocators:
        return "(L1 allocator has not been initialized)"
    total = max(int(item["total_bytes"]) for item in state.allocators.values())
    cells = ["·"] * width

    def cell(byte_offset: int) -> int:
        return min(width, max(0, int(byte_offset * width / max(1, total))))

    reserved_bytes = max(
        int(item["min_page_index"]) * int(item["entry_bytes_per_page"])
        for item in state.allocators.values()
    )
    for index in range(cell(reserved_bytes)):
        cells[index] = "R"

    # Pool identity decides the glyph; grow_direction only decides which byte
    # frontier moves. Production hybrid-Mamba uses Mamba at the low end growing
    # up and full-attention KV at the high end growing down, but replay should
    # also render test configurations that deliberately reverse those sides.
    for pool, allocator in state.allocators.items():
        glyph = "M" if "mamba" in pool.lower() else "K"
        start = cell(int(allocator["byte_low_frontier"]))
        stop = cell(int(allocator["byte_high_frontier"]))
        for index in range(start, stop):
            cells[index] = glyph

    return "|" + "".join(cells) + "|"


def _chunk_bar(state: ReplayState, limit: int = 32) -> str:
    if not state.chunks:
        return "(L2 typed-chunk arena has not been initialized)"
    capacity = int(state.l2_config.get("kv_pages_per_chunk", 1))
    parts = []
    for chunk_id in sorted(state.chunks)[:limit]:
        chunk = state.chunks[chunk_id]
        owner = chunk.get("owner", "FREE")
        if owner == "KV":
            label = f"K{len(chunk.get('kv_used_offsets', []))}/{capacity}"
        elif owner == "MAMBA":
            label = "M"
        else:
            label = "."
        if chunk.get("pending_release"):
            label += "~"
        elif int(chunk.get("pin_count", 0)):
            label += "!"
        parts.append(f"{chunk_id}:{label}")
    if len(state.chunks) > limit:
        parts.append(f"…+{len(state.chunks) - limit}")
    return "[" + "][".join(parts) + "]"


def _visual_frame(state: ReplayState, step: int, total_steps: int) -> dict[str, Any]:
    """Build browser state with L1 ownership and row markers as separate layers."""
    event = state.last_event
    total_bytes = max(
        (int(item["total_bytes"]) for item in state.allocators.values()), default=0
    )

    def percentage(byte_offset: int) -> float:
        if not total_bytes:
            return 0.0
        return max(0.0, min(100.0, byte_offset * 100.0 / total_bytes))

    allocators = []
    for pool, allocator in sorted(state.allocators.items()):
        start_pct = percentage(int(allocator["byte_low_frontier"]))
        end_pct = percentage(int(allocator["byte_high_frontier"]))
        allocators.append(
            {
                "pool": pool,
                "kind": "mamba" if "mamba" in pool.lower() else "kv",
                "start_pct": start_pct,
                "end_pct": end_pct,
                "grow_direction": allocator["grow_direction"],
                "watermark_pct": (
                    end_pct if allocator["grow_direction"] == "up" else start_pct
                ),
                "entry_bytes_per_page": int(allocator["entry_bytes_per_page"]),
                "watermark_page": int(allocator["watermark_page"]),
                "live_pages": int(allocator["live_pages"]),
                "hole_count": int(allocator["free_hole_pages"]),
                "pending_reuse_pages": int(allocator["pending_reuse_pages"]),
            }
        )

    reserved_bytes = max(
        (
            int(item["min_page_index"]) * int(item["entry_bytes_per_page"])
            for item in state.allocators.values()
        ),
        default=0,
    )
    intervals = [(0, reserved_bytes)] + [
        (int(item["byte_low_frontier"]), int(item["byte_high_frontier"]))
        for item in state.allocators.values()
    ]
    occupied_bytes = 0
    previous_end = 0
    for start, end in sorted(
        interval for interval in intervals if interval[1] > interval[0]
    ):
        if end <= previous_end:
            continue
        occupied_bytes += end - max(start, previous_end)
        previous_end = end

    grow_up = [
        item for item in state.allocators.values() if item["grow_direction"] == "up"
    ]
    grow_down = [
        item for item in state.allocators.values() if item["grow_direction"] == "down"
    ]
    frontier_gap_bytes = None
    if grow_up and grow_down:
        low_side_frontier = max(int(item["byte_high_frontier"]) for item in grow_up)
        high_side_frontier = min(int(item["byte_low_frontier"]) for item in grow_down)
        frontier_gap_bytes = max(0, high_side_frontier - low_side_frontier)
    for allocator in allocators:
        if frontier_gap_bytes is None:
            allocator["gap_capacity_pages"] = None
            allocator["next_page_fits"] = None
            allocator["next_watermark_pct"] = None
            continue
        page_bytes = allocator["entry_bytes_per_page"]
        allocator["gap_capacity_pages"] = frontier_gap_bytes // page_bytes
        allocator["next_page_fits"] = frontier_gap_bytes >= page_bytes
        direction = 1 if allocator["grow_direction"] == "up" else -1
        allocator["next_watermark_pct"] = allocator["watermark_pct"] + direction * (
            page_bytes * 100.0 / max(1, total_bytes)
        )
    markers: list[dict[str, Any]] = []

    def add_markers(pool: str | None, role: str, pages: Iterable[Any]) -> None:
        allocator = state.allocators.get(pool or "")
        if allocator is None:
            return
        page_bytes = int(allocator["entry_bytes_per_page"])
        sorted_pages = sorted(set(int(page) for page in pages))
        runs: list[tuple[int, int]] = []
        for page in sorted_pages:
            if runs and page == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], page)
            else:
                runs.append((page, page))
        for page, page_end in runs:
            markers.append(
                {
                    "pool": pool,
                    "role": role,
                    "page": page,
                    "page_end": page_end,
                    "page_count": page_end - page + 1,
                    "pct": percentage(page * page_bytes),
                    "width_pct": percentage((page_end - page + 1) * page_bytes),
                }
            )

    # Active transfer fences remain visible across frames. The F marker is an
    # overlay, never a replacement for the underlying KV/Mamba color.
    for fence in state.fences.values():
        if fence.get("status") == "active":
            add_markers(fence.get("pool"), "F", fence.get("protected_pages") or [])

    for pool, pages in state.known_holes.items():
        add_markers(pool, "H", pages)

    mamba_lru_spans = []
    mamba_allocator = next(
        (
            allocator
            for pool, allocator in state.allocators.items()
            if "mamba" in pool.lower()
        ),
        None,
    )
    if mamba_allocator is not None:
        page_bytes = int(mamba_allocator["entry_bytes_per_page"])
        for entry in state.mamba_lru.get("device", []):
            if entry.get("index_space") not in ("physical", "identity"):
                # Older traces recorded virtual slots without the v2p mapping;
                # showing those as physical rows would place labels in gaps.
                continue
            for page in sorted(set(int(index) for index in entry.get("indices", []))):
                mamba_lru_spans.append(
                    {
                        "node_id": int(entry["node_id"]),
                        "page": page,
                        "pct": percentage(page * page_bytes),
                        "width_pct": percentage(page_bytes),
                        "eviction_rank": entry.get("eviction_rank"),
                        "eviction_score": int(entry.get("eviction_score", 0)),
                        "protected": bool(entry.get("protected")),
                        "lock_ref": int(entry.get("lock_ref", 0)),
                    }
                )

    if event.get("event") in (
        "l1_compaction_decision",
        "l1_compaction_completed",
    ):
        pool = event.get("pool")
        add_markers(pool, "D", event.get("destination_pages") or [])
        add_markers(pool, "S", event.get("source_pages") or [])

    latest_plans = []
    for pool, compaction in sorted(state.last_compaction.items()):
        holes_all = [int(page) for page in compaction.get("free_pages") or []]
        sources_all = [int(page) for page in compaction.get("source_pages") or []]
        destinations_all = [
            int(page) for page in compaction.get("destination_pages") or []
        ]

        def sample(values: list[int], limit: int = 128) -> list[int]:
            if len(values) <= limit:
                return values
            half = limit // 2
            return values[:half] + values[-half:]

        holes = sample(holes_all)
        move_pairs_all = list(zip(sources_all, destinations_all))
        move_pairs = sample(move_pairs_all)
        sources = [source for source, _ in move_pairs]
        destinations = [destination for _, destination in move_pairs]
        invalid_preview_sources = (
            sorted(set(sources_all).intersection(state.known_holes.get(pool, [])))
            if compaction.get("event") == "l1_compaction_decision"
            else []
        )
        latest_plans.append(
            {
                "pool": pool,
                "decision": compaction.get("decision", "completed"),
                "urgent": bool(compaction.get("urgent")),
                "holes": holes,
                "holes_total": len(holes_all),
                "sources": sources,
                "sources_total": len(sources_all),
                "destinations": destinations,
                "destinations_total": len(destinations_all),
                "invalid_preview_sources": invalid_preview_sources,
                "moves": [
                    {"source": source, "destination": destination}
                    for source, destination in move_pairs
                ],
                "moves_total": len(move_pairs_all),
                "seq": int(compaction.get("seq", -1)),
            }
        )

    lazy_free = (
        event.get("event") == "l1_allocator_state"
        and event.get("reason") == "lazy_free"
    )
    hole_notices = []
    for pool, count in sorted(state.unresolved_holes.items()):
        allocator = next(item for item in allocators if item["pool"] == pool)
        hole_notices.append(
            {
                "pool": pool,
                "new_holes": (
                    int(event.get("free_hole_pages", 0))
                    - int(event.get("holes_before", 0))
                    if lazy_free and event.get("pool") == pool
                    else 0
                ),
                "total_holes": count,
                "position_known": False,
                "region_start_pct": allocator["start_pct"],
                "region_end_pct": allocator["end_pct"],
                "message": (
                    "A hole exists somewhere inside this pool's highlighted "
                    "region. Compaction can resolve it to an exact H page marker, "
                    "or a new allocation can reuse it first."
                ),
            }
        )
    hole_notice = next(
        (
            notice
            for notice in hole_notices
            if lazy_free and notice["pool"] == event.get("pool")
        ),
        hole_notices[0] if hole_notices else None,
    )

    host_mamba_lru = {
        int(index): entry
        for entry in state.mamba_lru.get("host", [])
        for index in entry.get("indices", [])
    }
    device_mamba_lru = {
        int(entry["node_id"]): entry for entry in state.mamba_lru.get("device", [])
    }
    chunks = []
    capacity = int(state.l2_config.get("kv_pages_per_chunk", 1))
    for chunk_id, chunk in sorted(state.chunks.items()):
        owner = chunk.get("owner", "FREE")
        pin_count = int(chunk.get("pin_count", 0))
        mamba_node_id = state.mamba_chunk_nodes.get(chunk_id)
        mamba_lru = host_mamba_lru.get(chunk_id) if owner == "MAMBA" else None
        mamba_status = None
        if owner == "MAMBA" and mamba_lru is None:
            device_entry = device_mamba_lru.get(mamba_node_id)
            node_state = state.nodes.get(mamba_node_id, {})
            if pin_count:
                mamba_status = "copying"
            elif device_entry is not None and device_entry.get("protected"):
                mamba_status = "l1_locked_backup"
            elif device_entry is not None or node_state.get("MAMBA") == "l1_resident":
                mamba_status = "l1_backup"
            elif node_state.get("MAMBA") == "l2_only":
                mamba_status = "l2_lru_pending"
            elif mamba_node_id is not None:
                mamba_status = "backup_not_ranked"
            else:
                mamba_status = "node_unrecorded"
        kv_used = len(chunk.get("kv_used_offsets", []))
        fill_pct = (
            min(100.0, kv_used * 100.0 / max(1, capacity))
            if owner == "KV"
            else 100.0 if owner == "MAMBA" else 0.0
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "owner": owner,
                "kv_used": kv_used,
                "kv_capacity": capacity,
                "fill_pct": fill_pct,
                "mamba_node_id": mamba_node_id,
                "mamba_status": "host_lru" if mamba_lru is not None else mamba_status,
                "mamba_lru": (
                    {
                        "node_id": int(mamba_lru["node_id"]),
                        "eviction_rank": mamba_lru.get("eviction_rank"),
                        "eviction_score": int(mamba_lru.get("eviction_score", 0)),
                        "protected": bool(mamba_lru.get("protected")),
                        "lock_ref": int(mamba_lru.get("lock_ref", 0)),
                    }
                    if mamba_lru is not None
                    else None
                ),
                "pin_count": pin_count,
                "pending_release": bool(chunk.get("pending_release")),
            }
        )

    recent_transfers = sorted(
        state.transfers.values(), key=lambda item: int(item.get("seq", -1))
    )[-8:]
    transfers = [
        {
            "id": transfer.get("transfer_id"),
            "direction": transfer.get("direction", "?"),
            "status": transfer.get("status", "?"),
            "node_ids": transfer.get("node_ids", []),
        }
        for transfer in recent_transfers
    ]
    nodes = [
        {
            "id": node_id,
            "full": node.get("FULL", "unknown"),
            "mamba": node.get("MAMBA", "unknown"),
            "l2": node.get("l2", "none"),
        }
        for node_id, node in sorted(
            state.nodes.items(), key=lambda item: item[1].get("_last_seq", -1)
        )[-8:]
    ]
    pressure = None
    if event.get("event") == "l1_watermark_pressure":
        pressure = {
            "pool": event.get("pool"),
            "reason": event.get("reason"),
            "requested_pages": int(event.get("requested_pages", 0)),
            "attempted_watermark_page": event.get("attempted_watermark_page"),
            "attempted_pct": percentage(int(event.get("attempted_frontier_bytes", 0))),
            "peer_pool": event.get("peer_pool"),
            "peer_pct": percentage(int(event.get("peer_frontier_bytes", 0))),
            "would_cross_bytes": int(event.get("would_cross_bytes", 0)),
            "current_gap_bytes": int(event.get("current_gap_bytes", 0)),
        }
    return {
        "step": step + 1,
        "total_steps": total_steps,
        "event": event.get("event", "unknown"),
        "seq": event.get("seq"),
        "pid": event.get("pid"),
        "eviction": (
            {
                "node_id": event.get("node_id"),
                "evicted_by_component": event.get("evicted_by_component", {}),
                "backup_required": event.get("backup_required"),
            }
            if event.get("event") == "l1_node_evicted"
            else None
        ),
        "l1": {
            "total_bytes": total_bytes,
            "reserved_pct": percentage(reserved_bytes),
            "occupied_bytes": occupied_bytes,
            "occupied_pct": percentage(occupied_bytes),
            "frontier_gap_bytes": frontier_gap_bytes,
            "frontier_gap_pct": (
                percentage(frontier_gap_bytes)
                if frontier_gap_bytes is not None
                else None
            ),
            "frontiers_full": frontier_gap_bytes == 0,
            "pressure": pressure,
            "allocators": allocators,
            "markers": markers,
            "mamba_lru_spans": mamba_lru_spans,
            "hole_notice": hole_notice,
            "hole_notices": hole_notices,
            "exact_holes": [
                (
                    {"pool": pool, "pages": list(pages)}
                    if len(pages) <= 128
                    else {
                        "pool": pool,
                        "pages": list(pages[:64]) + list(pages[-64:]),
                        "total": len(pages),
                    }
                )
                for pool, pages in sorted(state.known_holes.items())
                if pages
            ],
            "hole_resolution": state.last_hole_resolution,
            "latest_plans": latest_plans,
        },
        "l2": {"chunks": chunks},
        "mamba_lru": {
            "reason": state.mamba_lru.get("reason"),
            "seq": state.mamba_lru.get("seq"),
            "device": [dict(entry) for entry in state.mamba_lru.get("device", [])],
            "host": [dict(entry) for entry in state.mamba_lru.get("host", [])],
        },
        "transfers": transfers,
        "nodes": nodes,
    }


def render_state(state: ReplayState, step: int, total_steps: int) -> str:
    event = state.last_event
    lines = [
        f"HiCache trace — step {step + 1}/{total_steps}",
        f"event: {event.get('event')}  seq={event.get('seq')}  pid={event.get('pid')}",
        "",
        "L1 unified GPU byte arena",
        _byte_bar(state),
        "  low byte (left) → high byte (right)",
        "  base ownership: R=reserved K=KV M=Mamba ·=gap",
        "  row markers (separate layer): S=compaction-src "
        "D=compaction-dst F=transfer-fenced; holes are blank",
    ]
    for pool, allocator in sorted(state.allocators.items()):
        growth = "left→right" if allocator["grow_direction"] == "up" else "right→left"
        lines.append(
            f"  {pool}: grow={allocator['grow_direction']} ({growth}) "
            f"wm={allocator['watermark_page']} live={allocator['live_pages']} "
            f"holes={allocator['free_hole_pages']} "
            f"pending_reuse={allocator['pending_reuse_pages']}"
        )
    visual = _visual_frame(state, step, total_steps)
    lines.extend(["", "Current row markers (holes appear as blank space)"])
    grouped: dict[tuple[str, int], list[str]] = {}
    for marker in visual["l1"]["markers"]:
        if marker["role"] == "H":
            continue
        key = (marker["pool"], marker["page"])
        grouped.setdefault(key, []).append(marker["role"])
    if grouped:
        for (pool, page), roles in sorted(grouped.items()):
            lines.append(f"  {pool} page {page}: {'/'.join(sorted(set(roles)))}")
    elif visual["l1"]["hole_notice"]:
        notice = visual["l1"]["hole_notice"]
        lines.append(
            f"  {notice['pool']}: +{notice['new_holes']} hole(s), "
            f"total={notice['total_holes']}; exact positions become known at "
            "compaction preview unless allocation reuses them first"
        )
    elif visual["l1"]["hole_resolution"]:
        resolution = visual["l1"]["hole_resolution"]
        lines.append(
            f"  {resolution['pool']}: {resolution['count']} blank row(s) reused "
            "by this allocation"
        )
    else:
        lines.append("  (none at this event)")
    lines.extend(["", "L2 shared typed chunks", _chunk_bar(state)])
    lines.extend(["", "Mamba LRU eviction risk (100 = next candidate)"])
    if state.mamba_lru:
        for layer in ("device", "host"):
            entries = state.mamba_lru.get(layer, [])
            lines.append(f"  {layer}:")
            if not entries:
                lines.append("    (empty)")
                continue
            for entry in entries:
                status = (
                    f"protected lock={entry.get('lock_ref', 0)}"
                    if entry.get("protected")
                    else (
                        f"score={entry.get('eviction_score')} "
                        f"rank={entry.get('eviction_rank')}"
                    )
                )
                lines.append(
                    f"    node {entry.get('node_id')}: {status} "
                    f"indices={entry.get('indices', [])}"
                )
    else:
        lines.append("  (not recorded in this trace)")

    active = [
        transfer
        for transfer in state.transfers.values()
        if transfer.get("status") != "complete"
    ]
    lines.extend(["", "Transfers"])
    if active:
        for transfer in active[-6:]:
            lines.append(
                f"  {transfer.get('direction', '?')} #{transfer.get('transfer_id')} "
                f"{transfer.get('status')} nodes={transfer.get('node_ids', [])}"
            )
    else:
        lines.append("  (no in-flight transfers)")

    lines.extend(["", "Recently observed cache nodes (not the full radix tree)"])
    if state.nodes:
        recent_nodes = sorted(
            state.nodes.items(), key=lambda item: item[1].get("_last_seq", -1)
        )[-8:]
        for node_id, node in recent_nodes:
            lines.append(
                f"  node {node_id}: FULL={node.get('FULL', 'unknown')} "
                f"MAMBA={node.get('MAMBA', 'unknown')} "
                f"L2={node.get('l2', 'none')}"
            )
    else:
        lines.append("  (no cache-node lifecycle events)")

    decisions = list(state.last_compaction.values())
    lines.extend(["", "Latest compaction"])
    if decisions:
        for decision in decisions:

            def short_pages(values: Iterable[Any], limit: int = 18) -> str:
                pages = [int(page) for page in values]
                if len(pages) <= limit:
                    return str(pages)
                return f"{pages[:limit]} … +{len(pages) - limit}"

            lines.append(
                f"  {decision.get('pool')}: {decision.get('decision', 'completed')} "
                f"touched={short_pages(decision.get('touched_pages', []))} "
                f"src={short_pages(decision.get('source_pages', []))} "
                f"dst={short_pages(decision.get('destination_pages', []))}"
            )
    else:
        lines.append("  (none)")

    lines.extend(["", "Event payload", json.dumps(event, indent=2, sort_keys=True)])
    if state.errors:
        lines.extend(["", "Invariant errors", *[f"  ✗ {x}" for x in state.errors[-8:]]])
    return "\n".join(lines)


def replay_until(events: list[dict[str, Any]], step: int) -> ReplayState:
    state = ReplayState()
    for event in events[: step + 1]:
        state.apply(event)
    return state


def coverage_report(events: list[dict[str, Any]]) -> tuple[dict[str, bool], list[str]]:
    kinds = Counter(event.get("event") for event in events)
    pressure_state = ReplayState()
    page_pressure_observed = False
    for index, event in enumerate(events):
        pressure_state.apply(event)
        if len(pressure_state.allocators) >= 2 and any(
            allocator["next_page_fits"] is False
            for allocator in _visual_frame(pressure_state, index, len(events))["l1"][
                "allocators"
            ]
        ):
            page_pressure_observed = True
            break
    checks = {
        "l1_bidirectional_allocators": len(
            {
                event.get("grow_direction")
                for event in events
                if event.get("event") == "l1_allocator_initialized"
            }
        )
        >= 2,
        "l2_typed_chunk_arena": bool(kinds["l2_arena_initialized"]),
        "l2_kv_allocation": bool(kinds["l2_kv_allocated"]),
        "l2_mamba_allocation": bool(kinds["l2_mamba_allocated"]),
        "l2_chunk_pinning": bool(kinds["l2_transfer_pin_armed"]),
        "mamba_lru_eviction_risk": bool(kinds["mamba_lru_state"]),
        "d2h_async_lifecycle": bool(kinds["d2h_transfer_queued"])
        and bool(kinds["d2h_transfer_completed"]),
        "l1_eviction": bool(kinds["l1_node_evicted"]),
        "h2d_loadback_lifecycle": bool(kinds["h2d_transfer_queued"])
        and bool(kinds["h2d_transfer_completed"]),
        "watermark_page_pressure": page_pressure_observed
        or any(
            event.get("event") == "l1_watermark_pressure"
            and event.get("reason") == "peer_frontier_collision_prevented"
            for event in events
        ),
        "row_fence_overlap": any(
            event.get("event") == "l1_transfer_fence_checked"
            and bool(event.get("intersection_pages"))
            for event in events
        ),
        "row_fence_disjoint": any(
            event.get("event") == "l1_transfer_fence_checked"
            and not event.get("intersection_pages")
            for event in events
        ),
        "compaction_deferred": any(
            event.get("event") == "l1_compaction_decision"
            and event.get("decision") == "deferred_row_fence"
            for event in events
        ),
        "compaction_completed": bool(kinds["l1_compaction_completed"]),
    }
    state = replay_until(events, len(events) - 1) if events else ReplayState()
    unfinished = [
        transfer_id
        for transfer_id, transfer in state.transfers.items()
        if transfer.get("status") != "complete"
    ]
    errors = list(state.errors)
    if unfinished:
        errors.append(f"unfinished transfers at end of trace: {unfinished}")
    pinned_chunks = [
        chunk_id
        for chunk_id, chunk in state.chunks.items()
        if int(chunk.get("pin_count", 0)) or chunk.get("pending_release")
    ]
    if pinned_chunks:
        errors.append(f"pinned or pending-release L2 chunks at end: {pinned_chunks}")
    armed = Counter(
        (event.get("cuda_event_id"), tuple(event.get("chunk_ids", [])))
        for event in events
        if event.get("event") == "l2_transfer_pin_armed"
    )
    released = Counter(
        (event.get("cuda_event_id"), tuple(event.get("chunk_ids", [])))
        for event in events
        if event.get("event") == "l2_transfer_pin_released"
    )
    if armed != released:
        errors.append(
            "L2 transfer pin lifecycle mismatch: "
            f"armed_only={list((armed - released).elements())}, "
            f"released_only={list((released - armed).elements())}"
        )
    return checks, errors


def _row_fence_cases(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract one concise comparison card per row-fence check."""
    registrations: dict[int, int] = {}
    cases = []
    for index, event in enumerate(events):
        if event.get("event") == "l1_transfer_fence_registered":
            registrations[int(event["hazard_id"])] = index
            continue
        if event.get("event") != "l1_transfer_fence_checked":
            continue

        pool = event.get("pool")
        decision_index = index
        decision: dict[str, Any] = {}
        for candidate_index in range(index + 1, len(events)):
            candidate = events[candidate_index]
            if candidate.get("event") == "l1_transfer_fence_checked":
                break
            if (
                candidate.get("event") == "l1_compaction_decision"
                and candidate.get("pool") == pool
            ):
                decision_index = candidate_index
                decision = candidate
                break

        hazard_id = int(event["hazard_id"])
        cases.append(
            {
                "number": len(cases) + 1,
                "hazard_id": hazard_id,
                "start_step": registrations.get(hazard_id, index),
                "step": decision_index,
                "pool": pool,
                "protected_pages": [
                    int(page) for page in event.get("protected_pages") or []
                ],
                "touched_pages": [
                    int(page) for page in event.get("touched_pages") or []
                ],
                "intersection_pages": [
                    int(page) for page in event.get("intersection_pages") or []
                ],
                "blocking": bool(event.get("blocking")),
                "source_pages": [
                    int(page) for page in decision.get("source_pages") or []
                ],
                "destination_pages": [
                    int(page) for page in decision.get("destination_pages") or []
                ],
                "decision": decision.get("decision"),
            }
        )
    return cases


def _select_frame_indices(
    events: list[dict[str, Any]], *, compact_repeated_allocator_states: bool
) -> list[int]:
    """Choose replay frames while preserving every event in state evolution.

    Busy queued-request scenarios can emit the same Mamba allocation/free state
    hundreds of thousands of times. Compact mode renders the first occurrence
    of each allocator-state signature but still applies every skipped event to
    ``ReplayState``, so the next visible frame is an exact current snapshot.
    All non-allocator lifecycle events remain visible.
    """
    if not compact_repeated_allocator_states:
        return list(range(len(events)))

    selected = []
    seen_allocator_states = set()
    for index, event in enumerate(events):
        if event.get("event") != "l1_allocator_state":
            selected.append(index)
            continue
        signature = (
            event.get("pool"),
            event.get("reason"),
            event.get("watermark_page"),
            event.get("live_pages"),
            event.get("free_hole_pages"),
            event.get("pending_reuse_pages"),
            event.get("requested_pages"),
            event.get("urgent"),
        )
        if signature not in seen_allocator_states:
            seen_allocator_states.add(signature)
            selected.append(index)
    if events and selected[-1] != len(events) - 1:
        selected.append(len(events) - 1)
    return selected


def write_html(
    events: list[dict[str, Any]],
    output: Path,
    *,
    compact_repeated_allocator_states: bool = False,
) -> None:
    state = ReplayState()
    frames = []
    visual_frames = []
    selected_indices = _select_frame_indices(
        events,
        compact_repeated_allocator_states=compact_repeated_allocator_states,
    )
    selected_set = set(selected_indices)
    selected_events = [events[index] for index in selected_indices]
    previous_source_index = -1
    for index, event in enumerate(events):
        state.apply(event)
        if index not in selected_set:
            continue
        display_index = len(frames)
        frames.append(render_state(state, display_index, len(selected_indices)))
        visual = _visual_frame(state, display_index, len(selected_indices))
        visual["source_event_index"] = index
        visual["collapsed_events_before"] = index - previous_source_index - 1
        visual_frames.append(visual)
        previous_source_index = index
    raw_frames_json = json.dumps(frames).replace("</", "<\\/")
    visual_frames_json = json.dumps(visual_frames).replace("</", "<\\/")
    row_fence_cases_json = json.dumps(_row_fence_cases(selected_events)).replace(
        "</", "<\\/"
    )
    document = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiCache 동작 재생</title>
<style>
:root { color-scheme:dark; --bg:#07101f; --card:#101b2d; --line:#2b3b55; --muted:#93a4bd;
  --kv:#2f81f7; --mamba:#a371f7; --reserved:#64748b; --hole:#facc15; --src:#fb923c;
  --dst:#22c55e; --fence:#f43f5e; --free:#253249; }
* { box-sizing:border-box } body { margin:0;background:var(--bg);color:#e6edf7;font-family:Inter,ui-sans-serif,system-ui,sans-serif }
.top { position:sticky;top:0;z-index:20;background:rgba(7,16,31,.94);backdrop-filter:blur(12px);padding:18px 24px;border-bottom:1px solid var(--line) }
h1 { margin:0 0 12px;font-size:22px } .controls { display:flex;align-items:center;gap:9px;flex-wrap:wrap }
button { border:1px solid #40516d;background:#18263b;color:#f8fafc;border-radius:8px;padding:8px 12px;cursor:pointer }
button:hover { background:#243653 } input[type=range] { flex:1;min-width:260px;accent-color:#60a5fa }
#label { min-width:90px;text-align:right;font-variant-numeric:tabular-nums;font-weight:700 }
main { max-width:1500px;margin:auto;padding:22px 24px 60px }.headline { display:flex;justify-content:space-between;gap:16px;align-items:start;margin-bottom:14px }
.headline h2 { margin:0 0 5px;font-size:21px;overflow-wrap:anywhere }.muted { color:var(--muted) }.badge { display:inline-flex;border:1px solid #3c4d67;border-radius:999px;padding:5px 10px;background:#16243a;white-space:nowrap }
.rule { padding:12px 15px;border:1px solid #31527c;background:#0e2440;border-radius:10px;margin-bottom:15px }.rule strong { color:#8ec5ff }
.fence-guide { margin-bottom:15px;border:1px solid #44658d;background:#0a1b31;border-radius:12px;padding:16px }.fence-guide h3 { margin:0 0 5px }.case-tabs { display:flex;gap:8px;flex-wrap:wrap;margin:13px 0 }.case-button { text-align:left;line-height:1.35 }.case-button.active { border-color:#7dd3fc;background:#183c5d;box-shadow:0 0 0 1px #7dd3fc inset }.case-button b { display:block }.set-compare { display:grid;grid-template-columns:minmax(180px,1fr) auto minmax(180px,1fr) auto minmax(150px,.8fr);gap:10px;align-items:stretch;margin-top:12px }.set-card { padding:12px;border:1px solid #3a4d68;border-radius:9px;background:#0b1628 }.set-card b { display:block;margin-bottom:7px }.set-values { font-family:ui-monospace,monospace;font-size:17px;font-weight:850 }.math-symbol { align-self:center;font-size:24px;font-weight:900;color:#9fb5d1 }.verdict { padding:12px;border-radius:9px;border:1px solid;font-weight:750 }.verdict.safe { background:#0b3327;border-color:#23825b;color:#bbf7d0 }.verdict.blocked { background:#3c1020;border-color:#b43a55;color:#fecdd3 }.fence-detail { margin-top:11px;color:#c8d7ea;line-height:1.55 }
.grid { display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:15px;margin-bottom:15px }.card { background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0 }
.card h3 { margin:0 0 12px;font-size:15px;color:#c9d7ea }.arena-wrap { position:relative;height:174px;margin:8px 0 5px;padding-top:88px }
.arena { position:relative;height:54px;background:#080f1c;border:1px solid #42516a;border-radius:9px;overflow:visible }
.segment { position:absolute;top:0;height:100%;display:flex;align-items:center;justify-content:center;min-width:2px;border-right:1px solid rgba(255,255,255,.18);font-size:12px;font-weight:800;overflow:hidden }
.reserved { left:0;background:var(--reserved) }.kv { background:var(--kv) }.mamba { background:var(--mamba) }.axis { display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:7px }
.arena-summary { display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:5px }.usage { font-size:17px;font-weight:850 }.gap { color:#b8c7dc }.full-state { border:1px solid #51647f;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:850 }.full-state.open { color:#86efac }.full-state.pressure { color:#fdba74;border-color:#c76624;background:#371b0a }.full-state.closed { color:#fda4af;border-color:#9f3348 }
.watermark { position:absolute;top:0;height:54px;width:0;border-left:2px dashed rgba(255,255,255,.92);z-index:5;pointer-events:none }.watermark span { position:absolute;top:58px;left:0;transform:translateX(-50%);font-size:10px;font-weight:900;color:#eef6ff;background:#18263b;border:1px solid #536681;border-radius:4px;padding:2px 4px;white-space:nowrap }
.attempted-watermark { position:absolute;top:-49px;height:103px;width:0;border-left:3px dashed var(--fence);z-index:8;pointer-events:none }.attempted-watermark span { position:absolute;top:0;left:50%;transform:translateX(-50%);background:#3c1020;border:2px solid var(--fence);color:#fecdd3;border-radius:6px;padding:4px 7px;font-size:11px;font-weight:900;white-space:nowrap }
.row-span { position:absolute;top:1px;height:52px;z-index:4;min-width:2px;pointer-events:none;border:2px solid;box-sizing:border-box }.row-span-h { color:var(--hole);background:#080f1c;border-color:#080f1c }.row-span-s { color:var(--src);background:rgba(251,146,60,.36) }.row-span-d { color:var(--dst);background:rgba(34,197,94,.36) }.row-span-f { color:var(--fence);background:rgba(244,63,94,.34) }
.lru-risk-span { position:absolute;top:31px;height:20px;z-index:3;min-width:3px;border:1px solid rgba(254,202,202,.85);display:flex;align-items:center;justify-content:center;overflow:hidden;color:#fff;font-size:9px;font-weight:950;text-shadow:0 1px 2px #450a0a;pointer-events:none }.lru-risk-span.protected { background:#334155;border-color:#94a3b8;color:#e2e8f0;text-shadow:none }
.marker { position:absolute;top:-37px;height:91px;width:2px;z-index:6;background:currentColor;pointer-events:none }.marker span { position:absolute;top:0;left:50%;transform:translateX(-50%);padding:3px 5px;border-radius:5px;background:#07101f;border:2px solid currentColor;color:currentColor;font-size:11px;font-weight:900;white-space:nowrap }
.marker-s { color:var(--src) }.marker-d { color:var(--dst) }.marker-f { color:var(--fence) }
.legend { display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12px }.legend i { display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px }
.allocators { display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin-top:13px }.allocator { padding:10px;border-radius:8px;background:#0a1425;border-left:4px solid }.allocator.kv { border-color:var(--kv) }.allocator.mamba { border-color:var(--mamba) }.allocator b { text-transform:uppercase }
.resolution-notice { margin-top:12px;padding:11px 12px;border-radius:8px;background:#0d3326;border:1px solid #237552;color:#a7f3d0 }.resolution-notice strong { color:#d1fae5 }
.pressure-notice { margin-top:12px;padding:11px 12px;border-radius:8px;background:#3c1020;border:1px solid #a32f4b;color:#fecdd3 }.pressure-notice strong { color:#ffe4e8 }
.narrative { line-height:1.55 }.narrative .big { font-size:17px;font-weight:800;margin-bottom:8px }.plan { padding:10px 0;border-top:1px solid var(--line) }.plan:first-of-type { border-top:0 }.decision { font-weight:800 }.allowed { color:#4ade80 }.deferred { color:#fb7185 }.moves { display:flex;flex-wrap:wrap;gap:8px;margin-top:8px }.move { background:#0b1728;border:1px solid #344866;border-radius:8px;padding:7px 9px;font-family:ui-monospace,monospace }.hole-list { margin-top:7px;color:#fde047;overflow-wrap:anywhere }
.chunk-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:7px }.chunk { position:relative;min-height:82px;padding:8px;border:1px solid #3b4a62;border-radius:8px;background:#080f1c;overflow:hidden }.chunk.KV { border-color:#4089e8 }.chunk.MAMBA { border-color:#986ee0 }.chunk-fill { position:absolute;inset:0 auto 0 0;z-index:0;transition:width .2s ease }.chunk-fill.KV { background:linear-gradient(90deg,#123968,#1e5ca4) }.chunk-fill.MAMBA { background:linear-gradient(90deg,#34205d,#59358e) }.chunk-content { position:relative;z-index:1;text-shadow:0 1px 2px #030712 }.chunk .id { color:#d0dbea;font-size:11px }.chunk .owner { margin-top:5px;font-weight:800;font-size:12px }.chunk .percent { margin-top:3px;font-size:11px;font-weight:850;color:#f8fafc }.chunk-lru { margin-top:4px;padding:3px 5px;border-radius:4px;background:rgba(69,10,10,.82);color:#fecaca;font-size:10px;font-weight:900;display:inline-block;line-height:1.3 }.chunk-lru.protected { background:rgba(51,65,85,.9);color:#e2e8f0 }.chunk-lru.resident { background:rgba(30,64,175,.82);color:#dbeafe }.chunk-lru.copying { background:rgba(133,77,14,.88);color:#fef3c7 }.chunk-lru.pending-lru { background:rgba(88,28,135,.88);color:#f3e8ff }.lock { position:absolute;z-index:2;right:6px;top:5px;color:#ff8095;font-size:14px }.pending { position:absolute;z-index:2;right:6px;bottom:5px;color:#facc15;font-size:10px }
.mamba-lru-grid { display:grid;grid-template-columns:1fr 1fr;gap:12px }.lru-layer { border:1px solid #344660;border-radius:10px;background:#0a1425;padding:11px }.lru-layer h4 { margin:0 0 4px }.lru-entry { display:grid;grid-template-columns:74px 1fr auto;gap:9px;align-items:center;margin-top:7px;padding:8px;border:1px solid #6b2f3b;border-radius:8px;background:linear-gradient(90deg,rgba(127,29,29,.72),rgba(30,41,59,.7)) }.lru-entry.protected { border-color:#475569;background:#182235 }.risk-score { font-size:18px;font-weight:950;color:#fecaca }.lru-entry.protected .risk-score { color:#cbd5e1;font-size:12px }.lru-next { color:#fca5a5;font-weight:900 }.lru-location { font-family:ui-monospace,monospace;font-size:12px;color:#dbeafe }
.tables { display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px } table { width:100%;border-collapse:collapse;font-size:13px } td,th { padding:7px;border-bottom:1px solid #25354c;text-align:left;overflow-wrap:anywhere } th { color:#91a5bf }.empty { color:var(--muted);font-style:italic }
details { background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 16px } summary { cursor:pointer;font-weight:700 } pre { background:#030712;padding:16px;border-radius:8px;overflow:auto;line-height:1.35;font-size:12px;max-height:70vh }
@media(max-width:850px) { .grid,.tables,.set-compare,.mamba-lru-grid { grid-template-columns:1fr }.math-symbol { text-align:center }.headline { flex-direction:column }.top,main { padding-left:14px;padding-right:14px } }
</style></head><body>
<div class="top"><h1>HiCache 동작 과정 재생</h1><div class="controls">
<button id="prev">◀ 이전</button><button id="play">▶ 재생</button><button id="next">다음 ▶</button>
<input id="step" type="range" min="0" max="__MAX_STEP__" value="0"><span id="label"></span></div></div>
<main><div class="headline"><div><h2 id="event-title"></h2><div id="event-meta" class="muted"></div></div><span id="phase" class="badge"></span></div>
<section id="row-fence-guide" class="fence-guide" hidden></section>
<div class="grid"><section class="card"><h3>L1 통합 GPU 바이트 메모리</h3><div id="arena-root"></div><div id="allocator-root"></div><div id="hole-notice"></div></section>
<section class="card"><h3>현재 상태 / compaction 계획</h3><div id="narrative" class="narrative"></div></section></div>
<section class="card" style="margin-bottom:15px"><h3>L2 공유 typed chunk 메모리 <span class="muted">— KV는 token 사용률만큼 채웁니다. Mamba는 L1 상주 백업본인지, L2 전용 eviction 후보인지 배지로 구분합니다. 🔒는 비동기 복사 pin입니다.</span></h3><div id="chunks" class="chunk-grid"></div></section>
<section class="card" style="margin-bottom:15px"><h3>Mamba LRU 퇴출 위험도 <span class="muted">— 실제 LRU tail 순서를 환산하며, 100이 다음 eviction 후보입니다. LOCK은 lock_ref가 있어 순위에서 제외된 상태입니다.</span></h3><div id="mamba-lru"></div></section>
<div class="tables"><section class="card"><h3>최근 비동기 전송</h3><div id="transfers"></div></section><section class="card"><h3>최근 관찰된 cache node</h3><div id="nodes"></div></section></div>
<details><summary>원본 event/debug 텍스트</summary><pre id="frame"></pre></details></main>
<script>
const rawFrames=__RAW_FRAMES__;
const visualFrames=__VISUAL_FRAMES__;
const rowFenceCases=__ROW_FENCE_CASES__;
let timer=null; const slider=document.getElementById('step');
const requestedStep=new URLSearchParams(location.search).get('step');
let i=Math.max(0,Math.min(visualFrames.length-1,requestedStep===null&&rowFenceCases.length?rowFenceCases[0].step:Number(requestedStep||0)));slider.value=i;
const h=(value)=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pages=(items,limit=18,total=items.length)=>!total?'없음':total<=limit?items.join(', '):`${items.slice(0,limit).join(', ')} … 외 ${total-limit}개`;
const formatBytes=(n)=>n==null?'미확인':n>=1073741824?`${(n/1073741824).toFixed(2)} GiB`:n>=1048576?`${(n/1048576).toFixed(2)} MiB`:n>=1024?`${(n/1024).toFixed(1)} KiB`:`${n} B`;
const statusText=(value)=>({queued:'대기 중',in_flight:'복사 중',complete:'완료',unknown:'미확인',none:'없음',requested:'요청됨',ready:'준비 완료',copying:'복사 중',l1_resident:'L1 상주',l2_only:'L2에만 있음',allowed:'허용됨',deferred_row_fence:'row fence로 연기됨',completed:'완료'}[value]??value);
const directionText=(value)=>value==='D2H'?'D2H (L1→L2)':value==='H2D'?'H2D (L2→L1)':value;
const eventText=(value)=>({trace_started:'trace 기록 시작',mamba_lru_state:'Mamba LRU 순위 갱신',l1_allocator_initialized:'L1 allocator 초기화',l1_allocator_state:'L1 allocator 상태 갱신',l1_node_evicted:'L1 cache node eviction',l1_compaction_decision:'L1 compaction 판단',l1_compaction_completed:'L1 compaction 완료',l1_transfer_fence_registered:'L1 row fence 등록',l1_transfer_fence_checked:'L1 row fence 영향 범위 확인',l1_transfer_fence_deferred:'L1 compaction 연기',l1_transfer_fence_released:'L1 row fence 해제',l2_arena_initialized:'L2 typed chunk 메모리 초기화',l2_arena_cleared:'L2 typed chunk 메모리 초기화 해제',l2_chunk_owner_changed:'L2 chunk 소유 종류 변경',l2_kv_allocated:'L2 KV chunk 할당',l2_mamba_allocated:'L2 Mamba chunk 할당',l2_backup_requested:'L2 backup 요청',l2_backup_metadata_committed:'L2 backup metadata 반영',l2_backup_ready:'L2 backup 준비 완료',l2_chunks_pinned:'L2 chunk 고정',l2_chunks_unpinned:'L2 chunk 고정 해제',l2_transfer_pin_armed:'L2 전송 pin 등록',l2_transfer_pin_released:'L2 전송 pin 해제',loadback_requested:'L2 → L1 loadback 요청',loadback_metadata_committed:'Loadback metadata 반영',d2h_transfer_queued:'D2H 전송 대기',d2h_transfer_enqueued:'D2H 복사 시작',d2h_transfer_completed:'D2H 복사 완료',h2d_transfer_queued:'H2D loadback 대기',h2d_transfer_enqueued:'H2D loadback 시작',h2d_transfer_completed:'H2D loadback 완료'}[value]??value);
function phaseText(f) { if(f.l1.pressure)return 'WATERMARK 공간 부족'; if(f.event==='mamba_lru_state')return 'MAMBA LRU'; if(f.l1.hole_resolution?.kind==='recorded_on_free')return '빈 ROW 기록됨'; if(f.l1.hole_resolution?.kind==='reused_by_allocation')return '빈 ROW 재사용'; if(f.event==='l1_node_evicted')return 'L1 EVICTION'; if(f.event==='l1_compaction_decision'&&f.l1.latest_plans.some(p=>p.seq===f.seq&&p.urgent))return '상대편 공간 확보 COMPACTION'; if(f.event==='l1_compaction_decision')return 'COMPACTION 사전 계산'; if(f.event==='l1_compaction_completed')return 'COMPACTION 완료'; if(f.event.startsWith('h2d_'))return 'H2D LOADBACK'; if(f.event.startsWith('d2h_'))return 'D2H BACKUP'; if(f.event.includes('fence'))return 'ROW FENCE'; if(f.event.includes('transfer'))return '비동기 전송'; if(f.l1.allocators.some(a=>a.next_page_fits===false))return 'ROW 단위 공간 부족'; return '상태 갱신'; }
function renderArena(f) {
  const l1=f.l1;if(!l1.total_bytes){document.getElementById('arena-root').innerHTML='<div class="empty">L1 allocator가 아직 초기화되지 않았습니다.</div>';document.getElementById('allocator-root').innerHTML='';document.getElementById('hole-notice').innerHTML='';return;}
  const gapText=l1.frontier_gap_bytes==null?'watermark 사이 여유 공간 미확인':`watermark 사이 여유 ${formatBytes(l1.frontier_gap_bytes)} · ${l1.frontier_gap_pct.toFixed(2)}%`;
  const blocked=l1.allocators.filter(a=>a.next_page_fits===false);const stateClass=l1.frontiers_full?'closed':blocked.length?'pressure':'open';const stateText=l1.frontiers_full?'가득 참 · WATERMARK가 맞닿음':blocked.length?`${blocked.map(a=>a.kind==='kv'?'KV':'MAMBA').join(' + ')} 다음 ROW 할당 불가`:'다음 ROW 할당 가능';
  let html=`<div class="arena-summary"><span class="usage">${l1.occupied_pct.toFixed(2)}% 사용 중</span><span class="gap">${gapText}</span><span class="full-state ${stateClass}">${stateText}</span></div><div class="arena-wrap"><div class="arena">`;
  if(l1.reserved_pct>0)html+=`<div class="segment reserved" style="width:${l1.reserved_pct}%">R</div>`;
  for(const a of l1.allocators){const width=Math.max(0,a.end_pct-a.start_pct);html+=`<div class="segment ${a.kind}" style="left:${a.start_pct}%;width:${width}%">${a.kind==='kv'?'KV':'Mamba'}</div>`;}
  for(const r of l1.mamba_lru_spans||[]){const width=Math.max(0,Math.min(100-r.pct,r.width_pct));const label=r.protected?'LOCK':`E${r.eviction_score}`;const alpha=(.3+.65*r.eviction_score/100).toFixed(2);html+=`<div class="lru-risk-span ${r.protected?'protected':''}" style="left:${r.pct}%;width:${width}%;${r.protected?'':`background:rgba(185,28,28,${alpha})`}" title="Mamba node ${r.node_id} · row ${r.page} · ${r.protected?`lock_ref ${r.lock_ref}: eviction 제외`:`퇴출 위험도 ${r.eviction_score}, 순위 ${r.eviction_rank}`}">${label}</div>`;}
  for(const a of l1.allocators){const pct=Math.max(.5,Math.min(99.5,a.watermark_pct));html+=`<div class="watermark" style="left:${pct}%"><span>${a.kind==='kv'?'KV':'M'} WM</span></div>`;}
  if(l1.pressure?.reason==='peer_frontier_collision_prevented'){const pct=Math.max(.5,Math.min(99.5,l1.pressure.attempted_pct));html+=`<div class="attempted-watermark" style="left:${pct}%"><span>할당 차단 · ${h(l1.pressure.pool)} WM</span></div>`;}else for(const a of blocked){const pct=Math.max(.5,Math.min(99.5,a.next_watermark_pct));html+=`<div class="attempted-watermark" style="left:${pct}%"><span>다음 ${a.kind==='kv'?'KV':'M'} WM 이동 불가</span></div>`;}
  const grouped=new Map();for(const m of l1.markers){const key=`${m.pool}:${m.page}:${m.page_end??m.page}`;if(!grouped.has(key))grouped.set(key,{...m,roles:[]});grouped.get(key).roles.push(m.role);}
  for(const m of grouped.values()){const roles=[...new Set(m.roles)].sort();const classes=roles.map(role=>`row-span-${role.toLowerCase()}`).join(' ');const width=Math.max(0,Math.min(100-m.pct,m.width_pct));html+=`<div class="row-span ${classes}" style="left:${m.pct}%;width:${width}%"></div>`;}
  const markerBuckets=new Map();for(const m of grouped.values()){const signature=`${m.pool}:${[...new Set(m.roles)].sort().join('/')}`;if(!markerBuckets.has(signature))markerBuckets.set(signature,[]);markerBuckets.get(signature).push(m);}const visibleMarkers=[];for(const bucket of markerBuckets.values()){bucket.sort((a,b)=>a.page-b.page);const totalPages=bucket.reduce((sum,m)=>sum+(m.page_count??1),0);if(bucket.length<=8)visibleMarkers.push(...bucket);else for(const index of [0,Math.floor(bucket.length/2),bucket.length-1])visibleMarkers.push({...bucket[index],sampleTotal:totalPages});}
  let markerIndex=0;for(const m of visibleMarkers){const roles=[...new Set(m.roles)].sort().filter(role=>role!=='H');if(!roles.length)continue;const lane=markerIndex++%3;const primary=roles.includes('F')?'f':roles.includes('S')?'s':'d';const pct=Math.max(.4,Math.min(99.4,m.pct));const sample=m.sampleTotal?` · 전체 ${m.sampleTotal}개 중 표본`:'';const rowLabel=m.page_end!=null&&m.page_end!==m.page?`${m.page}–${m.page_end}`:`${m.page}`;html+=`<div class="marker marker-${primary}" style="left:${pct}%;top:${-32-lane*24}px;height:${86+lane*24}px"><span>${roles.join('/')} · ${h(m.pool)}:${rowLabel}${sample}</span></div>`;}
  html+='</div><div class="axis"><span>낮은 byte · 왼쪽</span><span>높은 byte · 오른쪽</span></div></div>';
  html+='<div class="legend"><span><i style="background:var(--reserved)"></i>예약 영역</span><span><i style="background:var(--mamba)"></i>Mamba 관리 범위</span><span><i style="background:var(--kv)"></i>KV 관리 범위</span><span><i style="background:#080f1c;border:1px solid #42516a"></i>검은 부분은 빈 row</span><span><i style="background:var(--src)"></i>S 이동 원본</span><span><i style="background:var(--dst)"></i>D 이동 목적지</span><span><i style="background:var(--fence)"></i>F 전송 보호 중</span></div>';
  document.getElementById('arena-root').innerHTML=html;
  document.getElementById('allocator-root').innerHTML='<div class="allocators">'+l1.allocators.map(a=>`<div class="allocator ${a.kind}"><b>${h(a.pool)} · ${a.kind==='kv'?'KV':'Mamba'}</b><div class="muted">증가 방향: ${a.grow_direction==='up'?'왼쪽 → 오른쪽':'오른쪽 → 왼쪽'}</div><div>사용 row ${a.live_pages} · hole ${a.hole_count} · 재사용 대기 ${a.pending_reuse_pages}</div><div class="muted">watermark row ${a.watermark_page} · 현재 gap에 다음 row ${a.gap_capacity_pages??'?'}개 할당 가능</div></div>`).join('')+'</div>';
  let notices='';const resolution=l1.hole_resolution;if(resolution?.kind==='reused_by_allocation')notices+=`<div class="resolution-notice"><strong>빈 row ${resolution.pages?.length?pages(resolution.pages):'?'} → 할당됨:</strong> 이번 할당이 ${h(resolution.pool)}의 빈 row ${resolution.count}개를 재사용했습니다.</div>`;const pressure=l1.pressure;if(pressure)notices+=`<div class="pressure-notice"><strong>${h(pressure.reason)}:</strong> ${h(pressure.pool)}이 ${pressure.requested_pages}개 row를 요청했지만 두 watermark 사이에는 ${formatBytes(pressure.current_gap_bytes)}만 남았습니다.${pressure.would_cross_bytes?` 빨간 가상 watermark는 ${h(pressure.peer_pool)} 영역을 ${formatBytes(pressure.would_cross_bytes)}만큼 침범하므로, 실제 상태를 바꾸기 전에 allocator가 할당을 차단했습니다.`:''}</div>`;document.getElementById('hole-notice').innerHTML=notices;
}
function renderNarrative(f) {
  const blocked=f.l1.allocators.filter(a=>a.next_page_fits===false);const latestTransfer=f.transfers.at(-1);let lead=f.event.startsWith('h2d_')?`cache node ${latestTransfer?.node_ids?.join(', ')||'?'}의 L2 → L1 loadback 상태는 '${statusText(latestTransfer?.status??'updating')}'입니다.`:f.event.startsWith('d2h_')?`cache node ${latestTransfer?.node_ids?.join(', ')||'?'}의 L1 → L2 backup 상태는 '${statusText(latestTransfer?.status??'updating')}'입니다.`:f.eviction?`cache node ${f.eviction.node_id}를 L1에서 eviction했습니다: ${Object.entries(f.eviction.evicted_by_component).map(([kind,count])=>`${kind} ${count}`).join(' · ')}. 반환된 row는 검은 빈 공간으로 표시됩니다.`:blocked.length?`${blocked.map(a=>a.kind==='kv'?'KV':'Mamba').join('와 ')}는 다음 row로 watermark를 이동할 수 없습니다. 남은 byte gap이 다음 row보다 작으므로 eviction이나 compaction으로 먼저 공간을 확보해야 합니다.`:'Allocator/cache 상태가 갱신되었습니다.';if(f.l1.pressure)lead=f.l1.pressure.reason==='peer_frontier_collision_prevented'?`요청한 ${h(f.l1.pressure.pool)} watermark가 ${h(f.l1.pressure.peer_pool)} 메모리 영역을 침범할 위치였습니다. 실제 watermark가 겹치기 전에 allocator가 빨간 가상 위치의 할당을 거부했습니다.`:`${h(f.l1.pressure.pool)}이 사용할 수 있는 page index를 모두 소진하여 watermark 확장을 거부했습니다.`;else if(f.l1.hole_resolution?.kind==='recorded_on_free')lead=`진단용 trace가 GPU 상태를 확인하여 ${h(f.l1.hole_resolution.pool)}의 정확한 빈 row를 기록했습니다: ${pages(f.l1.hole_resolution.pages)}.`;else if(f.l1.hole_resolution?.kind==='reused_by_allocation')lead=`새 ${h(f.l1.hole_resolution.pool)} 할당이 빈 row ${f.l1.hole_resolution.count}개를 채웠으므로, 검은 빈 공간이 할당된 메모리 색으로 바뀝니다.`;
  else if(f.event==='l1_compaction_decision'){const current=f.l1.latest_plans.find(p=>p.seq===f.seq);const invalid=f.l1.latest_plans.flatMap(p=>p.invalid_preview_sources||[]);if(invalid.length)lead=`이 사전 계산은 이미 hole인 row ${pages(invalid)}를 이동 원본으로 표시했습니다. 실제 복사가 아니라 수정 전 preview 기록의 불일치이며, 다음 compaction 완료 step의 실제 이동을 확인해야 합니다.`;else if(current?.urgent){const requester=String(current.pool).toLowerCase().includes('mamba')?'KV':'Mamba';lead=`${requester}가 다음 row를 할당할 공간이 부족하여 ${h(current.pool)} 쪽을 즉시 compaction합니다. hole을 채우고 ${h(current.pool)} watermark를 물려 ${requester}가 사용할 byte 공간을 확보합니다.`;}else lead='Allocator가 정확한 hole 위치를 확인하고, 어떤 사용 중 row를 그 자리로 옮길지 계획했습니다.';}
  else if(f.event==='l1_compaction_completed')lead='계획한 원본 row를 목적지 hole로 복사했습니다. 기본 소유 색상은 계속 KV 또는 Mamba로 유지됩니다.';
  else if(f.event==='l1_transfer_fence_checked')lead='Compaction 영향 row와 비동기 전송으로 보호 중인 row를 비교했습니다. 두 집합이 실제로 겹칠 때만 compaction을 연기합니다.';
  let html=`<div class="big">${lead}</div>`;
  if(!f.l1.latest_plans.length)html+='<div class="empty">아직 compaction 사전 계산이 없습니다.</div>';
  for(const p of f.l1.latest_plans){const cls=String(p.decision).includes('deferred')?'deferred':'allowed';html+=`<div class="plan"><div><b>${h(p.pool)}</b> · seq ${p.seq} · <span class="decision ${cls}">${h(statusText(p.decision))}</span>${p.urgent?' · <strong>긴급: 상대편 할당 요청</strong>':''}</div><div class="hole-list">빈 row: ${pages(p.holes,18,p.holes_total)}</div><div class="muted">S 이동 원본: ${pages(p.sources,18,p.sources_total)} · D 이동 목적지: ${pages(p.destinations,18,p.destinations_total)}</div>${p.invalid_preview_sources?.length?`<div class="pressure-notice"><strong>잘못된 preview:</strong> source ${pages(p.invalid_preview_sources)}는 이미 hole입니다. 이 화살표는 실제 복사가 아닙니다.</div>`:''}`;if(p.moves.length){const visible=p.moves.slice(0,12);html+='<div class="moves">'+visible.map(m=>`<span class="move"><b>S ${m.source}</b> &nbsp;→&nbsp; <b>D ${m.destination}</b> <span class="muted">(빈 row)</span></span>`).join('')+(p.moves_total>visible.length?`<span class="move">… 이외 이동 ${p.moves_total-visible.length}개</span>`:'')+'</div>';}html+='</div>';}
  document.getElementById('narrative').innerHTML=html;
}
function renderChunks(f) { const chunks=f.l2.chunks;document.getElementById('chunks').innerHTML=chunks.length?chunks.map(c=>{const pct=Math.max(0,Math.min(100,Number(c.fill_pct)||0));const pctText=pct.toFixed(pct>0&&pct<10?1:0);const usage=c.owner==='KV'?`${c.kv_used}/${c.kv_capacity} tokens`:c.owner==='MAMBA'?'고정 크기 state':'비어 있음';const lru=c.mamba_lru;let stateBadge='';if(lru)stateBadge=lru.protected?`<div class="chunk-lru protected">L2 LOCK · 퇴출 제외</div>`:`<div class="chunk-lru">퇴출 E${lru.eviction_score} · #${lru.eviction_rank}</div>`;else if(c.owner==='MAMBA'){const states={copying:['copying','D2H 복사 중'],l1_locked_backup:['protected','L1 LOCK · 백업본'],l1_backup:['resident','L1 상주 · 백업본'],l2_lru_pending:['pending-lru','L2 전용 · 순위 갱신 중'],backup_not_ranked:['resident','백업본 · LRU 제외'],node_unrecorded:['resident','백업본 · node 미확인']};const [cls,label]=states[c.mamba_status]||['resident','백업본'];stateBadge=`<div class="chunk-lru ${cls}">${label}</div>`;}const nodeText=c.mamba_node_id==null?'':` · node ${c.mamba_node_id}`;return `<div class="chunk ${h(c.owner)}" title="chunk ${c.chunk_id}${nodeText} · ${h(c.owner)} · ${pctText}%"><div class="chunk-fill ${h(c.owner)}" style="width:${pct}%"></div><div class="chunk-content"><span class="id">chunk ${c.chunk_id}${nodeText}</span><div class="owner">${h(c.owner)}</div><div class="percent">${pctText}% · ${usage}</div>${stateBadge}</div>${c.pin_count?`<span class="lock" title="비동기 복사 pin ${c.pin_count}">🔒${c.pin_count}</span>`:''}${c.pending_release?'<span class="pending">반환 대기</span>':''}</div>`;}).join(''):'<div class="empty">L2 메모리가 아직 초기화되지 않았습니다.</div>'; }
const lruReasonText=(value)=>({match:'prefix hit로 MRU 갱신',insert:'새 Mamba state 삽입',device_eviction_requested:'L1 eviction 직전',device_eviction_completed:'L1 eviction 완료',host_eviction_requested:'L2 eviction 직전',host_eviction_completed:'L2 eviction 완료',device_lock_acquired:'L1 state 보호 시작',device_lock_released:'L1 state 보호 해제',host_lock_acquired:'L2 state 보호 시작',host_lock_released:'L2 state 보호 해제',loadback_committed:'L2 → L1 loadback 반영'}[value]??value??'아직 기록 없음');
function renderMambaLru(f) { const root=document.getElementById('mamba-lru');const lru=f.mamba_lru||{};if(!lru.seq){root.innerHTML='<div class="empty">이 trace에는 Mamba LRU snapshot이 없습니다. LRU trace가 포함된 새 serving log가 필요합니다.</div>';return;}const layer=(title,where,entries)=>`<div class="lru-layer"><h4>${title}</h4><div class="muted">LRU 갱신 seq ${lru.seq} · ${h(lruReasonText(lru.reason))}</div>${entries.length?entries.map(e=>{const protectedState=e.protected;const locationKind=where==='physical row'&&!['physical','identity'].includes(e.index_space)?'virtual slot':where;const location=e.indices?.length?`${locationKind} ${e.indices.join(', ')}`:`${locationKind} 미확인`;const virtual=e.index_space==='physical'&&e.virtual_indices?.length?` · virtual slot ${e.virtual_indices.join(', ')}`:'';return `<div class="lru-entry ${protectedState?'protected':''}"><div class="risk-score">${protectedState?'LOCK':`E ${e.eviction_score}`}</div><div><b>node ${e.node_id}</b>${!protectedState&&e.eviction_rank===1?'<div class="lru-next">다음 eviction 후보</div>':''}<div class="lru-location">${location}${virtual}</div></div><div class="muted">${protectedState?`lock_ref ${e.lock_ref} · 퇴출 제외`:`#${e.eviction_rank}`}</div></div>`;}).join(''):'<div class="empty" style="margin-top:8px">현재 LRU에 Mamba state가 없습니다.</div>'}</div>`;root.innerHTML=`<div class="mamba-lru-grid">${layer('L1 · GPU Mamba LRU','physical row',lru.device||[])}${layer('L2 · Host Mamba LRU','chunk',lru.host||[])}</div>`; }
function table(rows,columns) { if(!rows.length)return '<div class="empty">아직 관찰된 항목이 없습니다.</div>';return '<table><thead><tr>'+columns.map(c=>`<th>${c[0]}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+columns.map(c=>`<td>${h(c[1](r))}</td>`).join('')+'</tr>').join('')+'</tbody></table>'; }
function renderRowFenceGuide(){const root=document.getElementById('row-fence-guide');if(!rowFenceCases.length){root.hidden=true;return;}root.hidden=false;let active=rowFenceCases[0];for(const item of rowFenceCases){if(i>=item.start_step)active=item;}const poolName=active.pool==='full'?'KV':'Mamba';const intersection=active.intersection_pages.length?`{ ${active.intersection_pages.join(', ')} }`:'∅ (없음)';const result=active.blocking?'COMPACTION 연기':'COMPACTION 허용';const explanation=active.blocking?`보호 row ${active.intersection_pages.join(', ')}를 compaction도 건드립니다. 비동기 복사가 끝날 때까지 compaction을 연기합니다.`:'두 집합이 겹치지 않습니다. 보호하지 않은 row만 이동하므로 비동기 복사와 compaction을 동시에 진행해도 안전합니다.';root.innerHTML=`<h3>Row fence가 판단하는 한 가지 질문</h3><div class="muted">이 페이지는 실제 serving 타임라인이 아니라 서로 독립된 테스트들을 모은 것입니다. 아래 버튼을 누르면 각 테스트의 결론이 나온 순간으로 바로 이동합니다.</div><div class="case-tabs">${rowFenceCases.map(c=>{const name=c.pool==='full'?'KV':'Mamba';const state=c.blocking?'겹침 · 연기':'영향권 밖 · 허용';return `<button class="case-button ${c.number===active.number?'active':''}" data-case="${c.number}"><b>${c.number}. ${name}</b>${state}</button>`;}).join('')}</div><div><b>시나리오 ${active.number}/${rowFenceCases.length} · ${poolName} · ${active.blocking?'보호 row와 겹침':'보호 row의 영향권 밖'}</b></div><div class="set-compare"><div class="set-card"><b>F · 비동기 복사로 보호 중</b><div class="set-values">{ ${active.protected_pages.join(', ')} }</div></div><div class="math-symbol">∩</div><div class="set-card"><b>S/D · compaction 영향 row</b><div class="set-values">{ ${active.touched_pages.join(', ')} }</div><div class="muted">S 원본 ${pages(active.source_pages)} → D 목적지 ${pages(active.destination_pages)}</div></div><div class="math-symbol">=</div><div class="verdict ${active.blocking?'blocked':'safe'}"><div>교집합 ${intersection}</div><div style="margin-top:7px">결론: ${result}</div></div></div><div class="fence-detail">${explanation}</div>`;for(const button of root.querySelectorAll('[data-case]'))button.onclick=()=>{const target=rowFenceCases.find(c=>c.number===Number(button.dataset.case));if(target){slider.value=target.step;show();}};}
function show(){i=Number(slider.value);const f=visualFrames[i];if(!f)return;document.getElementById('label').textContent=`${i+1} / ${visualFrames.length}`;document.getElementById('event-title').textContent=eventText(f.event);const collapsed=f.collapsed_events_before?` · 중간 반복 event ${f.collapsed_events_before}개 접음`:'';document.getElementById('event-meta').textContent=`원본 event: ${f.event} · seq ${f.seq??'?'} · pid ${f.pid??'?'}${collapsed}`;document.getElementById('phase').textContent=phaseText(f);renderRowFenceGuide();renderArena(f);renderNarrative(f);renderChunks(f);renderMambaLru(f);document.getElementById('transfers').innerHTML=table(f.transfers,[['ID',r=>r.id],['방향',r=>directionText(r.direction)],['상태',r=>statusText(r.status)],['노드',r=>r.node_ids.join(', ')]]);document.getElementById('nodes').innerHTML=table(f.nodes,[['노드',r=>r.id],['FULL/KV',r=>statusText(r.full)],['Mamba',r=>statusText(r.mamba)],['L2',r=>statusText(r.l2)]]);document.getElementById('frame').textContent=rawFrames[i]||'';}
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,i-1);show();};document.getElementById('next').onclick=()=>{slider.value=Math.min(visualFrames.length-1,i+1);show();};document.getElementById('play').onclick=(e)=>{if(timer){clearInterval(timer);timer=null;e.currentTarget.textContent='▶ 재생';return;}e.currentTarget.textContent='⏸ 일시정지';timer=setInterval(()=>{if(i>=visualFrames.length-1){clearInterval(timer);timer=null;e.currentTarget.textContent='▶ 재생';return;}slider.value=++i;show();},500);};slider.oninput=show;show();
</script></body></html>"""
    document = document.replace("__MAX_STEP__", str(max(0, len(frames) - 1)))
    document = document.replace("__RAW_FRAMES__", raw_frames_json)
    document = document.replace("__VISUAL_FRAMES__", visual_frames_json)
    document = document.replace("__ROW_FENCE_CASES__", row_fence_cases_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--html", type=Path)
    parser.add_argument(
        "--compact-repeated-allocator-states",
        action="store_true",
        help=(
            "collapse repeated allocator-state frames in HTML while applying "
            "every event to replay state"
        ),
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--require-coverage",
        action="store_true",
        help="fail if any required scenario coverage item is missing",
    )
    args = parser.parse_args(argv)

    events = load_events(args.trace)
    if not events:
        print("trace is empty", file=sys.stderr)
        return 2
    checks, errors = coverage_report(events)
    if args.require_coverage:
        errors.extend(
            f"missing required coverage: {name}"
            for name, passed in checks.items()
            if not passed
        )
    if args.validate or args.require_coverage:
        print("Coverage")
        for name, passed in checks.items():
            print(f"  {'✓' if passed else '✗'} {name}")
        print(f"Validation errors: {len(errors)}")
        for error in errors:
            print(f"  ✗ {error}")
    if args.html:
        write_html(
            events,
            args.html,
            compact_repeated_allocator_states=(args.compact_repeated_allocator_states),
        )
        print(f"wrote {args.html}")

    if args.play:
        for index in range(len(events)):
            print("\033[2J\033[H", end="")
            print(render_state(replay_until(events, index), index, len(events)))
            time.sleep(args.delay)
    elif not args.validate or args.step >= 0:
        step = args.step if args.step >= 0 else len(events) - 1
        step = min(step, len(events) - 1)
        print(render_state(replay_until(events, step), step, len(events)))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
