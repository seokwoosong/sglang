"""Admission policy helpers for recurrent states in the unified radix cache.

This module deliberately contains no allocator or radix-tree mutations.  It
classifies a checkpoint and decides whether it may become persistent; callers
keep ownership, stream ordering, and insertion mechanics in the existing
Mamba cache implementation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

MAMBA_STATE_ADMISSION_POLICIES = ("default", "marconi")


class MambaCheckpointKind(str, Enum):
    BRANCH = "branch"
    FINAL = "final"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True)
class MambaAdmissionDecision:
    kind: MambaCheckpointKind
    persist: bool


def decide_mamba_state_admission(
    *,
    policy: str,
    is_finished: bool,
    checkpoint_seqlen: Optional[int],
    branching_seqlen: Optional[int],
) -> MambaAdmissionDecision:
    """Classify one checkpoint and decide whether to persist it.

    ``marconi`` persists at most the branch candidate (when the existing
    branch-aware tracker captured that exact aligned position) and the final
    safe checkpoint.  Intermediate snapshots remain available to the running
    request but are not inserted into the radix tree.
    """
    if policy not in MAMBA_STATE_ADMISSION_POLICIES:
        raise ValueError(
            f"Unknown Mamba state admission policy {policy!r}; expected one of "
            f"{MAMBA_STATE_ADMISSION_POLICIES}."
        )

    if is_finished:
        return MambaAdmissionDecision(MambaCheckpointKind.FINAL, True)

    is_branch = (
        checkpoint_seqlen is not None
        and branching_seqlen is not None
        and checkpoint_seqlen == branching_seqlen
    )
    if is_branch:
        return MambaAdmissionDecision(MambaCheckpointKind.BRANCH, True)

    return MambaAdmissionDecision(
        MambaCheckpointKind.INTERMEDIATE,
        policy == "default",
    )


@dataclass
class MambaAdmissionStats:
    branch_candidates: int = 0
    final_candidates: int = 0
    intermediate_candidates: int = 0
    branch_admitted: int = 0
    final_admitted: int = 0
    intermediate_admitted: int = 0
    duplicate_candidates: int = 0
    intermediate_skipped: int = 0


class MambaAdmissionStatsTracker:
    """Request-scoped observability for admission-policy validation.

    A single structured DEBUG record is emitted at request completion.  The
    verification script starts SGLang with ``--log-level debug`` and compares
    these records between the default and Marconi policies.
    """

    LOG_PREFIX = "MAMBA_ADMISSION_STATS "

    def __init__(self, policy: str):
        self.policy = policy
        self._stats: dict[str, MambaAdmissionStats] = defaultdict(
            MambaAdmissionStats
        )

    def note_candidate(
        self, rid: str, decision: MambaAdmissionDecision
    ) -> None:
        stats = self._stats[rid]
        field = f"{decision.kind.value}_candidates"
        setattr(stats, field, getattr(stats, field) + 1)
        if not decision.persist:
            stats.intermediate_skipped += 1

    def note_result(
        self,
        rid: str,
        kind: MambaCheckpointKind,
        *,
        inserted: bool,
        duplicate: bool,
    ) -> None:
        stats = self._stats[rid]
        if inserted:
            field = f"{kind.value}_admitted"
            setattr(stats, field, getattr(stats, field) + 1)
        elif duplicate:
            stats.duplicate_candidates += 1

    def finish(self, rid: str) -> dict[str, object]:
        stats = self._stats.pop(rid, MambaAdmissionStats())
        payload: dict[str, object] = {"policy": self.policy, "rid": rid}
        payload.update(asdict(stats))
        logger.debug("%s%s", self.LOG_PREFIX, json.dumps(payload, sort_keys=True))
        return payload
