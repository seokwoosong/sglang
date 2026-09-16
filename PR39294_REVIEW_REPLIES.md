# PR #39294 — current reply drafts

These are unpublished drafts for the review branch. The published PR has not been updated. Local Codex review is not maintainer approval; posting these replies remains a separate step.

## C1 — Mutating recovery inside the eviction predicate

[Original comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049663)

Agreed. The follow-up now builds on the landed #36729 allocator dispatch and reclaim planner. The eviction callback is a pure sufficiency query; it does not flush deferred frees or compact pools. Preparation happens at explicit allocator boundaries, including after a reclaim phase only when that phase actually frees capacity.

The tri-pool path uses at most one token-reclaim phase and one state-reclaim phase, with demand-derived bounds. Its pure query can recognize capacity recoverable through permitted END compaction or a specific FLOAT move. The regression retains 95 cached tokens while reclaiming one FULL token, one SWA token and one state, then successfully allocates four tokens and verifies retained KV/state data. This removes per-victim recovery calls; it does not imply that eager frees themselves never move data.

## C2 — Whole-cache fallback for an infeasible demand

[Original comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049666)

Agreed. The whole-cache fallback has been removed. Allocator-owned ID limits and optimistic byte bounds reject provably impossible demands before a destructive eviction walk; the remaining tri-pool phases use finite quotas derived from the demand. Regressions check cache preservation for oversized demands and locked/exhausted cases.

Ordinary unsharded paged extend now forwards its exact new-page demand, so a conservative speculative probe cannot reject an allocation that fits. A passing byte bound is not treated as proof of FLOAT-layout feasibility: actual capacity is still checked after preparation.

## C3 — Overlap with #36729

[Original comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049670)

Thanks for pointing out the overlap. #36729 has now landed, and this follow-up is based on upstream `a3bf25dc62`, which contains its merged implementation (`2929a39927`). It uses that allocator-owned dispatch, shared-byte model and two-pool reclaim planner.

The original FULL 4 / SWA 4 / joint 3 issue is covered by that planner; I am no longer proposing a competing recovery capability. The follow-up retains the regression coverage and addresses additional deferred-reclaim/early-stop cases, bounded tri-pool recovery, and movement-gate/pending-event correctness. The two-pool integration is extended at the reclaim boundary, so this is reuse of the design with targeted changes, not a claim that every line from #36729 is unchanged.

## C4 — Redundant StreamingSession branch

[Original comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049675)

Agreed. StreamingSession now forwards the optional `allocation_reclaim_satisfied` callback unconditionally. The wrapper regression cases are included in the passing Python suite.

## C5 — Query-looking name hides mutation

[Original comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049679)

Agreed. `allocation_reclaim_satisfied` now receives a pure sufficiency query; mutation remains in explicit allocator preparation calls. The callback contract also states that a negative result is not proof that the allocation is impossible. This addresses the naming concern and the per-victim recovery issue together.

## Validation summary draft

Local validation of the review candidate is complete within the reported scope:

- Python: **296 tests / 1,669 subtests passed**, with 10 GPU skips.
- Rust: **26 tests / 139 subtests passed**, with one session exclusion.
- Synchronized CUDA: **10/10 cases passed**, including actual retained SWA/Mamba relocation and gate close/reopen.
- Same-event and distinct-event confirmation: **each 9 normal PASS + 3 expected negative detections** across FULL+Mamba, FULL+SWA and tri-pool layouts. All distinct normal cases observed an outstanding E2 urgent wait and actual two-source reuse. The distinct negative detects premature selection; it does not deliberately race a reader.
- The tri-pool ready-path CPU fixture met the predefined performance criteria; the pressure-path tradeoff is retained in the report.

These helper fixtures use FULL-owner moves, page size 1, lazy compaction and prebuilt indices. They do not establish natural production-flush reachability, overlapping writes, all moved owners, graphs/distributed operation or model-output equivalence. In all distinct confirmation cases E2 completed during allocation. Earlier serving results remain tied to their earlier revisions.

[Current evidence and exact scope](review/pr39294/gap-closure-20260917/RESULTS.md). These replies remain drafts; review-branch publication does not update the published PR or resolve its threads.
