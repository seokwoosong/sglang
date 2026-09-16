# Unified-memory follow-up validation — 2026-09-17

## Purpose and source

This work closes the ready-path CPU-cost question and adds sensitive, real-CUDA pending-event evidence for the follow-up to landed #36729. It does not replace the allocator design from that PR.

| Label | Source | Meaning |
| --- | --- | --- |
| B | `a3bf25dc620f31fc672aeced1465d6fe7c81d28f` | Upstream baseline containing #36729 squash `2929a39927a3943cee03e498f4e5f651185f1b1f` |
| A | `15ccf48036bf5de0ec0aa886dedaa74bbdea8123` | Existing follow-up, including dynamic-gate and shared-event fixes |
| C2 | A + `C2.patch`, SHA256 `4451ee8cc9ddb025f13a8ea954370c7d20ebac6310c454cd438379ff9433e5e0` | Tested immediate-ready optimization and regression |

C2 changes only `unified_hybrid_swa.py` and `test_unified_tri_joint_reclaim.py`. It checks immediate capacity before constructing a reclaim plan, retains both FULL/SWA ID limits and page rounding, flushes active deferred groups, skips a no-op flush helper when no group exists, and defers an unused import until recovery is needed. Its ready regression covers page sizes 1/4, eager/lazy compaction, inactive/active-empty free groups, closed movement gates, preserved cache bindings/payload, successful allocation and accounting.

## Completed correctness and performance

| Evidence | Result | Scope |
| --- | --- | --- |
| T14 Python | 296 tests / 1,669 subtests PASS; 10 GPU skips | 17 selected allocator/cache/interface test files |
| T14 Rust | 26 tests / 139 subtests PASS; one session exclusion | Three selected files using the verified compiled Rust backend |
| T17D CUDA | 10/10 worker PASS, all cleanup PASS | A/C2 ready, real retained SWA/Mamba relocation with reclaim, and FLOAT gate close/reopen |
| T16 CPU cost | 36/36 processes, 12 valid balanced B/A/C blocks, no exclusions | Five allocator fixture scenarios; original acceptance rule unchanged |
| Shared-event confirmation | 9 normal PASS + 3 expected negative detections | M2/S2/T3, actual pending E2 waits, two-source physical reuse and payload/accounting |
| Distinct-event decision confirmation | 9 normal PASS + 3 expected negative detections; all nine normals recorded an outstanding E2 urgent wait | New immutable historical-decision observer, separate registry |

Tests and subtests are different reporting units. Admission cases are part of their confirmation matrices, not additional independent coverage. Runner COMPLETE alone is never counted as functional coverage when PARTIAL is permitted.

### CPU cost

Each of 12 fresh-process blocks contains B/A/C in balanced order. Each process runs the same five scenarios with two warmups and five measured samples. Paired process medians, not individual calls, are the comparison units. Initial state, configuration, source and environment are checked. Profiling runs are separate from timing.

| Scenario | C2/A median ratio; delta | C2/B median ratio; delta |
| --- | --- | --- |
| M2 ready | 0.99224; −0.3525 µs | 1.01389; +0.6060 µs |
| S2 ready | 0.99928; −0.0535 µs | 0.98552; −1.0675 µs |
| T3 ready | 0.86634; −11.3515 µs | 1.00346; +0.2455 µs |
| S2 pressure | 0.99359; −1.8995 µs | 0.99351; −1.9235 µs |
| T3 pressure | 1.04418; +23.6355 µs | 1.07280; +38.1565 µs |

Ready-path C2/A 95% ratio interval: [0.83812, 0.88979]. C2/B: [0.87284, 1.04082], delta [−10.7405, +2.8505] µs. This meets the predefined improvement criterion and baseline upper limits of 10% / 5 µs. The first C1 candidate missed the 5 µs upper-delta limit (6.1715 µs) and was not adopted; C2 removed measured no-op dispatch/import work without dropping ID checks.

T3-pressure is not a no-overhead result: C2/A delta interval [−2.219, +60.064] µs; C2/B [11.8375, 65.5245] µs. Its median did not trigger the predefined combined >10% and >5 µs investigation rule. Retain this tradeoff; do not describe every path as faster. Intervals are descriptive and not adjusted for adaptive candidate development. These are CPU allocator fixtures, not serving throughput or latency measurements.

## What the asynchronous evidence means

All new helper fixtures use original move/drain helpers, controlled streams, prebuilt source/destination index tensors, FULL-owner moves, page size 1 and lazy compaction. M2 means FULL+Mamba/state, S2 means FULL+SWA, T3 means FULL+SWA+Mamba/state. These are real GPU memory pools created by production factories, with retained data markers and real reallocation; they are not model-serving runs.

The shared-event negative deliberately drops an earlier batch association and detects an unreclaimed completed-reader source in the actual free-list, then repairs the fixture only after reader completion. Normal and negative runs include real outstanding-event waits, actual two-source reuse and payload checks. This is sensitivity to lost reclamation, not an observed corruption result.

The original distinct-event observer synchronously read the GPU free-list before urgent drain. In this environment, that observation finished only after E2, losing the required interval. Those runs remain PARTIAL. Allocator-free late sleep/cat/cat-out/index-select diagnostics also lost the interval; equal-cycle sleep decomposition, event query polling, and requested queue/priority changes did not fix it. CUPTI failed initialization and produced no valid GPU trace. These observations do not prove a WSL/driver cause.

The new distinct protocol clones the actual free-list on the same consumer immediately after the nonurgent selection, without waiting for readback. E2 queries bracket that decision and clone submission. In normal cases, the original urgent drain then records a real consumer wait while E2 is pending. Allocation and payload writes precede fixture host-value checks; final readback inspects the immutable historical selection, both physical source reuses, reader/retained/new data, mappings and accounting.

The distinct negative assigns E1 to the second batch while independently scheduled E2 is pending. Its actual historical free-list includes source two, detecting premature **selection**. A real protective wait precedes writes; no racing overwrite is attempted. This does not claim the snapshot completed or an unsafe GPU write executed while E2 was pending. In all twelve distinct confirmation cases, E2 had already completed by the end of allocation, so it does not establish host-synchronization-free allocation or overlapping writes. All nine normal cases directly observed the original urgent drain wait on E2 while pending; the three negative controls used an explicit protective consumer wait before reuse.

## Environment and limits

One RTX 5090 32 GiB under WSL2; Python 3.12.3, torch 2.13.0+cu130, sglang-kernel 0.4.7, FlashInfer 0.6.14. Frozen package/source/context manifests accompany the results. Final confirmation uses the original default CUDA flags and native allocator backend. Queue/priority variants are diagnostics only and do not change production defaults.

The original eight-hour cutoff was reached while the conversation was paused; the resumed T17 launch was rejected before any child ran. The user then explicitly removed that time limit. A later prelaunch headroom rejection also launched no workload. Reviewed continuation retained finite child/ticket deadlines, resource checks and owned-process cleanup; initial admission changed from 2048 to 2560 MiB used while the sampled whole-device stop remained 3072 MiB. The ten CUDA regression cases recorded a peak of 2832 MiB. Sampling is not a continuous hardware memory cap.

Natural production flush/scheduler reachability, all moved owners, page sizes beyond the specific GPU fixtures, graphs and distributed concurrency are not established by this matrix. CPU modeled-event regressions cover additional layouts/page sizes but are separate evidence. Current-revision real-serving validation remains unavailable: the prior fixed baseline model/config-loader attempt failed before inference. Earlier serving output differences and timing results remain tied to their earlier revisions and are not evidence for this head. No general task-accuracy, output-equivalence or serving-performance claim is made.

## Final handoff

The canonical review branch is prepared with the exact tested C2 source/test bytes; all twelve changed source/test file hashes match TESTED_SOURCE_FILES.json. All-files pre-commit passed with the complete new evidence bundle staged and applied no edits; both staged and unstaged diff checks passed. The [designated-session final review](FINAL_REVIEW.md) approved scoped source/test adoption and review-branch preparation, conditional on two precise wording corrections, which have been applied. Final publication follows the verified source/check checkpoint. Local review approval is distinct from maintainer approval and GitHub CI. The published PR and review comments are not modified as part of this archival review-branch push.

The adopted code/regression commit is `e28633f640c49388cd491bc1f0eeb09c76c9bbb4`. Its binary source/test patch is byte-identical to the tested C2 patch; [adoption identity](ADOPTION.json) records the verification. The result bundle and reply drafts are an archival review-branch handoff. Required GitHub CI and maintainer decisions remain for the actual PR submission.
