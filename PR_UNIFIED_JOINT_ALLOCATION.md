Title: fix(unified-memory): bound reclaim and preserve tri-pool recovery state

## Motivation

This is a follow-up to the landed #36729 shared-byte allocator. Its two-pool planner already resolves the original case where separate FULL/SWA availability overcounts the shared arena. This patch keeps that design and addresses the remaining reclaim and recovery cases.

Allocation-driven eviction can stop before deferred frees become allocator-visible, continue after sufficient capacity has been reclaimed, or discard cached prefixes when permitted tri-pool layout recovery can satisfy the request. Additional regressions exposed stale capacity under dynamic movement gates and loss of earlier pending source batches sharing one unfinished event.

## Modifications

- Extend the existing two-pool reclaim boundary to expose grouped frees and stop reclaim once a pure allocator sufficiency callback reports enough capacity. Explicit count-based eviction retains its quota contract.
- Bound tri-pool token/state reclaim phases, reject provably impossible ID/byte demands before destructive walks, and prepare permitted END/FLOAT layouts at explicit boundaries. Preserve cached KV/state data when recovery is sufficient.
- Use exact new-page demand for ordinary unsharded extend; retain conservative demand for sharded paths.
- Respect FLOAT movement gates and avoid caching capacity that depends on a mutable gate. Preserve every pending source batch associated with the same unfinished event.
- Keep the tri-pool immediate-ready path short while retaining FULL/SWA ID limits and active deferred-group handling. Forward the optional eviction callback unconditionally through StreamingSession.

The dispatch, byte accounting and two-pool planning design from #36729 remain the foundation; this follow-up makes targeted changes to that implementation.

## Accuracy Tests

Validation focuses on allocation/reclamation behavior, retained KV/state payload, actual physical source reuse, movement gates and deferred event bookkeeping. Python: **296 tests / 1,669 subtests passed**, with 10 GPU skips. Rust: **26 tests / 139 subtests passed**, with one session exclusion. Synchronized CUDA: **10/10 cases passed**, including retained SWA/Mamba relocation and gate close/reopen. Each same-event and distinct-event confirmation matrix contains **9 normal passes and 3 expected negative detections** across FULL+Mamba, FULL+SWA and tri-pool layouts. Distinct normals all recorded a real urgent wait while E2 was pending; both physical sources were subsequently reused with payload/accounting checks. The distinct negative detects premature selection using a historical GPU free-list snapshot; it does not demonstrate a racing overwrite.

[Detailed results, tested revisions and limitations](review/pr39294/gap-closure-20260917/RESULTS.md) and [reproduction artifacts](review/pr39294/gap-closure-20260917/REPRODUCE.md).

No current-revision serving throughput, task-accuracy or general output-equivalence claim is made. Earlier serving measurements belong to the earlier implementation; a later serving attempt stopped at a baseline model/config-loader incompatibility before inference. The GPU helper fixtures do not establish natural scheduler/flush reachability, all moved owners, CUDA graphs or distributed execution.

## Speed Tests and Profiling

A fixed 12-block, 36-process B/A/C allocator CPU experiment compared the landed-#36729 baseline (B), the existing follow-up (A), and the immediate-ready candidate (C). All blocks were valid; none were excluded.

| Three-pool scenario | C versus A, median paired change | C versus B, median paired change |
| --- | --- | --- |
| Ready | −13.37%; −11.3515 µs | +0.35%; +0.2455 µs |
| Pressure | +4.42%; +23.6355 µs | +7.28%; +38.1565 µs |

Ready-path C/B upper 95% interval bounds were +4.08% and +2.8505 µs, within the predefined 10% / 5 µs limits; the C/A ratio upper bound was 0.8898. The pressure increase is retained in the report; not triggering the predefined combined median control rule is not a no-regression claim. The earlier C1 candidate missed its acceptance limit and was not adopted. These are local allocator fixture measurements, not serving-speed measurements, and intervals are not adjusted for adaptive candidate development.

## Checklist

- [x] Format code with pre-commit (all-files passed with the new evidence documents staged).
- [x] Add focused registered unit regressions.
- [x] Update current result and reviewer-response documents.
- [ ] Provide model accuracy and serving speed results (not established on this revision; scoped allocator evidence above).
- [x] Complete final code-style/source-adoption review.

Maintainer approval and required GitHub CI remain separate from local preparation.
