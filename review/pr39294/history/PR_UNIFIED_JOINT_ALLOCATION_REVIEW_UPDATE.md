<!-- Proposed replacement after agreeing on #36729 integration and publishing the reviewed follow-up. Not the current remote PR body. -->

## Motivation

This follows up on #33091. Unified FULL/SWA pools share bytes, so separate capacity checks can each pass while the combined allocation cannot fit. Recovery must consider the complete paired demand.

The revision also addresses review feedback on recovery cost and cache preservation. Calling a mutating recovery routine during every victim check hides repeated preparation work; attempting a whole-cache fallback can discard useful prefixes for a provably impossible demand. Conversely, an immediate-capacity-only stop check can miss recoverable geometry and evict too much: in the temporal tri-pool regression, an END-only query dropped all96 cached tokens where one checkpoint's reclaim and a bounded FLOAT move suffice to preserve95.

Integration proposal: build on #36729's allocator-owned dispatch/planner and land this as a focused follow-up after agreement with its author. The separately identified FLOAT movement-gate correction is kept as a distinct patch.

## Modifications

- Keep the eviction callback pure. The two-pool path queries its allocator planner; the tri path can certify permitted END compaction or a specific HIGH-side FLOAT move from host metadata, without reading FLOAT hole positions.
- Perform preparation at explicit allocator boundaries and stop after sufficient allocator-visible reclaim. Tri recovery uses finite token/state phases and accounts for state freed by token eviction; no whole-cache fallback or repeated phase escalation.
- Reject provably impossible namespace/optimistic-byte demand before destructive eviction. A negative sufficient geometry query remains unknown rather than a proof of impossibility.
- Match ordinary unsharded paged-extend recovery to its exact required pages.
- Forward the optional callback directly through StreamingSession.
- Revalidate movement gates and actual geometry before execution. FLOAT movement owners reject movement while gated; successful preparation is verified against actual joint capacity.

## Accuracy Tests

These changes affect allocation/reclaim policy and data movement, so validation checks live bindings, cache retention, K/V and state markers, and relevant movement dependencies. They do not add model-forward math changes.

On candidate tree `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`, based on #36729 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`:

- Python CPU:276 tests and1597 subtests passed;10 GPU cases skipped in this CPU lane.
- Supported Rust cache lane:36 tests and138 subtests passed;2 session-cursor cases excluded.
- Production geometry probe:864 layouts,477 sufficient positives, no positive allocation failure or payload/accounting error. General fallback is forbidden for positive certificates.
- Registered temporal state-first regression: exactly1 FULL/1 SWA/1 state reclaimed,95 cached tokens retained; eager/lazy, Python/Rust, actual allocation and retained K/V/conv/temporal payload checked.
- Local RTX5090 controls pass allocation/data checks for two-pool and tri-pool, page sizes1/4, and independent movement gates. Eager writer ordering is demonstrated at an unfinished-event wait, followed by actual movement and replay of the same captured translating graph.
- Lazy writer ordering and real pending-reader source reuse remain inconclusive in the tested environment. Their passing allocation/payload/graph controls and CPU pending tests do not establish those GPU overlap properties. See the separate validation report for the exact scope.

The earlier model-output comparisons and serving results remain tied to the earlier PR revision. They are not new measurements of this candidate and do not establish general model-output equivalence.

## Speed Tests and Profiling

Recovery work is bounded structurally and tested through actual reclaim, preparation, and movement observations. The temporal regression now stops after one checkpoint instead of walking into a94-token leaf. These counters are not a serving throughput or latency claim.

No serving-speed measurement of this new candidate is provided yet. An observer-free matched pressure benchmark and any selected serving validation must use the final agreed integrated revision; the historical serving table cannot be relabeled as evidence for this revision.

## Checklist

- [x] Run all-files pre-commit on the local candidate.
- [x] Add registered regression tests and run the selected Python/Rust lanes.
- [x] Follow the existing allocator/cache interface and source style.
- [ ] Complete any final integrated-revision documentation/benchmark requirements agreed with maintainers.
- [ ] Publish the agreed integration, obtain required reviewer approvals and pass required CI.
