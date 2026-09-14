# Proposed review replies — local drafts, not posted

These describe the integration candidate on #36729, not the currently published #39294 head. Link the reviewed patch and final GPU report before posting. Do not mark remote threads resolved until the change is published and the reviewer accepts it.

## Coordination message

Thanks for pointing out the overlap with #36729. I tested its current head and prepared a follow-up using its allocator-owned dispatch and two-pool planner.

The remaining changes make the reclaim callback pure, reject provably impossible demands before eviction, and stop reclaim once a specific bounded recovery can satisfy the joint allocation. The tri-pool path includes a FLOAT geometry check: a temporal-state regression showed that an END-only check could evict all 96 cached tokens, while the corrected path reclaims one token/state checkpoint and retains 95. I also kept an independently discovered FLOAT movement-gate fix separate.

Local CPU validation passes (276 tests / 1,597 subtests; supported Rust lane: 36 / 138; all pre-commit checks). These are allocator correctness checks; the old serving measurements are tied to the earlier patch.

@ZYHowell @ch-wan, would you prefer to land #36729 first and keep #39294 as the focused follow-up? I have separate patches and regression tests ready to share so we can agree on the integration before either change lands.

## C1 — mutating callback

Agreed. The integration candidate uses a pure allocator sufficiency query in the victim loop. Draining and movement occur at explicit preparation boundaries. The two-pool path uses your allocator-owned planner; the tri path can certify permitted END compaction or a specific HIGH-side FLOAT move without reading hole positions or moving pages in the callback.

The temporal state-first regression asserts that the first 1/1/1 FULL/SWA/state reclaim stops the walk, the matching explicit recovery allocates four tokens, and 95 cached tokens plus their KV/state payload survive. Preparation remains bounded by the existing token/state phases.

## C2 — destructive impossible-demand fallback

The candidate rejects provably impossible ID/optimistic-byte demands before the first destructive walk, and removes the whole-cache fallback. Token/state phases use finite demand-derived quotas. The byte check is an impossibility bound, not a complete FLOAT feasibility test.

The ordinary unsharded paged-extend path passes its exact page demand so an oversized conservative probe does not prevent a feasible allocation. Oversized and locked/pinned tests preserve cached bindings and payloads.

## C3 — overlapping PRs

Agreed. The local candidate is built on #36729 and preserves its dispatch and planner. Porting the actual allocator/cache regressions exposed remaining excess-reclaim and tri-pool cases, which the separate follow-up patches address. I propose agreeing on the landing order with @ZYHowell before updating the published implementation. This thread remains open pending that agreement.

## C4 — wrapper simplification

Applied in the candidate: `StreamingSession` forwards the optional callback unconditionally, matching the base interface and supported implementations.

## C5 — query side effects

Addressed together with C1. `_token_reclaim_satisfied` is a pure sufficient query; `ensure_capacity` owns preparation and verifies actual joint capacity. A negative sufficiency result remains unknown rather than being treated as allocation impossibility.
