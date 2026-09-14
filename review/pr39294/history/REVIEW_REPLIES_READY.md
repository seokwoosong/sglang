# Draft comments — not posted

These describe the local integration candidate on #36729, not the current remote #39294 head. Update source/validation links when publishing an authorized branch update.

## Coordination comment on #39294

Thanks for pointing out the overlap with #36729. I tested the regression cases against its current head and prepared an integration candidate on top of that allocator-owned design.

The candidate preserves #36729's dispatch/planner and adds the missing regression coverage, early stopping after sufficient reclaim, and exact paged-extend demand. It also handles the tri-pool cases that remain unresolved on that branch, using a pure reclaim predicate and bounded token/state eviction phases. During validation I found a separate FLOAT movement-gate issue; that fix is kept as an independent patch.

Local validation passes: 229 CPU tests (1,489 subtests), the supported Rust lane (31 tests / 97 subtests), and all pre-commit checks. Focused GPU validation is the next step; these results do not establish serving performance or model-output equivalence.

@ZYHowell @ch-wan, would it work to land #36729 first and keep #39294 as a focused follow-up on the agreed implementation? I can share the separate patches/tests so we can settle the overlap before either change lands.

## C1 — mutating per-victim predicate

Agreed. In the integration candidate the eviction callback is a pure allocator query. Group draining and capacity recovery happen at explicit allocator boundaries, with no recovery call inside the victim loop.

For tri-pool recovery, the predicate can also recognize capacity recoverable by permitted END compaction; checking only immediate capacity dropped an extra prefix in the lazy 96-token/8-state regression. The matching executor performs the certified END operation and checks its postcondition. Unknown FLOAT layouts use the existing recovery ladder at bounded phase boundaries. The regression retains 95 prefixes and successfully allocates the request.

## C2 — impossible demand drops the whole cache

Fixed in the candidate with an allocator-owned virtual-ID/optimistic-byte guard before the first destructive walk. The tri path uses finite demand-derived token/state quotas and has no whole-cache fallback or repeated phase escalation. Oversized and locked/pinned cases assert that cached IDs and payloads remain intact.

The byte bound is only an impossibility guard, not a complete FLOAT layout solver. Ordinary unsharded paged extend now passes its exact page demand so an oversized conservative probe does not strand a feasible allocation.

## C3 — overlap with #36729

Agreed; I have built the integration candidate on #36729 rather than merging the conflicting dispatch designs. Its two-pool planner covers the basic joint-shortfall case, but the ported tests also exposed excess eviction on larger requests and tri-pool cases that still failed allocation. The candidate keeps its architecture and adds focused fixes/tests for those cases.

I propose landing #36729 first, then rebasing #39294 onto the agreed result. The coordination comment above asks for agreement on that sequence before publishing a conflicting implementation.

## C4 — redundant wrapper branch

Fixed in the integration candidate: `StreamingSession` forwards the optional callback unconditionally. The base interface and supported implementations share the same contract.

## C5 — query name hides mutation

Resolved together with C1: the callback now performs a pure sufficiency query, while the allocator's explicit `ensure_capacity` path owns mutation. The tri callback can certify END recovery but never runs that recovery itself.
