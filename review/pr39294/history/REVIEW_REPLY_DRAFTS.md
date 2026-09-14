# Local reply drafts — not posted

## C1 and C5

Thanks, I agree that the eviction stop predicate should be side-effect free. I
separated the pure reclaim-satisfied check from explicit allocator preparation
in a local two-pool candidate based on #36729. The predicate accounts for capacity
recoverable by allowed compaction, while preparation happens after the walk.

I also checked the suggested immediate-readiness-only alternative. It over-evicts
one prefix in a lazy tri-pool case, so I am keeping that case as a regression and
working out the remaining tri-pool contract separately. The measurements distinguish
prepare calls, urgent flushes, and movement caused by ordinary eager eviction.

## C2

Agreed. An impossible exact demand should not drain the prefix cache. I reproduced
the cache loss and added preservation checks. I also found a related caller issue:
rejecting an oversized conservative prefill probe can leave a feasible exact
allocation without the reclaim it needs. The local two-pool candidate uses the
exact rounded page demand for ordinary unsharded extend and covers that path.

## C3

I compared the regression scenarios against #36729 at `6c8bbdf610`. Its two-pool
planner addresses the impossible-demand case, but the cache integration tests
also exposed excess eviction after cascades and queued frees. I prepared a local
candidate reusing that planner with a pure early-stop predicate, entry deferred
drain, and exact extend demand.

The tri-pool path still has a separate gap: the 96-token/8-state, demand-4 case
fails to allocate at that head while this PR succeeds. I have preserved the
reproductions and distinct-payload checks so we can agree on the final split
without dropping that coverage.

## C4

Agreed. The local candidate forwards the optional keyword unconditionally, and
the test now checks allocation and prefix preservation through the real session
wrapper rather than only checking a mock call.
