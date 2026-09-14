@ch-wan @ZYHowell, I’ve prepared a follow-up on top of #36729 that preserves its allocator-owned dispatch/planner and addresses the pure-query, bounded-reclaim, and wrapper comments. The remaining tri-pool changes include regression coverage for early stopping and retained KV/state data.

Local CPU/Rust tests and the scoped GPU allocation/data checks pass; the validation report explicitly retains the unresolved lazy-ordering and pending-reader coverage limits.

Would you prefer landing #36729 first and keeping #39294 as the focused follow-up? I have separate patches and tests ready to share so we can agree on the integration before updating the published implementation.
