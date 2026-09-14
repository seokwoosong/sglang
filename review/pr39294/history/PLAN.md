# PR #39294 reviewer analysis and execution plan — revision 1

Status: awaiting review by Codex session `01a09f85-77de-79d1-a50d-cb7a530616c1`.
The user authorized sending this plan to that session and executing experiments and fixes only after its approval. No GitHub reply, push, PR closure, or branch-history rewrite is authorized by this plan.

## Reviewed source and complete comment inventory

- PR #39294: `276771eefca951ae6d55608415f2d795cbfabdae`, branch `fix/unified-joint-allocation`, parent base `14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6`.
- Production checkout: `/home/sukwoo24/sglang-eval-worktrees/unified-joint-allocation`, clean at inspection.
- PR #36729: `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`; fetched to local `refs/codex/review/pr-36729`, no source checkout changed.
- All GitHub API pages checked: 5 review threads, each one comment, all unresolved/current; one COMMENTED review from ch-wan; its body is empty. One top-level conversation comment from the author links #33091. There is no additional reviewer top-level comment.
- Review submitted 2026-09-14 07:54:49 UTC / 16:54:49 KST. API response files are saved beside this plan.
- The historical human-review sweep scanned all 40,110 threads: path/risk query matched 6 threads across 5 PRs; conversation query matched 137 PRs. Recurring concerns: allocator ownership, preserve locked/live data and prefixes, avoid unnecessary hot-path work, and provide tests/scripts that establish the claimed behavior.

## What the PR currently does

1. `evict_from_tree_cache` asks the unified allocator whether the entire FULL+SWA token demand is ready.
2. It attempts deferred-free application and layout recovery before eviction.
3. It passes `lambda: prepare_token_allocation(demand)` as `allocation_ready` into eviction.
4. `target_reached` invokes this callback during individual tree steps; an additional full-cache pass follows if the initial component shortfalls do not satisfy the joint demand.
5. Decode uses the same complete capacity check.

## Comment-by-comment analysis

### C1 — Mutating predicate and repeated compaction risk (reviewer P1)

https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049663

Accept the architectural concern. `common.py:174` injects a mutating prepare callback; `_evict_components.target_reached` calls it at lines 840, 874, 907. The callback can close/reopen queued free groups, call urgent flush on every chain member, and move physical pages. A query-looking API makes future call-site changes dangerous and exposes repeated work during pressure.

Important qualification: a callback invocation is not an actual flush or move. `_flush_deferred_free_group` no-ops without pending groups; `prepare_token_allocation` returns early on ready capacity, occupied logical slots, or positive byte shortfall; eager end flush no-ops; float flush is boundary absorption, not relocation. CPU probes on the exact PR observed 92 prepare calls for 87 evicted single-page leaves but zero `_relieve_for_alloc` calls. This confirms repeated checking, not 87 physical compactions. Do not claim the existing Inkling timing is causally explained by this comment.

Required outcome: a pure readiness predicate; mutating preparation at explicit recovery boundaries with a justified work bound. Preserve actual allocator-visible reclaim and early stopping. A plain rename does not fix repeated work. A chain epoch cache only suppresses repeated checks at unchanged state: every eviction can change the epoch, so it is not by itself a compaction bound.

Reviewer suggestion to prepare only before/after the walk must be tested on asymmetric holes, deferred frees, live tri-pool state, page-grid alignment, and closed move gates; a capacity query must not miss recoverable space and evict extra prefixes. The initial grouped-leaf probes here did NOT reproduce that concern: both modes retained 95/96 entries. Treat it as a validation obligation, not a demonstrated regression.

### C2 — Impossible demand can destroy the prefix cache

https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049666

Accept the undesirable behavior and fix it. An impossible 101-token demand in the real production-factory fixture drains all 96 entries and remains unready. The committed `test_impossible_target_exhausts_without_claiming_readiness` explicitly asserts the cache is empty; it protects the current policy rather than cache preservation. The test should be replaced/updated for the new desired contract.

Qualification: an exact base-helper ablation against the same pool/cache also drained all 96 entries for this specific oversized request. It is not accurate to claim that every such case newly regressed in this PR. Establish a before/after case separately if asserting incremental regression.

Stopping when a step frees nothing is not sufficient: the current walker already stops on `node_id is None and not made_progress`, and a useless cache-draining walk can make progress on every victim. Also, checking only before the SECOND pass cannot promise an intact cache if the first pass already evicted entries.

Required outcome: an allocator-owned, pure impossibility check before the FIRST destructive pass. At minimum cover demand exceeding true virtual-ID capacity and an optimistic empty-arena byte upper bound. Passing an optimistic bound is not proof of layout feasibility. Further reclaim-aware checks must account for locked/live state and actual evictable components without double-counting cascades. Distinguish component shortfalls, cumulative eviction targets and absolute allocation demand.

Caller caveat: `alloc_paged_token_slots_extend` asks for `extend_num_tokens + batch_size * page_size * shard_size`, a conservative bound. A rejected oversized PROBE is not proof that the eventual exact allocation is impossible. Test this separately; do not introduce an unconditional caller failure based on that bound. Prefer allocator-owned exact page demand where necessary.

### C3 — Overlap and incompatible dispatch with #36729

https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049670

Accept the need for one coordinated architecture. #36729 changes 32 files, delegates common eviction to allocator methods, and implements two-pool `reclaim_plan(n,n)` plus `ensure_capacity`. Our callback path occupies the code it removes. A manual merge could keep textual code while dropping intended semantics.

Do not assume #36729 already replaces the entire PR:

- At inspected head, its tri-pool `evict_to_free_tokens` (line 1422 in saved source) STILL uses the four-iteration loop and calls `SWATokenToKVPoolAllocator.evict_to_free_tokens`, not the two-pool planner. Its tri-pool `can_reserve` rejects evictable-demand/empty-pool planning cases. We need live tri-pool/float tests before claiming equivalence.
- Its scheduler calls `check_decode_capacity` with `requests` and `spec_algorithm`; our override accepts only `num_tokens`/`tree_cache`. Carrying our override across this interface blindly can raise TypeError.
- It already has a live `reclaim_plan` exhaustive-target unit test in `test_multi_ended_allocator.py`; the missing piece is our real cache/controller joint-allocation interaction, not all planner testing.
- Factory arguments and capacity semantics changed (two-pool `unified_total_bytes` becomes `total_bytes`; dynamic capacities replace some static partition limits). Port the intended invariant, not old helper method names or obsolete static-cap assertions.

Recommended direction: first run a behavior-based test port on an isolated #36729 checkout. If its two-pool behavior covers the defect, prepare regression coverage there and isolate any remaining tri-pool, deferred-reclaim, decode, or recovery-cost fixes. Do not close #39294 or send coordination messages without separate user authorization. Locally prepare both a patch and a concise integration recommendation.

### C4 — Redundant StreamingSession branch

https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049675

Accept. Current in-tree definitions in BasePrefixCache, UnifiedRadixCache and StreamingSession all accept the optional keyword. Forwarding `allocation_ready=None` is semantically equivalent to omission for these implementations. If the callback interface is retained, replace the branch with unconditional forwarding. If the final shared architecture removes the callback, remove the added wrapper plumbing instead. Do not introduce a compatibility fallback for hypothetical external implementations.

### C5 — `allocation_ready` hides side effects

https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049679

Accept; solve together with C1. The pure readiness API and the mutating preparation API should be separate at both caller and callee. Documentation must state whether a capacity result is immediate, schedulable, or impossible after reclaim. Renaming to `try_recover_allocation` would improve readability but leave C1's work bound unanswered.

## Completed read-only probes (not new production tests)

`probe.py` and `probe-results.json` reproduce 12 CPU cases on current #39294. Both eager and lazy results match:

| Probe | Cached before/after | Ready | Prepare / recovery ladder calls |
|---|---|---|---|
| Current, demand 4 | 96 / 95 | true | 5 / 0 |
| Current, demand 90 | 96 / 9 | true | 92 / 0 |
| Current, impossible demand 101 | 96 / 0 | false | 100 / 0 |
| Exact base helper ablation, demand 101 | 96 / 0 | false | 0 / 0 |
| Current, free-group scope, demand 4 | 96 / 95 | true | 5 / 0 |
| Pure callback with before/after prepare, same grouped probe | 96 / 95 | true | 2 / 0 |

All final byte-accounting checks passed. These are CPU accounting/call-count observations, not GPU timing, payload-equivalence certification, or proof of worst-case compaction frequency.

## Proposed execution stages and acceptance gates

### A — Freeze source and build the regression/architecture comparison

After reviewer approval, create isolated worktrees for the exact #39294 head and #36729 head. Preserve the user worktrees and all existing experiments. Save source SHA, dirty diffs, environment identity, and every command/result.

1. Port the 13 logical regression scenarios to #36729, adapting only fixture/API differences and identifying any changed capacity CONTRACT explicitly. Preserve real production pools, cache entries, distinct payloads, and exact survivor assertions. Add no assertion on a method solely because #39294 introduced it.
2. Add focused negative scenarios for impossible empty-pool byte and virtual-ID demands, locked/non-evictable occupants, and conservative page-demand probes. Compare whether useful prefixes survive unsuccessful probes.
3. Instrument preparation attempts, actual urgent flush calls, actual moved pages/bytes and elapsed time separately. Test deep eviction, unequal FULL/SWA entry sizes, one-sided holes, page sizes 1/4/16, live tri-pool state, move gate disabled, and repeated unchanged-state probes. CPU first; no throughput claim from counters alone.
4. Record where pure before/after readiness succeeds and whether it causes excess eviction or misses recoverable layout. Preserve failing fixtures as evidence; do not assume a failure mode from the concern alone.

Output: `comparison.md`, per-case result JSON, a coverage matrix against C1-C5, proposed target architecture and exact implementation delta. No production branch rewrite or remote update.

### B — Architecture checkpoint with the specified reviewer session

Return the Stage A evidence to the SAME reviewer session and receive approval for the concrete implementation route. This is essential because C3 may change where the fix belongs; do not duplicate two allocators' capacity models or guess which PR maintainers will merge first.

Default route for consideration: use #36729's allocator-owned dispatch/planning for the two-pool overlap, port our real regression coverage, and implement only the verified remaining tri-pool/deferred/decode cases. An independently shippable #39294 fix remains a fallback if the reviewer identifies a concrete reason; any such design must satisfy C1/C2/C5 and clearly document its integration with #36729.

### C — Implement only the approved concrete delta

1. Reject provably impossible requests before destructive eviction; ensure a rejected conservative probe does not prevent an exact feasible allocation retry.
2. Keep readiness side-effect free. Use planned reclaim followed by explicit bounded preparation, or another reviewer-approved allocator-owned policy justified by Stage A. Do not merely add an arbitrary victim count or epoch cache and claim repeated compaction is solved.
3. Preserve normal victim selection, early stopping when a joint demand becomes satisfiable, locked entry protection, pending-free visibility, and the explicit count-based `evict` contract.
4. Preserve tri-pool float page-grid feasibility and safe payload movement/event ordering. Avoid transferring #36729's two-pool byte equation to tri-pool blindly.
5. Apply the wrapper simplification or remove unnecessary plumbing; update docs/PR text only to final implemented semantics.

### D — Validation and final review

- Run targeted CPU tests: joint allocation, allocation eviction, end/tri allocators, capacity memo/byte accounting, move gates, and actual session behavior as touched. Run Python and Rust TreeCore lanes if shared cache logic changes. Load current test guidance before adding registered tests.
- Existing baseline count is 162 tests/117 subtests for the original four-file suite (plus 35 page-interleave tests); do not carry those counts to a changed head without rerunning.
- GPU: distinct live FULL/SWA payloads through actual recovery in eager/lazy mode, live-state tri-pool cases, and graph replay only if the changed execution path needs it. Verify device availability/ownership before launching. Do not disturb unrelated model servers.
- Measure the reproduced pressure case with identical source/environment/workload and separate observer-free timing from instrumentation. The original 80-run serving data remain tied to old SHAs; do not repeat the entire matrix by default and do not claim the old Inkling latency is explained without evidence.
- Run diff checks, relevant formatting, and required all-files pre-commit in the isolated candidate; inspect all mutations.
- Send final patch/results to the specified reviewer session for acceptance. Prepare local review-reply drafts mapping C1-C5 to code/tests and limitations. No GitHub comments, push, rebase of the submitted branch, or PR closure as part of this authorization.

## Requested decision from the reviewer session

Please independently assess the five comments, the qualifications above, and this staged plan. Approve or revise Stage A's experiments and the architecture checkpoint; identify missing concrete success/failure criteria. If approving, state explicitly which stages may execute without another review. Review output should be written to `PLAN_REVIEW.md` beside this plan, with a clear decision and this plan's SHA256. Do not implement or mutate either PR yourself; the requesting session owns implementation after approval.
