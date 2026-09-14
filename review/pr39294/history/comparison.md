# Stage A CPU evidence and Stage B implementation proposal

Status: submitted for follow-up review; no production fix/GPU run performed.
Authority: PLAN_REVIEW.md approves isolated CPU comparisons and this checkpoint.

## Frozen arms and method

| Arm | HEAD | Worktree |
|---|---|---|
| #39294 | 276771eefca951ae6d55608415f2d795cbfabdae | /home/sukwoo24/sglang-eval-worktrees/review39294-stage-a |
| #36729 | 6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb | /home/sukwoo24/sglang-eval-worktrees/review36729-stage-a |
| Pure callback ablation | #39294 HEAD | same source, explicit harness-only callback substitution |

These are branch behavior comparisons, not proof of regression relative to each
branch's different base. The pure arm substitutes a side-effect-free immediate
FULL/SWA/joint predicate during eviction and performs explicit post-walk prepare;
the original pre-walk prepare remains. It intentionally has an extra post-prepare
even on a ready fast path, so it is an experimental arm, not a proposed patch.
The session-wrapper case in this arm intentionally exercises the unmodified
wrapper/helper and is not evidence of a pure callback session implementation.

Commands, for each arm (`pr39294`, `pr36729`, `pr39294_pure`), run from its worktree:

```bash
PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer \
 /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B \
 /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/stage_a.py ARM
PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer \
 /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B \
 /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/stage_a_edges.py ARM
```

The main harness records 45 case rows/arm; the focused supplement records 11/arm.
Some original tri test methods contain subcases, so rows are not comparable to
unittest test/subtest counts. A failed original method stops at its first failure;
the supplement constructs key tri variants independently. Artifact hashes and
counts are in stage-a-manifest.json. Source worktrees remain clean.

Two-pool paired fixtures have the same 6400 * page_size byte arena, 32 * page_size
FULL/SWA page bytes, sink/padding and 96 paired pages; exact per-arm geometry is
saved in case JSON. Tri fixtures also record exact geometry: 6784 bytes for the
conv-only live-state cases, 7808 bytes for temporal state. The unequal-hole case
uses explicit 8192 bytes and 32/128 byte entries. #36729's explicit-budget capacity
contract differs; old static-partition assertions are not a universal oracle.

## Observed behavior

| Case | #39294 | #36729 | Pure callback ablation |
|---|---|---|---|
| Ready 3-page request, 96 cached pages | allocates, retains 96 | same | same |
| Joint 4-page request | allocates, retains 95 | same | same |
| Joint 6-page request | retains 93 | retains 91 | retains 93 |
| Deep 90-page request | retains 9 | retains 0 | retains 9 |
| Impossible 101/200-page request | fails allocation, loses all 96 | fails allocation, retains 96 | loses all 96 |
| All entries locked | fails allocation, retains all | same | same |
| Queued ordinary/representative free, 94 cached | allocates, retains 94 | allocates, retains 93 | retains 94 |
| Real StreamingSession wrapper | allocates, retains 95 | same | unmodified path retains 95 |
| Unequal one-sided holes, move gate open | recovers and allocates; live payload preserved | same | same |
| Same holes with closed move gates | fails temporarily; allocates after gate opens | same | same |
| Tri lazy, 96 tokens + 8 live states, demand 4 | allocates, retains 95 | allocation fails, retains 96 | allocates, retains 94 |
| Tri eager, same resources/demand | allocates, retains 95 | allocation fails, retains 96 | allocates, retains 95 |
| Tri conv-only, 80 tokens + 24 states, demand 12 | allocates after leaf-sized eviction | allocation fails | allocates |
| Tri nonempty temporal states | allocates; live KV/state payload preserved | same in tested case | same |
| Overestimate 400, exact 396, empty page-4 arena | actual caller succeeds | succeeds | succeeds (unmodified caller) |
| Same extend, 2 reclaimable cached pages | actual caller succeeds | caller raises prefill OOM with 8 evictable tokens | succeeds (unmodified caller) |

The two-pool eviction and group results reproduce in eager/lazy modes and page
sizes 1/4/16 as applicable. Survivors are matched by original key and stable
virtual IDs and checked for distinct K/V values after the final real allocation.
Tri supplements check live K/V plus all still-bound conv/temporal state IDs,
including states outside the prefix cache. Byte accounting is also checked.
No payload corruption was observed in these executed comparisons. This is not
proof of GPU ordering or model-output equivalence.

The extend comparison runs the real Python caller and allocator path, replacing
only the external Triton `alloc_extend_kernel` with a documented CPU reference.
It establishes the Python eviction/retry failure, not GPU kernel correctness.
The exact 396-token request needs 99 pages; the conservative 400-token probe
exceeds the 99-pair arena limit. With two occupied pages the exact request needs
eviction. #36729 skips impossible-probe eviction but never retries reclaim using
the exact page demand, so the eventual alloc_extend fails.

## Recovery work, without throughput claims

Instrumentation covers helper through final allocation, including controller
drains, group drains, urgent flush calls/no-op returns and movement kernels.
For eager page-1 paired demand 90:

| Arm | prepare/ensure attempts | controller drains | actual moved pages | survivors |
|---|---:|---:|---:|---:|
| #39294 | 92 | 174 | 48 | 9 |
| #36729 | 2 | 192 | 48 | 0 |
| Pure ablation | 2 | 174 | 48 | 9 |

Eager eviction itself moves data. These are NOT 48 callback-triggered urgent
compactions. #36729's two ensure calls are helper plus alloc, not two expensive
compactions. Conversely, lazy tri demand 4 in #39294 runs one recovery ladder,
one urgent call per member, and moves one 48-byte Mamba state page; the pure arm
skips this recovery opportunity and evicts a second prefix instead. This is a
concrete counterexample to pure immediate-readiness plus boundary-only prepare.

Elapsed values in JSON are instrumented CPU timings and are not performance
claims. Observer-free GPU/serving timing is NOT_RUN, outside Stage A approval.
The pure immediate predicate leaves recorded mappings/layout/free queues unchanged
over 100 repeated queries. Mutation of internal capacity memo caches is not a
payload/layout mutation and is not treated as compaction.

## Mapping all 13 original test scenarios

| Original method suffix | Portable evidence / disposition |
|---|---|
| byte_shortfall_skips_recovery_until_first_sufficient_eviction | paired demand 4 across page/eager/lazy plus actual recovery counters |
| byte_bound_allows_recovery_from_unequal_peer_holes | original API-adapted case + distinct-payload open/closed gate supplement |
| joint_shortfall_enters_eviction_when_individual_targets_fit | paired demand 4, exact keys/payload/alloc |
| joint_shortfall_outlives_individual_eviction_quotas | paired demand 6; #36729 excess eviction observed |
| float_recovery_preserves_prefix_when_full_band_is_short | original prepare/eviction/decode case with distinct payload; #36729 fails prefix preservation |
| tri_pool_token_allocation_with_live_state | original port + 6 independent live KV/state supplements; pure lazy counterexample and #36729 allocation failure |
| sufficient_capacity_preserves_all_prefixes | paired demand 3; distinguishes cheap ensure from actual recovery |
| impossible_target_exhausts_without_claiming_readiness | observed old behavior retained as FAIL against requested preservation contract |
| queued_composite_free_avoids_cache_eviction | ordinary and representative groups fail prefix preservation on #36729; scope preserved |
| decode_gate_preserves_component_limit_after_swa_release | original old static-cap assertion fails on #36729; contract difference, not yet a proven corruption bug |
| locked_cache_cannot_satisfy_joint_shortfall | all locked prefixes and payload preserved, allocation fails |
| explicit_eviction_keeps_count_semantics | original count-based eviction test passes in all arms |
| streaming_session_forwards_allocation_readiness | replaces mock-only oracle with real wrapper/cache/allocation/survivor test; active cursor NOT_RUN |

Coverage still NOT_RUN: async pending-reuse event lifecycle; internal tombstone
`node_id=None, made_progress=True` followed by controller drain; active session
cursor lifecycle; sharded exact-extend/DCP paths; actual ScheduleBatch decode call
with requests/spec_algorithm; GPU event/graph ordering. These require targeted
fixtures beyond the executed allocation comparisons. They must be included in
the validation of an implementation touching the relevant paths; current rows
do not certify them. The grouped FULL-only case records that readiness was
already sufficient after SWA retirement; it is not a full-only shortfall repro.

## Stage B recommended route and bounded requested approval

Evidence rules out adopting either branch unchanged and rules out a bare
pure-immediate-predicate substitution. Recommend a local integration candidate
on #36729's frozen head, reusing its two-pool planner and preserving its
allocator-owned dispatch. This is a coordination artifact, not authorization to
rewrite/close #39294 or publish changes to the other author's branch.

### Concrete two-pool delta proposed for approval

1. `UnifiedSWATokenToKVPoolAllocator.evict_to_free_tokens`: apply queued composite,
   page-representative and FULL-only frees at the entry recovery boundary,
   preserving free-group scope. Retain pure impossible-demand planner rejection
   before any destructive tree walk.
2. Introduce optional PURE `allocation_reclaim_satisfied` plumbing on
   BasePrefixCache/UnifiedRadixCache/StreamingSession. Use
   `reclaim_plan(n,n) == (0,0)` with no evictable credit as the allocator predicate.
   This means the reclaimed state can satisfy demand after allowed compaction;
   it does not promise immediate readiness. The tree checks it after real
   controller reclaim, including node-less made-progress steps, and stops early
   even when the planner's conservative FULL-only quota is not exhausted.
   No compaction/flush/group-drain in the predicate. Count-based evict unchanged.
3. Perform explicit ensure_capacity after the walk; the alloc method may repeat
   the cheap check, but expensive flush should happen at most at the explicit
   preparation boundary and not per victim. Add state/no-move counter assertions
   and document the semantics. No redundant wrapper None branch.
4. For ordinary unsharded unified SWA extend, obtain exact rounded page demand
   before eviction via an allocator-owned demand hook (base default preserves
   current conservative behavior for other allocators; unified SWA override
   uses existing get_num_new_pages). This avoids the proven coarse-probe trap
   without adding cache ownership to alloc_extend. Do not alter DCP/DSV4 behavior
   in this first delta. Reviewer may prefer an equivalent exact retry hook.
5. Port regression behavior tests for these demonstrated failures and real
   wrapper behavior; validate existing planner/allocator/capacity/eviction tests.

Work bound: one entry free-group drain, pure planner queries per victim, one
post-eviction ensure. Repeated alloc ensure is a cheap readiness check if the
post-eviction ensure succeeded. Guard impossible requests before eviction.
Prove the count under holes/closed gates and include final allocation in counts.
The planner's current FULL-then-SWA optimization alone is not minimal under
cascade; the pure stop predicate avoids the demonstrated excess eviction.

### Tri-pool decision remains separate

Do NOT transplant the two-pool byte equation into tri-pool or claim the new
two-pool hook fixes it. The frozen #36729 tri loop demonstrably fails real joint
requests; the original #39294 fixes them. A pure immediate query loses an extra
prefix in lazy live-state recovery. A safe tri route needs a pure, page-grid-aware
post-recovery feasibility query or an explicit recovery checkpoint whose next
attempt is justified by concrete layout/byte progress. A renamed mutating callback
or arbitrary number of retries is not acceptable.

Request reviewer judgment: approve the concrete isolated two-pool delta and its
CPU validation now, while retaining Stage A authority for tri recovery-oracle
experiments, then review the concrete tri delta separately; or request a unified
tri design before any implementation. No GPU/serving work requested at this
checkpoint. No branch rewrite, GitHub comments, push or PR closure requested.

Please write STAGE_B_REVIEW.md with explicit scope/conditions. All limitations
above remain open; this document does not claim the full user task is complete.
