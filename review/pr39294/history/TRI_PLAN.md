# Tri-pool continuation plan — revision 1

Decision requested from session `01a09f85-77de-79d1-a50d-cb7a530616c1`: approve the concrete isolated production candidate and CPU validation below. GPU validation and remote publication will follow a separate concrete submission. User has reiterated that the objective is all five reviewer comments addressed and PR merge; partial implementation acceptance will not be reported as completion of that objective.

## Frozen composition

Base #36729 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb` plus immutable accepted two-pool patch SHA256 `cfc5b3c6c2593189b5a1c15a3f5b774eba55d19a1813d32013de0f68ec6cd2bf`. Create a distinct `review-tri-candidate` worktree, apply that patch, then retain the tri delta separately. Existing candidate and submitted branch remain untouched. Both GitHub heads checked unchanged on this continuation.

## Chosen implementation

Use a **pure sufficient END-recovery predicate, finite demand-derived component quotas, and explicit recovery at component-phase boundaries**. Do not implement a second FLOAT relocation planner or treat an optimistic byte bound as a successful reclaim certificate. This intentionally does not promise globally minimal eviction for FLOAT-dependent layouts.

Production scope: concrete `UnifiedMambaSWATokenToKVPoolAllocator` in `allocator/unified_hybrid_swa.py`, plus focused tri regression coverage. Reuse the already accepted cumulative callback interface without TreeCore or Rust algorithm changes. Extend the exact-demand override to the concrete tri class for ordinary unsharded paged extend; shared/sharded defaults remain unchanged.

1. Entry drains existing composite deferred free groups once. Reject a provably impossible exact demand before any cache walk using legacy tri conservation ceilings / FULL-owner ID namespace, optimistic byte capacity after **all** eligible component reclaim, and the minimum reserved sink. Passing this upper bound is only permission to investigate, never success. Pending reuse can be credited only in this optimistic rejection bound, never as immediately usable space.
2. Pure `_token_reclaim_satisfied(n)` recognizes immediate joint + conservation capacity or END-only recovery. Project allowed lazy FULL/Mamba holes into their frontiers, account for closed-gate FULL holes that remain reusable in place, check FULL-before-SWA allocation on the FLOAT page grid, current FLOAT holes/span/transparent state, and real owner/physical index limits. No tensor-to-host hole-position query, free/drain, wait, sort, relocation, or cached closure plan. False is unknown.
3. Explicit `_prepare_token_allocation(n)` checks the pure certificate and, when needed, flushes the two ENDs **without flushing FLOAT first**, then checks actual capacity. This binds the certificate to its executor: no unrestricted recovery may be used to hide a failed END certificate. If no certificate exists and the optimistic **current** live-byte shortfall is zero and conservation/ID headroom is adequate, use the existing tri recovery ladder once. It may absorb FLOAT boundary holes and relocate FLOAT using existing movement/event ownership. Do not attempt a ladder when live bytes alone prove it futile.
4. Try preparation once at entry. If still unsatisfied, perform one FULL/SWA cumulative walk with quotas of rounded requested FULL tokens and SWA tokens, Mamba quota zero. The pure callback stops at the first certified point. These are finite demand-derived upper quotas, not a whole-cache fallback and not claimed to be minimal. A request for n paired pages needs at most n pages returned on each token side to fund its new pages; paired/cascade credit is already visible to the callback and tracker.
5. Try explicit preparation after the token walk. If still unsatisfied and eligible state reclaim exists, perform one separate Mamba cumulative walk, with quota `ceil(requested_pair_bytes / state_entry_bytes)`, followed by one preparation. This order is deliberate: a combined three-component quota lets the existing Mamba-first component ordering discard valuable token prefixes before token-side recovery. Never repeat either component phase or enlarge a quota to whole-cache counts.
6. Return actual preparation result. Caller still verifies allocation. Failure with a closed gate/unrecognized layout is temporary/unsatisfied, not proven impossible. Existing donor/state-allocation, admission and speculative/sharded protocols are preserved. Include a real tri coarse-extend trap before copying the accepted exact-demand override; no automatic shared-base override.

The initial experimental quota variant used all three component quotas together and failed the fixed 95-prefix target. It is rejected; the invariant will not be weakened. Revised two-phase experiment is `tri_bounded_probe_v2.py`.

## Work bound

At most two cumulative cache walks: one token phase, one state phase. At most three explicit preparation boundaries: entry, after tokens, after states. Each preparation selects END certificate execution or one existing ladder; do not run both to rescue a false certificate. The final real `alloc` may independently invoke its existing recovery on failure; count this too, giving at most four ladder-equivalent boundaries across helper + caller. There is no expensive preparation in the victim predicate and no retry conditioned merely on epoch/live-byte progress.

Each boundary flushes each selected END at most once and invokes FLOAT relocation at most once; recovery moves are bounded by the sum of live END/FLOAT pages at those fixed phases (with bytes weighted per pool). Eager moves caused by actual frees remain separately counted, proportional to actual reclaimed victims; the patch cannot claim those disappear. Track actual movement calls/pages/bytes and bound work from all phases including final allocation. This is a structural bound tied to distinct resource phases, not a retry count chosen to stop an unbounded loop.

## Evidence and unresolved domain

`tri_geometry_probe_v4.py`: 864 legal CPU cases, page 1/4/16 × ratios 1:1/1:4/4:1 × eager/lazy × six hole patterns × gates open/closed × demand 0/5/7/10 pages. 451 sufficient positives; all actually allocated under the existing ladder; 53 additional feasible cases were unrecognized, 0 certificate false positives, 0 payload/accounting errors. `tri_geometry_end_only.py` executes the END sequence only for the 451 positives and succeeds for all. Its unrecognized cases are NOT_RUN for recovery; a zero false-negative counter there is not a completeness result.

Early geometry harness versions had invalid fake-payload addressing (physical envelope vs kernel ID, and page vs token); preserved v1/v2 logs are harness errors, not allocator evidence. v3 passed but mixed FULL-only release was not preceded by SWA release at the same slot; v4 corrects that lifecycle. Compare v4 and END-only artifacts, not those earlier versions.

Already established factory cases: 96 tokens/8 states lazy demand4 should allocate and retain exactly95 cached tokens; 80 tokens/24 states joint failure on #36729 must allocate after legal eviction; strict request-owned KV plus nonempty conv/temporal states [3,4] must remain mapped and unchanged. The revised bounded experiment must meet all these before acceptance.

The predicate intentionally does not recognize all FLOAT-dependent opportunities. The existing ladder at fixed phase boundaries preserves a concrete path for such recovery, with finite quota fallback. Tests must identify over-eviction relative to an earliest feasible checkpoint, especially hole-position permutations. If the finite policy exhibits unacceptable retention or fails a formerly working supported case, do not call C3 closed; report the counterexample and refine the plan with reviewer guidance. No unconditional whole-FLOAT packing.

## Required CPU tests and review evidence

- Original three factory geometries × eager/lazy; fixed 95 survivor regression; predetermined request-owned KV and conv/nonempty temporal payload.
- Impossible demand leaves all cache IDs/payload intact, including repeats; locked prefixes and oversized demand with insufficient reclaimable bytes. Actual FULL ID and static tri caps tested separately from raw bytes.
- Immediate-ready avoids preparation and eviction; sufficient END executes only the two END flushes; no FLOAT movement can rescue a false certificate.
- Unequal page grids, empty/one-live/many-live FLOAT, equal hole counts with different positions, FULL-only and SWA-only release, Mamba cascades; compare prediction with actual selected recovery and report unrecognized successes.
- Pending reuse with unfinished→finished event transition at mocked external CUDA boundary; gate-closed→open, both reusable-hole and unavailable-pending accounting. Queries leave mappings, queues, holes, frontiers and payload unchanged. Real event ordering remains GPU validation.
- Ordinary/representative/FULL-only deferred groups remain scoped and visible; real Python session wrapper/cursor; Rust path excluding unsupported session mode explicitly.
- Actual normal extend with conservative probe larger than exact feasible demand, partial page and zero-new-page; real ScheduleBatch decode signature and rank-consensus enabled numeric arguments/callback presence.
- Worst deep eviction and forced-unsatisfied cases prove bounded recovery independent of victim count, including final allocation. Existing count eviction semantics unchanged.
- Re-run changed tri/new joint tests, relevant existing allocator/capacity/accounting/session tests, Python + actual Rust extension, full required pre-commit. Preserve exact source/log hashes and C1–C5 status.

## Merge path after local acceptance

Refresh #36729/#39294 and prepare maintainer coordination text and the exact proposed branch/PR delta. C3 requires agreement with the other author, not just local tests. Submit a concrete GPU/serving validation plan to the same session, including GPU event/compaction/live-payload checks, bounded-work regression and fair baseline comparison on the exact composed candidate. Do not reuse old model parity/performance numbers as results of this patch. Once implementation, evidence and integration choice are reviewable, obtain any still-missing user authority for publishing/pushing/commenting. Track CI and maintainer review after authorized publication; actual merge depends on maintainer approval.
