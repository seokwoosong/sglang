# Tri-pool architectural advice

**Recommendation: use a narrowly scoped version of A as the core contract. Consider B only as an explicitly separate recovery-checkpoint policy for unresolved geometry.**

This is advisory input for `TRI_PLAN.md`, **not approval for production tri edits**. The accepted two-pool patch remains immutable. Continue the already authorized CPU geometry experiments and submit the concrete tri delta for review.

- Reviewer session: `01a09f85-77de-79d1-a50d-cb7a530616c1`
- Date: 2026-09-14 (Asia/Seoul)
- Source inspected: frozen #36729 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`
- Accepted two-pool patch SHA256: `cfc5b3c6c2593189b5a1c15a3f5b774eba55d19a1813d32013de0f68ec6cd2bf`
- Experimental oracle inspected: `tri_oracle_probe.py`, SHA256 `e87abc70bb158ce838dd85eda0106158d6148c7a89e535618b52b0f37801b640`

## Why A is the preferred core

The cache needs to know when it can stop discarding prefixes. That requires a sound statement: **without another victim, a specified permitted recovery operation can make this complete demand allocatable**. A sufficient predicate can provide that statement without being a complete solver for every float layout.

The END-only result is useful: it identifies a state where immediate capacity is insufficient but a concrete END flush permits allocation while retaining 95 prefixes. Keep that mechanism narrow. Do not turn its false result into “impossible,” and do not generalize it to FLOAT relocation by adding total free bytes.

The minimal architecture has three separate responsibilities, all owned by the allocator:

1. Reject demand that is provably impossible, before destructive cache eviction.
2. Purely recognize already-ready or certified-recoverable states.
3. Execute the recovery justified by that recognition, then check actual capacity/allocation.

These do not require a new generic planner framework, a scheduler byte model, or a new TreeCore geometry model. A small internal result/plan representation is sufficient if more than a boolean is necessary. Choose names and concrete types in `TRI_PLAN.md`; do not add public APIs merely to represent every conceptual state below.

## Separate the meanings of the results

| Result | What it establishes | Permitted consequence |
| --- | --- | --- |
| Ready now | The next exact demand fits current usable resources | Skip eviction/recovery and allocate |
| Certified recoverable | A specified permitted recovery sequence will make the demand fit, under stated preconditions | Stop eviction; execute that sequence; verify |
| Proven impossible | Even an optimistic safe upper bound cannot serve the exact demand | Reject before destructive eviction |
| Unknown / temporarily blocked | Neither success nor impossibility is established | Use an explicit bounded policy or return unsatisfied; never label it permanent impossibility |

The existing `allocation_reclaim_satisfied` callback may represent the first two states only. **Do not return true merely because an optimistic bound says “worth trying recovery now.”** That is a checkpoint request, not satisfaction of reclaim. If B is needed, its pause reason must remain distinct from successful joint reclaim in both code and reporting.

For token allocation, preserve the demand vector FULL pages + SWA pages + zero new Mamba state pages. Existing Mamba state still consumes bytes and divides the available bands. State allocation and donor-ID recovery are separate demands and remain separate paths.

## Keep prediction tied to execution

For END recovery, define exactly which ENDs may flush, whether float boundary absorption is included, and the post-flush layout assumptions. The pure predicate and executor must agree on this sequence. They must account for immediate reusable holes, free virtual IDs, physical index limits, sink/padding, pending reuse, movement gates and the actual FULL-before-SWA allocation order.

Do not certify END-only recovery and then silently invoke an unrestricted fallback ladder that can relocate FLOAT to make the test pass. Either prove that the normal ladder terminates at the predicted END phase, or make the intended execution boundary explicit in the proposed delta. Verify the phase and moves, not only final allocation success.

A cached plan needs validity conditions. Relevant free/allocate operations, frontier or hole changes, and gates/events can invalidate it. An epoch may help invalidate geometric state, but it is not automatically a version for external gate/event state. Revalidate before movement; after a precondition changes, report stale/blocked rather than treating the prediction as still binding. A pure callback can compute or cache metadata, but must not wait, drain queues, move pages, or mutate allocation state.

Avoid maintaining a second copy of relocation logic in a composite allocator. If certification of existing FLOAT relocation requires exact placement reasoning, prefer a small pure planning primitive beside that movement implementation, with the executor consuming the same decision. This may be more code than END-only recovery and therefore belongs in the concrete plan before production work starts.

## FLOAT is the unresolved part

At the inspected source, `FloatMultiEndedAllocator.pages_in_band` (`unified_sub_pool.py:2146`) clips and rounds on the FLOAT pool's own page grid. `take_physical_pages` (`:2206`) drains holes and takes remaining extension from one side; a live FLOAT does not allocate by summing both gaps. An empty FLOAT has different transparent/repositioning behavior.

`make_room(side, min_bytes)` (`:2372`) receives an **absolute contiguous-band target**, not a byte deficit. Its partial move (`:2417–2444`) selects boundary live sources, uses only holes on the far side of every selected source, and requires enough far-gap destinations for the remainder. Thus the aggregate capacity check at `:2393–2397` is not a certificate for the partial move. Distinct hole placements with equal counts and bytes can take different branches.

For any FLOAT-capable version of A, show the exact post-recovery configuration in which FULL allocates first and SWA then succeeds. Certifying only that `make_room` opens its requested side is insufficient if the move consumes the opposite band needed by the same joint request.

Do not use unconditional whole-FLOAT packing as an easy substitute for proving the existing partial move. Whole packing can require a different number/order of copies; `_relocate_to_positions` also has an ordered-overlap path. A different movement strategy needs an explicit correctness and cost argument, not just a successful CPU allocation.

The initial plan may intentionally certify only END recovery. However, **END-only coverage cannot close C3 or establish complete replacement of #39294** while FLOAT-dependent failures remain. State exactly how those cases remain handled or remain unresolved. A correct conservative predicate can still evict more prefixes than necessary; soundness alone does not establish acceptable cache retention.

## If B is needed, make its work bound real

B is acceptable for investigation, but “retry after concrete progress” is not yet an adequate bound. Every victim can produce concrete progress, reproducing per-victim recovery under another name.

The plan must specify:

- The first threshold that permits a recovery attempt and why recovery cannot succeed earlier under the supported geometry.
- What information the attempt returns: success, stale/gated state, exact unsatisfied placement constraint, or unresolved geometry.
- What **new resource milestone** can make another expensive attempt useful. Identify the pages/IDs or band change required, rather than merely checking a changed epoch or any drop in live bytes.
- Why the same unmet condition cannot trigger another attempt after each victim. State a structural upper bound or an amortized work argument for attempts and moved pages/bytes. Count the final allocation's recovery too.
- What happens when the optimistic bound is already feasible but the geometry attempt fails. Do not continue re-attempting on every still-feasible observation. If no defensible next threshold can be derived, record the case as unresolved and bring it to review.

A finite cache bounds the number of victims, but does not by itself satisfy C1's recovery-cost concern. Likewise, a fixed retry count can stop a loop without proving either adequate recovery or acceptable prefix retention. Do not hide this tradeoff in the implementation.

A checkpoint policy must preserve the existing cumulative tracker/cascade contract. Separate rounds must not double-charge prior component frees, restart victim selection carelessly, or lose session cursor cleanup. If implementing a pause/resume API would require TreeCore changes, identify them explicitly; they are outside the previously accepted two-pool patch.

## Impossibility checks and caller behavior

Start with cheap, allocator-owned, pure guards for exact virtual-ID and optimistic empty-arena byte limits. A conservative total-byte upper bound is useful for rejection; passing it proves nothing about the FLOAT bands. Locked/live state and reclaimable components may tighten rejection only when their accounting is sound and cascades are not double-counted.

Closed gates and unfinished readers/writers are temporary constraints, not global impossibility. Pending reuse is not usable capacity merely because logical ownership was released. Preserve the existing event and move-gate owners rather than putting synchronization in the query.

Keep exact demand separate from caller overestimates. The accepted exact-extend hook applies only to concrete two-pool ordinary unsharded extend. Do not assume tri inherited that fix. If tri rejection exposes the same coarse-probe trap, include a concrete tri caller-demand/retry delta in `TRI_PLAN.md`; it is not silently authorized by the two-pool change.

## Minimal geometry experiment set

Use small exhaustive legal layouts where practical, alongside the production-factory reproductions. Enumerating arbitrary nonphysical states can produce misleading counterexamples. Compare the proposed prediction with the **actual selected recovery sequence and real allocation**, and preserve both false positives and feasible-but-unrecognized states.

- Reproduce 96 tokens/8 states, lazy, demand 4 with 95 retained prefixes and live state payloads.
- Hold bytes and hole counts constant while changing FLOAT hole positions; exercise partial move success and failure plus the whole-move boundary.
- Cross page sizes 1/4/16 with unequal END/FLOAT entry bytes and off-grid frontiers; include empty, one-live-page and nonempty FLOAT.
- Exercise FULL-only, SWA-only and state cascades, asymmetric holes, all-locked state, and independent ID pressure; count physical reclaim once.
- Include gate-closed then reopened and pending-reuse layouts. CPU mocks establish accounting/control flow only; real event ordering remains later GPU evidence.
- Keep predetermined request-owned KV and conv/nonempty temporal states alive outside cache ownership. Assert their bindings and payloads even if all evictable cache entries disappear.
- Repeat unchanged-state queries and separately count planner CPU work, host/device synchronization boundaries, explicit preparation, each movement phase and final allocation. A pure query that copies/sorts a GPU hole list per victim may still be an unacceptable hot-path cost.

For A, **zero false-positive certificates** is mandatory within the declared supported domain. For B, require both eventual allocation/prefix-retention behavior on the targeted cases and the claimed recovery-work bound. Do not collapse these into a single PASS count.

## What to submit in TRI_PLAN.md

1. Exact base and composition with the immutable accepted two-pool patch; a separately reviewable proposed tri delta.
2. Chosen supported domain and explicit remaining unknown cases. Prefer one concrete route over two uncommitted alternatives.
3. Source symbols for query, demand/guard, execution and caller failure handling; any shared primitive extraction or TreeCore/API change.
4. Prediction/execution preconditions, state validity, units and recovery-work bound.
5. Geometry results, minimum reproducers, required survivor/payload assertions and tests that fail before the proposed fix.
6. CPU validation scope and the later GPU/event checks needed before a broader merge-readiness claim.

There is no need to implement a universal tri planner before asking for review. There does need to be an honest, concrete path for the reported FLOAT-dependent failures and an explicit boundary around what an END-only improvement does not solve.

This advisory does not undo the user's broader intent to address comments and pursue merge. It supplies the requested design guidance; the separate production tri checkpoint remains in force. No remote action or implementation was performed by this reviewer. Only this advice artifact was created; accepted artifacts were not modified.
