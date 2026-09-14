# Supplemental plan: pure sufficient HIGH-side FLOAT recovery

Status: submitted for approval; production edits and GPU resume are not authorized by this document.

Frozen source is composed tree `0b1bd85dd84bad2fb6bc0d4b3c79429dc510304b`, patch `fb1484263d9f170ff26b87764af5bcb9987b9251c8948ea23637cb3580a144ec`, on #36729 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`. Keep it and accepted two-pool artifacts immutable. Implement only after the required review in a new isolated worktree layered on the composed tree.

## Counterexample and purpose

Legal temporal state `(1,4,8)`: allocate eight states before 96 token pairs; three independent single-state checkpoints cover 1, 1, and 94 tokens. A four-token request makes the END-only stopping query miss a recoverable FLOAT layout after the first victim. The current candidate evicts all 96 tokens; #39294 retains 95 in both eager and lazy modes. CPU reproduces this without streams or graph; this is excessive eviction, not a demonstrated CUDA corruption.

The proposed CPU-only ablation retains exactly 95 in both modes. It executes no unrestricted recovery ladder. The supplemental trace records each query, actual component reclaim, reusable/pending holes, bindings and frontiers; it checks all 95 retained KV tokens plus retained conv and temporal state payload. It must show the successful query at 95 cached tokens and first-phase reclaim exactly one FULL, one SWA and one state.

## Certificate and exact executor

All quantities below are physical page counts or byte frontiers, not nominal scheduling conservation caps. Let N be the equal FULL/SWA demand in pages and eF/eS/eM the member bytes per page. Require the existing actual index-space guard, nontransparent FLOAT, and an open FLOAT move gate. Transparent FLOAT retains the existing END certificate.

Let hF be reusable FULL holes if lazy, otherwise zero. Let cF/cM be reusable END holes credited only if lazy and that owner's movement gate permits compaction; pending reuse is never credited. Define:

- F0 = current FULL low byte frontier + cF*eF (after permitted END flush).
- F1 = F0 - max(0, N - (hF-cF))*eF (after FULL allocation).
- M1 = current Mamba high byte frontier - cM*eM.
- lo = max(ceil(M1/eS), FLOAT.min_page_index).
- hi = min(floor(F0/eS), FLOAT.num_pages).
- hi1 = min(floor(F1/eS), FLOAT.num_pages).
- L/H = current FLOAT low/high page watermarks.
- limit = hi1 - N; R = H - limit.

Reject the certificate if F1 falls below the FULL physical floor, R <= 0, or L-lo < R. Otherwise certify the exact absolute request `FLOAT.make_room(side="high", min_bytes=(hi-limit)*eS)` after flushing FULL then Mamba only. No FLOAT boundary absorption occurs before this plan; no FLOAT hole credit is used.

Proof: current high gap is hi-H, hence the requested additional high gap is R pages. L-lo >= R reserves fresh destinations strictly below every live FLOAT source even if no reusable FLOAT holes exist at favorable positions. Thus the existing partial relocation can move its R highest live pages (possibly fewer when holes help), and the far-destination check cannot fail. Moving those highest live pages retreats H by at least R. If R >= live FLOAT pages, the existing whole-pool branch packs against lo and the same reserved-gap inequality implies its capacity check and final target. In either branch H' <= limit <= hi1-N, which leaves at least N fresh SWA pages above FLOAT *after* FULL's extension. LOW destinations and HIGH subsequent allocations are disjoint. Unknown hole locations cannot invalidate this conservative certificate; holes can only reduce movement or improve the retreat. Different page grids are rounded inward for every usable region.

The callback computes END sufficiency OR this FLOAT sufficiency using only existing host metadata, tensor lengths and gate queries. It performs no device hole-list reads, mutation, waits, or movement. False continues to mean unknown. This is sufficient, not a complete planner or minimum-movement promise.

Executor: check ready and index guard; keep the END-only branch explicit; otherwise select the FLOAT branch only on the new certificate. Flush FULL and Mamba, check ready, recompute the FLOAT certificate using fresh frontiers/gates, execute one matching HIGH make_room, then require actual joint capacity and index headroom. A failed positive plan must not fall through to an unrelated ladder. Closed or changed gates may cause a temporary false result. Existing fallback remains only for certificate-negative byte-eligible layouts.

Use one shared END-frontier projection helper for the two certificates. Rename the existing END query explicitly and keep the callback entry point as their disjunction; ensure_capacity must not confuse FLOAT-positive with END-only. Do not store a stale movement plan in the callback closure. No TreeCore/API/pool movement implementation change, per-victim preparation, additional phase, or widened quota is proposed. The existing fixed preparation-call bound is preserved.

## Evidence and validation

CPU ablation is in `tri_float_certificate_probe.py`; it monkeypatches the candidate in an isolated process, with no production edits. Initial 864 geometry cases have 477 positive predictions, 26 FLOAT-only positives, zero errors and zero positive allocation failures; 41 negative predictions still allocate via the existing fallback, so completeness is explicitly not claimed. The ablation can change fixture construction placement; do not infer an exact coverage delta from old counts alone.

`tri_high_strict_geometry.py` repeats with the unrestricted fallback replaced by an assertion for positive predictions. `tri_high_float_retention.py` covers equal two-hole counts at positions (4,7)/(5,6), asymmetric layers and eager/lazy layouts against preserved comparator results. The geometry matrix covers page sizes 1/4/16, asymmetric (1,1)/(1,4)/(4,1), gates, transparent FLOAT and several hole patterns. Run pending-reuse variants with genuine pending metadata and no credit. Add dedicated nontrivial HIGH-positive tests in legal production fixtures, exact first-victim 95 retention, callback mutation/source guards, matching movement target and payload assertions.

After approval: implement as a new source layer; run the existing 229 Python CPU cases and 31 Rust cases with supported single-state nodes, including unchanged two-pool and gate tests; targeted new cases and all-files checks. Submit the source patch and fresh results for renewed candidate acceptance. Do not weaken retention or geometry assertions to obtain green tests.

GPU remains stopped. Correct the independent writer ownership/original-physical-destination issues from TRI_GPU_HARNESS_REVIEW.md, then resume the bounded GPU lane only under the renewed checkpoint's authorization. These CPU results do not establish event/graph safety or model output accuracy. C3 coordination, current-main conflict resolution, public responses and eventual merge remain separate outstanding tasks.

## Completed supplemental evidence

Strict 864-case geometry: zero errors/false positives with unrelated ladder forbidden for every positive. Equal-count/different-hole-position 12 cases: all allocate, all four prefixes retained, zero ladders. Detailed temporal first-victim replay in both eager/lazy modes: exactly one FULL token, one SWA token and one state reclaimed, 95 cached tokens retained, all retained KV/conv/temporal payload checked, no ladder. Query snapshots unchanged. Evidence files and hashes:

- `tri_float_certificate_probe.py`: `e6ff690d6c6a8032e5d4225e7bee60ea20015dba4ab97e118a306590daad78c5`
- `tri_high_retention_trace_v2.py`: `7108a044b5deb1009ee063fd253f63b8f30155b999ecb3c77b83b64e7a2b0a74`
- `tri-high-trace-v2-temporal-float-repro.json`: `d7f5cbb1dec046455610150270b2c390c577e4d01325c8426769504753267921`
- `tri-high-strict-geometry-cases.json`: `b84ebb4817b5809edaaeb02ed4f78cc620a4caa1845bd1b77ed84822d145ceaa`
- `tri-high-float-retention-cases.json`: `ea785853bbb7b705c872ff56727e63039ffc824a6cdd44e5a557e539550c82d7`
