# Final focused source/results/documentation review

**APPROVED for scoped C2 source/test adoption and local review-branch preparation, conditional on the two documentation corrections and final checks below.** No production-source must-fix was found in the submitted C2 delta. This is not universal readiness, maintainer approval, required CI completion or authorization to update the published PR branch.

## Candidate and source assessment

Reviewed canonical worktree: `/home/sukwoo24/sglang-eval-worktrees/pr39294-review`, starting HEAD `e8f51a123bb69bf03b4076eede5557536d686e3d`, including staged and unstaged contents. Its prior difference from tested A is documentation/evidence only. The current source/test binary delta hashes exactly to C2: `4451ee8cc9ddb025f13a8ea954370c7d20ebac6310c454cd438379ff9433e5e0`. All twelve changed source/test files relative to B match TESTED_SOURCE_FILES.json; no untested source transplant was found.

The immediate-ready branch rounds demand and checks both FULL/SWA ID headroom plus physical available capacity, matching the existing immediate-success condition in ensure_capacity. It avoids reclaim-size queries, preparation and the unused import when allocation already fits. The `free_group is not None` guard preserves active empty/nonempty group handling; the called deferred-free helper already returns immediately when no group exists. The existing slow path and rank-consensus decorator remain intact. The new regression covers page sizes 1/4, eager/lazy and active-empty/inactive groups with allocation, cache bindings, retained payload and accounting checks. Its direct closed-gate setup is a unit fixture, not new public gate/API or general temporal-state coverage.

## Results accepted within scope

T14's recorded Python/Rust counts, T16's complete 36-arm/12-block result, and T17D's ten synchronized CUDA passes support the disclosed source checkpoint. C1's miss and C2's T3-pressure increases remain visible; meeting the predefined ready-path criterion does not establish every path is faster or serving performance improved.

T27 is COMPLETE with eight normal PASS and two EXPECTED_FAULT_DETECTED results, exit zero and cleanup PASS throughout. I checked T26V2/T27 manifest-command/source bindings and all twelve raw distinct-result hashes against the matrix. Combined results are nine normal passes and three expected negative detections. Every normal records an outstanding E2 wait on the actual consumer. Every case preserves the required decision interval and actually reuses both sources. Every negative records the exact premature-selection signature from the historical GPU free-list. All twelve have E2 completed after allocation and before stamping finishes. These results support the new selection/dependency/reuse observer; they do not satisfy or replace the old blocking-observer criterion.

The documents properly retain the FULL-owner/page1/lazy/prebuilt-index/original-helper boundary, natural-flush/scheduler and other-owner limitations, prior PARTIALs, absent overlapping-write evidence, unavailable current serving validation and separate maintainer/CI status. Existing same-event evidence remains a separate matrix, not duplicated distinct coverage.

## Two required documentation corrections

1. **PR_UNIFIED_JOINT_ALLOCATION.md:11:** “stop using a pure allocator sufficiency callback” reads as removing the callback. Replace with, for example: “Extend the existing two-pool reclaim boundary to expose grouped frees and stop reclaim once a pure allocator sufficiency callback reports enough capacity.” Retain the count-quota sentence.
2. **review/pr39294/gap-closure-20260917/RESULTS.md:52–54:** explicitly restrict original urgent-drain observation to the nine normal distinct cases. Replace the last sentence of the negative paragraph with: “All nine normal cases directly observed the original urgent drain wait on E2 while pending; the three negative controls used an explicit protective consumer wait before reuse.” Similarly qualify the preceding “original urgent drain” sentence with “In normal cases”. Negative `waits` arrays are empty because their protective wait is not the wrapped original urgent-drain path; do not imply twelve observed original waits.

These are precise reporting fixes, not requests for more experiments. Their application and routine status/review-artifact additions need no new architectural review if source, hashes and claims otherwise remain unchanged.

## Preservation and final publication gates

The compressed archive matches its outer hash, and all 392 listed members match their byte counts and SHA256 values; the 393rd tar member is ARCHIVE_MANIFEST.json itself. It preserves historical evidence without rewriting outcomes. REPRODUCE.md appropriately requires explicit path/environment rebinding and distinguishes replay from the original run.

FINAL_CHECKS.json's pre-commit log hash matches the retained all-files PASS log. There are subsequent unstaged documentation changes and an untracked FINAL_CHECKS.json, so the existing check is not certification of the final index. Apply the two wording corrections, preserve this review, stage intended artifacts, recheck the final staged/working diffs and source manifest, and complete the planned final checks. Commit code/regression separately from documents/evidence as proposed. Do not silently include source changes or unrelated work under this approval.

The user's review-branch publication authorization remains effective. Before push, verify the actual push destination is contributor `https://github.com/seokwoosong/sglang.git` and only `review/pr39294-unified-joint-allocation`. Recheck the remote tip; the proposed explicit lease expects `a81393946af98619c6f669dbfdc7dd3eada1080a`. If it differs, stop and inspect the intervening work rather than refreshing the expected SHA blindly. Preserve the published PR branch and unpublished reply status. Verify the resulting remote tip, save the final outcome and stop owned workloads before the separately authorized final shutdown.

## Review bindings

- FINAL_REVIEW_REQUEST.md: `0c76df7b59b17678411336b2a582fae69da652500565cba14b5c01a4410c3691`
- PR_UNIFIED_JOINT_ALLOCATION.md, before corrections: `d29ec047da2d80d878659fe199d2eab433ad98fb99cbeb3e26a4c1d48adaccd1`
- PR39294_REVIEW_REPLIES.md: `1074226ded74fca8b2ec3f8643d342a5768ff591c7437c5c3deec4e2a882c4e8`
- RESULTS.md, before corrections: `8de123378bd563665359394b0e04d32adb088b2c74759a1beebec4333abf35cd`
- REPRODUCE.md: `38e61930c47b60911cc24765c1c82dd06cbdbcf25b18886fac28d4ef0ef07c19`
- evidence.tar.gz: `fb6a70e8e87f3d35ec06fbc8c52137f3ca4af397862a51140f6df81e683487f8`
- T27 STATUS: `3bc5943887003eda5bdc20c7160a91ed596c1387285162a2fb728c72f338cfd0`
- DISTINCT_EVENT_DECISION_MATRIX.json: `1a7fb58bacf8b843e46f6086b21c11db0929d51f64b5a93bf8076b0f299e790d`

Static source/document/JSON/hash/archive inspection only. No experiments, test execution, source edits, remote queries/mutations, push or shutdown were performed. Only this review artifact was written.
