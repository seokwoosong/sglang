# PR39294 bounded hardening final disposition — 2026-09-16

**ACCEPTED: final bounded hardening report and P2 results. APPROVED: the three requested local preservation/adoption steps, subject to the exact checks below.** No further experiment or source correction is required for this local checkpoint. This does not mean every original coverage objective was demonstrated, all PR comments are resolved, or the PR is push/merge-ready.

## Results accepted

The final report correctly distinguishes the tested pending-fix tree from historical random/GPU/graph evidence, preserves partial and inconclusive outcomes, and records the index-oracle corrections, initial diagnostic timing overrun and blocked serving. The copied pending-fix and startup reviews match the reviewer's accepted artifacts by hash.

P2's saved command contains exactly the two approved files. Its log reports14 tests/25 subtests passed; guard exit0, reason=null, elapsed14.04 seconds and peak RSS1434038272 bytes are within the120-second/8GiB allowance. Source provenance identifies the accepted pending tree. This completes the selected local interface/rejection lane; it establishes no real-model or distributed serving result.

Retain the final evidence as separate reporting units: CPU281/1636 with10 GPU skips; focused1/27; Rust25/131 with1 session exclusion; all-files pre-commit PASS; corrected CUDA same-event4 PASS and different-event4 INCONCLUSIVE; P2 14/25. Focused cases overlap the broad suite and must not be added as unique coverage. Serving remains BLOCKED before inference, with candidate/client/output/pressure/cancellation checks NOT_RUN. The old white-box reader-address claim remains withdrawn.

The report's223-second initial diagnostic wall-time deviation is acknowledged, not retroactively authorized. The timing addendum distinguishes process runtime from reduction wall time; preserve it alongside the final report or link it explicitly. This scheduling deviation does not erase the reproduced capacity values or the subsequent separately bounded validations.

## Verified local state

At this static review:

- Isolated branch: `review/pr39294-gate-memo-hardening`, HEAD `8b15837f2e1d125081b427d1296a6f92fbecf670`.
- Exactly two staged paths: `python/sglang/srt/mem_cache/allocator/unified_sub_pool.py` and `test/registered/unit/mem_cache/test_unified_pending_event_batches.py`; no unstaged diff.
- Index equals tested tree `bbf48421d7d23dfd888b687a718e8714b39cea16`.
- Canonical worktree `/home/sukwoo24/sglang-eval-worktrees/pr39294-review` is clean on `review/pr39294-unified-joint-allocation` at `f92803e8f6d88d6d48ad9bceae1c8bcdcbe266fa`.
- That canonical HEAD is an ancestor of the isolated HEAD. The requested backup ref is currently absent.

These checks support the proposed fast-forward path; repeat them immediately before mutation because this review does not freeze concurrent local state.

## Authorized preservation steps

1. **Pending code commit:** commit only the two staged paths on the isolated branch, with title `Preserve all pending move batches sharing an event`. Do not amend/squash the gate commit or stage unrelated files. Verify the new code commit's tree is exactly `bbf48421d7d23dfd888b687a718e8714b39cea16`, then record its full commit identity. Abort preservation if bytes/tree differ unexpectedly; do not silently repair or broaden the code change.
2. **Separate documentation commit:** preserve the report, accepted gate/pending/index/startup/final reviews and final result/source/guard summaries under `review/pr39294/hardening-20260916`. Keep raw logs and harnesses immutable in the original artifact directory. Retain original artifact hashes, and create a manifest for copied documents; distinguish any edited presentation copy from the original report hash. Link raw evidence with absolute local artifact paths, and fix copied relative links so they do not point to nonexistent repository locations. Record the new code commit, tested tree and explicit not-pushed/not-merge-ready status. Add the dated pointer to `review/pr39294/README.md` without removing/replacing prior content. The docs commit must have no executable or test-source changes; its tree will differ from the tested code tree only through these documentation additions. No new behavioral assertions or reclassification of results is permitted as a docs cleanup.
3. **Local canonical adoption:** recheck canonical cleanliness and HEAD, then create `backup/pr39294-review-before-hardening-20260916` at `f92803e8f6d88d6d48ad9bceae1c8bcdcbe266fa` only if absent. If it already exists, preserve it; require that it identifies the intended pre-adoption state or stop for review. From the canonical worktree, perform only `git merge --ff-only review/pr39294-gate-memo-hardening`. Abort on dirtiness, an unexpected HEAD/ref or non-fast-forward; no reset, rebase, force update or conflict-resolution merge. Verify the resulting canonical HEAD equals the intended isolated docs commit and the worktree is clean. Preserve the isolated branch and backup.

The existing report is acceptable; no substantive result correction blocks these steps. Link repair, provenance additions and the dated pointer are the authorized documentation work, not permission to rewrite historical evidence. A new unexpected failure or source difference requires review rather than an automatic experiment/retry.

## Remaining boundaries

The narrow gate and pending-source fixes are accepted within their demonstrated scope. Unmodified CUDA same-event flush/scheduler reachability, unfinished urgent ordering, early reuse safety, different-event GPU overlap and negative-control sensitivity remain limited/unestablished as reported. The final pending tree did not rerun the historical472Python/96Rust matrix or production GPU/graph corpus. Real two-pool serving was blocked by config-loader compatibility; complete pretrained tri serving and distributed DCP/PD were not demonstrated.

C3 author/design coordination and actual PR CI remain separate prerequisites. Local adoption preserves reviewed work; it does not authorize modifying the published PR branch, pushing the archival branch, posting comments, triggering remote CI, changing dependencies/models or claiming PR merge readiness. Submit any subsequent remote/integration request separately. No additional experiment is authorized here.

## Reviewed hashes

Artifact root: `/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916`.

| Input | SHA256 |
| --- | --- |
| final-disposition/REQUEST.md | `44d34887b2f99c29724ca467db077ea5849ea6076560c64a3febc6db341ba062` |
| HARDENING_RESULTS.md | `0309849878122f4b1dab238ad3a2f4d7aae5639233d8dd8d07562cfb6d4b4d9e` |
| p2-final/SOURCE.json | `e630d703d8dd4bf536933685052439327dd38f0828d4321f5e411a908c53bfbb` |
| p2-final/p2.run.json | `16e0858af4dfecb611186f8fcfb26d4dadcabd9239fa30f7682d215baaa77b9c` |
| p2-final/p2.log | `6fe55ffb6e135bc07eac5be4bd1923aa0301639bddcd44fed44232908f456bf9` |
| pending-fix-final/REVIEW.md | `2331759bf257ef8791957e732cd45975596ede920c438ca122e78ef96627defe` |
| serving-startup-disposition/REVIEW.md | `6f2045bd38816f278130c96be3a0921d02fae61999c30a48d46f89a6542b4a9d` |

Static file/log/hash and read-only Git inspection only. The reviewer performed no commit, staging, branch change, source edit, test, server launch or experiment. This final review artifact is the sole output change.
