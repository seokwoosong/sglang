# PR39294 pending index correction review — 2026-09-16

**APPROVED: the exact CPU assertion correction and updated GPU white-box probe.** The existing CPU completion gate and budgets remain unchanged. This is not final pending-fix acceptance or permission to resume serving/P2. No additional runtime source change is authorized.

## Static disposition

The focused log reports12 FULL same/different-mode subcase failures at the final physical-reuse comparison, with15 other subcases passing. The examples240/242 versus120/121 and328/330 versus164/165 are consistent with the kernel-page multiplier2. The guard records an assertion exit rather than a resource stop. Earlier assertions were reached, but the failing cases did not complete their subsequent fresh-marker/survivor checks; do not count those as completed passes.

The allocator source distinguishes these spaces:

- A physical token is `physical_page * page_size + offset`.
- A kernel-facing index uses stride `page_size * kernel_page_multiplier` after virtual-to-physical translation.
- `UnifiedMHATokenToKVPool.move_kv_cache` consumes real physical token IDs and copies whole page envelopes; indexing its exposed per-layer buffer requires kernel-facing IDs.

Therefore dividing a kernel-facing index by logical page size does not yield a physical page when the multiplier differs from1. The changed CPU assertion, `member.virtual_to_physical[fresh // member.pool_page_size]`, compares actual physical pages with the independently derived physical source set. It retains exact reuse rather than weakening the requirement. The complete test diff contains only that conversion. Classify these observed failures as an index-unit error in the new test, not a new allocator failure. Final allocator correctness still depends on completing the corrected validation.

The GPU diff is likewise appropriate: it retains `physical_indices` for original movement warmup and physical page arguments for the commit helper, while constructing reader indices through `physical_to_virtual[source_page]`, virtual-token expansion, and `translate_kv_loc_for_kernel`. Those indices are materialized before remapping/hazard submission and remain fixed for the forward reader. Expected payload is read from the same correctly translated source addresses. There is no new host observation inside the hazardous interval; event timing, commit logic, ledger oracle and delays are otherwise unchanged.

The inspected production diff remains the approved pending accumulation plus the two requested comment corrections. This review's earlier acceptance missed the index-space error; the corrected address interpretation here supersedes that portion of the prior review.

## Evidence correction

- Historical white-box ledger loss and candidate/base original-flush CPU reproductions remain valid bookkeeping evidence. However, the old white-box reader indices did not establish reads from the declared moved sources. Withdraw that reader-address claim; do not retroactively treat the corrected probe as validating the old run.
- In the old main GPU harness, `new_pages`/reader reuse eligibility used kernel-facing indices as physical pages. Preserve its reader trials as INCONCLUSIVE; neither promote nor silently rerun them under this approval. The inspected writer decision uses physical source pages and physical pre-move mappings, and its fixed buffer accesses use kernel translation, so this specific reuse-eligibility error does not invalidate that separate writer oracle. This targeted review does not independently recertify the submitted historical result counts.
- The new probe remains direct-helper white-box coverage with substituted prebuilt index tensors, the prior move-direction limitation, a source-union rather than complete per-event association oracle, and host synchronization before final drain. It does not establish unmodified CUDA flush reachability, unfinished urgent-wait ordering, or early physical reuse safety. Both event observations must remain unfinished for a conclusive concurrent ledger result; otherwise retain INCONCLUSIVE.

## Execution conditions

1. Corrected CPU validation may use only the remainder of the existing new-fix300-second phase. Its saved deadline is `17190.550198927` on the recorded monotonic epoch; preserve startup time, failed attempt and all subsequent work against that deadline. The inspected resumption script reads this saved deadline rather than resetting it. No new CPU allowance is granted here. If it expires before all required checks pass, retain NOT_RUN/PARTIAL and request a completion disposition; do not infer the GPU gate passed.
2. After the required CPU checks pass, the previously approved, not-yet-started new-fix GPU phase remains300 seconds total, four sequential layout2/3 × page1/4 cells, different-event then same-event once each, at most60 seconds per process or remaining phase time. Use the approved absolute3GiB whole-device guard, at least4GiB free preflight, less-than1GiB fixture allocation,8GiB RSS and32MiB logs. No parallel serving/GPU work, new trials, delay escalation or extension of the closed original GPU batch.
3. Use the corrected probe hash below instead of the prior probe hash. Preserve failed/unexecuted predecessor artifacts and write fresh results. Record the corrected test and actual loaded source in a new manifest before GPU execution. The saved `pending-fix-cpu/SOURCE.json` contains the old test hash and pre-correction tree; it must remain historical provenance, not be presented as the corrected candidate identity.
4. Keep the accepted gate commit `8b15837f2e1d125081b427d1296a6f92fbecf670` separate from pending changes. No production behavior change beyond the approved pending fix is included. New functional failures stop dependent expansion; successful CPU/GPU records still require final fix-results review before serving/P2 resume. A provisional row PASS is not a completed result if later assertions or the guard fail.

## Reviewed hashes

Directory: `/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/pending-fix-gpu`.

| Input | SHA256 |
| --- | --- |
| REVIEW_REQUEST.md | `212992346184efe697dfb0a2d811b22a96edaab9eeacae4a384aba0284fa33db` |
| MANIFEST.json | `1076afdfc0e7027c96e2482426a5ad80edb2fcc33c66e5a0af61d714fd78ab31` |
| probe.py | `b03ea44481ac942110bd02248d19332066baf9022a2584d40c69f28cbe7c6d69` |
| Worktree test_unified_pending_event_batches.py | `22ad126c7a18f55d3edcabe109a3ce68d386c76a80282e24002db8450c1ed286` |

All manifest entries match their corresponding files. Static source/diff/log inspection and hashes only. No test, probe, GPU command or server was executed; no source or harness was edited. This review artifact is the sole output change. Historical coverage limitations and C3/CI/merge prerequisites remain unchanged.
