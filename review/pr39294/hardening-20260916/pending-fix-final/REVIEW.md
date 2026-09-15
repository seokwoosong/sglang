# PR39294 pending fix results review — 2026-09-16

**ACCEPTED for the final narrow pending-source fix. APPROVED to lift the serving/P2 hold and proceed with serving v4 and the previously selected P2 checks under their unchanged budgets.** No mandatory production correction remains from this finding. This is acceptance of the bounded fix checkpoint, not universal asynchronous safety, review-branch adoption, push, CI acceptance or PR merge readiness.

## Accepted source and validation

- HEAD: `8b15837f2e1d125081b427d1296a6f92fbecf670` (separate accepted gate commit).
- Final index tree: `bbf48421d7d23dfd888b687a718e8714b39cea16`.
- Pending patch SHA256: `1e1afc2ff57e207a92f149c554edf5b4feb4805b5e75412be51ee0d79377b911`.

The current HEAD/diff hash match the submission; the index compares equal to the stated tree and there is no unstaged diff. The staged scope remains the allocator pending accumulation and the registered regression, including approved comment and index-unit corrections. Keep these exact bytes frozen through serving/P2 and record their identity in every result.

Saved final CPU logs and guards show **281 tests/1636 subtests passed,10 GPU skips**, followed by all-files pre-commit exit0. Both guards have no failure reason; completion-phase elapsed107.70 seconds is within the approved180-second allowance. The earlier corrected focused1/27 and Rust25/131 with1 documented session exclusion remain applicable to the same test/source bytes. The old12 fixture index failures and timed-out broad attempt remain historical failures/incomplete runs; they are not counted as passes or combined into additional successful coverage.

The corrected GPU probe hash matches its approval. All four cell processes exit0 with no guard failure, each in roughly12–15 seconds and with maximum reported whole-device usage2812MiB. Saved source pins match the accepted tree. The eight scenarios are classified as follows:

| Scenario | Result | Scope |
| --- | --- | --- |
| Same event, layouts2/3 × page sizes1/4 | 4 PASS | Both observations unfinished after both commits; event ledger and CPU pending set retain both sources; corrected fixed-address reader, retained payload and final drain/accounting checks complete |
| Different events, same four geometries | 4 INCONCLUSIVE | Both events already completed at the checkpoint; functional checks finish, but concurrent pending coverage is not established |

Because each process completes successfully, these are final results rather than provisional row statuses preceding later failures. This is adequate focused GPU evidence for the changed accumulation path. Further repetitions to force different-event overlap are not required or authorized for this checkpoint.

## Finding disposition and limits

The pre-existing bookkeeping defect is resolved within the demonstrated scope: original direction-valid CPU flushes with modeled unfinished events and real-CUDA direct-helper same-event pending batches. The fix retains every batch under its event, and the CPU regression additionally checks independent mapping-derived sources, per-event association, intermediate drain and actual physical reuse.

Preserve these limits in reports and eventual PR replies:

- The GPU probe substitutes prebuilt indices and invokes the helper directly; its direction/timing limitations remain. It is not proof of unmodified production CUDA flush/scheduler reachability.
- Final GPU drain follows host synchronization, so unfinished urgent wait ordering, early reuse safety and negative-control sensitivity are not newly demonstrated.
- Different-event GPU overlap remains INCONCLUSIVE. CPU modeled-event tests do not replace that evidence.
- Withdraw the old white-box moved-source reader-address claim and retain the old main reader reuse trials as INCONCLUSIVE. The old ledger failure and candidate/base CPU reproductions remain valid independent evidence.
- The472Python/96Rust random sequences and earlier production GPU/graph counts belong to the pre-pending-fix checkpoint. They must not be reported as reruns on this final tree. Historical PARTIAL/NOT_RUN and unavailable model/distributed coverage remain unchanged.

No payload corruption was established by the original finding, and this acceptance introduces no such retrospective claim.

## Serving/P2 authorization

The complete serving-v3→v4 batch diff changes only candidate HEAD and expected pending patch hash. Client, comparator and run sheet hashes are unchanged. The baseline remains `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`; the whole-device24GiB guard and all workload/oracle controls remain in place. **The pending-fix stop gate is now cleared for these independent bounded lanes**, despite the explicitly retained historical coverage limits.

- Serving v4: at most six sequential launches **B,C,C,B,B,C**, fixed local Granite snapshot, one owned server group, original separate30-minute phase including startup, no models/downloads/config search added. Retain startup180s, whole-client90s, request30s, drain10s, at least20GiB free preflight, absolute whole-device24GiB/combined owned RSS24GiB ceilings,32MiB logs and bounded cleanup. No other GPU test runs in parallel. Use fresh serving-v4 outputs.
- Preserve exact input/sampling/output comparison and first failure/divergence stop rules. Map comparator operands to actual launch labels: pair `(2-C,3-B)` reverses the literal `baseline`/`candidate` field meanings. Keep runtime window NOT_EXPOSED, missing/zero eviction evidence PARTIAL, joint recovery UNESTABLISHED, and cancellation reachability INCONCLUSIVE when appropriate. Queue drain is not a complete ownership-leak proof; sequence parity is not task accuracy.
- P2: only `test/registered/unit/disaggregation/test_unified_memory_move_gate.py` and `test/registered/unit/server_args/test_unified_prefill_cuda_graph_gate.py`, with the approved CPU environment/guard,120 seconds total,8GiB RSS and32MiB log limit. These are local interface/rejection checks, not distributed DCP/PD or graph-performance validation.
- New functional failures, unexplained hangs, output divergence or cleanup failure restore the relevant dependent stop gate. No automatic retry, tolerance relaxation, added budget or source modification follows this acceptance. Preserve the existing rule for clean resource-only unavailability without relabelling it PASS.

No local pending commit, accepted review-branch adoption, PR rewrite, push, remote comment or merge is authorized by this disposition. Those decisions remain separate after serving/P2 results and outstanding C3/CI requirements.

## Reviewed hashes

| Input | SHA256 |
| --- | --- |
| pending-fix-final/REVIEW_REQUEST.md | `b2167a4192884b1f6236cee544a2b49c7d70b864302df223b3faf6347eb70200` |
| pending-fix-final/MANIFEST.json | `88c4ce57584d9a81e3198c3efb586b1ab77fd968f1fa7a06c5ca2319d25626e2` |
| pending-fix-final/RESULTS.json | `f81c4475f56db7a9e8298354dc83896b4102eb83c7ca4b67b2019f77e6eaed89` |
| pending-fix-final/pending-final.patch | `1e1afc2ff57e207a92f149c554edf5b4feb4805b5e75412be51ee0d79377b911` |
| pending-fix-gpu/probe.py | `b03ea44481ac942110bd02248d19332066baf9022a2584d40c69f28cbe7c6d69` |
| serving-v4/MANIFEST.json | `641816b88a1abf6011899f937c2e7d4922b11ceeb306d66a0b684b6b366853e4` |
| serving-v4/serving_batch.py | `631a644dd59d75772406684c36103900030802f56e3b3927c5c7c9339690e5d9` |
| serving-v4/serving_client.py | `091072935ea1902a19d7e970aa937483712a7951ca85d18c1d1863959cbdd5af` |
| serving-v4/compare.py | `80a02ecd61479792c9adcd4d2f8dc4d1d24f1aef3569d2a126d445a556b97ffc` |
| serving-v4/SERVING_RUN_SHEET.md | `9f147f270b04b4f10d382537d2e0ccd73ad1b5b5ce8f41dc8040bea22fad2643` |

All manifest entries match. Static inspection of files, saved results/guards, diffs and hashes only. No tests, pre-commit, GPU workloads or server were executed by the reviewer; no source or harness was edited. This review artifact is the sole output change.
