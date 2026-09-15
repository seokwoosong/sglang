# PR39294 gate fix and resumption review — 2026-09-16

**Patch: APPROVED. Prepared CPU/GPU resumption: CHANGES_REQUESTED; do not launch these exact harnesses yet.** The memo correction and the experimental harnesses have separate dispositions. The harness findings below are static review findings, not observed production corruption or results of new experiments.

Reviewed identity:

| Item | SHA / SHA256 |
| --- | --- |
| Worktree HEAD/base | `f92803e8f6d88d6d48ad9bceae1c8bcdcbe266fa` |
| Submitted index tree | `e390f8a18b5ca65d8b5235e90cb00a5c30fd501a` |
| Exact staged patch | `4adbc22d25e973542ea8022c4394bda8855a7335efeb4e3cf0e464e7f1f05333` |
| GATE_FIX_MANIFEST.json | `79fb9f577ad5b86aa05d6c736f87a746faef9cfc88ff9e4101904592957833c6` |
| GATE_FIX_REVIEW_REQUEST.md | `ca461cf02867aaadc1aba8963233133720e9453c06d1fd93391a37a0d5815421` |
| stateful-resumed.py | `4667160d02b00cf12114cd818038cf7a9074d9f533f8dafcc10869a5523f5592` |
| gpu_hardening.py | `8d9a58b19595c215c3ee31a22a7b7b1d315e30ac9dc13034163f7df0fafd8e4d` |

All listed source/artifact hashes match the manifest. Independently hashing `git diff --cached --binary` gives the exact patch hash above. The acceptance is bound to those staged bytes; no write-tree or experiment was needed for this review.

Patch acceptance rationale:

- The query bypasses only schedulable memoization when relevant lazy peers have a dynamic disaggregation gate installed. The gate-presence check does not invoke the callback. END uses its growth-side nontransparent peer; FLOAT checks both sides using the same traversal as its credit computation.
- A bypassed query clears its schedulable memo sentinel. The verifier excludes gate-dependent entries from epoch-valid memo checks, while preserving immediate and ungated coherence checks. Installation/replacement after priming is handled; mutable callback results need no scheduler invalidation hook.
- Immediate and composite capacity semantics, actual movement protection and host-transfer gate behavior are unchanged. The additional short peer walk is explicit; ungated computation remains memoized.
- The real public-setter, pre-query verifier, transparent/live FLOAT and prefill-queue tests address the reproduced defect. The submitted reduced replay now reports `47/7/7` with no verifier error. Logs show final focused 4 tests/12 subtests PASS, earlier identical production logic across 79 tests/807 subtests and actual Rust 39 tests/150 subtests with 2 session exclusions, and final all-files pre-commit PASS. Broader counts preceded the documented docstring/test-only additions and are accepted with that qualification; they are not all claimed to have run on the final test file.

No mandatory production patch correction was found. Preserve this as a separate accepted local correction to the existing gate-memo defect. This is not a branch adoption, push, PR merge or hardening-completion decision.

Mandatory harness corrections before resumption:

1. **Fix marker arithmetic in both CPU and GPU harnesses.** `stateful-resumed.py:18–20` and `gpu_hardening.py:38–41` construct `16 ** torch.arange(width)` in signed int64. At channel 16, the divisor is 2^64 and overflows to zero; larger widths cannot support that encoding. The GPU temporal shape `(1,4,8)` has width 32, and several CPU tri geometries are wider still. The Python `16**width` assertion does not protect the tensor arithmetic. Bound extraction to the number of significant digits supported by the ordinal's integer range, then fill remaining coordinates with a defined exact pattern, or use another overflow-safe encoding. Validate injectivity over the actual domain and distinct component/in-page/generation information. Do not classify an invalid marker division as an allocator failure.

2. **Make CPU coverage assertions match the claimed lifecycle.** In `stateful-resumed.py:103–106`, assigning `refs[2]=2` does not create two real reference holders, and the sole acquired lock is released before any protected eviction/check. The `refs` ledger is never consulted. Add a real shared-prefix holder/lock lifecycle with an intervening reclaim and survival check, or explicitly leave alias ownership coverage PARTIAL. Include the required group drain, lock/unlock and state-generation reuse transitions in the coverage predicate where applicable; state alloc/free counters alone do not prove reuse with retained payload. Keep proposed random steps distinct from the deterministic prefix and record the prefix in reproducible failure evidence. Assert the instantiated tree backend and record the loaded extension: `--backend rust` currently only selects rows/names and does not itself enable Rust. Do not claim Python/Rust differential agreement without an explicit legal-trace comparison.

3. **Tie GPU pending PASS to the same event, source and release boundary.** At `gpu_hardening.py:89–110`, `bool(pending_set)`, reuse of any moved source and any unfinished wait on the scheduler stream can independently satisfy `reached`. That does not prove the reused source was quarantined for the observed event or released through the urgent drain being tested. Record operation phases and event identity on each original wait. Require a nonempty intersection of the reader's original sources, actually moved sources, sources pending for that event, and the exact newly written allocation. Observe that event unfinished at the original urgent-drain wait. Preserve the nonurgent quarantine assertion for that same set, then record the ordered reuse stream. Otherwise report the missing prerequisite as INCONCLUSIVE. Keep functional result and sensitivity separate.

4. **Validate all retained allocations and generation changes on GPU.** `hazard()` stamps `vs` but validates only `chosen`; recovery or arrangement B can affect other retained pools. After the hazard, compare all retained KV and nonempty conv/temporal payloads, allowing the writer delta only on its declared set. The graph loop currently rewrites neither recycled loose allocations nor protected generations: fixed `gold` can miss stale-generation reads. Add bounded generation-changing exact markers and assert the same captured reader sees them after actual owner-specific movement. Retain fixed buffers/shapes, masks and sentinels. Report per-owner movement cycles separately; a single mapping change cannot be described as 100 movement cycles.

5. **Make the process guard fail closed and always clean up.** `run_guard.py:18` can raise an uncaught `TimeoutExpired`/launch error from `nvidia-smi`; child cleanup is not in a finally block. `members` may also be unset if process inspection failed, and nonzero/invalid GPU-query output can currently appear as zero usage. Any monitor failure must record an explicit reason and TERM/KILL the owned process group before exiting. Preserve cleanup even if the group leader exits while descendants remain. This must not depend on CUDA synchronization. Use a monotonic shared deadline, cap each launch by remaining budget, and preserve atomic/streamed per-row evidence before a watchdog kill. The GPU script currently prints successful rows only after all trials in the process finish.

6. **Preserve evidence files and bind launch records.** The resumed CPU failure path writes `failure-python.json`/`failure-rust.json`, overwriting the prior stopped batch's hashed artifacts. Use a new immutable run directory or unique resumed names for failures, rows, logs and manifests. Record cwd, HEAD plus patch/source hashes, backend/module paths, guard command and deadline. GPU `source_root` alone does not pin the staged source. Record seed/cell/arrangement, device/software versions and limits without inserting device reads in the hazard window.

Scope/priority for the next submission:

- Keep the accepted gate patch frozen. Correct only test/observer/guard issues and submit new harness hashes. Do not modify production synchronization or capacity rules to make a hazard reachable.
- A new explicit 20-minute CPU batch after this source correction is reasonable, preserving the prior stopped batch. The matrix, two-process/8-GiB-per-tree limits and finite prefix/reduction budget remain ceilings. CPU-only resumption can be approved separately once its encoding, coverage, provenance and cleanup issues are corrected; it need not wait for GPU reachability.
- For GPU, first review the corrected production-hazard and graph harness. Retain the single shared 20-minute deadline, 180-second process timeout plus 5-second kill grace, one process, at least 4 GiB free, less than 1 GiB fixture/harness/graph allocations and the 3 GiB process-memory ceiling. A final PyTorch peak assertion alone is not live enforcement of the fixture limit; bounded construction sizes and guard monitoring must support that claim.
- Arrangement B's independent `a.alloc` is an allocator-level scheduling arrangement, not full scheduler execution; retain that explicit label. Correct nonurgent writer deferral or production H2D settling remains INCONCLUSIVE rather than grounds to remove production protection. The current scope has no implemented negative or white-box controls; those remain NOT_RUN pending the promised concrete setup review.
- Existing PARTIAL/NOT_RUN labels, historical GPU19 evidence, incomplete tri serving coverage, C3/CI and merge prerequisites remain unchanged. A harness setup failure is not a new production defect.

The user has authorized continuing hardening work, but the exact prepared harnesses have not passed this required resumption checkpoint. Submit the bounded corrections above for targeted re-review before launching new experiments. No implementation, test, build, GPU workload, source modification or remote action was performed by this reviewer; only static inspection, hash verification and this artifact were completed.
