# Production FLOAT supplement: renewed candidate review request

Worktree `/home/sukwoo24/sglang-eval-worktrees/review-tri-float-candidate`, base #36729 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`, index tree `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`, ten staged files and no unstaged delta. Frozen failed candidate tree `0b1bd85dd84bad2fb6bc0d4b3c79429dc510304b` is preserved. No commit, submitted-branch rewrite or remote mutation.

`tri-float-supplement-final.patch` changes only concrete tri allocator and its registered tests: 349 insertions/21 deletions relative to the frozen candidate. `tri-float-composed-final.patch` is the full overlay on #36729. Exact hashes in `tri-float-final-manifest.json`.

## Source contracts

- Shared `_token_end_frontiers` projects only permitted reusable END holes; no pending credit or FLOAT hole positions.
- `_token_end_reclaim_satisfied` preserves the original END proof. `_token_reclaim_satisfied` is END OR `_token_float_high_target` and remains pure.
- HIGH target uses approved opposite fresh-gap reservation and inward page grids. Executor flushes FULL then Mamba, checks IDs/readiness, recomputes with **flush_ends=False** so it cannot credit another unexecuted END flush, performs at most one matching HIGH move, verifies capacity. No ladder rescues a failed positive plan.
- Existing two phases, quotas and two-pool/gate source are unchanged. Supplemental partial-move comments make no minimum-copy claim.

## Completed validation

| Run | Result |
| --- | --- |
| `run_tri_float_cpu.sh` / `tri-float-cpu-final-1.log` | 276 passed, 1597 subtests, 10 GPU skips |
| Rust joint + tri + gate, excluding session_cursor / `tri-float-rust-final-1.log` | 36 passed, 138 subtests, 2 deselected |
| `pre-commit run --all-files` / `tri-float-precommit-final-1.log` | all PASS including Rust fmt/clippy |
| Production strict geometry, `tri_float_production_geometry.py` | 864 cases,477 positives,26 HIGH-only,zero errors/false positives; unrelated ladder forbidden for positives |
| Production pending metadata, `tri_float_production_pending.py` |18 PASS |

The selected CPU script records all13 files and the exact environment/command. It is larger than the earlier229-test selection; do not describe this count change as solely five new methods. All previously added joint/tri/gate tests remain selected. No model/server/download/GPU run in these commands.

Environment: `CUDA_VISIBLE_DEVICES=999 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer`; interpreter `/home/sukwoo24/.venv_sglang_upstream_full/bin/python -B -m pytest -q`, cwd is new worktree. Rust adds `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`, selects `test_unified_joint_allocation_eviction.py test_unified_tri_joint_reclaim.py test_unified_float_move_gate.py -k 'not session_cursor'`. The actual Rust extension is unchanged from the prior proven build; no Rust API/source changes.

Five new registered methods cover:

1. Exact temporal state-first legal factory case, eager/lazy, supported Python/Rust single-state nodes: first-phase1/1/1, exactly95 cache tokens, KV/conv/temporal markers, pure query snapshot, one freshly matched256-byte target, <=3 preparations including allocation, no fallback.
2. Whole FLOAT movement with equal two-hole counts at (4,7)/(5,6), eager/lazy and asymmetric FULL:SWA4:2; verifies R>=V, actual whole branch once, retained payload and FULL-only payload.
3. Exact fresh-gap boundary and one-page-short rejection at page sizes1/4/16. The short case legally binds an existing FULL-only ID into LOW, never edits watermarks or counterfeits holes. Equality cases execute the partial branch, allocate and preserve payload without a fallback. Unknown is not called impossible.
4. Geometry matrix positives across page sizes1/4/16, ratios1:1/1:4/4:1, both compaction modes, holes/transparent layouts and closed gates. Nontrivial HIGH coverage required at every page size.
5. Independently close FULL/Mamba/FLOAT gates during preparation after an initial256-byte query. FULL closed causes fresh target224 and succeeds with reusable holes; Mamba/FLOAT closed cause temporary false with no FLOAT move/fallback. Reopening each allows successful allocation with retained payload. Existing END-only tests now explicitly query the renamed END predicate; they still require zero FLOAT movement. Existing pending metadata checks are preserved and the additional18-case production probe passes.

The first test-writing run put the new factory method in the wrong test class and failed with AttributeError before source execution. It was moved to the correct existing class; no production regression was concealed or expected assertion weakened. `tri-float-targeted-2.log` then passes20 methods; the gate-change method was added afterward and the final selected suites pass21 tri methods.

## GPU and integration status

GPU remains stopped. `gpu_tri_validation_v2.py` corrects the separate harness review: real FULL/SWA/state lock receipt protects last cached KV; only this KV and request-owned last state are written; separate original physical destinations are cloned before the delay; only those markers get+17; captured read keeps virtual translation; settle-entry event observations distinguish inconclusive overlap. Lock receipt released after synchronize; old harness/logs retained. Syntax checked, not executed. Exact hash is in manifest.

Request renewed production-candidate acceptance and a decision on resuming the bounded GPU lane with corrected harness. The initial8 rows do not alone cover pending readers, independent gates, two-pool CUDA or all page sizes; do not close the full GPU plan from8 passes. No merge-ready claim, C3 agreement or public reply follows from this review.
