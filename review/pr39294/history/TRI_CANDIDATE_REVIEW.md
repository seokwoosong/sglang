# Tri candidate acceptance submission

Request: accept the isolated CPU candidate below, identify any remaining mandatory CPU/code changes, and review `TRI_GPU_PLAN.md` for the next limited GPU lane. This submission does not claim #39294 is merge-ready; integration agreement with #36729 and GPU/serving evidence remain outstanding.

## Exact composition

- Frozen #36729 HEAD/base: `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`.
- Accepted two-pool patch unchanged: SHA256 `cfc5b3c6c2593189b5a1c15a3f5b774eba55d19a1813d32013de0f68ec6cd2bf`; accepted index tree `e4adfed934c78ebd659152d2990de85f6e3e17dc`.
- Separate FLOAT guard plus its registered tests: `tri-float-gate-final.patch`, SHA256 `17f4315c16427607c2de7787b5db265bd8ad05b4a9c5a142b2ec4f596fc0a682`.
- Separate tri policy/tests: `tri-policy-final.patch`, SHA256 `17f1fb3de3fbcf6af687b5b6ede17cc2ac4a36bcf9a7ba5d9516a84484e268bd`.
- Complete composed patch: `tri-composed-final.patch`, SHA256 `fb1484263d9f170ff26b87764af5bcb9987b9251c8948ea23637cb3580a144ec`.
- Candidate index tree: `0b1bd85dd84bad2fb6bc0d4b3c79429dc510304b`; worktree `/home/sukwoo24/sglang-eval-worktrees/review-tri-candidate`.
- Applying guard then tri patches to the accepted tree reproduces that exact candidate tree: `tri-patch-composition-proof.json`. No new commits or remote mutations. Original submitted branch remains clean at `276771eefca951ae6d55608415f2d795cbfabdae`.

## Implementation

The tri allocator now separates physical readiness/END certification from mutation. `_token_reclaim_satisfied` uses host metadata, actual FULL owner IDs, SWA physical-page limits, allowed reusable END holes, and FLOAT page-grid geometry. It never reads FLOAT hole positions to the host, moves pages, drains groups or waits. Pending reuse is excluded from its credit. False remains unknown.

`ensure_capacity` first checks actual index capacity and immediate feasibility. A positive END certificate selects only FULL/Mamba flushes and returns their actual capacity postcondition; no FLOAT fallback rescues a failed END certificate. A negative certificate can select the old ladder only if an optimistic current-byte bound allows it. Nominal partition/conservation caps were removed from these physical paths per the approved amendment. Existing reporting and admission APIs are unchanged.

`evict_to_free_tokens` drains existing composite frees, rejects provably impossible ID/byte demands before a cache walk, tries entry ensure, then executes one demand-bounded FULL/SWA phase and at most one state phase. The callback is pure. State quota subtracts actual Mamba cascades returned by the token phase. No-progress phases do not trigger another expensive ensure. At most three helper ensure boundaries plus the caller's own ensure occur in one helper+alloc attempt. The bound is not global across scheduler retries. Eager free-induced movement is distinct from recovery preparation.

Concrete tri ordinary-unsharded extend uses the accepted exact-page demand hook. Shared/sharded defaults remain unchanged; the existing two-pool dispatch test is updated only to acknowledge tri's now-authorized concrete override.

FLOAT `make_room` and `compact_holes` check the initialized `disagg_move_gate` at their owning boundary before event settling/movement. Closed returns current-side gap or zero moved pages. Open behavior and copy/event algorithms are unchanged. Direct production `make_room` callers are the two `_float_open_short_side` branches; no direct production `compact_holes` callers were found under mem_cache at this SHA. Tests exercise its public movement owner directly.

## Final CPU validation

Environment: `CUDA_VISIBLE_DEVICES=999`, `PYTHONPATH=python`, `FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer`, `/home/sukwoo24/.venv_sglang_upstream_full/bin/python -B -m pytest`. The Rust lane additionally sets `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`.

| Final artifact | Observed result |
| --- | --- |
| `tri-cpu-suite-final-2.log` | **229 passed, 1,489 subtests passed, 8 GPU tests skipped** |
| `tri-rust-final-2.log` | **31 passed, 97 subtests passed, 2 session-cursor tests deselected** |
| `tri-precommit-final-3.log` | **All-files pre-commit PASS**, including Rust fmt/clippy |
| `tri-guard-only-registered-final.log` | Separate registered FLOAT guard tests on the guard-only baseline; see final log |

The CPU command covers multi-ended allocator, tri pool, allocation eviction, capacity memo, byte accounting, Python session/streaming, accepted two-pool joint tests, new tri policy and separate FLOAT gate tests. Existing 72 `alloc(available_size())` geometry subcases pass unchanged. The two Rust exclusions are the existing unsupported session-cursor mode; both pass in Python. The Rust backend is real, with extension proof already recorded for the same frozen Rust source; the guard/tri changes add no Rust source changes.

Final registered tests cover 95-prefix retention, strict request-owned KV and conv/nonempty temporal state, impossible/locked cache preservation, state-only recovery with locked token bindings, real ordinary extend (400 probe vs396 exact), deferred ordinary/representative/FULL-only groups and wrapper scope, session cursor, actual ScheduleBatch decode, physical allocation above nominal partition headroom, deep96-leaf6/90 demand retention, rank-consensus enabled stable numeric/callback-presence traces, partial and exact state-quota cascade credit, END-only recovery, stale gate changes, pending-reuse completion/wait, and both FLOAT movement owners/sides.

Mamba checkpoint eviction can make `match_prefix` stop earlier even when locked FULL/SWA bindings survive. State-only tests therefore assert exact FULL component values by node ID plus actual KV payload, rather than incorrectly requiring a reusable Mamba prefix after its checkpoint is evicted. Outside state IDs are predetermined and must remain mapped.

### Preserved failed iterations and corrected interpretation

- Initial candidate nominal-cap checks broke72 existing cases; `tri-existing-initial.log` is preserved. The approved capacity amendment fixes the source, without weakening those tests.
- Initial new-fixture failures were missing `UnifiedSWAKVPool.k_buffer` (use `layer_num`), expected broadcast marker shape, and unsuitable allocation order for a deliberately tight state fixture. Final tests explicitly assert successful setup and use physical/kernel units correctly.
- A paged tri fixture initially tried to enable Mamba checkpoints without extra buffers; that configuration is unsupported. The page4 exact-extend case uses the real tri allocator and request-owned Mamba state, with FULL/SWA cache components. It does not claim page4 Mamba-cache support.
- Initial grouped checkpoint values2/3 were accepted by Python but **Rust requires exactly one state slot per Mamba node** and rejected them. Those artificial cases are removed from final tests; `tri-new-rust-final.log` remains a failed historical log. Final partial/exact cases use legal single-state nodes on both backends. A greater-than-state-quota grouped synthetic case is **not** claimed as valid runtime coverage; saturation of the remaining quota has source inspection plus exact-zero coverage, not a valid independent greater-than-quota runtime case. Please identify if further realizable coverage is mandatory.
- Test file extraction moved guard-only tests/helpers out of the tri policy test into `test_unified_float_move_gate.py`. `split_tri_gate_tests.py`, saved input and `tri-test-split-proof.txt` verify unchanged test/helper ASTs at extraction. The subsequent legal-single-state correction is intentionally separate from that relocation proof.

## Geometry and retention evidence

- `tri-final-geometry-cases.json`:864 legal layouts;451 positive production certificates, all allocate;484 total actual allocations, so33 unrecognized successes;0 payload/accounting errors. The majority of positives are immediate-ready controls. Previous gated successes that depended on forbidden movement are not safe references for the new guard.
- `tri-final-end-pressure-cases.json`:297 cases;15 initially unready positive certificates across page sizes1/4/16, all realize END-only capacity and allocation with FLOAT flush/move excluded. Negative-query rows are not allocation-success evidence.
- `tri-final-pending-cases.json`:18 PASS cases create genuine CPU pending metadata through nonurgent compaction and verify pure queries, external event completion/wait transitions, allocation and preserved payloads. These are mocked GPU-event control-flow checks, not GPU ordering.
- `tri-candidate-retention-cases.json` versus `tri-pr39294-retention-cases.json`: all six real96-leaf cases allocate, preserving exactly95/93/9 prefixes for demand4/6/90 in eager/lazy modes, with all request-owned state retained. This fixture uses state-before-KV layout; the original state-after-KV factory case is separately tested.
- `tri-candidate-float-retention-cases.json` versus `tri-pr39294-float-retention-cases.json`:12 cases vary equal-count FLOAT holes at positions(4,7)/(5,6), demand4/6/12 and eager/lazy. All retain all4 cached prefixes, protected extra FULL-only bindings, and live payloads; each uses one explicit ladder. Entry recovery succeeds, so this proves first-feasible recovery at entry on these layouts, not a complete mid-walk FLOAT optimizer.
- Guard-only `tri-guard-only-cases.json`: all three baseline closed-gate move reproductions now have zero FLOAT copies in ensure and final alloc and return temporarily unsatisfied. Separate registered tests assert payload, both owner methods/sides, hole reuse, and state/token recovery after reopening.

## Review comment status and remaining limits

| Comment | Candidate response | Still needed outside local CPU scope |
| --- | --- | --- |
| C1 mutating per-victim predicate | Pure callback; fixed resource-phase preparation boundaries; deep/live-state tests | Focused actual GPU event/movement validation |
| C2 whole-cache fallback on impossible demand | Pure ID/optimistic-byte rejection before first walk; finite request quotas; no whole-cache escalation | No claim of complete infeasibility solver for all FLOAT layouts |
| C3 overlap with #36729 | Candidate composes on exact #36729 allocator dispatch, preserves two-pool regressions, adds tri policy separately | Author/maintainer agreement on integration and refreshed heads |
| C4 redundant wrapper branch | Already fixed by accepted unconditional forwarding | Nothing further in local code |
| C5 misleading mutating query name | Callback now actually pure; movement is allocator-owned ensure | Nothing further in local code |

No universal FLOAT planner, DCP/spec/sharded exact-demand expansion, real distributed communication, serving performance/accuracy, live PD/RDMA traffic, or full-model CUDA-graph validation is claimed. Greater-than-quota single-state cascade and a fully blocked no-progress-after-real-walk layout are not each isolated as independent final tests; current source paths are finite and use real returned counts, and reviewed primary cases cover their surrounding behavior. Request explicit acceptance judgment on these remaining corners rather than silently treating them as executed.

## Next requested decision

Please write `TRI_CANDIDATE_ACCEPTANCE.md` with mandatory changes, or acceptance of this CPU candidate and its stated limits. Also review `TRI_GPU_PLAN.md`: one bounded local GPU harness lane, no serving/model downloads or remote actions. User's objective remains all comments addressed and PR merged; neither local acceptance nor a GPU marker pass settles C3 or maintainer merge approval.
