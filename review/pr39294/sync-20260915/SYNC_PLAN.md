# PR39294 integration plan — 2026-09-15

Status: awaiting approval from designated review session.

## Exact inputs
- Archival review branch: 68a70f6cc8ca174dd75606723424b210cb664ed3 (approved follow-up code plus historical review documents).
- Latest fetched PR36729: a6eb82dc77353249fff1b08356533c2dc16ea6ec, still OPEN; prior tested head 6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb.
- Latest fetched upstream main: 832ec39cc0324cb0e7823dc8385e27a30c356bdd.
- Published PR39294 remains 276771eefca951ae6d55608415f2d795cbfabdae.

Read-only merge-tree checks: review/latest36729, review/main and latest36729/main each merge cleanly. Published39294/main conflicts in unified_radix_cache.py. Pairwise results do not establish a clean or correct combined integration.

## Implementation scope
1. Create isolated integration worktree/branch from archival review HEAD, preserving all frozen candidates and original PR branch.
2. Merge pinned latest36729, then pinned main using merge commits. Preserve PR36729 allocator-owned shared-byte planner and all accepted bounded reclaim, FLOAT gates and tri-pool HIGH certificate behavior. Do not transplant the older published implementation or restore its conflicting callback design.
3. Inspect complete combined result. Incoming PR36729 changes the uncapped draft-free profiled-byte-budget condition in kv_cache_configurator.py. Incoming main changes session ownership, request-attempt handles, row-free coalescing and Python/Rust tree/controller interfaces. Check our callback/eviction/session forwarding against these changes, including ID/byte distinction, controller-visible progress, finite quotas, no movement in pure predicates and session ownership.
4. Fix only concrete integration regressions, with focused meaningful tests. Material allocator algorithm redesign or newly required production concurrency changes require a supplemental review first.

## Validation
- Run the previous 13-file CPU lane (run_tri_float_cpu.sh in prior artifact directory) against the integrated worktree. Preserve new logs separately; old 276/1597 results remain historical.
- Run affected upstream unified radix cache/session/row-release and Rust integration tests as supported by their CPU fixtures. Inspect changed Rust source/extension provenance and rebuild locally if the loaded extension is stale; do not count a Python fallback as Rust.
- Run actual Rust joint/tri/gate lane (previously 36/138, two known session-cursor exclusions); reassess exclusions if upstream changed the applicable contract.
- Inspect coverage for latest36729 byte-budget selection: uncapped non-spec preserves profiled remainder; capped and draft/spec paths preserve intended caps. Add a focused real configurator test only if current coverage misses the changed branch.
- Run relevant formatting/static checks and repository all-files pre-commit after final changes.
- Rerun bounded existing corrected GPU allocation/data controls, paged cases and independent movement gates against merged code if CUDA is available: one process, >=4GiB free, <1GiB fixture, external 180s timeout. Reuse accepted harness semantics and record exact new paths/hashes. Stop on CUDA errors/timeouts/retention failures. No model/server/download or throughput experiment.
- Do not repeat inconclusive overlap-forcing probes. Previous lazy writer ordering and pending-reader GPU limitations remain explicit; this integration does not upgrade them to PASS.

## Completion and scope boundaries
Submit final combined source/result manifest to the same reviewer. On approval, fast-forward local archival review branch to integration result and add current integration status plus new review documents, preserving historical snapshots/manifests. No actual PR39294 branch rewrite, remote PR merge, GitHub/Slack reply or new PR. Preserve current remote archival snapshot until validated; any subsequent archival push must remain a normal fast-forward to the same separately authorized review branch.

When PR36729 actually merges, verify the merged commit and remaining delta again; this refresh uses an OPEN PR head, not a claimed merged baseline.
