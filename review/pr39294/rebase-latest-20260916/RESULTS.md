# PR39294 review branch: latest-upstream rebase (2026-09-16)

The review branch was rebased onto fetched upstream/main `a3bf25dc620f31fc672aeced1465d6fe7c81d28f`. This upstream contains the actual squash merge of #36729, `2929a39927a3943cee03e498f4e5f651185f1b1f`. The old PR36729 development history and integration merge were not replayed. Only our eight follow-up commits were selected.

- Previous review HEAD: `a81393946af98619c6f669dbfdc7dd3eada1080a`.
- Backup: `backup/pr39294-review-before-upstream-20260916` at that previous HEAD.
- Old combined base: `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`.
- Rebased/tested HEAD: `15ccf48036bf5de0ec0aa886dedaa74bbdea8123`.
- Tested code tree: `ac086271d5e3bee1f32f84595f9a860ca5151ed4`.
- Worktree: `/home/sukwoo24/sglang-eval-worktrees/pr39294-review`.
- Branch: `review/pr39294-unified-joint-allocation`.

Rebase command: `git rebase --onto a3bf25dc620f31fc672aeced1465d6fe7c81d28f 0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`.

All eight commits replayed without conflicts, empty commits or manual source changes. Range-diff marks every commit `=`. New HEAD descends from the pinned upstream. Existing history documents retained their old source identities and test claims; they are not new-base test evidence.

## New-base validation

| Lane | Result | Scope |
| --- | --- | --- |
| CPU | 295 tests /1661 subtests PASS,10 GPU skips | Previous15 selected allocator/cache files plus2 P2 files; guarded78.29s |
| Rust first startup | NOT_RUN | No current-source cached/bundled extension; build disabled; pytest not entered |
| Offline Rust source build | PASS | Existing loader, cached dependencies/toolchain only, no installation/download; guarded48.75s |
| Rust after build | 25 tests /131 subtests PASS,1 session exclusion | Pending-event, dynamic-gate and tri-joint-reclaim tests; guarded52.21s |
| First all-files pre-commit | Formatting edits required | Five archived JSON files lacked final newline; all other hooks passed |
| Final all-files pre-commit | PASS | Approved rerun completed in21.08s; no hook edits |

Counts are separate reporting units; do not add subtests to tests or infer unique coverage by summing overlapping lanes. CPU includes existing FULL KV+Mamba, FULL KV+SWA, and tri-pool regressions, but these are not equivalent GPU/serving coverage claims.

Python is `/home/sukwoo24/.venv_sglang_upstream_full/bin/python -B`, version3.12.3; torch2.13.0+cu130. CPU runs use CUDA_VISIBLE_DEVICES=999, SGLANG_USE_CPU_ENGINE=1, PYTHONPATH pinned to this worktree/python and FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer. Rust explicitly selects SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust. The build uses SGLANG_RUST_BUILD_MODE=auto, CARGO_NET_OFFLINE=true and RUSTUP_AUTO_INSTALL=0; tests use build mode never. All-files hooks use GITHUB_BASE_REF=upstream/main pinned to the SHA above, with pre-commit4.6.1 at `/home/sukwoo24/.venv_sglang/bin/pre-commit`.

The loaded Rust artifact has fingerprint `7fae3ea16abf42d08df52e7d594d6f5bcdcd7ac0bc2ea096e1ecfa199ebc5674`, SHA256 `1709a59dc2fb15a0f27e6bbce0dcb0d4906e39c0ec5e42a50bac19b886b6d42d`. The test process loaded the same artifact path and its hash was rechecked. Source loader and allocator/cache paths were recorded. No Python fallback is counted as a Rust result.

Each invocation retained its fixed deadline and8GiB RSS/32MiB log guard. The build and successful Rust rerun used explicitly approved supplemental allowances, not a reset of the missing-cache attempt. Final hooks likewise require the separate approved allowance. Original attempts/logs remain preserved.

## Documentation formatting and remaining limits

The only post-test edits are review documentation. The five automatic EOF edits add exactly one newline to their HEAD contents. Their manifest preserves original source hashes and original copied hashes, records new presentation-copy hashes and points to the pre-format commit. No recorded experiment value changes. Raw artifacts remain immutable.

No GPU, random matrix or real-model serving was rerun on this rebased tree. Historical blocked Granite startup, incomplete/unestablished asynchronous timing, real tri-model and distributed coverage remain unresolved. CPU/Rust passes do not prove those paths. C3 coordination and PR CI remain separate requirements.

This operation is local. The archival remote remains at the previously pushed HEAD until separately authorized; the published PR branch is untouched. Rebase completion is not push/merge readiness.

Raw commands, complete logs, guards, source identities, artifact checks and review approvals are in `/home/sukwoo24/sglang-eval-results/pr39294-rebase-latest-20260916/`. Repository copies are indexed by the accompanying MANIFEST.json; raw logs and binaries are not committed.
