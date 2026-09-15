# PR39294 rebased integration results

Candidate worktree: /home/sukwoo24/sglang-eval-worktrees/pr39294-rebase-20260915
HEAD: `276f389f0410a5ddc57fe71a5ea40fb13bf25c66`; tree: `d8f56aeb88c02a056f8defe4b8f47c68f6c777bf`.
Combined base: `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`; parents: a6eb82dc77353249fff1b08356533c2dc16ea6ec / 832ec39cc0324cb0e7823dc8385e27a30c356bdd.

Latest fetched OPEN PR36729 a6eb82dc77 and upstream main 832ec39cc0 were combined with a merge commit. Our four follow-ups were rebased from explicit old base 6c8bbdf610. All four range-diff entries are `=`; no conflict or production/test integration edit was necessary. Working tree is clean. Published PR39294 and remote archive remain unchanged. Backup branch points to old review HEAD 68a70f6cc8.

## New validation on this candidate

| Lane | Result | Evidence |
| --- | --- | --- |
| Existing 13 CPU files | 276 PASS / 1597 subtests / 10 GPU skips | cpu-13.log |
| Actual Rust joint/tri/gate plus upstream Rust integration | 141 PASS / 138 subtests / 1 skip / 2 deselected | rust.log |
| Upstream session-lock, streaming-session, row-coalescing and pool configurator | 56 PASS / 7 subtests | upstream-compat-cpu.log |
| Python session_id/cache_salt metadata | 2 PASS / 2613 deselected | session-metadata.log |
| Same-rid distinct attempts through real cache + StreamingSession | PASS: success preserves, abort isolates old attempt, direct release forwards | attempt-probe.py/log |
| Real config_from_budget -> derive sizes -> real SWA pool factory | 4 PASS: uncapped/capped/speculative/draft | config-budget-probe.py/json; config-budget-final-v3.log |
| CUDA tri allocation/retained payload | 6 PASS | tri-gpu-v3-cases.json |
| CUDA two-pool and paged tri | 10 PASS | gpu-extra-cases.json |
| CUDA independent movement gates including allocation | 3 PASS | gpu-gate-pending-v3-cases.json |
| Whole-repository pre-commit, correct upstream comparison base | PASS | precommit-upstream-base.log |

Rust provenance is in rust-provenance.json: fingerprinted extension includes session_id argument; upstream insertion/event tests passed against it. Manifest hashes integrated radix sources and loaded binary. Two session-cursor exclusions were reassessed: adapter.py explicitly rejects enable_session_radix_cache (lines 298–302); session metadata does not add cursor support. One Rust integration skip is the CUDA-required test at test_rust_tree_core_integration.py:1193; it is not claimed passed.

Budget fixture uses CPU, page256, head_dim10, budget140001. Uncapped actual pool139264 vs rounded target133120 (6144-byte remainder). Capped/speculative/draft each allocate61440; capped full tokens<=1024 and profiled budgetNone, draft carried target envelopeNone. The actual config factory, token constraints, size derivation and pool factory execute; only runner configuration is a fixture. Initial smaller fixtures either lacked remainder or violated the existing bs1 floor; those failed exploratory assertions/logs are preserved and are not reported as production regressions or PASS evidence.

An initial broad upstream cache-file run under hidden CUDA without CPU-engine selection failed device fixture detection and was interrupted. No assertions were weakened. Focused applicable tests were rerun with SGLANG_USE_CPU_ENGINE=1; unrelated full GPU fixture matrix remains unrun.

Initial all-files precommit failed taxonomy checks because the hook defaults to stale origin/main b276a9ac. Its _changed_registered_files reads GITHUB_BASE_REF, tries origin/<base> then <base>. Re-run used GITHUB_BASE_REF=upstream/main (832ec39cc0), the actual intended base; all hooks passed, no source edits. No hooks skipped or lint rules modified.

GPU controls are exact prior accepted implementations with only top-level selection changed: omit event_graph row, omit superseded gates/pending from extra harness, retain corrected three gate rows. 19 total PASS, sequential processes, RTX5090, >=4GiB free checked in each script, <1GiB allocated assertion, external180s timeout. No delay-forcing/pending overlap experiments rerun. Previous lazy ordering/pending-reader limitations remain unresolved; no new ordering, graph, serving, accuracy, performance or complete distributed coverage claimed.

## Commands

All commands ran in candidate worktree. Python: /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B. PYTHONPATH=python; FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer.
- CPU13: bash /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/run_tri_float_cpu.sh (script sets CUDA_VISIBLE_DEVICES=999).
- Rust: CUDA_VISIBLE_DEVICES=999 SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust python -m pytest -q test/registered/unit/mem_cache/{test_unified_joint_allocation_eviction,test_unified_tri_joint_reclaim,test_unified_float_move_gate,test_rust_tree_core_integration}.py -k 'not session_cursor'.
- Compatibility: CUDA_VISIBLE_DEVICES=999 SGLANG_USE_CPU_ENGINE=1 python -m pytest -q -x test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py::TestStreamingSessionLockLifecycle test/registered/unit/mem_cache/test_streaming_session_unit.py test/registered/unit/mem_cache/test_free_kv_row_coalesce.py test/registered/unit/model_executor/test_pool_configurator.py.
- Metadata: same CPU-engine environment, pytest -q -x test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k 'session_id_is_attributed or session_id_and_cache_salt'.
- Probes: CUDA_VISIBLE_DEVICES=999 python <artifact-dir>/config-budget-probe.py and attempt-probe.py.
- GPU: timeout --signal=TERM 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer python <artifact-dir>/gpu_tri_controls.py, gpu_duo_paged_controls.py, gpu_gate_controls.py, sequentially.
- Lint: GITHUB_BASE_REF=upstream/main /home/sukwoo24/.venv_sglang/bin/pre-commit run --all-files.

## Requested disposition

Approve adopting this tested candidate as local review/pr39294-unified-joint-allocation, retaining backup and remote archive. Then append current integration documentation and unchanged decision artifact; historical manifests/docs stay intact. No source changes needed, no push. Please write final review to /tmp/PR39294_REBASE_ACCEPTANCE.md (review session cannot write artifact directory); primary agent will copy it unchanged.

Documentation note: the acceptance hashes REVIEWED_REBASE_RESULTS.md. This current copy clarifies the absolute CPU command and identifies the existing CUDA-only Rust skip; code and validation results are unchanged.

The branch copy of rust-provenance.json has a final newline added by pre-commit; its data is unchanged. candidate-manifest.json records hashes of the original local artifacts, including the original provenance JSON.
