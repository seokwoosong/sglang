# Two-pool candidate: implementation and CPU validation for acceptance review

This implements only the scope approved in STAGE_B_REVIEW.md. Tri production
behavior remains unchanged. The complete #39294 follow-up is not merge-ready on
the strength of this candidate alone.

## Source and patch

- Isolated candidate: `/home/sukwoo24/sglang-eval-worktrees/review-joint-candidate`.
- HEAD/base: `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb` (#36729 frozen head).
- Candidate consists of a staged local diff, no new commit or submitted-branch rewrite.
- Full patch: `two-pool-candidate.patch`; hashes: `candidate-manifest.json`.
- Seven files: six production Python files, 102 added/10 removed lines, plus
  567-line registered regression fixture. No CUDA/Rust engine source changes.

## Behavior and ownership

`UnifiedSWATokenToKVPoolAllocator.evict_to_free_tokens` now drains its existing
deferred-free groups once at entry, computes the existing reclaim plan, rejects
unsatisfiable plans before the tree walk, and provides a pure zero-credit
`reclaim_plan(n,n) == (0,0)` predicate. After the bounded quota walk it performs
explicit `ensure_capacity`; actual alloc can repeat the cheap readiness check.

The optional `allocation_reclaim_satisfied` mode is explicitly a cumulative
reclaim-quota contract. It bypasses component shortfall/availability targets and
uses the pure joint predicate to stop after visible controller frees. It does
not restore a full-cache fallback. Default eviction and Mamba donor paths retain
their existing semantics. Unsupported base-cache callback use fails explicitly.
StreamingSession forwards the optional argument without a redundant None branch.

The allocator base has a typed extend-demand hook whose default retains the
existing conservative demand. Only the concrete two-pool implementation replaces
ordinary unsharded demand with the existing exact CPU page-rounding calculation.
The shared SWA base, tri-pool, and shard-size != 1 retain the previous demand.

Rank-consensus input selection compares quota params and callback presence,
excluding process-local closure addresses. The allocator-owned boundary checks
the numeric `num_tokens` and result, preserving demand semantics. The enabled-mode
test exercises the real decorators and compares independently created callbacks'
event traces plus actual allocation and surviving payloads.

## Validation results

| Validation | Result |
|---|---|
| New Python regression file | 14 passed, 39 subtests passed |
| New Rust-backed regression file | 13 passed, 39 subtests passed; 1 session-cursor test deselected because Rust explicitly rejects session-radix-cache |
| Existing allocator/tri/capacity/byte-accounting/eviction/session tests | 196 passed, 1,392 subtests passed, 8 GPU tests skipped |
| All-files pre-commit, including Rust formatting/clippy and registration checks | Passed |
| Diff whitespace checks | Passed |
| Candidate comparison harness | All executed two-pool failures from Stage A resolved; old tri/static-cap-contract failures remain as before |
| GPU/serving/graph execution | NOT_RUN: not authorized in this review stage |

Relevant logs are named in candidate-manifest.json. The broad CPU suite log also
contains 13 new-fixture setup failures from an earlier revision: setting an empty
CUDA_VISIBLE_DEVICES exposed test_utils indexing, then masking CUDA exposed
implicit device/model resolution. Those were fixture setup errors, not 196
existing-test failures. The final registered fixture uses explicit CPU device
and cached Qwen3 config, and the final Python/Rust logs above supersede those
new-test errors. No weights/server/GPU kernel launch is needed for these tests.
CUDA_VISIBLE_DEVICES=999 masks devices without triggering the empty-string parser
bug. `rust-extension-proof.txt` confirms zero CUDA devices and the actual loaded
fingerprinted `mem_cache.cpython-312-x86_64-linux-gnu.so` path.

The first all-files pre-commit runs performed formatting; the final run passes.
Only intended seven files are changed. Staging the new test makes the incremental
registration checker examine that file instead of unrelated historical branch
layout migrations; no lint checker was disabled or modified.

## Acceptance criteria covered

- Page 1/4/16, eager/lazy, demand 3/4/6/90: actual allocation succeeds and exact
  deterministic key/virtual-ID sets 96/95/93/9 retain distinct K/V payloads.
- Impossible byte demand and all-locked requests preserve prefixes and fail
  allocation. An independent asymmetric SWA-restore ID-limit case has ample
  physical bytes/SWA pages but exceeds the FULL owner's virtual namespace;
  both pure plan and ensure reject it. Equal-pair oversized requests also hit
  the byte bound, so they are not described as independent ID-only cases.
- Ordinary/representative groups preserve all 94 prefixes, restart/end scope,
  and leave zero live allocations after explicit cleanup. A new FULL-only case
  really is short until its pending free is applied and also cleans up fully.
- The actual extend caller handles exact 396/coarse 400 with two cached pages,
  zero new pages, page boundaries, mixed prefix lengths and a reclaim case that
  retains 93 prefixes. Output tail indices, unique output slots, new page counts
  and retained data are checked. Only external Triton index generation is replaced
  by a CPU reference, as authorized. Sharded/default/tri dispatch remains default;
  DCP/DSV4 end-to-end behavior is NOT_RUN.
- Real ScheduleBatch.check_decode_mem passes requests/spec_algorithm through the
  existing allocator interface and allocates successfully with expected survivors.
- Pure planner queries preserve independent deep snapshots of nonempty group
  contents, v2p/p2v, holes, live counts and byte frontiers under open/closed gates.
- Internal SWA tombstones produce `node_id=None, made_progress=True`; after their
  real controller frees, the new predicate stops on the FIRST sufficient step.
  Initial arena half-pair slack means one SWA page is sufficient, so the second
  quota victim is preserved. Same case exercises the active Python session cursor.
  The next independent leaf's key/virtual ID and K/V payload survive actual alloc.
- Existing callback-free eviction, Mamba donor/ID recovery, explicit count-based
  eviction and session tests pass. Rust session references are explicitly unsupported
  by the frozen adapter (ValueError at adapter.py:306), not silently treated as PASS.
- Helper-through-alloc instrumentation still separates preparation attempts,
  actual group/controller drains, urgent flush/no-op results and movement.
  Candidate paired pressure uses two cheap ensure calls (helper and alloc), with
  no per-victim prepare. Eager free movement remains attributable to eviction.

## Tri-pool supplement and remaining work

`strict_temporal_live.py` improves the earlier temporal evidence: it predeclares
two required live state IDs outside the cache and one request-owned FULL/SWA KV
token. It asserts they remain bound, rather than skipping negative mappings,
and validates distinct conv, nonempty temporal and K/V payloads. All six
eager/lazy/branch-arm cases pass; results are `*-strict-temporal-cases.json`.

`tri_oracle_probe.py` is still only an experimental sufficient predicate for
END-hole compaction. It recovers the lazy 95-prefix result in the six tested tri
cases, but does not model float relocation or establish completeness over page
grids/pending reuse. It is not in production source. Closed gates and unequal
holes were tested in the two-pool comparison, not as a complete tri oracle sweep.

Still NOT_RUN/incomplete for a full PR: tri oracle geometry/completeness and its
production delta; candidate-specific pending-reuse event transition under the
new predicate (existing allocator event-lifecycle tests passed); actual GPU
movement/event/graph validation; sharded exact-extend e2e; maintainer coordination
on where these patches should land. No global task-accuracy/output-equivalence
claim is made.

## Requested reviewer decision

Please review the staged source/test diff and evidence, and write
`CANDIDATE_ACCEPTANCE.md`. State whether the approved two-pool CPU work is accepted,
or list concrete required revisions. Treat tri production and GPU/remote actions
as separate scope. If more work is necessary to accept this candidate, specify
the bounded authorized delta/validation. The user asked us to follow this session's
review decision; no GitHub comment/push/rewrite/PR closure has been performed.
