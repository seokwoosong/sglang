# PR39294 hardening results — 2026-09-16

## Decision
Two additional pre-existing allocator bugs were reproduced on both the combined base and candidate and corrected. The final narrow fixes passed bounded CPU/Rust and focused CUDA checks. This is not a declaration of universal two/three-pool safety or push/merge readiness. Real serving could not start with the fixed local model/environment; distributed integration and several asynchronous timing conditions remain unestablished.

## Source
- Combined base: `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`, including PR36729 `a6eb82dc77353249fff1b08356533c2dc16ea6ec` and upstream `832ec39cc0324cb0e7823dc8385e27a30c356bdd`.
- Accepted review branch before hardening: `f92803e8f6d88d6d48ad9bceae1c8bcdcbe266fa`.
- Isolated candidate: `/home/sukwoo24/sglang-eval-worktrees/pr39294-gate-memo-hardening`, branch `review/pr39294-gate-memo-hardening`.
- Gate fix commit: `8b15837f2e1d125081b427d1296a6f92fbecf670`.
- Final tested pending fix index tree: `bbf48421d7d23dfd888b687a718e8714b39cea16`.
- Pending diff against gate HEAD SHA256: `1e1afc2ff57e207a92f149c554edf5b4feb4805b5e75412be51ee0d79377b911`.

## Fixes
1. A mutable move gate can change admission without changing the allocator epoch. Cached schedulable capacity then overstates the available capacity (reproducer: cached47 versus actual7). Bypass only gate-dependent schedulable memoization, retaining ungated caching. Registered regression covers mutable callbacks, gate replacement, real prefill queue state, FLOAT neighbors and page sizes1/4/16.
2. Multiple move batches sharing one unfinished event overwrote that event's pending source list, while the CPU pending set accumulated it. Preserve all source lists and tensors under the event. Registered regression covers two-pool FULL and three-pool FULL/Mamba, page sizes1/4/16, same/different/fired-between events, event association, drain and physical reuse with retained payload markers.

Both baseline and candidate CPU reproductions establish that these defects predate this follow-up. No payload corruption was observed in the original pending finding. GPU direct-helper evidence does not establish unmodified production CUDA flush reachability.

## Validation on final pending-fix tree
| Lane | Result | Scope/limit |
| --- | --- | --- |
| Full selected CPU lane | 281 tests /1636 subtests PASS;10 GPU skips | 15 selected files; final log in pending-fix-cpu-completion |
| Focused new regression | 1 test /27 subtests PASS | CPU modeled events, independent mapping-derived source oracle |
| Rust selected lane | 25 tests /131 subtests PASS | One documented session exclusion |
| All-files pre-commit | PASS | No formatter delta; GITHUB_BASE_REF=upstream/main |
| CUDA same-event helper | 4 PASS | Layouts2/3 × pages1/4; both observations unfinished; corrected reader addresses; source union, retained payload, final drain |
| CUDA different-event helper | 4 INCONCLUSIVE | Functional checks complete, events already fired; no concurrent pending evidence |
| Real serving | BLOCKED before inference | First baseline AutoConfig startup rejects granite_swa; no client/candidate/output/pressure/cancellation results |
| P2 local interfaces | 14 tests /25 subtests PASS | Exactly the two approved CPU files;14.04s guarded elapsed; local interface/rejection only |

Tests and subtests are different reporting units and are not added together. Focused cases overlap the broad suite. GPU maximum observed whole-device memory was2812MiB, below3072MiB. Sampling is not continuous memory measurement or process attribution.

Final fix acceptance: [pending-fix-final/REVIEW.md](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/pending-fix-final/REVIEW.md). Raw outputs and hashes: [pending-fix-final/RESULTS.json](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/pending-fix-final/RESULTS.json), [pending-fix-gpu](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/pending-fix-gpu), [pending-fix-cpu-completion](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/pending-fix-cpu-completion).

## Earlier checkpoint evidence — not reruns of final pending tree
- Stateful Python:472 completed sequences,60416 random proposals;181 PASS and291 PARTIAL because required KV ID reuse was not observed. Remaining104 planned sequences NOT_RUN at the bounded deadline.
- Stateful Rust:96 completed sequences,12288 proposals;24 PASS and72 PARTIAL for the same reuse coverage condition. No invariant failure in completed rows. These are proposal counts, including rejected operations, not counts of successful allocations.
- Production GPU hazard harness:48 trials with functional checks complete;3 writer-ordering PASS and45 INCONCLUSIVE. All reader reuse trials remain INCONCLUSIVE. Negative-control sensitivity UNESTABLISHED.
- Graph harness:4 cells ×100 replays; functional checks complete, all4 PARTIAL for missing required owner-specific movement.400 replays do not mean400 movement cycles.
- CPU controls detected the deliberately injected retained-payload and stale-memo defects; they do not establish GPU race sensitivity.

## Corrections and bounded-run history
The original diagnostic wall duration was223s, exceeding its2-minute reduction allowance. TIMING_ADDENDUM.md records this deviation; it was not silently treated as within budget. Later supplemental phases were reviewed with explicit separate limits.

A physical-page versus kernel-page indexing mistake invalidated the old direct-helper moved-source reader-address claim; that claim is withdrawn. The pending ledger failure and separate baseline/candidate CPU flush reproductions remain valid. The new GPU probe uses physical-to-virtual mapping followed by kernel translation for actual reader addresses. Old main reader reuse checks remain INCONCLUSIVE. Initial new CPU test failures were fixture index-unit errors, corrected before the final27-subcase pass. Historical failed and timed-out logs are retained and not relabelled PASS.

The corrected broad CPU attempt timed out under its existing300s phase. A separately reviewed180s completion allowance ran the broad suite and pre-commit once, finishing in107.70s. The focused/Rust lanes were not repeated to inflate coverage.

## Remaining limits
- No demonstrated unmodified CUDA same-event flush/scheduler reachability, unfinished urgent-wait ordering, or safe early reuse proof from the corrected helper probe. Final GPU drain follows host synchronization.
- Different-event GPU overlap and negative-control sensitivity remain unresolved; CPU modeled-event coverage does not substitute for these.
- Fixed local Granite-SWASH snapshot has model type unsupported by the installed Transformers. Startup failure occurred on the baseline before inference, so it supplies no allocator comparison. No package upgrade, config shim, alternate model or automatic retry was performed.
- No complete real three-pool pretrained checkpoint was available. A reduced Inkling fixture is not a model-quality evaluation.
- P2 is limited to local interface/rejection checks, not multi-process DCP/PD or graph performance.
- Reviewer C3 coordination/design agreement and PR CI remain separate requirements. Token sequence parity, if later obtained, would not establish general task accuracy.
- Published PR branch and remote archival review branch have not been pushed or modified by this experiment series.

## Final P2 record

Exact command and guard limits are saved in [p2-final/p2.run.json](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/p2-final/p2.run.json), with [test output](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/p2-final/p2.log) and [source identity](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/p2-final/SOURCE.json). Guard exit0, no failure reason, maximum owned RSS1434038272 bytes. Serving closure and independent P2 authorization are in [serving-startup-disposition/REVIEW.md](/home/sukwoo24/sglang-eval-results/pr39294-hardening-20260916/serving-startup-disposition/REVIEW.md).

## Local preservation

Pending fix code commit: `88988f6ed8552e1f1df15238da8c13365e0c3721`; its tree exactly equals the tested `bbf48421d7d23dfd888b687a718e8714b39cea16`. The gate fix remains a separate commit. This is an edited presentation copy: local artifact links and this provenance section were added; original report bytes remain in the artifact root and its hash is recorded in MANIFEST.json.

The final review authorizes local archival adoption only. No push or published PR update has been performed. This checkpoint is not PR merge readiness. Canonical adoption is recorded separately in the local artifact root after fast-forward.
