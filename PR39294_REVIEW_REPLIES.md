# PR #39294 — reviewer 답변 후보

대상: https://github.com/sgl-project/sglang/pull/39294

아래 영어 문단은 각 review thread에 붙일 답변 후보다. 수정과 테스트는 **로컬 후보에서 완료**했으며, 현재 GitHub PR에 반영됐다고 표현하지 않는다. C3의 통합 방식은 아직 합의 전이다. 게시 후 실제 commit 링크를 추가하면 reviewer가 변경을 확인하기 쉽다.

## C1 — Mutating recovery inside the eviction predicate

[Reviewer comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049663)

**답변 후보**

Agreed—the eviction predicate should not perform allocation recovery. I’ve implemented and tested a local revision on top of #36729 that keeps the callback pure and moves preparation to explicit allocator boundaries.

The two-pool path uses #36729’s reclaim planner. For tri-pool allocation, the callback can recognize capacity recoverable through permitted END compaction or a specific FLOAT move, without executing either operation. Checking only immediate capacity was insufficient in a temporal-state regression: the revised path stops after reclaiming one FULL token, one SWA token, and one state, then allocates four tokens while preserving the remaining 95 cached tokens and their KV/state data.

The tri path has at most one token-reclaim phase and one state-reclaim phase, with preparation after a phase only when it reports actual reclaim. This removes per-victim recovery calls. The bound applies to explicit preparation attempts; it does not imply that eager frees themselves never move data.

## C2 — Whole-cache fallback for an infeasible demand

[Reviewer comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049666)

**답변 후보**

Agreed. The local revision removes the whole-cache fallback and rejects provably impossible demands before the first destructive walk, using allocator-owned ID-capacity and optimistic byte bounds that account for eligible reclaim. The remaining tri-pool phases use finite, demand-derived quotas.

I added regression coverage asserting that rejected oversized demands leave the cache intact, including locked/pinned cases. I also addressed the speculative-probe concern: ordinary unsharded paged extend now passes its exact required page demand, so an oversized conservative probe does not prevent a feasible allocation.

Passing the byte bound is not treated as proof that the FLOAT layout is feasible; the allocator still verifies actual joint capacity after preparation.

## C3 — Overlap and incompatible dispatch with #36729

[Reviewer comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049670)

**답변 후보 — 권장**

Thanks for pointing out the overlap. I agree that we should settle on one allocator-owned design. I ported the shared-pool regression cases onto #36729 and built the local follow-up around its dispatch and reclaim planner.

Its two-pool planner covers the basic FULL 4 / SWA 4 / joint 3 case. The ported tests also exposed remaining early-stopping and tri-pool recovery cases, for which I have separate fixes and regression coverage ready.

@ZYHowell @ch-wan, would landing #36729 first and keeping #39294 as a focused follow-up work for you? I can share the patches and tests so we can agree on the integration before updating the published implementation. The tri-pool changes are separately reviewable if you prefer to split that work out.

**짧은 대안 — 통합 순서만 먼저 조율할 때**

Agreed. I’ve ported the regression cases onto #36729 and prepared a local follow-up that preserves its allocator-owned dispatch and planner. @ZYHowell @ch-wan, would you prefer landing #36729 first and keeping #39294 for the remaining fixes and tests? I can share the separate patches so we can settle the overlap before either change lands.

## C4 — Redundant StreamingSession branch

[Reviewer comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049675)

**답변 후보**

Agreed. I’ve simplified the local revision to forward the optional callback unconditionally. The wrapper regression tests pass with the shared interface.

## C5 — Query-looking name hides mutation

[Reviewer comment](https://github.com/sgl-project/sglang/pull/39294#discussion_r4003049679)

**답변 후보**

Agreed. I addressed this together with the per-victim recovery issue: `allocation_reclaim_satisfied` now receives a pure allocator sufficiency query, while `ensure_capacity` owns preparation at explicit boundaries. The callback does not drain or compact pools. Its contract also makes clear that a negative result is not a proof of allocation impossibility.

## 별도 validation 요약 댓글 후보

테스트 결과는 각 thread에 반복하기보다, 필요하면 아래 내용을 PR의 일반 댓글로 한 번 공유한다. GitHub CI나 maintainer 승인을 받은 것처럼 표현하지 않는다.

I’ve completed local validation of the integration candidate:

- Python CPU: **276 tests and 1,597 subtests passed**, with 10 GPU cases skipped in this CPU lane.
- Rust cache backend: **36 tests and 138 subtests passed**, with two session-cursor cases excluded.
- All-files pre-commit: **passed**.
- Scoped CUDA validation: **20 scenarios passed; three remain inconclusive**. The passing cases cover allocation, retained KV/state data, movement gates, and an eager writer-dependency case followed by replay of the same captured graph after relocation.

The inconclusive cases are lazy writer ordering and pending-reader source reuse at page sizes 1 and 4: the intended outstanding-event condition was not exercised successfully in this environment. Those results are not counted as asynchronous-safety coverage. These checks also do not establish model-output equivalence or serving performance, and the earlier serving measurements remain tied to the earlier PR revision.

## 로컬 근거 — 게시용 본문에 포함하지 않음

- Tested worktree: `/home/sukwoo24/sglang-eval-worktrees/review-tri-float-candidate`
- Candidate tree: `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`
- #36729 base used for validation: `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`
- [CPU candidate acceptance](/home/sukwoo24/sglang-eval-results/pr39294-review-20260914/TRI_FLOAT_CANDIDATE_ACCEPTANCE.md)
- [GPU results acceptance and limitations](/home/sukwoo24/sglang-eval-results/pr39294-review-20260914/GPU_RESULTS_ACCEPTANCE.md)
- [Detailed status and remaining integration work](/home/sukwoo24/sglang-eval-results/pr39294-review-20260914/FINAL_STATUS_KO.md)

위 acceptance 문서는 지정 Codex 세션의 로컬 검토 기록이다. GitHub reviewer/maintainer의 승인이나 PR merge 승인을 의미하지 않는다.
