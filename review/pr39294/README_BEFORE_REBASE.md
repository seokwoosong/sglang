# PR #39294 review branch

이 branch는 reviewer 대응을 검토하기 위한 로컬 묶음이다. #36729의 검증한 head를 기반으로 최종 코드, 회귀 테스트와 관련 Markdown 문서를 포함한다. GitHub PR에 반영됐거나 maintainer의 merge 승인을 받은 상태를 뜻하지 않는다.

## 먼저 읽을 문서

- [Reviewer C1–C5 영어 답변 후보](../../PR39294_REVIEW_REPLIES.md)
- [최종 대응 상태와 남은 통합 작업](history/FINAL_STATUS_KO.md)
- [최종 CPU 후보 검토 승인](history/TRI_FLOAT_CANDIDATE_ACCEPTANCE.md)
- [최종 GPU 결과 수용 및 검증 한계](history/GPU_RESULTS_ACCEPTANCE.md)
- [#36729 통합 조율 댓글](history/COORDINATION_COMMENT_READY.md)
- [통합 후 사용할 PR 본문 후보](history/PR_UNIFIED_JOINT_ALLOCATION_REVIEW_UPDATE.md)

## 코드와 기준 revision

Base: #36729의 `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`.

| Commit | 내용 |
| --- | --- |
| `7b98cd9a86` | Two-pool joint reclaim, 순수 callback, impossible-demand 대응과 회귀 테스트 |
| `899fe03521` | FLOAT movement gate 수정과 테스트 |
| `c3903a5be8` | Tri-pool reclaim 및 FLOAT recovery, 95토큰 보존 회귀 테스트 |

세 번째 코드 commit의 전체 Git tree는 검증·수용한 `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`와 동일하다. 그 다음 commit은 검토 문서만 추가한다. 코드 수정 없이 기존 검증 결과에 대응하는 내용을 branch로 구성했다.

```bash
git diff 6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb c3903a5be8 -- python test
```

## 검증 결과와 미확정 범위

- Python CPU: 276 tests / 1,597 subtests PASS, 10 GPU skips.
- 실제 Rust backend: 36 tests / 138 subtests PASS, 2 session-cursor exclusions.
- 검증된 코드의 전체 pre-commit: PASS.
- GPU의 23개 고유 시나리오: 20 PASS / 3 INCONCLUSIVE. Lazy writer ordering과 page size 1/4의 실제 pending-reader source reuse는 입증되지 않았다.
- 지정 Codex 세션의 로컬 검토에서는 이 한계를 명시해 수용했다. 모델 정확도·출력 동등성·serving 속도를 새로 검증했다는 뜻은 아니다.

## 문서 이력 읽는 법

루트의 PR/초기 review 문서 4개와 `history/`의 관련 Markdown 33개를 포함한다. 아래 이력에는 초기 계획, 실패한 중간 후보, 수정 요청, 이후 승인과 이전 답변 초안이 함께 들어 있다. **최종 상태는 위의 최종 CPU/GPU 승인과 최신 답변 후보를 기준으로 읽는다.** 과거 문서의 미승인·실패 상태는 당시의 기록이다.

기존 문서 본문의 절대 경로와 command는 원래 실행 환경/시점을 가리킨다. 로컬 결과 디렉터리의 raw 로그·실험 스크립트 전체를 branch에 복제한 것은 아니다. 문서 간 탐색에는 이 README의 상대 링크를 사용할 수 있다.

[DOCUMENT_MANIFEST.json](DOCUMENT_MANIFEST.json)은 포함된 37개 원본 문서의 경로·SHA256, branch 사본의 SHA256과 코드 commit/tree를 기록한다.

## 전체 review 문서

- [CANDIDATE_ACCEPTANCE.md](history/CANDIDATE_ACCEPTANCE.md)
- [CANDIDATE_REVIEW.md](history/CANDIDATE_REVIEW.md)
- [COORDINATION_COMMENT_READY.md](history/COORDINATION_COMMENT_READY.md)
- [FINAL_STATUS_KO.md](history/FINAL_STATUS_KO.md)
- [GPU_JOINT_EXTRA_REVIEW.md](history/GPU_JOINT_EXTRA_REVIEW.md)
- [GPU_RESULTS_ACCEPTANCE.md](history/GPU_RESULTS_ACCEPTANCE.md)
- [GPU_RESULTS_REVIEW_REQUEST.md](history/GPU_RESULTS_REVIEW_REQUEST.md)
- [PLAN.md](history/PLAN.md)
- [PLAN_REVIEW.md](history/PLAN_REVIEW.md)
- [PR_UNIFIED_JOINT_ALLOCATION_REVIEW_UPDATE.md](history/PR_UNIFIED_JOINT_ALLOCATION_REVIEW_UPDATE.md)
- [REVIEW_KO.md](history/REVIEW_KO.md)
- [REVIEW_REPLIES_FLOAT_DRAFT.md](history/REVIEW_REPLIES_FLOAT_DRAFT.md)
- [REVIEW_REPLIES_READY.md](history/REVIEW_REPLIES_READY.md)
- [REVIEW_REPLY_DRAFTS.md](history/REVIEW_REPLY_DRAFTS.md)
- [STAGE_B_REVIEW.md](history/STAGE_B_REVIEW.md)
- [TRI_CANDIDATE_ACCEPTANCE.md](history/TRI_CANDIDATE_ACCEPTANCE.md)
- [TRI_CANDIDATE_REVIEW.md](history/TRI_CANDIDATE_REVIEW.md)
- [TRI_CAPACITY_AMENDMENT.md](history/TRI_CAPACITY_AMENDMENT.md)
- [TRI_DESIGN_ADVICE.md](history/TRI_DESIGN_ADVICE.md)
- [TRI_FLOAT_CANDIDATE_ACCEPTANCE.md](history/TRI_FLOAT_CANDIDATE_ACCEPTANCE.md)
- [TRI_FLOAT_CANDIDATE_REVIEW_REQUEST.md](history/TRI_FLOAT_CANDIDATE_REVIEW_REQUEST.md)
- [TRI_FLOAT_SUPPLEMENT_PLAN.md](history/TRI_FLOAT_SUPPLEMENT_PLAN.md)
- [TRI_FLOAT_SUPPLEMENT_REVIEW.md](history/TRI_FLOAT_SUPPLEMENT_REVIEW.md)
- [TRI_GPU_ACCEPTANCE.md](history/TRI_GPU_ACCEPTANCE.md)
- [TRI_GPU_HARNESS_REVIEW.md](history/TRI_GPU_HARNESS_REVIEW.md)
- [TRI_GPU_PLAN.md](history/TRI_GPU_PLAN.md)
- [TRI_GPU_V3_REVIEW.md](history/TRI_GPU_V3_REVIEW.md)
- [TRI_PLAN.md](history/TRI_PLAN.md)
- [TRI_PLAN_ADDENDUM.md](history/TRI_PLAN_ADDENDUM.md)
- [TRI_PLAN_REVIEW.md](history/TRI_PLAN_REVIEW.md)
- [TRI_RETENTION_REGRESSION_REVIEW.md](history/TRI_RETENTION_REGRESSION_REVIEW.md)
- [comparison.md](history/comparison.md)
- [stage-a-supplement.md](history/stage-a-supplement.md)
