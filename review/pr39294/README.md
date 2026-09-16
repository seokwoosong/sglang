# PR #39294 review branch

## 2026-09-16 최신 upstream rebase

#36729가 실제 merge된 upstream `a3bf25dc62` 위에 후속 8개 commit만 rebase했다. 충돌 없이 range-diff 모두 동일 패치다. 새 기반에서 CPU 295 tests/1,661 subtests(10 GPU skips), Rust 25/131(1 session 제외)를 통과했다. GPU·실서빙은 재실행하지 않았다. [새 기반 결과·제한](rebase-latest-20260916/RESULTS.md), [최종 검토](rebase-latest-20260916/FINAL_REVIEW.md), [커밋 매핑](rebase-latest-20260916/commit-map.json). 이번 rebase는 로컬 변경이며 아직 push하지 않았다. 아래 내용은 이전 checkpoint 기록이다.

## 2026-09-16 hardening 추가 검증

[최종 결과와 한계](hardening-20260916/HARDENING_RESULTS.md), [지정 세션의 최종 승인](hardening-20260916/final-disposition/REVIEW.md), [문서·원본 SHA manifest](hardening-20260916/MANIFEST.json). 동적 gate capacity 캐시와 같은 이벤트의 pending source 누락을 별도 수정했다. 최종 CPU 281 tests/1,636 subtests, Rust 25/131, P2 14/25를 통과했고 CUDA same-event 4 PASS / different-event 4 INCONCLUSIVE다. 실서빙은 baseline config-loader 호환성 문제로 추론 전에 중단됐다. 로컬 보존이며 push·PR 반영·merge 완료를 뜻하지 않는다. 아래 내용은 이전 rebase checkpoint 기록으로 보존한다.

이 branch를 후속 수정의 작업 기준으로 사용한다. 최신으로 fetch한 #36729와 upstream main을 합친 기반 위에 기존 후속 수정 4개 커밋을 rebase했다. GitHub PR #39294 반영 또는 maintainer의 merge 승인을 뜻하지 않는다.

## 현재 기준과 변경

- #36729: `a6eb82dc77353249fff1b08356533c2dc16ea6ec` (확인 시 OPEN)
- upstream main: `832ec39cc0324cb0e7823dc8385e27a30c356bdd`
- 두 이력을 합친 기반: `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`
- 검증한 rebase 결과: `276f389f0410a5ddc57fe71a5ea40fb13bf25c66`

| 이전 커밋 | Rebase 후 커밋 | 내용 |
| --- | --- | --- |
| `7b98cd9a86` | `95c4989210` | Bounded joint reclaim와 회귀 테스트 |
| `899fe03521` | `2cbee5ec44` | FLOAT 이동 제한과 테스트 |
| `c3903a5be8` | `a62664863b` | Tri-pool prefix 보존과 테스트 |
| `68a70f6cc8` | `276f389f04` | 기존 검토 문서 묶음 |

충돌 없이 완료했고 range-diff 네 항목 모두 동일 패치로 확인했다. Rebase 과정의 별도 production/test 수정은 없다. 이후 커밋은 현재 통합 상태를 기록하는 문서 변경이다.

## 새 기반에서 수행한 검증

- 기존 CPU 회귀: 276 PASS / 1,597 subtests / 10 GPU skips.
- 실제 새 Rust 확장의 회귀·통합: 141 PASS / 138 subtests / 1 CUDA skip / 2 미지원 session-cursor 제외.
- 추가 upstream CPU 호환성: 56 PASS / 7 subtests; session metadata 2 PASS.
- 실제 configurator byte budget 4개 경로, 동일 rid의 attempt 분리와 abort/finish forwarding 확인.
- CUDA allocation/data 및 독립 이동 제한: 19 PASS.
- 전체 pre-commit: 실제 upstream/main을 비교 기준으로 지정해 PASS.

[통합 결과·명령·제한](sync-20260915/REBASE_RESULTS.md), [계획 승인](sync-20260915/REBASE_PLAN_REVIEW.md), [최종 검토](sync-20260915/PR39294_REBASE_ACCEPTANCE.md), [검증 manifest](sync-20260915/candidate-manifest.json).

이 검증은 serving 정확도·속도 또는 모든 비동기 실행의 검증을 뜻하지 않는다. 이전 lazy-writer ordering 및 pending-reader GPU 결과의 미확정 한계를 유지한다. #36729가 실제 merge되면 landed revision과 후속 delta를 다시 확인해야 한다. 실제 PR 반영, C3 조율, CI와 merge는 별도 단계다.

## 보존된 검토 이력

[Rebase 이전 README와 전체 문서 목록](README_BEFORE_REBASE.md), [기존 reviewer 답변 후보](../../PR39294_REVIEW_REPLIES.md), [원본 문서 manifest](DOCUMENT_MANIFEST.json).

기존 문서의 SHA·경로·테스트 수치는 당시 snapshot을 가리킨다. 이전 manifest와 원본 문서를 변경하지 않았다. Raw 실행 로그와 probe/harness는 통합 결과 문서에 기록한 로컬 artifact 디렉터리에 있다.
