# PR #39294 reviewer 대응 — 로컬 검증 완료, 통합 합의 필요

## 현재 결론

지정 Codex 세션 `01a09f85-77de-79d1-a50d-cb7a530616c1`에서 단계별 계획 승인 후 실험/수정을 진행했다. 최종 production CPU 후보와 제한을 명시한 GPU 검증 범위를 수용받았다. GitHub reviewer 승인이나 merge 승인을 받은 것은 아니다.

현재 원격 #39294는 원래 `276771eefca951ae6d55608415f2d795cbfabdae`이며 변경하지 않았다. OPEN / REVIEW_REQUIRED / DIRTY 상태이고 ch-wan의 기존5개 thread가 모두 unresolved다. 새 댓글은 없다. #36729 역시 OPEN, head `6c8bbdf610fae5a3ddba826162bc7ca91e79dbfb`로 동일하다.

## Reviewer 항목별 상태

| 항목 | 로컬 대응 | 남은 일 |
| --- | --- | --- |
| C1: victim마다 mutating prepare | 순수 sufficiency callback + 명시적/유한 preparation 단계 | 합의된 branch에 게시 후 reviewer 확인 |
| C2: impossible demand로 cache 전체 제거 | 첫 destructive walk 전 ID/optimistic-byte guard, 전체-cache fallback 제거, 정확한 paged extend demand | 동일 |
| C3: #36729와 충돌/중복 | #36729 dispatch/planner 위에 별도 후보와 regression patch 구성 | 저자/maintainer와 landing order 합의 필요 |
| C4: StreamingSession 불필요 분기 | optional callback 무조건 전달 | 게시 후 확인 |
| C5: query 이름 아래 mutation | C1과 함께 query/ensure contract 분리 | 게시 후 확인 |

## 추가로 발견하고 수정한 tri-pool 회귀

temporal state8개를 먼저 배치하고96토큰을 캐시에 저장한 경우, END-only 순수 판단식은4토큰 요청에서96토큰을 모두 제거했다. 기존 #39294는95토큰을 보존했다. 이를 CPU로 재현하고 별도 수정 계획 승인을 받았다.

수정 후보는 FLOAT의 반대편 이동 목적지와 FULL 이후 SWA 공간을 보수적으로 예약하는 순수 판단식을 사용한다. 첫 FULL/SWA/state1/1/1 회수 직후 멈추고,256-byte 절대 HIGH band target의 실제 이동 후4토큰을 할당하며95토큰과 KV/conv/temporal payload를 보존한다. 준비 도중 gate가 바뀌면 실제 frontier에서 다시 계산한다. 전체 FLOAT feasibility나 최소 이동량을 보장한다고 주장하지 않는다.

## 최종 검증

- Python CPU:276PASS /1597subtests /10GPU skips. `run_tri_float_cpu.sh`에13개 파일과 실행 환경 기록.
- 실제 Rust cache backend:36PASS /138subtests /2session-cursor exclusions.
- 전체 pre-commit:PASS, Rust fmt/clippy 포함.
- 실제 production geometry864사례:477positive,26HIGH-only, positive allocation failure와 payload/accounting error0.
- production CPU pending18사례:PASS. 실제 GPU pending 증거로 해석하지 않는다.
- GPU: two-pool/paged tri/개별 gate/데이터 보존/graph 재생 및 eager writer dependency 검증 수용. lazy writer ordering과 실제 pending-reader 재사용2사례는INCONCLUSIVE. 기본/별도 scheduler stream 모두에서 warmed host→device tensor 생성이 먼저 event를 완료시키는 환경 관측을 기록했다. 추가 지연 반복이나 production 동기화 변경은 하지 않았다.
- 새로운 serving 성능/모델 정확도 수치는 없다. 이전PR 결과는 이전SHA에 귀속된다.

## 결과물

- 최종 작업 트리: `/home/sukwoo24/sglang-eval-worktrees/review-tri-float-candidate`
- 최종 index tree: `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`;10개 staged파일,unstaged변경 없음. commit/push 안 함.
- `tri-float-composed-final.patch`: #36729 위의 전체 후보.
- `tri-float-supplement-final.patch`: 실패한 tri 후보 위의2파일 추가 수정.
- `TRI_FLOAT_CANDIDATE_ACCEPTANCE.md`: production CPU 후보 승인.
- `TRI_GPU_ACCEPTANCE.md`: GPU 검증 범위 수용과 명시적 한계.
- `GPU_RESULTS_REVIEW_REQUEST.md`, `gpu-final-manifest.json`: GPU 상세 결과/명령/hash.
- `REVIEW_REPLIES_FLOAT_DRAFT.md`:5개 항목별 답변 초안.
- `COORDINATION_COMMENT_READY.md`: #39294에 게시할 조율 댓글 원문.
- `PR_UNIFIED_JOINT_ALLOCATION_REVIEW_UPDATE.md`: 통합 합의 후 사용할PR본문 초안.

## 다음 순서

1. 준비된 조율 댓글을 게시해 #36729 먼저 merge하고 #39294를 follow-up으로 유지할지 합의한다. 아직 외부 게시 권한을 받지 않았으므로 댓글은 로컬에만 있다.
2. 합의된 결과와 최신 main을 기준으로 최종 통합하고 충돌을 해결한다. 이미 게시된 branch를 지금 임의로 rewrite하지 않는다.
3. 최종 통합 revision의 필요한 회귀 검증과 합의된 성능/serving 범위를 실행하고 변경을 게시한다.
4. reviewer5개 항목에 게시된 commit 근거로 답변하고 required CI/maintainer approval을 확보한다.

현재 상태를 'reviewer 모두 대응 완료' 또는 'merge-ready'로 표시하지 않는다. 로컬 구현/검증 단계는 수용됐고, 다음 blocker는 C3의 외부 통합 합의다.
