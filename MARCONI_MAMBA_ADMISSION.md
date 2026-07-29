# Marconi-style Mamba cache admission for SGLang

이 문서는 `marconi-mamba-admission` 브랜치에 구현된 Mamba state admission
정책과 CUDA 머신에서 이어서 검증하는 방법을 설명한다.

## 1. 구현 범위

이번 구현의 목표는 `--enable-unified-memory`를 사용하는 hybrid
full-attention + Mamba/GDN 모델에서 request당 영구 Mamba checkpoint 후보를
다음 두 종류로 제한하는 것이다.

1. 기존 SGLang이 발견하고 정렬해서 추적한 **branch point** 최대 1개
2. request 종료 시 기존 코드가 제공하는 **latest safe final checkpoint** 1개

chunked prefill 또는 decode tracking interval에서 생성되는 그 밖의 중간
checkpoint는 실행 중인 request의 working state로는 유지하지만 shared radix
tree에는 삽입하지 않는다.

이 브랜치는 다음을 구현하지 않는다.

- FLOP-aware cache eviction
- Marconi 논문의 전체 eviction/admission 알고리즘
- 별도의 “동일 input prefix가 세 번째 관측됐는가” reuse counter
- legacy `MambaRadixCache` 경로의 admission 변경

따라서 정확한 표현은 **Marconi-style branch/final admission for the unified
Mamba cache**이다.

## 2. 실행 옵션

새 server argument:

```text
--mamba-state-admission-policy {default,marconi}
```

| 값 | 동작 |
|---|---|
| `default` | 기존 SGLang checkpoint admission을 유지한다. |
| `marconi` | 정렬된 branch checkpoint와 request final checkpoint만 영구 삽입한다. |

`marconi`는 현재 의도적으로 `--enable-unified-memory`와 함께 사용할 때만
허용된다. unified memory 없이 지정하면 시작 단계에서 `ValueError`가 발생한다.

## 3. admission 동작

정책 판정은 allocator나 radix tree를 직접 수정하지 않는 별도 파일
`python/sglang/srt/mem_cache/mamba_admission_policy.py`에 있다.

판정 순서는 다음과 같다.

```text
is_finished?
  yes -> FINAL, persist
  no  -> checkpoint_seqlen == mamba_branching_seqlen?
           yes -> BRANCH, persist
           no  -> INTERMEDIATE
                    default -> persist
                    marconi -> skip
```

정상적인 Marconi request에서 영구 checkpoint 후보는 따라서 최대 2개다.

```text
0 or 1 branch candidate + 1 final candidate
```

이미 존재하는 radix node와 중복된 후보는 새 Mamba slot을 소비하지 않으며
통계상 `duplicate_candidates`로 기록된다.

## 4. 기존 SGLang 코드 재사용

### 4.1 Branch point

새 코드가 branch 위치를 다시 계산하지 않는다.

기존 unified `MambaComponent.finalize_match_result_in_tree_core()`가 full-KV hit가
현재 재사용 가능한 Mamba boundary보다 길 때 `mamba_branching_seqlen`을
계산한다. 기존 scheduler는 그 위치가 다음 extend batch 안에 있으면 정확한
aligned recurrent state를 ping-pong tracking buffer에 캡처한다.

새 admission layer는 다음 unfinished-cache handoff에서

```text
mamba_last_track_seqlen == mamba_branching_seqlen
```

인지 확인하고, 일치할 때만 branch checkpoint로 분류한다. 삽입 또는
deduplication이 끝난 branch marker는 Marconi 모드에서 제거해 같은 request가
같은 branch 후보를 반복 admission하지 않도록 한다.

### 4.2 Final checkpoint

request가 끝났을 때의 state donation/copy/free 로직은 기존
`MambaComponent.prepare_for_caching_req(..., is_finished=True)`와
`cleanup_after_caching_req()`를 그대로 사용한다.

extra-buffer 전략에서는 `mamba_last_track_seqlen`에 해당하는 latest safe
ping-pong snapshot을 삽입한다. 이것은 반드시 마지막 생성 token과 정확히 같은
위치는 아니며, 기존 SGLang이 안전하게 재사용할 수 있다고 보장하는 가장 최신
checkpoint라는 의미의 final이다.

### 4.3 Intermediate checkpoint 거부

Marconi가 intermediate checkpoint를 거부하면 다음을 수행한다.

1. `InsertParams.skip_radix_insert = True` 설정
2. Mamba slot을 radix cache에 donate/copy하지 않음
3. request-owned KV indices와 ping-pong working snapshot은 유지
4. `mamba_last_track_seqlen`을 지우지 않아 request가 곧 종료되더라도 final
   checkpoint fallback으로 사용할 수 있게 함

unified radix node는 full KV와 대응하는 recurrent state가 함께 있어야 재개
가능한 hybrid checkpoint다. 따라서 Mamba state만 거부하고 해당 위치의 full
KV node만 영구 삽입하지 않고, multi-component radix insert 전체를 생략한다.

## 5. 수정 파일

| 파일 | 역할 |
|---|---|
| `python/sglang/srt/mem_cache/mamba_admission_policy.py` | 순수 admission 판정 및 request별 통계 |
| `python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py` | 기존 snapshot donation/cleanup 흐름에 정책 hook 연결 |
| `python/sglang/srt/mem_cache/base_prefix_cache.py` | admission kind와 `skip_radix_insert` 전달 필드 |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 거부된 multi-component checkpoint의 기존 early-return 경로 |
| `python/sglang/srt/server_args.py` | CLI argument와 unified-memory validation |
| `scripts/verify_marconi_mamba_admission.py` | 실제 모델 default/Marconi 자동 비교 |
| `test/registered/unit/mem_cache/test_mamba_admission_policy_core.py` | dependency-free 정책 단위 테스트 |
| `test/registered/unit/mem_cache/test_mamba_admission_policy.py` | component 및 server-arg 테스트 |
| `test/registered/unit/mem_cache/test_marconi_verification_script.py` | 실험 결과 판정기 테스트 |

기존 핵심 코드 수정은 policy hook, shared insert flag, early-return 조건으로
제한했다. 정책 판정과 관측 코드는 별도 파일에 있어 upstream version update 때
충돌 범위를 줄였다.

## 6. 객관적 관측 지표

각 request 종료 시 DEBUG log에 다음 prefix를 가진 JSON record가 출력된다.

```text
MAMBA_ADMISSION_STATS {"policy": "marconi", "rid": "...", ...}
```

필드:

| 필드 | 의미 |
|---|---|
| `branch_candidates` | branch로 분류된 후보 수 |
| `final_candidates` | final로 분류된 후보 수 |
| `intermediate_candidates` | 중간 후보 수 |
| `branch_admitted` | 실제 새 branch state가 삽입된 수 |
| `final_admitted` | 실제 새 final state가 삽입된 수 |
| `intermediate_admitted` | 실제 새 intermediate state가 삽입된 수 |
| `duplicate_candidates` | 이미 state가 있어 새 slot을 쓰지 않은 후보 수 |
| `intermediate_skipped` | 정책에 의해 거부된 중간 후보 수 |

Marconi가 의도대로 동작하기 위한 핵심 조건:

```text
intermediate_candidates > 0
intermediate_skipped > 0
intermediate_admitted == 0
branch_candidates > 0              # branch workload에서
branch_candidates + final_candidates <= 2  # 각 request에서
```

## 7. CUDA 머신에서 이어서 검증하기

### 7.1 Checkout과 설치

```bash
git clone https://github.com/seokwoosong/sglang.git
cd sglang
git checkout marconi-mamba-admission

pip install --upgrade pip
pip install -e "python"
```

이미 SGLang CUDA 개발 image를 사용한다면 repository를 mount하고 editable
install만 다시 수행해도 된다.

CUDA 확인:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

첫 번째 값이 `True`여야 한다.

Hugging Face 모델 접근 권한이나 인증이 필요한 환경에서는 서버 실행 전에
`hf auth login` 또는 `HF_TOKEN` 설정을 완료한다. 토큰은 log나 결과 JSON에
포함하지 않는다.

### 7.2 권장 자동 비교

```bash
python scripts/verify_marconi_mamba_admission.py \
  --model-path Qwen/Qwen3.5-0.8B \
  --extra-server-args \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton
```

스크립트는 이 checkout의 `python/`을 `PYTHONPATH` 앞에 추가한 다음 다음
작업을 순서대로 수행한다.

1. `default` server 실행
2. 긴 공통 prefix와 서로 다른 suffix를 가진 deterministic request 3개 실행
3. server 종료 및 structured admission log 수집
4. `marconi` server에서 동일 workload 실행
5. generated text와 admission 통계를 비교
6. PASS/FAIL JSON 생성

기본 출력:

```text
/tmp/sglang-marconi-verification/
├── default.log
├── marconi.log
└── comparison.json
```

OOM이 발생하면 두 server에 동일한 값을 적용하도록 마지막에 다음 인자를
추가한다.

```text
--mem-fraction-static 0.75
```

예:

```bash
python scripts/verify_marconi_mamba_admission.py \
  --model-path Qwen/Qwen3.5-0.8B \
  --extra-server-args \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton \
  --mem-fraction-static 0.75
```

### 7.3 자동 판정 기준

`comparison.json`의 `verdict`가 `PASS`가 되려면 다음 조건을 모두 만족해야
한다.

| 검사 | 기대 결과 |
|---|---|
| default intermediate admission | `intermediate_admitted > 0` |
| Marconi intermediate workload | `intermediate_candidates > 0` |
| Marconi 거부 동작 | `intermediate_skipped > 0` |
| Marconi 중간 삽입 방지 | `intermediate_admitted == 0` |
| branch workload 유효성 | `branch_candidates > 0` |
| request당 persistent 후보 | `branch_candidates + final_candidates <= 2` |
| correctness smoke test | default와 Marconi generated text가 동일 |

`branch_admitted`는 radix tree에 동일 state가 이미 존재하면 0일 수 있으므로
branch 발견 여부는 `branch_candidates`로 판정하고, 실제 slot 신규 할당 여부는
`branch_admitted`와 `duplicate_candidates`를 함께 본다.

### 7.4 수동 서버 실행

Default:

```bash
sglang serve \
  --model-path Qwen/Qwen3.5-0.8B \
  --enable-unified-memory \
  --mamba-state-admission-policy default \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton \
  --log-level debug
```

Marconi:

```bash
sglang serve \
  --model-path Qwen/Qwen3.5-0.8B \
  --enable-unified-memory \
  --mamba-state-admission-policy marconi \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton \
  --log-level debug
```

현재 SGLang CLI는 `sglang serve`가 권장 entrypoint다. 수정된 checkout을
확실히 사용하려면 editable install 후 실행하거나 다음 entrypoint를 사용한다.

```bash
PYTHONPATH=python python -m sglang.launch_server ...
```

## 8. 테스트

dependency-free policy와 판정기 테스트:

```bash
python test/registered/unit/mem_cache/test_mamba_admission_policy_core.py
python test/registered/unit/mem_cache/test_marconi_verification_script.py
```

전체 SGLang test dependencies가 설치된 CUDA 개발 환경:

```bash
python -m pytest -q \
  test/registered/unit/mem_cache/test_mamba_admission_policy_core.py \
  test/registered/unit/mem_cache/test_mamba_admission_policy.py \
  test/registered/unit/mem_cache/test_marconi_verification_script.py
```

## 9. 현재까지 확인된 결과

Apple arm64 개발 머신에서 확인한 항목:

- dependency-free admission policy test 5개 통과
- verification result validator test 2개 통과
- 수정 Python 파일 `py_compile` 통과
- 신규 파일 대상 Ruff import/error 검사 통과
- `git diff --check` 통과

실제 `Qwen/Qwen3.5-0.8B + --enable-unified-memory` 비교는 CUDA/Triton 경로가
필요하므로 Apple 머신에서는 `SKIP`으로 기록됐다. 아직 실제 GPU 성능이나
메모리 감소 수치를 확인한 것으로 간주하면 안 된다.

## 10. CUDA 검증 후 권장 후속 작업

1. `comparison.json`, `default.log`, `marconi.log` 보존
2. request별 candidate/admitted/skipped 수가 예상과 맞는지 확인
3. default/Marconi output equality를 여러 prompt에서 반복 확인
4. 실제 Mamba slot 수 또는 unified-pool free bytes를 추가 측정
5. long-prefix concurrency workload에서 throughput/TTFT/ITL 비교
6. 필요하면 별도 변경으로 FLOP-aware eviction 구현

admission correctness와 eviction/performance 실험은 별도 단계로 유지하는 것이
회귀 원인을 구분하기 쉽다.
