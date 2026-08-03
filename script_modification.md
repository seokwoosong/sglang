# Unified Memory + HiCache: 다른 GPU 환경 실행 기록

이 문서는 `feat/unified-memory-hicache`의 qualification suite를 다른 GPU에서
재현할 때 실제로 확인한 실패 원인과 `run_alltest.sh`의 대응을 기록한다. 현재
checkout에는 기존 `script_modification.md`가 남아 있지 않아, 확인 가능한 실행
artifact와 오류만으로 새로 작성했다.

## 확정된 시험 행렬

- 모델: `Qwen/Qwen3.5-4B`
- tensor parallel: TP=1, TP=2, TP=4
- 정책: write-through, write-back
- CUDA Graph: off, on(L2/L3 integrity)
- stress: concurrent L1/L2/L3 churn
- `--page-size 1`, Triton attention/linear-attention/Mamba, kernel HiCache I/O

Qwen3.5-27B TP=4는 이 작업의 시험 행렬이 아니다. `run_alltest.sh`는 기본 모델을
4B로 두고, serving qualification에서 Qwen3.5-4B가 아니면 시작 전에 실패한다.

## 관측한 오류와 해결

### 1. TP 수보다 visible GPU가 적음

TP=2/4를 GPU 1개에서 실행하면 invalid device ordinal, NCCL 초기화 오류 또는
worker launch 실패가 발생한다. 이를 skip하면 전체 matrix가 통과한 것처럼 보이므로
runner는 serving 시작 전에 visible GPU가 4개 미만이면 종료한다.

```text
ERROR: the TP=1,2,4 serving sweep requires at least four visible GPUs
```

의도적으로 CPU/unit/kernel subset만 실행할 때만 다음을 사용한다.

```bash
./run_alltest.sh --skip-serving
```

### 2. 4B에서 L2-only candidate를 찾지 못함

Unified typed-chunk 구현에서 `--hicache-size`는 KV와 Mamba 각각의 크기가 아니라
두 type이 공유하는 총 host budget이다. 4B에 1 GB를 지정하면 pressure 후 모든
candidate가 L3까지 내려가 L2-only hit assertion이 실패할 수 있었다. 로컬 32 GB
GPU profile은 shared total 2 GB를 기본값으로 사용한다.

```bash
export SGLANG_UNIFIED_HICACHE_SIZE=2
```

장비/모델에 따라 L2 hit가 전혀 없거나 L3 pressure가 부족하면 size,
`MEM_FRACTION_STATIC`, pressure request 수를 함께 조정해야 한다. Test는 counter로
실제 tier를 확인하므로 단순 output 성공만으로 통과하지 않는다.

### 3. 소수 `--hicache-size` 사용

현재 CLI의 `--hicache-size`는 정수 GB다. `--hicache-size 0.25`는 argument parsing
단계에서 실패한다. Ratio 기반 크기를 원하면 size를 0으로 두고
`--hicache-ratio`를 사용한다. Full 4B qualification은 모호성을 피하기 위해 정수
2 GB를 사용한다.

### 4. cache/extension directory 권한 문제

Container나 read-only home에서는 FlashInfer, Torch extension, Triton compile
cache 생성이 실패할 수 있다. Runner는 writable directory를 만들고 다음 환경을
각 invocation에 전달한다.

```text
FLASHINFER_WORKSPACE_BASE=<cache>/flashinfer
TORCH_EXTENSIONS_DIR=<cache>/torch-extensions
TRITON_CACHE_DIR=<cache>/triton
```

기본 root는 `/tmp/sglang-alltest-cache`이며
`SGLANG_ALLTEST_CACHE_DIR`로 바꿀 수 있다.

### 5. CUDA Graph disable flag deprecation

`--disable-cuda-graph`는 deprecated warning을 낸다. Test는 현재 다음 phase별
옵션을 사용한다.

```text
--cuda-graph-backend-decode disabled
--cuda-graph-backend-prefill disabled
```

Graph qualification invocation은 두 옵션을 제거하고
`SGLANG_UNIFIED_HICACHE_ENABLE_CUDA_GRAPH=1`을 전달한다.

### 6. 4B write-back에서 greedy token 하나가 달라짐

동기와 비동기에서 같은 위치가 반복적으로 달라져 async race와 분리했다. 최초
계산에서는 token 24가 token 23보다 logprob 0.125 높았고, restore에서는 두 token이
동률이 되어 argmax tie-break가 token 23을 골랐다. Test는 exact match를 우선하고,
양쪽 token이 서로의 top-5에 있으며 양쪽 gap이 0.15 이하인 첫 divergence만
near-tie로 기록한다. KV/Mamba payload 자체는 별도 bit-exact serialization,
raw-pointer metadata, CUDA round-trip test가 검증한다.

## Runner 동작

```bash
PYTHON_BIN=/path/to/python \
./run_alltest.sh --model /path/to/Qwen3.5-4B
```

Runner는 다음 원칙을 지킨다.

1. unit → CUDA kernel/race → TP=1/2/4 serving 순서다.
2. 각 TP에서 graph-off integrity, graph-on integrity, concurrent stress를 실행한다.
3. 하나라도 non-zero exit이면 즉시 전체 실행을 중단하고 그 시점 summary/log를
   남긴다.
4. serving sweep은 visible GPU 4개를 요구한다. 불완전한 TP matrix를 PASS로
   보고하지 않는다.
5. `--list`로 실제 19개 invocation을 실행 없이 확인할 수 있다.

Artifact와 log는 기본적으로 `/tmp/unified-hicache-alltests/<timestamp>` 아래에
저장된다. 다른 장비에서 실패하면 summary의 첫 실패 log와 같은 directory의
structured artifact를 함께 보존해야 한다.
