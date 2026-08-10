# Qwen3.5 HiCache component transfer evaluation

> 이 보고서의 component-direct 구현은 이후 whole-slot raw-copy로 대체됐다.
> 최종 비교와 선택은 `QWEN35_MAMBA_RAW_COPY_EVALUATION.md`를 참고한다.

## 결론

RTX 5090에서 static+HiCache와 unified-memory+typed-L2 HiCache(U3)의 KV/Mamba
D2H·H2D를 같은 payload로 비교했다.

- 기존 U3 Mamba D2H는 `24.83 GiB/s`로 static의 `42.54 GiB/s`보다 느렸다.
- 원인은 unified L1 row를 1-slot GPU staging buffer에 gather한 뒤 slot마다
  `copy_()`를 호출하는 two-hop 경로였다.
- U3도 static과 같은 mapped-host Mamba kernel을 직접 사용하고 실제 L1/L2
  stride를 전달하도록 변경했다.
- 수정 후 실제 서버의 U3 Mamba D2H median은 `45.84 GiB/s`였다. 패치 전보다
  `1.85x` 빨라졌고 static보다 `1.08x` 빨랐다.
- 통제된 microbenchmark에서는 Mamba D2H와 H2D 모두 static과 U3가 사실상
  같았다. KV는 U3 D2H가 더 빠르고 H2D는 같았다.

따라서 현재 U3의 순수 KV/Mamba 양방향 전송 구현에는 static 대비 본질적인
대역폭 저하가 남아 있지 않다.

## 수정 내용

패치 전 unified Mamba D2H는 다음 순서였다.

1. strided unified L1 row를 contiguous GPU staging slot으로 gather
2. staging slot에서 registered host row로 slot별 `copy_()`

staging capacity를 1에서 8 slot로 늘려도 batch 8 D2H는
`23.45 -> 29.78 GiB/s`에 그쳤다. 같은 payload를 mapped-host kernel로 직접
보내면 `48.76 GiB/s`였다. 병목은 작은 staging capacity가 아니라 불필요한
GPU gather와 두 번째 복사였다.

패치 후에는 static과 동일한 `transfer_kv_mamba_lf_pf` kernel을 사용한다.
Unified L1의 source slot stride와 typed L2의 destination chunk stride를 실제
tensor에서 읽어 kernel에 전달한다. H2D는 기존부터
`transfer_kv_mamba_pf_lf`를 직접 사용했으므로 변경하지 않았다.

Production 수정은 최종 typed-L2 commit `16fb67cc9`에 포함되어 있다.

## 통제된 component microbenchmark

환경은 Qwen3.5-0.8B의 실제 Mamba 크기인 18 layers, temporal row 1 MiB,
conv row 36 KiB, 총 19,537,920 bytes/slot이다. U3 조건은 실제
`SharedTypedChunkHostArena`와 `UnifiedChunkMambaPoolHost`를 사용한다. 각 값은
30회 CUDA-event 측정의 median이다.

### Mamba

Contiguous slot 결과다. Fragmented 결과도 최대 차이가 1% 미만이었다.

| Batch slots | Direction | static+HiCache | U3 typed L2 | U3/static |
|---:|---|---:|---:|---:|
| 1 | D2H | 48.14 GiB/s | 48.09 GiB/s | 0.999x |
| 4 | D2H | 48.60 GiB/s | 48.60 GiB/s | 1.000x |
| 8 | D2H | 48.71 GiB/s | 48.72 GiB/s | 1.000x |
| 16 | D2H | 48.72 GiB/s | 48.77 GiB/s | 1.001x |
| 1 | H2D | 33.29 GiB/s | 33.89 GiB/s | 1.018x |
| 4 | H2D | 44.85 GiB/s | 44.84 GiB/s | 1.000x |
| 8 | H2D | 47.21 GiB/s | 46.92 GiB/s | 0.994x |
| 16 | H2D | 48.30 GiB/s | 48.21 GiB/s | 0.998x |

### KV

같은 4,096-token payload를 옮겼다. Page 8이 실제 서버 검증 조건이다.

| Page size | Direction | static+HiCache | U3 typed L2 | U3/static |
|---:|---|---:|---:|---:|
| 1 | D2H | 4.58 GiB/s | 47.29 GiB/s | 10.31x |
| 8 | D2H | 23.48 GiB/s | 48.70 GiB/s | 2.07x |
| 32 | D2H | 37.12 GiB/s | 48.80 GiB/s | 1.31x |
| 1 | H2D | 48.98 GiB/s | 45.54 GiB/s | 0.93x |
| 8 | H2D | 48.94 GiB/s | 48.82 GiB/s | 0.997x |
| 32 | H2D | 48.99 GiB/s | 48.86 GiB/s | 0.997x |

U3 KV D2H는 unified page envelope를 한 번에 복사한다. Static D2H는 page가
작을수록 더 많은 page/layer 작업이 필요하므로 page size 1에서 차이가 가장
크다. H2D는 두 조건 모두 layer별 kernel을 사용하므로 page 8과 32에서 같다.

## 실제 Qwen3.5 서버 검증

Qwen3.5-0.8B, page size 8, write-back, CUDA graph off, 10k input, 64 output,
40 groups x 2 rounds, concurrency 4로 각 조건을 3회 실행했다. 모든 run은
eviction, D2H backup, H2D load-back, host hit가 실제로 발생해야 성공하도록
검증했다. 아래는 component CUDA interval의 3회 median이다.

| Component | static+HiCache | U3 typed L2 | U3/static |
|---|---:|---:|---:|
| KV D2H | 22.95 GiB/s | 47.25 GiB/s | 2.06x |
| Mamba D2H | 42.54 GiB/s | 45.84 GiB/s | 1.08x |
| KV H2D | 48.23 GiB/s | 48.00 GiB/s | 0.995x |
| Mamba H2D | 25.23 GiB/s | 22.53 GiB/s | 0.893x |

U3 Mamba D2H는 패치 전 같은 profiler에서 `24.83 GiB/s`였고 수정 후
`45.84 GiB/s`가 되어 `1.85x` 개선됐다.

서버에서 관측한 Mamba H2D effective rate는 U3가 약 10.7% 낮다. 이것은
순수 copy path의 차이가 아니다. 동일 batch와 동일 payload만 실행한 위
microbenchmark에서는 typed L2와 static H2D가 1% 이내로 같았다. 실제 서버의
두 조건은 load-back 횟수와 batch 분포가 다르고, U3에는 unified layout 및
compaction kernel이 함께 실행되어 transfer kernel의 GPU 자원 경쟁도 다르다.
따라서 server interval은 end-to-end 실행 중 effective rate이고 독립적인 PCIe
copy ceiling은 아니다. H2D production kernel은 이미 baseline과 동일하므로
이 결과를 근거로 별도 copy 경로를 추가하지 않았다.

참고로 같은 서버 workload의 total token throughput median은 static
`51,043 tokens/s`, U3 `57,092 tokens/s`였다. 이 workload는 component 전송을
발생시키기 위한 검증 workload이므로 일반 성능 결론으로 확대하지 않는다.

## Correctness 및 artifact 감사

| 검증 | 결과 |
|---|---:|
| 관련 CUDA/unit tests | 241 passed, 1 skipped, 26 subtests passed |
| 실제 서버 profile | 6/6 passed |
| 서버 요청 | 각 run 80/80 completed, failed 0 |
| eviction/backup/load-back/host-hit | 모든 run에서 관측 |
| dropped token 및 CUDA 오류 | 0 |
| pre-commit | passed |

## 재현

```bash
source ~/.venv_sglang/bin/activate

PYTHONPATH=python python benchmark/hicache/bench_mamba_transfer_paths.py \
  --batches 1 4 8 16 \
  --patterns contiguous fragmented \
  --warmup 5 \
  --repetitions 30 \
  --output artifacts/hicache_component_transfer/mamba/final

PYTHONPATH=python python benchmark/hicache/bench_unified_typed_page_transfer.py \
  --page-sizes 1 8 32 \
  --tokens 512 4096 \
  --patterns contiguous fragmented \
  --warmup 5 \
  --repetitions 20 \
  --output artifacts/hicache_component_transfer/kv/final

for repeat in 1 2 3; do
  PYTHONPATH=python python benchmark/hicache/run_qwen35_hicache_matrix.py \
    transfer \
    --model-size 0.8b \
    --pages 8 \
    --variants eval-s1 eval-u3 \
    --repetition "$repeat" \
    --artifact-root artifacts/hicache_component_transfer/server
done

PYTHONPATH=python python benchmark/hicache/analyze_component_transfer.py \
  --output artifacts/hicache_component_transfer/summary.json
```

Raw local artifacts는 `artifacts/hicache_component_transfer/` 아래에 있다.
