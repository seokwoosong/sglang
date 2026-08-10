# Qwen3.5 unified HiCache Mamba raw-copy evaluation

## 결론

Unified-memory typed-L2의 Mamba D2H는 slot 전체를 한 번에 옮기는 indexed
raw-copy를 최종 구현으로 선택했다.

- 기존 component-direct는 temporal과 conv를 각각 전송하여 kernel 2회가
  필요했다.
- raw-slot은 dtype, layer pointer, component loop 없이 slot payload 전체를
  kernel 1회로 전송한다.
- 통제 실험에서 raw-slot은 component-direct보다 D2H가 약 0.7~1.0% 빨랐고,
  batch 8의 enqueue 비용은 `58.28 -> 37.81 us`로 35% 감소했다.
- 실제 서버의 Mamba D2H는 `45.84 -> 47.29 GiB/s`로 3.2% 개선됐다.
- 관련 correctness와 page size 1, 8, 32 실제 서버 검증을 모두 통과했다.

최종 production commit은 `16fb67cc9`이다.

## 비교한 세 구현

| 이름 | GPU L1 | CPU L2 | Mamba D2H |
|---|---|---|---|
| static+HiCache | static layer-first | static page-first | temporal/conv direct kernel 2회 |
| U3 component-direct | unified page-major | typed chunks | temporal/conv direct kernel 2회 |
| U3 raw-slot | unified page-major | typed chunks | 전체 slot indexed raw-copy 1회 |

두 U3 구현의 H2D는 동일한 기존 per-layer Mamba kernel을 사용한다. 이번
변경은 D2H만 단순화한다.

## Layout 수정

기존에는 두 payload의 논리 데이터는 같았지만 byte 순서가 달랐다.

```text
Unified L1: [conv components][temporal]
Typed L2:   [temporal][conv components]
```

Typed L2를 L1과 같은 순서로 변경했다.

```text
Unified L1: [conv components][temporal]
Typed L2:   [conv components][temporal][alignment padding]
```

L2 chunk는 KV page의 정수배여야 하므로 끝에 padding이 존재할 수 있다.
따라서 raw-copy kernel은 source와 destination stride를 별도로 받는다.

```text
src = unified_l1_raw + physical_slot * mamba_slot_bytes
dst = typed_l2_raw   + host_chunk   * typed_chunk_bytes
copy_bytes = mamba_slot_bytes
```

Padding은 복사하지 않는다. 선택된 physical slot과 host chunk가 흩어져 있어도
한 indexed kernel invocation에서 모두 처리한다.

## 통제된 component microbenchmark

Qwen3.5-0.8B의 실제 Mamba 크기인 18 layers, temporal row 1 MiB, conv row
36 KiB, 총 19,537,920 bytes/slot을 사용했다. 각 값은 CUDA event 30회
측정의 median이다.

### D2H bandwidth

| Batch slots | static+HiCache | U3 component-direct | U3 raw-slot | Raw/component |
|---:|---:|---:|---:|---:|
| 1 | 48.15 GiB/s | 48.11 GiB/s | 48.58 GiB/s | 1.010x |
| 4 | 48.60 GiB/s | 48.58 GiB/s | 48.98 GiB/s | 1.008x |
| 8 | 48.71 GiB/s | 48.65 GiB/s | 49.07 GiB/s | 1.009x |
| 16 | 48.76 GiB/s | 48.73 GiB/s | 49.07 GiB/s | 1.007x |

Fragmented source/destination에서도 raw-slot은 `48.66~49.09 GiB/s`였으며
결과가 동일했다.

### D2H enqueue 비용

| Batch slots | static+HiCache | U3 component-direct | U3 raw-slot |
|---:|---:|---:|---:|
| 1 | 57.53 us | 57.30 us | 38.56 us |
| 4 | 58.11 us | 69.09 us | 38.07 us |
| 8 | 57.62 us | 58.28 us | 37.81 us |
| 16 | 59.34 us | 58.68 us | 51.12 us |

Raw-slot은 두 component kernel을 한 raw kernel로 합치므로 launch와 Python
dispatch 비용이 감소한다.

H2D는 세 조건 모두 동일한 경로다. Batch 8에서 static `47.07 GiB/s`,
component-direct `46.80 GiB/s`, raw-slot `46.64 GiB/s`였고 차이는 1% 이내다.

## 실제 Qwen3.5 서버 비교

Qwen3.5-0.8B, page size 8, write-back, CUDA graph off, 10k input, 64 output,
40 groups x 2 rounds, concurrency 4로 각 조건을 3회 실행했다. 표는 measured
phase의 median이다.

| 구현 | Mamba D2H | Mamba H2D | KV D2H | KV H2D | Total tokens/s |
|---|---:|---:|---:|---:|---:|
| static+HiCache | 42.39 | 25.68 | 23.30 | 48.64 | 50,255 |
| U3 component-direct | 45.84 | 22.53 | 47.25 | 48.00 | 57,092 |
| U3 raw-slot | 47.29 | 21.95 | 47.57 | 47.77 | 57,602 |

단위는 transfer column은 GiB/s, throughput은 tokens/s다. Raw-slot은
component-direct 대비 Mamba D2H가 3.2%, total throughput이 0.9% 높았다.

Mamba H2D는 두 U3에서 동일한 코드다. 서버에서 관측되는 차이는 서로 다른
load-back batch와 동시 GPU workload에 의한 effective-rate 변동이며, 통제된
H2D 결과에서는 차이가 없었다.

## 검증

| 검증 | 결과 |
|---|---:|
| 관련 CUDA/unit tests | 242 passed, 1 skipped, 26 subtests passed |
| 기존 component 서버 profile | 6/6 passed |
| raw-slot page 8 서버 profile | 6/6 passed |
| raw-slot page 1/32 추가 profile | 2/2 passed |
| 각 서버 run | 80/80 requests completed |
| eviction/backup/load-back/host-hit | 모든 run에서 관측 |
| dropped token, CUDA error | 0 |
| artifact analyzer | passed |
| pre-commit | passed |

## 재현

```bash
source ~/.venv_sglang/bin/activate

PYTHONPATH=python python benchmark/hicache/bench_mamba_transfer_paths.py \
  --batches 1 4 8 16 \
  --patterns contiguous fragmented \
  --warmup 5 \
  --repetitions 30 \
  --output artifacts/hicache_component_transfer/mamba/raw-comparison

for repeat in 1 2 3; do
  PYTHONPATH=python python benchmark/hicache/run_qwen35_hicache_matrix.py \
    transfer \
    --model-size 0.8b \
    --pages 8 \
    --variants eval-s1 eval-u3 \
    --repetition "$repeat" \
    --artifact-root artifacts/hicache_component_transfer/server_raw
done

PYTHONPATH=python python benchmark/hicache/analyze_component_transfer.py \
  --output artifacts/hicache_component_transfer/summary_raw.json
```

Raw artifacts는 `artifacts/hicache_component_transfer/` 아래에 있다.
