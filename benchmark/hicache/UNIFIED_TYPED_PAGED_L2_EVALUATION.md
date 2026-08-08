# Unified typed-page HiCache evaluation

## 결론

Unified-memory + typed L2 HiCache가 `page_size=1,2,4,8,16,32`에서 동작하도록 구현했고, 모든 크기에서 실제 L1 eviction, L2 backup, L2 load-back을 거친 출력의 token ID가 동일함을 확인했다.

순수 KV 전송에서는 OURS의 raw-page D2H가 non-unified HiCache baseline보다 1.57–18.59배 빨랐다. H2D는 두 구현 모두 layer별 direct kernel을 사용하므로 거의 동률이었다. 실제 pressure workload에서는 작은 page의 unified 관리 비용 때문에 P1–P4가 baseline보다 느렸지만, P8과 P16에서는 input throughput이 각각 5.4%, 8.4% 높았다.

후속 최적화에서 unified Mamba H2D의 row별 staging을 stride-aware direct kernel로 교체했다. 실제 server에서 Mamba H2D는 P1 4.23→19.18 GiB/s, P16 4.21→20.43 GiB/s로 개선됐다. P1 전체 처리량은 compaction 병목 때문에 동일했고, P16은 2회 평균 2.3% 높아졌다.

이어서 lazy compaction의 survivor mapping을 row별 `.item()` 대신 batch gather로 바꿨다. P1 compaction CPU는 11.42→1.54 s로 줄었고 전체 처리량은 29,263→40,408 tok/s(+38.1%)로 개선됐다. P16 compaction CPU도 27.8% 줄었지만 전체 처리량 차이는 -1.1%로 측정 변동 범위였다.

## 구현

- Unified L1과 typed L2의 KV page envelope를 동일하게 만들었다.
  - 한 page는 `[L0 K tokens][L0 V tokens] ... [Ln K tokens][Ln V tokens]` 순서다.
  - token ID 기반 radix/controller API는 유지하되 typed allocator는 완전한 page 단위로 할당하고 해제한다.
- KV chunk는 arena의 낮은 주소부터, Mamba chunk는 높은 주소부터 할당한다. 두 타입은 고정 비율로 나뉘지 않고 같은 L2 arena를 동적으로 공유한다.
- Unified KV D2H는 page 전체를 한 번에 복사한다.
- Unified KV H2D는 layer readiness를 유지하면서 page envelope의 layer offset을 직접 복사한다.
- 등록된 host mmap의 CPU 주소와 GPU mapped alias가 다른 환경을 위해 JIT kernel 입력 주소를 `cudaHostGetDevicePointer`로 변환한다.

관련 구현 커밋:

- `61b05e439` — unified-memory backup/load-back와 direct Mamba H2D
- `969f4bf4b` — row-aware asynchronous transfer fence
- `f6982dadc` — shared typed L2, multi-token page, raw-page KV 전송
- `310370741` — batched compaction mapping lookup
- `4c886ec59` — lifecycle trace와 memory-path profiler
- `1fe9f9504` — benchmark와 결과 집계 도구

## 환경과 workload

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB
- PyTorch: 2.11.0+cu130
- 모델: `Qwen/Qwen3.5-0.8B`
- dtype: BF16
- 비교 대상:
  - baseline: `--enable-unified-memory` 없이 standard HiCache `page_first`
  - OURS: `--enable-unified-memory` + shared typed L2 + raw-page transfer
- 공통 server 설정:
  - `--enable-hierarchical-cache`
  - `--hicache-write-policy write_back`
  - `--hicache-io-backend kernel`
  - `--hicache-size 12`
  - `--max-total-tokens 120000`
  - `--mem-fraction-static 0.27`
  - `--max-running-requests 4`
  - `--chunked-prefill-size 4096`
  - `--context-length 65536`
- `--max-mamba-cache-size`는 사용하지 않았다.
- Pressure workload: 10k-token prefix 40개, 3 rounds, 64 output tokens, concurrency 4
- 각 P1–P16 조건은 3회 반복했다. P32 OURS도 3회 반복했다.

## Correctness

관련 kernel/allocator test 전체 결과:

```text
239 passed, 1 skipped, 26 subtests passed
```

Server parity는 40개의 10k-token prefix로 eviction을 만든 뒤, L2에서 실제 복원된 요청을 원본 실행과 비교했다. 허용한 BF16 logprob 오차는 0.02다.

| page size | 구성 | L2 복원 | output IDs | logprob token IDs | 복원 max abs logprob diff |
|---:|---|---:|---|---|---:|
| 1 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 2 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 4 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 8 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 16 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 32 | OURS | 5,890 tokens | same | same | 5.83e-5 |
| 4 | baseline | 9,986 tokens | same | same | 8.82e-6 |

모든 pressure 성능 run도 120/120 요청 성공, L1 eviction/L2 backup/L2 load-back/host hit 발생, dropped token 0을 만족했다.

## 순수 CPU↔GPU KV 전송

Qwen3.5-0.8B KV shape인 6 MHA layers, 2 heads, head dim 256을 사용했다. 아래 값은 contiguous 4096-token payload를 process 3회 × 각 15회 측정한 전체 45 sample의 median이다. `speedup`은 `baseline time / OURS time`이므로 1보다 크면 OURS가 빠르다.

| P | D2H baseline (ms) | D2H OURS (ms) | D2H speedup | H2D baseline (ms) | H2D OURS (ms) | H2D speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.441 | 0.992 | 18.59x | 0.962 | 1.030 | 0.934x |
| 2 | 8.972 | 0.989 | 9.08x | 0.959 | 1.026 | 0.935x |
| 4 | 4.931 | 0.965 | 5.11x | 0.959 | 0.970 | 0.989x |
| 8 | 2.957 | 0.965 | 3.06x | 0.960 | 0.962 | 0.998x |
| 16 | 1.942 | 0.964 | 2.01x | 0.960 | 0.962 | 0.999x |
| 32 | 1.512 | 0.962 | 1.57x | 0.969 | 0.960 | 1.010x |

Baseline D2H는 최대 64 page씩 staging하므로 P가 작을수록 호출 수가 많다. OURS는 선택된 unified page들을 한 raw-page kernel로 전송한다. 따라서 P가 커질수록 baseline의 staging 호출 수가 줄어들어 D2H 차이도 작아진다. H2D는 양쪽 모두 layer별 direct 전송이어서 거의 같다.

Fragmented page 패턴에서도 같은 경향이 나왔다. 전체 결과는 `artifacts/unified_typed_page_summary/micro_speedups.csv`에 있다.

## 실제 server pressure 성능

아래 값은 3회 run의 median이다.

| P | baseline input tok/s | OURS input tok/s | 변화 | baseline TTFT p50 (ms) | OURS TTFT p50 (ms) | baseline TPOT p50 (ms) | OURS TPOT p50 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37,013 | 31,681 | -14.4% | 364.2 | 335.3 | 8.921 | 8.701 |
| 2 | 38,003 | 35,732 | -6.0% | 339.8 | 248.0 | 9.069 | 8.852 |
| 4 | 39,190 | 38,189 | -2.6% | 324.6 | 202.9 | 9.297 | 8.932 |
| 8 | 39,810 | 41,964 | +5.4% | 304.1 | 160.7 | 9.109 | 8.818 |
| 16 | 39,850 | 43,181 | +8.4% | 301.2 | 153.7 | 9.060 | 8.815 |
| 32 | crash | 44,091 | N/A | crash | 157.4 | crash | 8.794 |

Server metric의 byte/duration으로 계산한 effective transfer rate는 다음과 같다. 이는 KV DMA만 측정한 microbenchmark와 달리 Mamba 전송과 load-back orchestration 시간을 포함한다.

| P | baseline backup D2H GiB/s | OURS backup D2H GiB/s | baseline load-back H2D GiB/s | OURS load-back H2D GiB/s |
|---:|---:|---:|---:|---:|
| 1 | 3.93 | 30.97 | 34.84 | 12.86 |
| 2 | 7.16 | 30.87 | 35.57 | 13.31 |
| 4 | 12.73 | 30.65 | 34.51 | 13.29 |
| 8 | 19.40 | 30.55 | 35.56 | 13.10 |
| 16 | 26.81 | 29.85 | 34.72 | 13.35 |
| 32 | crash | 31.13 | crash | 13.31 |

두 표의 H2D 결과는 측정 범위가 다르다. isolated 표는 KV만 측정하지만 server metric은 같은 load stream의 KV와 Mamba state를 모두 포함한다. 아래 component profile에서 server H2D 차이는 allocator나 row fence가 아니라 Mamba H2D에 의해 발생함을 확인했다.

또한 baseline과 OURS는 동적 L1/L2 정책 때문에 전송량 자체가 다르다. median 기준 baseline은 약 18.6 GB를 backup하고 7.5 GB를 load-back한 반면, OURS는 약 8.5–8.8 GB를 backup하고 10.2–11.2 GB를 load-back했다. Server throughput은 이 전체 정책 효과를 반영하며 순수 DMA 비교는 위 microbenchmark를 기준으로 봐야 한다.

## 실제 server component profiling

Clean 성능 run과 분리하여 P1/P16을 각각 2회 profiling했다. profiler는 CUDA synchronize 없이 event interval을 모으며 다음을 분리한다.

- 전체 D2H/H2D transfer-stream 구간
- KV와 Mamba의 개별 D2H/H2D CUDA 구간 및 logical payload
- Python enqueue와 index 준비
- unified allocator, virtual-to-physical translation, compaction, row fence
- page-first Mamba prefill gather/scatter

아래 전송률과 시간은 2회 run의 median이다. `allocator CPU`는 내부에서 실행한 compaction을 포함하므로 `compaction CPU`와 더하면 안 된다.

| P | 구성 | KV D2H GiB/s | Mamba D2H GiB/s | KV H2D GiB/s | Mamba H2D GiB/s | 전체 H2D GiB/s | allocator CPU (s) | compaction CPU (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 3.86 | 34.52 | 37.49 | 20.25 | 29.59 | 0 | 0 |
| 1 | OURS | 32.97 | 20.62 | 34.24 | 4.23 | 11.81 | 13.03 | 11.31 |
| 16 | baseline | 28.06 | 33.71 | 36.96 | 19.68 | 28.83 | 0 | 0 |
| 16 | OURS | 33.76 | 21.21 | 36.54 | 4.21 | 12.12 | 3.44 | 1.78 |

핵심 해석은 다음과 같다.

1. **KV H2D는 병목이 아니다.** P1/P16 모두 baseline과 OURS가 34–38 GiB/s로 비슷하다. isolated KV benchmark의 결론을 실제 server에서도 재현했다.
2. **server H2D 저하의 직접 원인은 unified Mamba H2D다.** Static L1은 destination row가 contiguous라 한 layer의 batch를 `transfer_kv_mamba_pf_lf`로 바로 옮긴다. Unified L1은 1 MiB Mamba row 사이 간격이 전체 18.63 MiB slot envelope라서, 현재 구현이 pinned host row를 한 slot짜리 GPU staging으로 복사한 뒤 `index_copy_`로 scatter한다. 이 per-row staging 경로가 약 4.2 GiB/s에 머물며 baseline보다 약 4.7배 느리다.
3. **KV D2H 최적화는 의도대로 동작한다.** P1에서 baseline 3.86 GiB/s 대비 OURS 32.97 GiB/s다. P16에서는 baseline도 staging 호출을 더 잘 amortize하여 차이가 28.06 대 33.76 GiB/s로 줄어든다.
4. **P1 server 성능 저하의 가장 큰 추가 비용은 compaction CPU다.** OURS는 P1과 P16에서 비슷하게 약 6.1–6.2 GiB를 재배치했고 실제 GPU relocation은 각각 약 44/68 ms뿐이었다. 그러나 compaction CPU는 11.31 s에서 1.78 s로 줄었다. 따라서 이 pressure workload의 P1 비용은 이동 bandwidth가 아니라 작은 page 단위의 mapping/compaction control overhead다.
5. **Row fence는 이 workload의 병목이 아니었다.** 약 225개의 asynchronous transfer fence가 등록됐지만 compaction defer, urgent wait, measured fence GPU wait는 모두 0이었다.
6. **Mamba forward gather/scatter는 존재하지만 주원인은 아니다.** 별도 CUDA-event run에서 OURS P1은 6.91 GiB를 gather/scatter하며 각각 88.8/84.4 ms, P16은 6.59 GiB에 77.7/78.3 ms를 사용했다. 이는 P1의 약 11 s compaction CPU 비용보다 훨씬 작다.

따라서 측정 시점의 최적화 우선순위는 다음과 같았다.

1. Mamba H2D의 한-slot staging + row별 `copy_()`를 batched pinned transfer/scatter kernel 또는 더 큰 contiguous staging으로 교체한다.
2. P1 compaction의 per-page CPU mapping/control을 batch화한다.
3. 그 다음 Mamba prefill gather/scatter를 stride-aware kernel로 제거한다.

Profiling 자체의 overhead 때문에 이 run의 throughput은 clean 성능표와 섞지 않았다. 다만 방향은 동일했다: profiling run에서 P1 OURS는 baseline보다 17.8% 느렸고 P16은 5.9% 빨랐다. 모든 8개 aggregate profile run과 2개 layout profile run은 120/120 요청, required eviction/backup/load-back/host hit, dropped token 0을 통과했다.

PyTorch profiler trace도 P1 baseline/OURS 각 50 forward step을 수집했지만, 이 RTX 5090 환경에서는 CUPTI가 `CUPTI_ERROR_UNKNOWN (999)`로 초기화되지 않아 CUDA activity가 빠졌다. 따라서 GPU 결론에는 해당 trace의 CPU duration을 사용하지 않았고, 정상 수집된 CUDA-event aggregate만 사용했다.

## 최적화 iteration: direct Mamba H2D

### 문제와 변경

Unified L1의 Mamba destination은 각 1 MiB row 사이가 18.63 MiB slot stride로 떨어져 있다. 기존 H2D는 선택된 host row마다 다음 두 작업을 반복했다.

1. pinned host row를 한-slot GPU staging buffer로 `copy_()`한다.
2. staging row를 실제 unified destination에 `index_copy_()`한다.

실제 load-back batch는 1, 4, 6, 8 slot이었지만 staging capacity가 1이라 batch를 전혀 amortize하지 못했다. RTX 5090 microbenchmark에서 기존 경로는 5.78–6.75 GiB/s, multi-row staging은 최대 24.10 GiB/s, 기존 mapped-host Mamba JIT kernel은 35.15–38.06 GiB/s였다.

따라서 H2D만 staging을 제거하고 registered host row에서 strided unified destination으로 JIT kernel이 직접 읽도록 변경했다. Typed L2에서는 Mamba slot 사이에 shared-chunk padding이 있으므로 source stride도 `item_size * num_layers`로 추정하지 않고 실제 `src.stride(0)`을 전달한다. D2H는 GPU kernel이 host memory에 직접 store하는 방향의 제약이 달라 기존의 bounded staging 경로를 유지한다.

첫 구현은 source slot을 연속 Mamba 배열로 가정해 typed-chunk roundtrip test 2개가 실패했다. 해당 코드로 수집한 `iter06_mamba_direct_h2d` 결과는 모두 폐기했다. 실제 tensor stride를 전달하도록 수정한 뒤 전체 test와 server 실험을 처음부터 다시 수행했다.

### 결과

아래 값은 수정 전 2회와 수정 후 2회 profile run의 평균이다. 모든 run은 같은 P1/P16 pressure workload를 사용했다.

| P | 항목 | 수정 전 | 수정 후 | 변화 |
|---:|---|---:|---:|---:|
| 1 | Mamba H2D | 4.23 GiB/s | 19.18 GiB/s | 4.53x |
| 1 | 전체 H2D | 11.81 GiB/s | 27.43 GiB/s | 2.32x |
| 1 | H2D enqueue CPU | 168.12 ms | 55.12 ms | -67.2% |
| 1 | 전체 처리량 | 29,364 tok/s | 29,263 tok/s | -0.3% |
| 16 | Mamba H2D | 4.21 GiB/s | 20.43 GiB/s | 4.86x |
| 16 | 전체 H2D | 12.12 GiB/s | 29.48 GiB/s | 2.43x |
| 16 | H2D enqueue CPU | 175.60 ms | 57.11 ms | -67.5% |
| 16 | 전체 처리량 | 39,519 tok/s | 40,440 tok/s | +2.3% |

P1에서는 약 11.4 s의 compaction CPU가 남아 있어 전송 개선이 end-to-end 처리량으로 나타나지 않았다. P16은 compaction이 약 1.8 s라 H2D 개선 일부가 전체 성능에 반영됐다.

수정 후 전체 관련 test는 `162 passed, 1 skipped, 14 subtests passed`였다. 별도 P1/P16 parity run은 각각 L2에서 5,890 token을 load-back했고 output ID와 logprob token ID가 모두 같았다. 원본 대비 최대 absolute logprob 차이는 두 조건 모두 `5.83e-5`였다.

유효한 raw artifact는 다음 경로에 있다.

```text
artifacts/u2_optimization/iter06_mamba_direct_h2d_fixed/
  fixed-stride-p{1,16}-r{1,2}/
  fixed-stride-parity-p{1,16}/
```

### 결정

이 변경을 유지한다. 측정된 Mamba H2D 병목과 Python enqueue 비용을 제거했고, typed-chunk stride와 실제 L2 load-back correctness를 함께 검증했다. 다음 최적화 대상은 P1 compaction CPU다.

## 최적화 iteration: batched compaction mapping

### 문제와 변경

P1 profile에서 Full/KV urgent compaction만 약 11 s를 사용했다. 실제 GPU relocation은 약 46 ms였으므로 data movement가 아니라 CPU control path가 병목이었다. 원인은 survivor를 선택할 때마다 다음 mapping lookup을 실행한 것이다.

```python
v_moved = int(self.physical_to_virtual[src].item())
```

P1은 한 run에서 약 25만 row를 이동하므로 GPU→CPU synchronization도 같은 횟수만큼 발생했다. 변경 후에는 survivor source/destination을 CPU list에 모으고 commit 시 다음 한 번의 indexed gather로 모든 virtual ID를 가져온다.

```python
v_moveds = self.physical_to_virtual[src_pages]
```

잘못된 `p2v=-1` survivor를 검출하는 invariant는 유지하되 row마다 확인하지 않고 commit batch마다 한 번 확인한다. Row-aware fence가 move batch를 나누는 경우에도 각 batch가 자체 mapping을 gather하므로 기존 ordering을 보존한다. Scalar p2v lookup을 거부하는 tensor proxy 회귀 테스트도 추가했다.

### 결과

비교 기준은 direct Mamba H2D 최적화가 적용된 동일 source다. 아래 값은 수정 전 2회와 수정 후 2회 profile run의 평균이다.

| P | 항목 | 수정 전 | 수정 후 | 변화 |
|---:|---|---:|---:|---:|
| 1 | compaction CPU | 11.42 s | 1.54 s | -86.5% |
| 1 | allocator CPU, compaction 포함 | 13.04 s | 3.23 s | -75.2% |
| 1 | 전체 처리량 | 29,263 tok/s | 40,408 tok/s | +38.1% |
| 1 | 평균 TTFT | 650.3 ms | 343.4 ms | -47.2% |
| 16 | compaction CPU | 1.83 s | 1.32 s | -27.8% |
| 16 | allocator CPU, compaction 포함 | 3.23 s | 2.59 s | -19.6% |
| 16 | 전체 처리량 | 40,440 tok/s | 39,993 tok/s | -1.1% |

P1 수정 전/후의 relocated payload는 각각 평균 6.31/6.05 GiB였고, Mamba H2D는 19.18/19.34 GiB/s였다. 따라서 +38.1%는 transfer 경로가 달라진 결과가 아니라 작은 page의 per-row mapping synchronization을 제거한 효과다. P16은 한 physical page가 16 token이라 mapping 횟수가 이미 적어 component CPU 감소가 전체 처리량 개선으로 이어지지 않았다.

P1/P16 parity는 각각 L2에서 5,890 token을 load-back한 뒤 output ID와 logprob token ID가 모두 같았다. 최대 absolute logprob 차이는 `5.83e-5`였다. 두 performance 조건도 각각 120/120 요청, required eviction/backup/load-back/host hit, dropped token 0을 통과했다.

현재 전체 관련 test 결과는 `239 passed, 1 skipped, 26 subtests passed`다.

Raw artifact는 다음 경로에 있다.

```text
artifacts/u2_optimization/iter07_batch_compaction_mapping/
  batch-mapping-p{1,16}-r{1,2}/
  batch-mapping-parity-p{1,16}/
```

### 결정

이 변경을 유지한다. P1의 최상위 CPU 병목을 직접 제거하고 end-to-end 성능을 크게 개선했으며, P16에서는 성능 regression 없이 component 비용을 줄였다. 이 최적화는 HiCache 전용이 아니라 unified-memory allocator 자체의 개선이며, 별도 upstream PR용 commit `66535cef3`에서도 독립적으로 검증했다.

## 최적화 iteration: urgent preview-plan reuse

### 가설

Row-aware fence는 compaction 전에 정확한 source/destination footprint를 preview한다. Urgent compaction은 forward를 먼저 drain하므로 preview 뒤에 write hazard로 plan이 잘리지 않는다. 따라서 urgent pass가 같은 two-pointer survivor walk를 다시 수행하지 않고 preview의 src/dst를 실제 move plan으로 재사용할 수 있다.

### 결과

P1은 batch-mapping control 2회와 preview-reuse 2회를 비교했고, P16은 preview-reuse pilot 1회를 실행했다. 모든 run은 120/120 요청과 required cache-pressure validation을 통과했다.

| P | 항목 | control | preview reuse | 변화 |
|---:|---|---:|---:|---:|
| 1 | compaction CPU | 1.537 s | 1.474 s | -4.0% |
| 1 | allocator CPU, compaction 포함 | 3.229 s | 3.058 s | -5.3% |
| 1 | 전체 처리량 | 40,408 tok/s | 40,295 tok/s | -0.3% |
| 16 | compaction CPU | 1.325 s | 1.251 s | -5.5% |
| 16 | 전체 처리량 | 39,993 tok/s | 41,081 tok/s | +2.7% pilot |

P16 control 두 번 자체가 39,154–40,833 tok/s 범위였으므로 한 번의 +2.7% pilot은 효과를 입증하지 못한다. P1에서는 component CPU 감소가 반복됐지만 end-to-end 성능은 개선되지 않았다.

Raw artifact와 각 run의 exact experimental worktree diff는 다음 경로의 manifest에 있다.

```text
artifacts/u2_optimization/iter08_urgent_preview_reuse/
  preview-reuse-p1-r{1,2}/
  preview-reuse-p16-r1/
```

### 결정

변경을 폐기하고 production code를 원복했다. 약 60–170 ms의 누적 CPU 감소는 측정됐지만 end-to-end 이득이 없었고, conservative fence preview를 실행 plan으로 승격하면 allocator correctness의 결합도가 커진다. 이 위험을 감수할 만큼의 성능 근거가 아니다.

## 최종 optimized OURS와 baseline 비교

아래 값은 동일 profiler와 pressure workload에서 baseline 2회, direct Mamba H2D + batched compaction mapping이 적용된 OURS 2회의 평균이다.

| P | 구성 | 처리량 (tok/s) | 평균 TTFT (ms) | 전체 D2H GPU (ms) | 전체 H2D (GiB/s) |
|---:|---|---:|---:|---:|---:|
| 1 | non-unified HiCache baseline | 35,728 | 462.1 | 2,203.1 | 29.59 |
| 1 | optimized OURS | 40,408 | 343.4 | 233.5 | 28.08 |
| 16 | non-unified HiCache baseline | 37,319 | 410.1 | 428.2 | 28.83 |
| 16 | optimized OURS | 39,993 | 331.7 | 230.3 | 29.31 |

최종 OURS 처리량은 baseline보다 P1에서 13.1%, P16에서 7.2% 높았다. H2D는 두 구성 모두 약 28–29 GiB/s로 동률이고, P1 D2H는 unified raw-page 전송이 baseline의 layer/page staging보다 크게 빠르다. Row fence wait/defer는 여전히 0이었고, 최종 parity와 전체 unit/kernel test도 통과했다.

## P32 baseline crash

Non-unified baseline P32는 첫 KV eviction에서 CUDA illegal memory access로 종료됐다. P32에서 한 page copy가 192 KiB가 되어 128 KiB 이상에만 사용하는 upstream `cudaMemcpyBatchAsync` fast path가 활성화된다. 전송 index는 device `8224..9503`, host `0..1279`로 모두 유효하고 page도 완전히 연속이었다.

진단용으로 batch fast path만 끄고 기존 `cudaMemcpyAsync` fallback을 사용하자 같은 P32 eviction workload가 20/20 요청으로 통과했다. 이 변경은 baseline 정의를 바꾸므로 본 성능표에는 섞지 않았고 production 코드에도 반영하지 않았다. 작은 host allocation을 사용하는 microbenchmark의 P32 baseline은 정상 동작했다.

## 재현

```bash
source ~/.venv_sglang/bin/activate

PYTHONPATH=python:test python -m pytest -q --tb=short \
  test/registered/kernels/ops/kvcache/test_unified_chunk_hicache_transfer.py \
  test/registered/kernels/ops/kvcache/test_unified_hicache_transfer.py \
  test/registered/kernels/ops/kvcache/test_unified_row_transfer_fence.py \
  test/registered/kernels/ops/kvcache/test_hicache_page_first_write_back.py \
  test/registered/kernels/ops/kvcache/test_hicache.py \
  test/registered/kernels/ops/mamba/test_transfer_mamba.py \
  test/registered/unit/mem_cache/test_typed_chunk_host.py \
  test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py \
  test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py \
  test/registered/unit/mem_cache/test_multi_ended_allocator.py

python benchmark/hicache/bench_unified_typed_page_transfer.py \
  --page-sizes 1 2 4 8 16 32 \
  --tokens 512 4096 \
  --patterns contiguous fragmented \
  --warmup 3 \
  --repetitions 15 \
  --output artifacts/unified_typed_page_transfer/full-run1/results.json

python benchmark/hicache/run_unified_ablation.py \
  --variant paged-ours \
  --scenario steady \
  --run-name paired-r1-p4 \
  --artifact-root artifacts/unified_typed_page_server/matrix \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --max-total-tokens 120000 \
  --page-size 4 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --context-length 65536 \
  --hicache-size 12 \
  --hicache-write-policy write_back \
  --input-len 10000 \
  --output-len 64 \
  --groups 40 \
  --group-order-start 1 \
  --rounds 3 \
  --shared-ratio 0.95 \
  --prime-output-len 1 \
  --prime-repeats 1 \
  --max-concurrency 4 \
  --reverse-group-order \
  --require-eviction \
  --require-loadback \
  --require-backup \
  --require-host-hit \
  --forbid-dropped \
  --no-profile-memory-breakdown \
  --server-extra-arg=--mem-fraction-static=0.27 \
  --server-extra-arg=--language-only \
  --server-extra-arg=--mm-feature-transport=cpu

# baseline은 --variant paged-baseline으로 바꾸고 같은 조건에서 실행한다.
python benchmark/hicache/analyze_unified_typed_page_results.py

# component profiler는 기본 활성화다. P1 예시:
python benchmark/hicache/run_unified_ablation.py \
  --variant paged-ours \
  --scenario steady \
  --run-name profile-p1 \
  --artifact-root artifacts/unified_typed_page_profile \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --max-total-tokens 120000 --page-size 1 \
  --max-running-requests 4 --chunked-prefill-size 4096 \
  --context-length 65536 --hicache-size 12 \
  --hicache-write-policy write_back \
  --input-len 10000 --output-len 64 \
  --groups 40 --group-order-start 1 --rounds 3 --shared-ratio 0.95 \
  --prime-output-len 1 --prime-repeats 1 --max-concurrency 4 \
  --reverse-group-order --require-eviction --require-loadback \
  --require-backup --require-host-hit --forbid-dropped \
  --server-extra-arg=--mem-fraction-static=0.27 \
  --server-extra-arg=--language-only \
  --server-extra-arg=--mm-feature-transport=cpu

python benchmark/hicache/summarize_memory_breakdown.py \
  artifacts/unified_typed_page_profile/profile-valid-r1-p1 \
  artifacts/unified_typed_page_profile/profile-valid-r2-p1 \
  --markdown-output artifacts/unified_typed_page_profile/MEMORY_BREAKDOWN.md
```

Server run의 exact command, environment, git SHA, GPU snapshot, request별 latency와 transfer metrics는 아래 디렉터리에 저장되어 있다.

- `artifacts/unified_typed_page_server/matrix`
- `artifacts/unified_typed_page_server/parity`
- `artifacts/unified_typed_page_transfer/full-run{1,2,3}`
- 집계 CSV: `artifacts/unified_typed_page_summary`
- component profile: `artifacts/unified_typed_page_profile`
