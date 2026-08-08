# Qwen3.5 unified-memory + HiCache final evaluation

## 결론

RTX 5090에서 unified-memory + HiCache를 page size 1, 8, 32와
write-back/write-through로 검증했다. Qwen3.5-0.8B는 성능 3회 반복,
정확도 parity, 내부 component profiling을 완료했고, 전체 소요 시간이 4시간
미만이어서 Qwen3.5-4B의 page size 1, 32 핵심 조건도 추가했다.

- 최종 성능 run은 231/231개가 성공했다: 0.8B 189개와 4B 42개다.
- 모든 run에서 client 실패와 HiCache dropped token은 0이었다.
- 0.8B parity 21/21개가 통과했다. 총 861개 비교에서 output token ID와
  logprob token ID가 모두 같았다.
- U3 typed L2는 0.8B에서 U2 split L2보다 평균 3.9–23.6%, 4B에서는
  1.8–29.2% 높은 total token throughput을 보였다.
- U3의 실제 server transfer는 D2H 약 31–35 GiB/s, H2D 약 37 GiB/s로
  static HiCache와 비슷한 범위다. U2의 H2D는 약 6 GiB/s였다.
- 가장 큰 차이는 긴 prompt의 write-back이었다. U3는 S1 static HiCache보다
  0.8B에서 평균 189.2%, 4B에서 49.3% 높았다. 이는 DMA만의 효과가 아니라
  unified L1/L2의 동적 용량과 cache reuse까지 포함한 end-to-end 결과다.

이번 결과는 기본 구현과 주요 전송 경로가 안정적으로 동작한다는 근거다.
다만 4B는 추가 확인용 1회 측정이므로 절대 성능 차이에 대한 통계적 결론은
0.8B 3회 결과를 우선해야 한다.

## 비교 구성

| 이름 | L1 | HiCache | L2 |
|---|---|---|---|
| S1 | static memory partition | async | standard static HiCache |
| U0 | unified memory | 없음 | 없음 |
| U2 | unified memory | async + row fence | legacy split L2 |
| U3 | unified memory | async + row fence | shared typed L2 |

S1, U2, U3는 write-back과 write-through를 각각 측정했다. U0에는 write
policy가 없으므로 동일한 U0 결과를 두 정책 표에 비교 기준으로 표시했다.

## 환경과 workload

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB
- 모델: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`
- dtype: BF16
- clean 성능 측정: CUDA graph 사용
- profile 측정: CUDA graph를 끄고 별도 실행
- 공통 설정:
  - `--max-total-tokens 120000`
  - `--max-running-requests 8`
  - `--chunked-prefill-size 4096`
  - `--context-length 65536`
  - `--hicache-size 12`
  - `--mamba-radix-cache-strategy extra_buffer`
  - 0.8B `--mem-fraction-static 0.27`
  - 4B `--mem-fraction-static 0.55`
- `--max-mamba-cache-size`는 사용하지 않았다.

| workload | input/output | groups × rounds | concurrency |
|---|---:|---:|---:|
| short | 3,000 / 256 | 120 × 2 | 8 |
| middle | 10,000 / 256 | 40 × 2 | 8 |
| long, 0.8B | 50,000 / 128 | 8 × 2 | 4 |
| long, 4B | 50,000 / 128 | 8 × 2 | 2 |

각 workload는 95% shared prefix와 역순 두 번째 round를 사용해 L1 eviction,
L2 backup, host hit, L2 load-back을 강제로 발생시켰다. HiCache 구성은 네
event가 모두 관측되지 않으면 실패하도록 했다.

0.8B clean 1회분은 약 63–65분이었다. clean 3회, parity, profile의 유효
artifact 실행시간 합계는 약 3시간 48분이었다. 따라서 사전에 정한 4시간
조건에 따라 4B를 추가했고, 4B 42개 유효 run은 약 1시간 35분이었다.

## Correctness와 artifact 감사

| 검증 | 결과 |
|---|---:|
| 0.8B clean | 63조건 × 3회 = 189/189 passed |
| 4B clean | 42/42 passed |
| 0.8B parity | 21/21 passed |
| parity 비교 | 861/861 output IDs and logprob token IDs equal |
| 전체 최대 absolute logprob diff | 0.0179786, tolerance 0.02 |
| 실제 L2 복원 조건의 최대 diff | 0.0001553 |
| HiCache 실제 load-back | 5,890–9,986 tokens |
| 관련 CUDA unit tests | 53 passed |
| pre-commit, 실험/분석 스크립트 | passed |

모든 231개 clean 결과는 `validation.passed=true`, 요청 실패 0, dropped
token 0이었다. 0.8B 첫 반복은 실험 중 발견한 paged transfer 문제를
수정하며 세 SHA로 나뉘었다. P1은 수정의 영향을 받지 않는 기존 경로이고,
P8 U2/U3 및 P32 전체는 해당 수정 후 다시 실행했다. 두 번째와 세 번째
반복 126개는 최종 SHA `34b8af8f2`로 고정했다.

원본 artifact와 위 SHA들은 재현을 위해
`origin/archive/unified-hicache-paged-l2-eval-20260809`에 보존했다. 커밋
정리 후 동일한 production/test 트리의 최종 SHA는 `7e41fa422`다. 정리
과정에서는 paged legacy transfer와 dense staged layout 수정을 typed
multi-token L2 기능 커밋에 합쳤다. archive 대비 production/test diff는
pre-commit이 적용한 formatting뿐이며 동작 변경은 없다.

## 0.8B end-to-end 성능

아래 값은 3회 total token throughput 평균이다. 단위는 token/s다. 전체
평균의 run-to-run CV 중앙값은 2.7%, 최대는 13.3%였다. 평균과 표준편차,
TTFT, TPOT, transfer rate의 원수치는
[`clean_summary.csv`](../../artifacts/qwen35_hicache_matrix/summary/clean_summary.csv)에
있다.

### Write-back

| workload / P | S1 static | U0 unified only | U2 split L2 | U3 typed L2 |
|---|---:|---:|---:|---:|
| 3k / 1 | 28,514 | 24,771 | 27,965 | 29,491 |
| 3k / 8 | 28,828 | 24,251 | 29,654 | 30,584 |
| 3k / 32 | 30,680 | 25,721 | 29,839 | 30,739 |
| 10k / 1 | 48,663 | 41,704 | 53,918 | 61,319 |
| 10k / 8 | 50,950 | 41,079 | 58,015 | 62,526 |
| 10k / 32 | 54,017 | 43,192 | 60,267 | 65,403 |
| 50k / 1 | 44,167 | 41,779 | 97,897 | 127,804 |
| 50k / 8 | 45,165 | 39,446 | 111,802 | 135,096 |
| 50k / 32 | 48,386 | 42,137 | 113,123 | 135,075 |

U3의 page 평균 변화는 S1 대비 3k +3.2%, 10k +23.3%, 50k +189.2%다.
U2 대비로는 각각 +3.9%, +10.0%, +23.6%다.

### Write-through

| workload / P | S1 static | U0 unified only | U2 split L2 | U3 typed L2 |
|---|---:|---:|---:|---:|
| 3k / 1 | 26,247 | 24,771 | 27,337 | 29,072 |
| 3k / 8 | 26,766 | 24,251 | 28,673 | 33,057 |
| 3k / 32 | 28,300 | 25,721 | 29,057 | 33,092 |
| 10k / 1 | 44,938 | 41,704 | 65,462 | 72,318 |
| 10k / 8 | 47,437 | 41,079 | 66,004 | 73,691 |
| 10k / 32 | 50,126 | 43,192 | 67,372 | 72,804 |
| 50k / 1 | 39,568 | 41,779 | 109,998 | 133,011 |
| 50k / 8 | 40,314 | 39,446 | 114,031 | 139,042 |
| 50k / 32 | 42,878 | 42,137 | 115,606 | 139,443 |

U3의 page 평균 변화는 S1 대비 3k +17.1%, 10k +53.8%, 50k +235.4%다.
U2 대비로는 각각 +11.8%, +10.1%, +21.2%다.

### 0.8B P32 latency

각 값은 `TTFT p50 / TPOT p50`이며 단위는 ms다.

| policy | workload | S1 | U0 | U2 | U3 |
|---|---|---:|---:|---:|---:|
| WB | 3k | 87.1 / 2.67 | 207.3 / 2.84 | 142.0 / 2.70 | 102.6 / 2.69 |
| WB | 10k | 394.5 / 3.67 | 536.8 / 4.23 | 347.9 / 3.45 | 171.6 / 3.45 |
| WB | 50k | 2819.9 / 7.03 | 2149.6 / 17.14 | 847.6 / 7.08 | 517.2 / 7.10 |
| WT | 3k | 119.7 / 2.70 | 207.3 / 2.84 | 134.5 / 2.73 | 91.5 / 2.74 |
| WT | 10k | 457.3 / 3.67 | 536.8 / 4.23 | 316.3 / 3.48 | 151.3 / 3.49 |
| WT | 50k | 3688.6 / 6.83 | 2149.6 / 17.14 | 894.8 / 7.07 | 540.0 / 7.06 |

## 4B 추가 실험

4B는 의미 있는 양 끝점 P1/P32를 1회 측정했다. 따라서 아래 값은 모델
크기 확장 시 기능과 성능 방향을 확인하는 결과이며 variance 추정치는 아니다.

### Write-back total token throughput

| workload / P | S1 static | U0 unified only | U2 split L2 | U3 typed L2 |
|---|---:|---:|---:|---:|
| 3k / 1 | 7,700 | 7,413 | 7,562 | 7,848 |
| 3k / 32 | 7,785 | 7,357 | 7,583 | 7,808 |
| 10k / 1 | 13,371 | 12,526 | 13,867 | 15,716 |
| 10k / 32 | 13,640 | 12,419 | 13,916 | 15,682 |
| 50k / 1 | 11,731 | 11,346 | 13,591 | 17,556 |
| 50k / 32 | 11,756 | 11,087 | 13,552 | 17,502 |

U3는 page 평균으로 S1 대비 3k +1.1%, 10k +16.3%, 50k +49.3%,
U2 대비 +3.4%, +13.0%, +29.2%였다.

### Write-through total token throughput

| workload / P | S1 static | U0 unified only | U2 split L2 | U3 typed L2 |
|---|---:|---:|---:|---:|
| 3k / 1 | 7,489 | 7,413 | 7,280 | 7,445 |
| 3k / 32 | 7,538 | 7,357 | 7,338 | 7,439 |
| 10k / 1 | 13,023 | 12,526 | 12,758 | 13,379 |
| 10k / 32 | 13,258 | 12,419 | 12,980 | 13,291 |
| 50k / 1 | 11,312 | 11,346 | 11,180 | 12,520 |
| 50k / 32 | 11,334 | 11,087 | 11,173 | 12,336 |

U3는 page 평균으로 S1 대비 3k -1.0%, 10k +1.5%, 50k +9.8%,
U2 대비 +1.8%, +3.6%, +11.2%였다. 3k의 -1.0%는 1회 측정에서
통계적 regression으로 판단할 수 없는 크기다.

### 4B P32 latency

각 값은 `TTFT p50 / TPOT p50`이며 단위는 ms다.

| policy | workload | S1 | U0 | U2 | U3 |
|---|---|---:|---:|---:|---:|
| WB | 3k | 590.4 / 9.40 | 874.7 / 9.18 | 886.1 / 9.16 | 733.6 / 9.08 |
| WB | 10k | 2039.8 / 13.12 | 2249.7 / 14.01 | 1856.5 / 12.20 | 1417.2 / 11.78 |
| WB | 50k | 4683.2 / 13.83 | 4833.7 / 14.71 | 4851.4 / 13.52 | 2106.5 / 13.45 |
| WT | 3k | 871.2 / 9.05 | 874.7 / 9.18 | 894.9 / 9.31 | 895.3 / 9.28 |
| WT | 10k | 2219.6 / 12.52 | 2249.7 / 14.01 | 2268.3 / 12.81 | 2265.7 / 12.82 |
| WT | 50k | 4854.2 / 13.50 | 4833.7 / 14.71 | 4914.1 / 13.50 | 4876.2 / 13.44 |

## Component profiling

0.8B 10k workload를 CUDA graph 없이 별도 실행했다. 아래 전송률은 CUDA
event interval의 logical bytes/time이며, throughput은 profiling overhead가
포함되므로 clean 표와 직접 섞지 않는다.

### Write-back

| P | 구성 | D2H total GiB/s | H2D total GiB/s | KV H2D GiB/s | transfer control CPU s | compaction CPU s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | S1 | 6.9 | 37.9 | 47.8 | 1.55 | 0.00 |
| 1 | U2 | 6.2 | 6.0 | 4.8 | 2.06 | 0.73 |
| 1 | U3 | 34.2 | 36.6 | 47.3 | 0.39 | 0.73 |
| 8 | S1 | 28.1 | 37.7 | 48.1 | 0.73 | 0.00 |
| 8 | U2 | 22.4 | 6.0 | 4.7 | 1.59 | 0.68 |
| 8 | U3 | 35.0 | 37.0 | 47.6 | 0.40 | 0.70 |
| 32 | S1 | 39.7 | 38.5 | 48.6 | 0.67 | 0.00 |
| 32 | U2 | 29.5 | 6.2 | 4.9 | 1.52 | 0.68 |
| 32 | U3 | 34.4 | 36.9 | 47.9 | 0.44 | 0.69 |

### Write-through

| P | 구성 | D2H total GiB/s | H2D total GiB/s | KV H2D GiB/s | transfer control CPU s | compaction CPU s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | S1 | 6.7 | 37.9 | 47.8 | 3.46 | 0.00 |
| 1 | U2 | 6.9 | 6.1 | 4.8 | 2.64 | 0.40 |
| 1 | U3 | 30.7 | 36.9 | 47.2 | 0.34 | 0.39 |
| 8 | S1 | 25.7 | 36.8 | 46.2 | 1.76 | 0.00 |
| 8 | U2 | 22.4 | 6.6 | 5.3 | 1.71 | 0.39 |
| 8 | U3 | 31.0 | 37.4 | 48.1 | 0.37 | 0.40 |
| 32 | S1 | 33.2 | 38.2 | 48.2 | 1.62 | 0.00 |
| 32 | U2 | 28.0 | 6.4 | 5.1 | 1.66 | 0.40 |
| 32 | U3 | 31.2 | 38.1 | 48.6 | 0.38 | 0.39 |

U3의 typed raw-page 경로는 U2의 가장 큰 전송 병목을 제거했다. 특히 H2D
total이 U2 약 6 GiB/s에서 U3 약 37 GiB/s로 회복되어 S1과 거의 같다.
P1 D2H도 U2 6.2–6.9 GiB/s에서 U3 30.7–34.2 GiB/s로 개선됐다.

U2와 U3의 allocator/compaction 시간은 비슷하므로 U3의 차이는 allocator가
아니라 typed transfer에서 나온다. U3 profile마다 약 373–548개의 row fence
등록이 실제 발생했다. write-through에서는 영향권 밖 compaction 허용도
52–67회 관측됐고, 측정된 fence wait/defer는 0이었다. 이 workload에서는
row fence가 정확히 작동했지만 성능 병목은 아니었다.

상세 operation별 결과는
[`profile_summary.csv`](../../artifacts/qwen35_hicache_matrix/summary/profile_summary.csv)와
[`profile_breakdown.csv`](../../artifacts/qwen35_hicache_matrix/summary/profile_breakdown.csv)에
있다. Parity 집계는
[`parity_summary.csv`](../../artifacts/qwen35_hicache_matrix/summary/parity_summary.csv)에
있다.

## 실험 중 발견한 문제와 처리

1. U2 P8에서 legacy relayout이 page size 1을 가정해 전송이 실패했다.
   multi-token page envelope와 4D H2D view를 지원하도록 수정하고 paged
   roundtrip 회귀 테스트를 추가했다.
2. P32 S1에서 RTX 5090/WSL의 registered mmap destination에
   `cudaMemcpyBatchAsync`를 사용하면 API 성공 뒤 illegal memory access가
   발생했다. WSL에서는 stream-ordered `cudaMemcpyAsync` fallback을 자동
   사용하도록 했다.
3. 4B U0의 최초 50k concurrency 4는 120k L1에 약 200k 실행 working set을
   요구해 OOM이 났다. 로그상 full usage 99%, 가용 1,084 token에서 4,096
   token을 요청한 설정 한계였다. U0에는 L2가 없으므로 실행 중 row를
   offload할 수도 없다. 모든 4B 50k 구성을 concurrency 2로 통일해 다시
   측정했으며 42/42개가 통과했다. 최초 c4 결과는 최종 비교에서 제외했다.

## 재현

```bash
source /home/sukwoo24/.venv_sglang/bin/activate

# 0.8B clean 3회
for repeat in 1 2 3; do
  python benchmark/hicache/run_qwen35_hicache_matrix.py clean \
    --model-size 0.8b --pages 1 8 32 --repetition "${repeat}" \
    --mem-fraction-static 0.27 --resume
done

# 0.8B parity와 component profile
python benchmark/hicache/run_qwen35_hicache_matrix.py parity \
  --model-size 0.8b --pages 1 8 32 --mem-fraction-static 0.27 --resume
python benchmark/hicache/run_qwen35_hicache_matrix.py profile \
  --model-size 0.8b --pages 1 8 32 --mem-fraction-static 0.27 --resume

# 4B 핵심 subset
python benchmark/hicache/run_qwen35_hicache_matrix.py clean \
  --model-size 4b --pages 1 32 --repetition 1 \
  --mem-fraction-static 0.55 --long-concurrency 2 --resume

# CSV 재생성
python benchmark/hicache/analyze_qwen35_hicache_matrix.py
```
