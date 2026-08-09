# Qwen3.5-0.8B CUDA graph ablation

## 결론

RTX 5090에서 static memory + HiCache(S1), unified memory only(U0), unified
memory + typed HiCache(U3)를 CUDA graph ON/OFF로 비교했다. Page size 8,
write-back, 3K와 50K workload를 각 3회 반복했다.

- 성능 run 36/36, parity run 6/6이 통과했다.
- Short에서는 CUDA graph가 세 구현의 처리량을 2.38–2.63배 높였다.
- Long에서는 S1이 1.10배, U3가 1.17배 높아졌다. U0의 1.03배 차이는
  반복 편차와 비슷해 명확한 향상으로 보기 어렵다.
- U3는 CUDA graph의 이점을 잃지 않았다. Short graph ON에서는 S1과
  사실상 동률이고, long에서는 U3/S1 처리량 비가 2.70배에서 2.89배로
  증가했다.
- Graph ON batch의 99.19–99.81%가 실제 graph를 사용했다. Capture에는
  평균 약 6.1–6.4초와 0.64 GB가 추가로 필요했다.
- 모든 run에서 request failure, dropped token, assertion, OOM은 0이었다.

## 실험 구성

| 이름 | L1 | HiCache | L2 |
|---|---|---|---|
| S1 | static memory partition | async, write-back | standard static HiCache |
| U0 | unified memory | 없음 | 없음 |
| U3 | unified memory | async + row fence, write-back | shared typed L2 |

공통 설정은 다음과 같다.

- Model: `Qwen/Qwen3.5-0.8B`, BF16
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB
- `--page-size 8`
- `--max-total-tokens 120000`
- `--max-running-requests 8`
- `--mem-fraction-static 0.27`
- `--hicache-size 12`
- `--chunked-prefill-size 4096`
- `--context-length 65536`
- `--max-mamba-cache-size`는 지정하지 않음
- Graph ON: prefill=`breakable`, decode=`full`
- Graph OFF: prefill/decode=`disabled`

| workload | input/output | groups × rounds | concurrency |
|---|---:|---:|---:|
| Short | 3,000 / 256 | 120 × 2 | 8 |
| Long | 50,000 / 128 | 8 × 2 | 4 |

두 workload 모두 95% shared prefix와 역순 두 번째 round를 사용했다. 모든
조건에서 L1 eviction을 요구했고, S1/U3는 L2 backup, host hit, load-back도
모두 발생해야 통과하도록 했다. 반복마다 variant, workload, graph mode의
실행 순서를 회전해 순서와 GPU 온도 편향을 줄였다.

## 처리량

아래 값은 total token throughput 3회 평균이다. Speedup은 같은 반복의
ON/OFF를 짝지어 계산한 뒤 평균했다.

| workload | 구현 | OFF token/s | ON token/s | ON/OFF | ON 이득 |
|---|---|---:|---:|---:|---:|
| Short | S1 | 9,986 | 25,550 | 2.559× | +155.9% |
| Short | U0 | 8,907 | 21,173 | 2.377× | +137.7% |
| Short | U3 | 9,690 | 25,448 | 2.626× | +162.6% |
| Long | S1 | 35,082 | 38,427 | 1.095× | +9.5% |
| Long | U0 | 33,157 | 34,299 | 1.035× | +3.5% |
| Long | U3 | 94,787 | 110,891 | 1.170× | +17.0% |

각 조건의 throughput CV는 0.79–4.93%였다. Paired speedup 표준편차는
Short 0.023–0.034, long 0.019–0.034였다. U0 long은 세 반복이 1.061,
0.996, 1.047배여서 작은 평균 차이를 확정적인 graph 이득으로 해석하지
않는다.

## TTFT와 TPOT

| workload | 구현 | TTFT OFF→ON | 변화 | TPOT OFF→ON | 변화 |
|---|---|---:|---:|---:|---:|
| Short | S1 | 252.1→173.3 ms | -31.3% | 9.05→3.25 ms | -64.1% |
| Short | U0 | 485.5→306.9 ms | -36.8% | 9.38→3.54 ms | -62.2% |
| Short | U3 | 237.3→169.2 ms | -28.7% | 9.57→3.33 ms | -65.2% |
| Long | S1 | 3475.3→3271.8 ms | -5.9% | 14.36→12.28 ms | -14.5% |
| Long | U0 | 2807.3→2924.8 ms | +4.2% | 25.52→23.01 ms | -9.8% |
| Long | U3 | 617.4→623.6 ms | +1.0% | 11.78→9.31 ms | -20.9% |

Short는 출력 256 token이어서 decode graph 효과가 크게 드러난다. Long은
대부분의 시간이 50K prefill, cache reuse, eviction/load-back에 쓰이므로
전체 graph 이득이 작다. U0/U3 long TTFT의 작은 증가는 반복 변동 범위이며,
TPOT과 전체 처리량은 개선됐다.

## 구현 간 비교에 미치는 영향

| workload | 비교 | Graph OFF | Graph ON |
|---|---|---:|---:|
| Short | U3 / S1 | 0.970× | 0.996× |
| Short | U3 / U0 | 1.088× | 1.202× |
| Long | U3 / S1 | 2.704× | 2.890× |
| Long | U3 / U0 | 2.861× | 3.235× |

CUDA graph를 켜도 U3가 상대적으로 불리해지지 않는다. Short에서는 S1과
동률이고 U0보다 빠르다. Long에서 U3의 큰 이점은 graph 자체만의 효과가
아니라 typed L2의 동적 용량과 95% prefix reuse를 포함한다. 다만 ON/OFF의
eviction, backup, load-back token 수가 비슷하고 모든 HiCache 필수 event가
발생했으므로, graph ON 결과가 cache pressure를 회피해 얻은 결과는 아니다.

HiCache 전송과 allocator/row-fence orchestration은 graph 밖에서 실행된다.
그럼에도 U3의 graph speedup이 S1/U0 이상이므로, 현재 구현의 off-graph
orchestration이 CUDA graph 이득을 상쇄한다는 증거는 없다.

## Graph 적용 감사

- ON: prefill/decode capture가 모든 18개 성능 run에서 완료됐다.
- OFF: graph 사용 batch와 capture는 모든 18개 run에서 0이었다.
- ON graph hit rate: 99.19–99.81%.
- 평균 capture: prefill 약 5.0–5.3초, decode 약 1.0–1.2초.
- Capture memory: 모든 ON run에서 합계 0.64 GB.
- Graph capture를 포함한 server startup은 대체로 5–7초 증가했다. 이 시간은
  위 measured request throughput에는 포함하지 않았다.

## Correctness

10K input, 64 output의 parity를 각 구현의 graph ON/OFF에서 한 번씩 실행했다.

| 검증 | 결과 |
|---|---:|
| Performance | 36/36 passed |
| Parity | 6/6 passed |
| Promotion/replay 비교 | 246/246 output IDs and logprob token IDs equal |
| 최대 absolute logprob diff | 0.0179786, tolerance 0.02 |
| S1 실제 L2 load-back | ON/OFF 각각 9,986 tokens |
| U3 실제 L2 load-back | ON/OFF 각각 5,890 tokens |
| Request failure / dropped token | 0 / 0 |

Graph ON/OFF 사이의 restored reference output token ID도 세 구현에서 모두
동일했다.

## 재현

Server source는 `7e41fa422`에 고정했다. Artifact는
`artifacts/qwen35_cuda_graph_ablation`에 저장한다.

```bash
source /home/sukwoo24/.venv_sglang/bin/activate

for repetition in 1 2 3; do
  python benchmark/hicache/run_qwen35_hicache_matrix.py graph \
    --model-size 0.8b --pages 8 \
    --variants eval-s1 eval-u0 eval-u3 \
    --repetition "${repetition}" \
    --artifact-root artifacts/qwen35_cuda_graph_ablation \
    --max-total-tokens 120000 --max-running-requests 8 \
    --long-concurrency 4 --hicache-size 12 \
    --mem-fraction-static 0.27 --resume
done

python benchmark/hicache/run_qwen35_hicache_matrix.py graph-parity \
  --model-size 0.8b --pages 8 \
  --variants eval-s1 eval-u0 eval-u3 \
  --repetition 1 \
  --artifact-root artifacts/qwen35_cuda_graph_ablation \
  --max-total-tokens 120000 --max-running-requests 8 \
  --hicache-size 12 --mem-fraction-static 0.27 --resume

python benchmark/hicache/analyze_qwen35_cuda_graph_ablation.py
```

분석 결과는 다음 파일로 생성된다.

- `summary/run_summary.csv`
- `summary/aggregate_summary.csv`
- `summary/paired_runs.csv`
- `summary/paired_speedup_summary.csv`
- `summary/parity_summary.csv`
- `summary/audit.json`
