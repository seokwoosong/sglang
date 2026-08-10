# Qwen3.5 unified-memory + HiCache 4-way evaluation

Date: 2026-08-11 (KST)

## Conclusion

The unified-memory + typed-L2 HiCache implementation passed every correctness
and functional gate on Qwen3.5-0.8B. It supports page sizes 1, 8, and 32,
write-back and write-through, and CUDA graph ON and OFF without dropped
requests or tokens.

For the tested eviction-heavy workload, U3 retained static-HiCache-level KV
H2D bandwidth while making KV D2H 1.25-9.57x faster. U3 also improved the
end-to-end 10K throughput by 1.16-1.61x and the 50K throughput by 2.83-3.42x
over S1, depending on page size and write policy. The large 50K improvement
primarily came from lower TTFT through effective host-cache restoration; TPOT
was similar between S1 and U3.

The full 0.8B suite took about 5 hours 5 minutes. Qwen3.5-4B was therefore not
started, following the pre-declared rule that 4B would only be added if the
complete 0.8B suite finished within four hours.

## Evaluated configurations

| ID | Configuration |
|---|---|
| S0 | Static GPU memory, no HiCache |
| S1 | Static GPU memory + HiCache |
| U0 | Unified GPU memory, no HiCache |
| U3 | Unified GPU memory + typed-L2 HiCache |

Common server settings:

- Model: `Qwen/Qwen3.5-0.8B`
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- Driver / PyTorch / CUDA: 610.62 / 2.11.0 / 13.0
- Production commit: `87b7a062c213308f4d9dd289ec4a14fc7d142835`
- Evaluation server source: `fcdd52f4e1835bdb4996ac8c87c83d50c3fe55c2`
- `--mem-fraction-static 0.27`
- `--max-total-tokens 120000`
- `--max-running-requests 8`
- `--chunked-prefill-size 4096`
- `--context-length 65536`
- `--hicache-size 12` for S1/U3
- No `--max-mamba-cache-size`
- Page sizes: 1, 8, 32
- HiCache policies: write-back and write-through
- Main performance matrix: CUDA graph ON

Workloads used two rounds with 95% shared prefixes and reversed group order in
the second round. Every run required actual eviction. S1/U3 additionally
required backup, load-back, and a host-cache hit.

| Workload | Input / output | Groups | Rounds | Concurrency |
|---|---:|---:|---:|---:|
| Short | 3,000 / 256 | 120 | 2 | 8 |
| Middle | 10,000 / 256 | 40 | 2 | 8 |
| Long | 50,000 / 128 | 8 | 2 | 4 |

## Correctness and functional validation

| Gate | Runs | Result |
|---|---:|---|
| Page-size/policy/variant parity | 18/18 | PASS |
| CUDA graph ON/OFF parity at page 8 | 8/8 | PASS |
| Main performance runs | 162/162 | PASS |
| CUDA graph performance runs | 72/72 | PASS |
| Component-profile runs | 36/36 | PASS |

- All output token IDs and logprob token IDs matched S0 exactly.
- Each parity run made 41 output comparisons.
- Maximum absolute logprob difference: 0.0179786 (threshold: 0.02).
- No request failed and `hicache_dropped_tokens_total` remained zero.
- All HiCache runs observed backup, load-back, and a host-cache hit.
- All no-HiCache runs observed eviction but no HiCache transfer.
- CUDA graph ON runs recorded graph batches; OFF runs recorded fallback
  batches.

## End-to-end throughput

Values are mean total tokens/s over three independent server runs. Standard
deviation was below 5% for every group.

| Page | Workload | S0 | S1 WB | S1 WT | U0 | U3 WB | U3 WT |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 3K | 27,148 | 29,767 | 27,371 | 25,820 | 30,534 | 30,496 |
| 1 | 10K | 44,354 | 50,816 | 46,909 | 43,596 | 64,312 | 75,368 |
| 1 | 50K | 30,558 | 46,093 | 41,056 | 43,529 | 134,381 | 140,151 |
| 8 | 3K | 26,889 | 30,257 | 27,977 | 25,302 | 30,404 | 32,851 |
| 8 | 10K | 43,515 | 53,026 | 49,326 | 42,793 | 61,693 | 74,627 |
| 8 | 50K | 30,068 | 46,651 | 41,769 | 41,390 | 133,935 | 136,681 |
| 32 | 3K | 26,860 | 30,415 | 28,036 | 25,325 | 30,562 | 32,980 |
| 32 | 10K | 43,605 | 53,532 | 49,326 | 42,700 | 64,862 | 73,438 |
| 32 | 50K | 30,333 | 47,250 | 41,902 | 42,254 | 134,191 | 137,604 |

Representative page-size 8 latency:

| Workload | Configuration | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---|---:|---:|
| 3K | S0 / U0 | 165.9 / 208.0 | 2.79 / 2.90 |
| 3K | S1 WB / U3 WB | 91.4 / 102.2 | 2.70 / 2.74 |
| 3K | S1 WT / U3 WT | 120.1 / 89.2 | 2.73 / 2.77 |
| 10K | S0 / U0 | 514.9 / 537.7 | 4.47 / 4.28 |
| 10K | S1 WB / U3 WB | 413.5 / 185.6 | 3.67 / 3.51 |
| 10K | S1 WT / U3 WT | 464.6 / 156.2 | 3.72 / 3.53 |
| 50K | S0 / U0 | 5,067.7 / 2,182.6 | 8.55 / 17.57 |
| 50K | S1 WB / U3 WB | 2,791.9 / 531.8 | 7.10 / 7.18 |
| 50K | S1 WT / U3 WT | 3,817.9 / 552.4 | 6.92 / 7.20 |

Interpretation:

- U3 was close to S1 WB for 3K, while U3 WT was 1.11-1.18x faster than S1 WT.
- U3 was 1.16-1.61x faster than S1 at 10K.
- U3 was 2.83-3.42x faster than S1 at 50K. The dominant difference was TTFT,
  not TPOT, which is consistent with more effective L2 restoration.
- U3 throughput was stable across page sizes 1, 8, and 32.
- U0's 50K total throughput was higher than S0, but U0's TPOT was worse. Total
  token throughput alone must therefore not be used as a decode-speed metric.

## CUDA graph

Page size 8 was measured with graph ON and OFF for all four configurations.
S1 and U3 used write-back in this paired ablation. Each pair was repeated three
times; the run order was rotated between repetitions.

| Configuration | 3K | 10K | 50K |
|---|---:|---:|---:|
| S0 | +158.0% | +73.7% | +8.6% |
| S1 | +172.2% | +83.4% | +11.2% |
| U0 | +152.6% | +76.2% | +6.1% |
| U3 | +176.2% | +102.9% | +17.2% |

These are throughput changes from graph OFF to ON. The CUDA graph audit passed
for all 72 performance runs and all 8 token-parity runs.

## KV and Mamba transfer profile

The table shows mean GiB/s over three server runs at the fixed 10K workload.

| Page | Policy | Variant | KV D2H | Mamba D2H | KV H2D | Mamba H2D |
|---:|---|---|---:|---:|---:|---:|
| 1 | WB | S1 | 4.6 | 42.8 | 48.5 | 26.3 |
| 1 | WB | U3 | 44.1 | 47.4 | 47.0 | 23.1 |
| 1 | WT | S1 | 4.6 | 41.9 | 48.5 | 23.5 |
| 1 | WT | U3 | 41.4 | 46.5 | 46.7 | 22.5 |
| 8 | WB | S1 | 23.7 | 42.4 | 48.5 | 25.9 |
| 8 | WB | U3 | 47.4 | 46.3 | 48.2 | 22.7 |
| 8 | WT | S1 | 21.5 | 41.6 | 48.4 | 23.1 |
| 8 | WT | U3 | 46.6 | 47.1 | 47.2 | 22.5 |
| 32 | WB | S1 | 38.2 | 41.6 | 48.0 | 25.8 |
| 32 | WB | U3 | 47.7 | 46.9 | 48.0 | 22.4 |
| 32 | WT | S1 | 30.6 | 39.7 | 48.1 | 22.8 |
| 32 | WT | U3 | 47.0 | 47.3 | 47.6 | 22.6 |

U3/S1 ratios:

| Page | Policy | KV D2H | Mamba D2H | KV H2D | Mamba H2D | Throughput |
|---:|---|---:|---:|---:|---:|---:|
| 1 | WB | 9.57x | 1.11x | 0.97x | 0.88x | 1.19x |
| 1 | WT | 9.05x | 1.11x | 0.96x | 0.96x | 1.33x |
| 8 | WB | 2.00x | 1.09x | 0.99x | 0.87x | 1.14x |
| 8 | WT | 2.17x | 1.13x | 0.98x | 0.98x | 1.27x |
| 32 | WB | 1.25x | 1.13x | 1.00x | 0.87x | 1.12x |
| 32 | WT | 1.54x | 1.19x | 0.99x | 0.99x | 1.25x |

The KV result matches the expected layout behavior: U3 can transfer the typed
page envelope directly on D2H, while the static baseline pays increasingly less
layer-first overhead as page size grows. Both use an efficient per-layer H2D
path, so H2D remains approximately equal.

The server profile showed U3 Mamba H2D at 0.87-0.88x S1 for write-back. A
production-layout microbenchmark did not reproduce a primitive-level deficit:
for contiguous batch 16, baseline versus U3 raw-slot was 48.30 versus 48.10
GiB/s on H2D and 48.76 versus 49.11 GiB/s on D2H. The isolated paths were also
equivalent for batch sizes 1, 4, and 8 and for fragmented rows. The server-only
difference therefore comes from the actual transfer-size mix, scheduling, and
GPU contention rather than an avoidable staging or layout conversion. U3's
end-to-end throughput remained 1.12-1.19x higher in those write-back runs, so no
production-path change was justified by this result.

The KV microbenchmark independently reproduced the server trend for 4,096
contiguous tokens:

| Page | D2H speedup | H2D speedup |
|---:|---:|---:|
| 1 | 10.14x | 0.929x |
| 8 | 2.08x | 0.997x |
| 32 | 1.29x | 0.996x |

## Reproduction

From the `feat/unified-hicache-paged-l2` worktree:

```bash
source /home/sukwoo24/.venv_sglang/bin/activate
eval_artifacts=artifacts/qwen35_unified_hicache_4way

python benchmark/hicache/run_qwen35_hicache_matrix.py parity \
  --model-size 0.8b --pages 1 8 32 --repetition 1 \
  --mem-fraction-static 0.27 --artifact-root "$eval_artifacts"

python benchmark/hicache/run_qwen35_hicache_matrix.py graph-parity \
  --model-size 0.8b --pages 8 --repetition 1 \
  --mem-fraction-static 0.27 --artifact-root "$eval_artifacts"

for repetition in 1 2 3; do
  python benchmark/hicache/run_qwen35_hicache_matrix.py clean \
    --model-size 0.8b --pages 1 8 32 --repetition "$repetition" \
    --mem-fraction-static 0.27 --artifact-root "$eval_artifacts"
  python benchmark/hicache/run_qwen35_hicache_matrix.py graph \
    --model-size 0.8b --pages 8 --repetition "$repetition" \
    --mem-fraction-static 0.27 --artifact-root "$eval_artifacts"
  python benchmark/hicache/run_qwen35_hicache_matrix.py profile \
    --model-size 0.8b --pages 1 8 32 --variants eval-s1 eval-u3 \
    --repetition "$repetition" --mem-fraction-static 0.27 \
    --artifact-root "$eval_artifacts"
done

python benchmark/hicache/analyze_qwen35_hicache_matrix.py \
  --artifact-root "$eval_artifacts" \
  --output-dir "$eval_artifacts/final-summary"
python benchmark/hicache/analyze_qwen35_cuda_graph_ablation.py \
  --artifact-root "$eval_artifacts" --expected-repetitions 3
```

Raw artifacts occupy 79 MiB under
[`artifacts/qwen35_unified_hicache_4way`](../../artifacts/qwen35_unified_hicache_4way).
The main summaries are in
[`final-summary`](../../artifacts/qwen35_unified_hicache_4way/final-summary),
and CUDA graph summaries are in
[`summary`](../../artifacts/qwen35_unified_hicache_4way/summary).
