# Qwen3.5 unified-memory + HiCache post-rebase evaluation

Date: 2026-08-12 (KST)

## Conclusion

The rebased unified-memory + typed-L2 HiCache implementation passed the full
Qwen3.5-0.8B regression and performance campaign. The planned matrix completed
386/386 runs, and one additional targeted P8 padding-page reproduction also
passed. There were no failed requests or dropped HiCache tokens.

The same-backend comparison remains the canonical memory-system result. With
all variants pinned to the Triton attention, linear-attention, and Mamba
backends, U3 was approximately equal to S1 write-back at 10K for page sizes 8
and 32, but was 2.47-2.93x faster at 50K. With write-through, U3 was 1.54-1.73x
faster at 10K and 3.21-3.45x faster at 50K. The short write-back case was a
trade-off: U3 was equal at page size 1 and 9.7-10.2% slower at page sizes 8 and
32.

Removing the three explicit backend flags from static S0/S1 resolved the full
attention backend to FlashInfer while linear attention and Mamba remained on
Triton. This product-ceiling control improved static throughput by an average
of 1.5% at 3K, 11.8% at 10K, and 71.5% at 50K. It is not a layout-only
comparison with U3, because the unified page-major MHA layout currently
requires the Triton attention backend.

The campaign exposed two integration bugs that smaller unit/parity tests had
not reached. Both were fixed before the final matrix was restarted:

- deferred Full-pool frees are now flushed before an immediate Mamba-slot
  retry inside a scheduler free group;
- static paged transfers now accept the physically backed tail padding slots
  up to `size + page_size - 1`, while unified pools continue to report their
  actual shared-arena limit.

## Evaluated source and environment

- Branch under test: `feat/unified-hicache-paged-l2`
- Rebase-resolution commit: `95c881d01e0d3a55912172e7979118823aaf059f`
- Mamba deferred-free fix: `f69cbc3afb928d2f8f6a6984f53bc16bb9703c5e`
- Final server source: `743cae224c5bc28687457558a074736776350392`
- Evaluation worktree:
  `/home/sukwoo24/sglang-eval-worktrees/qwen08-post-rebase-743cae2`
- Model: `Qwen/Qwen3.5-0.8B`, BF16
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 610.62
- Python / PyTorch / CUDA: 3.12.3 / 2.11.0+cu130 / 13.0
- FlashInfer: 0.6.14

Every final artifact records the clean worktree, exact server command,
environment overrides, source SHA, resolved server arguments, hardware
snapshots, request records, metrics, and logs. All 387 manifests in the final
artifact root reference the same final server SHA and have status `completed`.

Common settings:

- `--mem-fraction-static 0.27`
- `--max-total-tokens 120000`
- `--max-running-requests 8`
- `--chunked-prefill-size 4096`
- `--context-length 65536`
- `--hicache-size 12` for S1/U3
- no explicit `--max-mamba-cache-size`
- page sizes 1, 8, and 32
- HiCache write-back and write-through
- CUDA graph ON for the main clean matrix

The existing evaluation environment uses the torch-2.11-compatible SGL kernel
0.4.5 wheel. The server version guard was skipped because the available
0.4.6.post1 wheel targets a different torch ABI; the evaluated Qwen paths use
Triton/FlashInfer and do not call the new AOT operations.

## Configurations

| ID | L1 | HiCache/L2 | Kernel backends |
|---|---|---|---|
| S0-T | static layer-first | disabled | Triton / Triton / Triton |
| S1-T | static layer-first | static HiCache | Triton / Triton / Triton |
| U0-T | unified page-first | disabled | Triton / Triton / Triton |
| U3-T | unified page-first | typed shared L2 | Triton / Triton / Triton |
| S0-D | static layer-first | disabled | resolved defaults |
| S1-D | static layer-first | static HiCache | resolved defaults |

For S0-D/S1-D, the resolved attention / linear-attention / Mamba backends were
FlashInfer / Triton / Triton. The server command omitted all three backend
arguments; these values were not substituted by the benchmark runner.

The clean workloads used two rounds, a 95% shared prefix, and reversed group
order. Every run required an actual L1 eviction. HiCache runs additionally
required backup, load-back, and host-hit evidence.

| Workload | Input / output | Groups | Rounds | Concurrency |
|---|---:|---:|---:|---:|
| Short | 3,000 / 256 | 120 | 2 | 8 |
| Middle | 10,000 / 256 | 40 | 2 | 8 |
| Long | 50,000 / 128 | 8 | 2 | 4 |

## Correctness and functional audit

| Gate | Result |
|---|---:|
| Page/policy/configuration parity | 27/27 PASS |
| CUDA graph ON/OFF parity | 8/8 PASS |
| Pinned-Triton clean performance | 162/162 PASS |
| CUDA graph performance | 72/72 PASS |
| Component profile | 36/36 PASS |
| Static/default clean performance | 81/81 PASS |
| Targeted P8 padding-page reproduction | 1/1 PASS |

- The 27 general parity runs made 1,107 comparisons; the 8 graph parity runs
  made another 328.
- Output IDs and logprob token IDs matched S0-T across every configuration and
  graph mode.
- Maximum absolute logprob difference was 0.0179786 with a tolerance of 0.02.
- No measured request failed and `hicache_dropped_tokens_total` remained zero.
- Every HiCache performance run exercised backup, load-back, and a host hit.
- The final analyzers reported zero invalid artifacts and the CUDA graph audit
  passed.

## End-to-end throughput

Values are mean total tokens/s over three independent server launches. `T`
means the pinned three-Triton configuration and `D` means backend arguments
were omitted. The maximum coefficient of variation among all 81 groups was
4.09%.

| Page | Workload | S0-T | S1-T WB | S1-T WT | U0-T | U3-T WB | U3-T WT | S0-D | S1-D WB | S1-D WT |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3K | 29,291 | 33,589 | 28,764 | 27,807 | 33,552 | 32,365 | 29,536 | 34,135 | 28,931 |
| 1 | 10K | 47,642 | 57,340 | 47,697 | 46,652 | 69,456 | 82,390 | 53,091 | 62,219 | 52,383 |
| 1 | 50K | 32,238 | 49,416 | 43,207 | 45,080 | 144,988 | 148,855 | 54,028 | 80,219 | 68,858 |
| 8 | 3K | 29,017 | 36,362 | 29,984 | 27,200 | 32,823 | 35,498 | 29,328 | 37,101 | 30,398 |
| 8 | 10K | 46,765 | 67,040 | 52,520 | 45,562 | 67,025 | 81,013 | 52,998 | 75,561 | 58,681 |
| 8 | 50K | 31,922 | 56,174 | 44,246 | 43,252 | 146,036 | 147,498 | 53,011 | 103,491 | 78,268 |
| 32 | 3K | 28,731 | 36,632 | 30,064 | 27,220 | 32,892 | 35,541 | 29,288 | 37,627 | 30,416 |
| 32 | 10K | 46,350 | 69,217 | 52,609 | 45,617 | 69,864 | 82,716 | 52,594 | 78,212 | 59,047 |
| 32 | 50K | 31,420 | 58,551 | 46,124 | 43,484 | 144,823 | 148,117 | 54,354 | 106,599 | 79,503 |

### Same-backend memory comparison

U3-T/S1-T throughput ratios:

| Page | Workload | Write-back | Write-through |
|---:|---|---:|---:|
| 1 | 3K | 0.999x | 1.125x |
| 1 | 10K | 1.211x | 1.727x |
| 1 | 50K | 2.934x | 3.445x |
| 8 | 3K | 0.903x | 1.184x |
| 8 | 10K | 1.000x | 1.543x |
| 8 | 50K | 2.600x | 3.334x |
| 32 | 3K | 0.898x | 1.182x |
| 32 | 10K | 1.009x | 1.572x |
| 32 | 50K | 2.473x | 3.211x |

The long-context advantage remains the strongest result. The P8/P32 10K
write-back result is now parity rather than a large U3 advantage, while
write-through still favors U3. Short write-back shows a roughly 10% cost for
U3 at the larger page sizes.

Compared with the previous pre-rebase four-way artifact set, every one of the
54 pinned-Triton groups improved. The range was 1.017-1.293x and the unweighted
mean improvement was 1.091x. This comparison uses the exact same workload
definitions, capacity, backends, and three-run aggregation.

### Static default-backend ceiling

Default/Triton ratios for S0 and both S1 policies ranged as follows across page
sizes:

| Workload | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| 3K | 1.006x | 1.027x | 1.015x |
| 10K | 1.085x | 1.135x | 1.118x |
| 50K | 1.594x | 1.842x | 1.715x |

The FlashInfer default makes static S1 write-back faster than U3-T at the 3K
P8/P32 and 10K P8/P32 points. U3-T remains 1.36-1.81x faster for 50K
write-back and 1.86-2.16x faster for 50K write-through. These ratios include a
backend difference and therefore answer the product-ceiling question, not the
pure L1-layout question.

## CUDA graph

Page size 8 was measured with graph ON and OFF for all four pinned-Triton
configurations. S1/U3 used write-back. Each pair was repeated three times and
run order was rotated or reversed between repetitions.

| Configuration | 3K | 10K | 50K |
|---|---:|---:|---:|
| S0-T | +167.2% | +80.0% | +7.9% |
| S1-T | +197.4% | +108.3% | +14.0% |
| U0-T | +164.3% | +81.2% | +4.2% |
| U3-T | +186.2% | +111.2% | +19.1% |

All 72 performance runs and all 8 parity runs passed the graph audit. Graph ON
runs recorded real graph batches and both captures; OFF runs recorded neither.
U3 did not lose the CUDA graph benefit after the upstream rebase.

## Transfer and control-path profile

The table shows mean GiB/s over three graph-OFF 10K profile runs. CPU allocator,
compaction, and translation times overlap GPU/model work and are diagnostic,
not additive end-to-end latency.

| Page | Policy | Variant | KV D2H | Mamba D2H | KV H2D | Mamba H2D |
|---:|---|---|---:|---:|---:|---:|
| 1 | WB | S1-T | 2.8 | 42.1 | 48.5 | 22.3 |
| 1 | WB | U3-T | 43.6 | 47.3 | 46.6 | 22.0 |
| 1 | WT | S1-T | 2.8 | 40.8 | 48.4 | 22.2 |
| 1 | WT | U3-T | 41.6 | 47.2 | 47.0 | 22.3 |
| 8 | WB | S1-T | 16.2 | 41.6 | 48.0 | 22.8 |
| 8 | WB | U3-T | 47.3 | 46.9 | 47.5 | 21.7 |
| 8 | WT | S1-T | 15.8 | 40.8 | 48.0 | 23.0 |
| 8 | WT | U3-T | 46.8 | 45.4 | 47.9 | 21.1 |
| 32 | WB | S1-T | 31.1 | 41.5 | 48.0 | 23.1 |
| 32 | WB | U3-T | 47.2 | 47.0 | 48.3 | 21.7 |
| 32 | WT | S1-T | 26.3 | 40.8 | 48.2 | 22.8 |
| 32 | WT | U3-T | 47.0 | 46.7 | 48.0 | 22.2 |

U3's KV D2H advantage was 15.6x/14.8x at page size 1, 2.93x/2.96x at page
size 8, and 1.52x/1.79x at page size 32 for WB/WT. KV H2D remained within
about 4% and Mamba transfer rates remained close. The expected unified
allocator/compaction/translation work was present in every U3 profile, and
row-aware fence registrations were recorded without dropped transfers.

## Integration fixes validated by the campaign

### Deferred Full free before Mamba retry

The first full run reached a state where a Full-tree eviction happened inside
a free group. The tree mutation was visible immediately, but the allocator's
physical free was intentionally deferred until the group ended. The scheduler
then retried a Mamba-slot allocation before that boundary and saw no shared
gap. `UnifiedMambaTokenToKVPoolAllocator.flush_deferred_frees()` now provides
the narrow synchronization point, and the Mamba eviction path calls it before
the retry. The exact 240-request failure workload and the complete final
matrix passed afterward.

### Static tail padding page

At page size 8, static KV storage physically contains the complete tail page,
so valid transfer indices can extend through `size + page_size - 1`. The
rebased transfer validator incorrectly used `pool.size` as an exclusive bound
and rejected index 120007 for a size-120000 pool. KV pools now expose an
explicit transfer-index limit: static pools include the padding page and
unified pools override it with the actual shared-arena capacity. The boundary
test accepts 120007, rejects 120008, and the exact static P8 write-back
workload plus all later P8/P32 runs passed.

## Reproduction

```bash
source /home/sukwoo24/.venv_sglang/bin/activate

bash benchmark/hicache/run_post_rebase_hicache_evaluation.sh

python benchmark/hicache/analyze_qwen35_hicache_matrix.py \
  --artifact-root artifacts/qwen35_unified_hicache_post_rebase_743cae2 \
  --output-dir \
    artifacts/qwen35_unified_hicache_post_rebase_743cae2/final-summary

python benchmark/hicache/analyze_qwen35_cuda_graph_ablation.py \
  --artifact-root artifacts/qwen35_unified_hicache_post_rebase_743cae2 \
  --expected-repetitions 3 \
  --variants post-s0 post-s1 post-u0 post-u3
```

Raw artifacts occupy approximately 100 MiB under
`artifacts/qwen35_unified_hicache_post_rebase_743cae2`. They are intentionally
gitignored; the reproducible runner, analyzers, and this result table are
tracked.
