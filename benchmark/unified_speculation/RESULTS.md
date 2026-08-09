# Qwen3.5 Unified-Memory Speculative-Decoding Validation

## Scope

- Date: 2026-08-09
- GPU: NVIDIA GeForce RTX 5090
- Model: `Qwen/Qwen3.5-0.8B`, bfloat16, TP=1
- Goal: compare static and unified memory with no speculation, NGRAM, and the
  model's built-in MTP layer.
- The experiments did not set `--max-mamba-cache-size`. Unified memory derived
  its Mamba capacity automatically (163 slots in this run).

| ID | Memory | Speculative decoding |
|---|---|---|
| S0 | Static | Disabled |
| U0 | `--enable-unified-memory` | Disabled |
| S-N | Static | NGRAM, 4 draft tokens |
| U-N | Unified | NGRAM, 4 draft tokens |
| S-M | Static | Built-in MTP, 3 steps, top-k 1, 4 draft tokens |
| U-M | Unified | Built-in MTP, 3 steps, top-k 1, 4 draft tokens |

Common server settings included page size 1, 60,000 total tokens, 16 maximum
running requests, 4,096-token chunked prefill, 65,536 context length, Triton
attention/linear-attention/Mamba backends, Mamba `extra_buffer`, CUDA graph, and
disabled overlap scheduling.

## Implementation and debugging

NGRAM required only removing the old unified-memory rejection and validating
the existing target verification path.

Built-in MTP required the draft KV layer to share the target's unified physical
page envelope. For Qwen3.5-0.8B, each physical full-attention page now contains
six target layers plus one draft layer. Target compaction moves all seven layer
views atomically, while both workers resolve the same live virtual-to-physical
mapping.

GPU integration exposed two CUDA-graph-specific correctness faults:

1. TARGET_VERIFY used a stale batch sequence-length sum and could leave the tail
   of its KV read indices untranslated. It now uses the verify iteration's own
   length mirror.
2. Multi-step draft-decode CUDA graph still produced incorrect output when the
   built-in draft layer shared the unified envelope. The final correctness-safe
   implementation disables only this draft-decode graph. Target verification,
   prefill, and draft-extend graphs remain enabled. External EAGLE drafts with a
   separate KV pool are unaffected.

Before the fallback, U-M matched S-M in only 5/21 exact-output cases. After the
sequence-length fix it matched 14/21, and after the targeted graph fallback it
matched 20/21.

## Correctness

The validation set contained four short/repetitive prompts, one 8K retrieval
prompt, and 16 deterministic GSM8K few-shot prompts.

| Static/unified pair | Exact token sequences | GSM8K primary-answer parity |
|---|---:|---:|
| S0 vs U0 | 21/21 | 16/16 |
| S-N vs U-N | 21/21 | 16/16 |
| S-M vs U-M | 20/21 | 16/16 |

The one S-M/U-M token mismatch occurred only after both outputs had emitted the
same primary GSM8K answer (`#### 6`); the model then continued into an unrelated
few-shot-style tail. Running that case alone produced exact token equality. The
8K retrieval output was exactly equal in every static/unified pair.

Absolute GSM8K primary-answer scores were 2/16 for S0/U0 and 3/16 for all four
speculative configurations. This small model is not used here as an accuracy
benchmark; the relevant result is static/unified parity.

## Performance

Each cell is the mean of three runs. Every run used 24 fixed random prompts,
128 output tokens, maximum concurrency 8, and a cache flush plus one warm-up
request. Standard deviations are included for input throughput and TPOT.

| Config | Input | Input tok/s | Output tok/s | TTFT ms | TPOT ms | Accept length |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 3K | 36,441 ± 171 | 1,555 | 164.2 | 3.87 ± 0.00 | — |
| U0 | 3K | 34,752 ± 168 | 1,483 | 168.7 | 4.08 ± 0.02 | — |
| S-N | 3K | 23,470 ± 1,381 | 1,001 | 87.3 | 6.20 ± 0.35 | 1.56 |
| U-N | 3K | 23,106 ± 234 | 986 | 87.8 | 6.26 ± 0.01 | 1.56 |
| S-M | 3K | 42,422 ± 89 | 1,810 | 166.6 | 3.08 ± 0.01 | 3.97 |
| U-M | 3K | 39,643 ± 584 | 1,691 | 171.9 | 3.32 ± 0.03 | 3.98 |
| S0 | 8K | 47,311 ± 239 | 757 | 493.2 | 5.57 ± 0.01 | — |
| U0 | 8K | 50,606 ± 65 | 810 | 449.3 | 6.38 ± 0.01 | — |
| S-N | 8K | 31,010 ± 131 | 496 | 397.1 | 11.27 ± 0.05 | 1.54 |
| U-N | 8K | 31,214 ± 138 | 499 | 242.2 | 12.19 ± 0.06 | 1.54 |
| S-M | 8K | 45,411 ± 154 | 727 | 539.7 | 5.59 ± 0.02 | 3.99 |
| U-M | 8K | 46,050 ± 140 | 737 | 508.5 | 6.77 ± 0.01 | 3.99 |

Key comparisons:

- At 3K, MTP increased output throughput by 16.4% for static memory and 14.1%
  for unified memory relative to their no-spec baselines.
- At 8K, MTP output throughput was 4.0% below S0 and 9.0% below U0. The longer
  prefill dominates this workload despite an approximately 3.99 acceptance
  length.
- U-M was 6.6% below S-M at 3K. At 8K its aggregate output throughput was 1.4%
  higher, but its TPOT was 21.1% worse. The disabled shared-envelope draft
  decode graph is the clearest remaining optimization target.
- Random prompts had only about 1.55 NGRAM acceptance length, so NGRAM was slower
  in both memory modes. The close static/unified results show that this is a
  workload/algorithm effect rather than a unified-memory regression.

## Verification

- Related unit tests: 143 passed, 17 subtests passed.
- Pre-commit on all modified Python and test files: passed.
- All six servers launched successfully and completed 21 correctness requests
  plus six performance runs (3K and 8K, three repetitions).

Raw results are stored at:

- `/home/sukwoo24/sglang-eval-results/unified-spec-qwen35/final/accuracy_final_raw.json`
- `/home/sukwoo24/sglang-eval-results/unified-spec-qwen35/final/perf_final_raw.jsonl`
- `/home/sukwoo24/sglang-eval-results/unified-spec-qwen35/final/perf_u_m_corrected.jsonl`

The reproducible runner is `run_qwen35_unified_spec_eval.py` in this directory.
