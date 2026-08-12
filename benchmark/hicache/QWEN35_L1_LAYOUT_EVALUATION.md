# Qwen3.5 L1 layer-first versus page-first evaluation

Date: 2026-08-12 (KST)

## Conclusion

Changing only the static L1 device-pool layout from layer-first (LF) to
page-first (PF) did not improve end-to-end throughput on Qwen3.5-0.8B. Across
54 no-eviction resident pairs, PF-static/LF averaged 0.9959x (-0.41%). Across
27 radix-pressure pairs that all performed real L1 eviction, it averaged
0.9927x (-0.73%). The per-condition three-run means ranged from -1.63% to
+1.15% in the resident matrix and from -2.26% to +1.13% under pressure.

The practical conclusion is that the current PF view is approximately
performance-neutral with a small regression tendency. Its value is the
page-contiguous, all-layer envelope required by unified memory and efficient
whole-page transfer; it should not be justified as an L1 compute-speed
optimization by itself.

A third PF-unified configuration deliberately separated allocator effects
from layout effects. It was 2.15% slower than PF-static in resident workloads,
5.35% slower at 3K pressure, and 1.26% slower at 10K pressure, but 39.13%
faster at 50K pressure. That long-context result is not a layout result. The
shared allocator retained about 247K device tokens instead of 90K and evicted
about 653K instead of 970K on average in those runs, by lending otherwise-idle
Mamba capacity to the Full/KV side.

All 288 planned runs completed. The final audit passed with no invalid
configuration, request failure, missing repetition, parity error, unexpected
HiCache activity, or pressure run without eviction.

## What was compared

The primary pair fixes the allocator and changes only one server flag:

| ID | Device allocator | L1 storage layout | Unified memory | HiCache |
|---|---|---|---:|---:|
| `l1-lf` | static | layer-first | off | off |
| `l1-pf-static` | static | page-first | off | off |
| `l1-pf-unified` | shared Full/Mamba arena | page-first | on | off |

For LF, each layer owns an ordinary contiguous K/V tensor. PF-static allocates
the same number of bytes but exposes per-layer strided views over page
envelopes of the form:

```text
page = [L0 K slots | L0 V slots | L1 K slots | L1 V slots | ...]
```

Thus `l1-pf-static / l1-lf` is the layout-only result. The
`l1-pf-unified / l1-pf-static` ratio is reported only as a secondary allocator
decomposition and is not treated as a fair LF/PF layout comparison.

## Evaluated source and environment

- Evaluation branch: `bench/l1-layer-vs-page-first`
- Pinned server source: `743cae224c5bc28687457558a074736776350392`
- Evaluation worktree:
  `/home/sukwoo24/sglang-eval-worktrees/qwen08-post-rebase-743cae2`
- Model: `Qwen/Qwen3.5-0.8B`, BF16
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 610.62
- Python / PyTorch / CUDA: 3.12.3 / 2.11.0+cu130 / 13.0
- Attention / linear-attention / Mamba backends: Triton / Triton / Triton
- `--mem-fraction-static 0.27`
- `--max-total-tokens 120000`
- `--max-running-requests 8`
- `--chunked-prefill-size 4096`
- `--context-length 65536`
- page sizes 1, 8, and 32

The analyzer checked the resolved server arguments in every artifact, rather
than trusting only the requested command line. It verified the pinned SHA,
layout flag, allocator flag, HiCache-off state, all three Triton backends,
120K nominal token capacity, radix-cache mode, request validation, and
HiCache metric deltas.

The static LF and PF logs both report a 120K-token KV cache (0.69 GB K + 0.69
GB V) and 164 Mamba slots (0.10 GB conv + 2.90 GB SSM). Only the PF view and
strides differ. The unified control reports one 4,678,778,880-byte (4.36 GB)
arena, while retaining the same `max_total_num_tokens=120000` and
`max_mamba_cache_size=164` scheduler settings.

## Matrix design

Each performance point was launched from a fresh server three times. Page,
workload, CUDA-graph, and variant order were reversed or rotated by repetition
to reduce launch-order and thermal bias.

| Workload | Input / output | Groups | Rounds | Resident concurrency | Pressure concurrency |
|---|---:|---:|---:|---:|---:|
| Short | 3,000 / 256 | 120 | 2 | 8 | 8 |
| Middle | 10,000 / 256 | 40 | 2 | 8 | 8 |
| Long | 50,000 / 128 | 8 | 2 | 2 | 4 |

The stages were:

- Parity: 3 page sizes x graph ON/OFF x 3 variants = 18 runs.
- Resident: radix cache disabled, no priming, concurrency constrained to fit
  the common 120K L1 budget; 162 runs.
- Pressure: radix enabled, primed shared prefixes, reversed group order, graph
  ON, and `--require-eviction`; 81 runs.
- Profile: graph OFF 10K pressure with memory-path instrumentation; 27 runs.

HiCache was disabled for every stage, so there were no host transfers, backup,
load-back, or L2-capacity effects in this experiment.

## Correctness and audit

| Gate | Result |
|---|---:|
| Cross-layout and graph parity | 18/18 PASS |
| Resident performance | 162/162 PASS |
| Pressure performance with real eviction | 81/81 PASS |
| Memory-path profile | 27/27 PASS |
| Final configuration/completeness audit | PASS, 0 errors |

- The parity stage made 738 comparisons. Output IDs and logprob token IDs
  matched across LF, PF-static, PF-unified, and graph ON/OFF.
- Maximum absolute logprob difference was 0.0179786 with a tolerance of 0.02.
- All measured requests passed, and the runner's dropped-request guard never
  fired.
- HiCache metric deltas were zero in every run.
- Every pressure run recorded eviction. Across all workloads, mean evicted
  tokens were 925,832 for LF, 929,170 for PF-static, and 728,921 for
  PF-unified.
- Mean throughput coefficient of variation across the 81 aggregated groups
  was 1.09%. The maximum was 6.45% in the PF-unified P1/3K graph-ON resident
  group; the maximum among the two primary static variants was 4.01%.

## Primary result: static PF versus static LF

The following values are the arithmetic mean of paired PF-static/LF
throughput ratios over all page sizes and three repetitions. Raw throughput is
the unweighted mean total tokens/s over those same nine runs.

| Stage | Graph | Workload | LF tok/s | PF-static tok/s | PF/LF |
|---|---|---|---:|---:|---:|
| Resident | OFF | 3K | 10,779 | 10,817 | 1.0036x |
| Resident | OFF | 10K | 25,848 | 25,656 | 0.9926x |
| Resident | OFF | 50K | 28,368 | 28,205 | 0.9942x |
| Resident | ON | 3K | 27,250 | 27,098 | 0.9944x |
| Resident | ON | 10K | 43,213 | 43,112 | 0.9980x |
| Resident | ON | 50K | 29,535 | 29,312 | 0.9924x |
| Pressure | ON | 3K | 28,305 | 28,240 | 0.9978x |
| Pressure | ON | 10K | 45,830 | 45,223 | 0.9868x |
| Pressure | ON | 50K | 31,232 | 31,009 | 0.9936x |

The layout-only effect by page size was also small:

| Stage | P1 | P8 | P32 |
|---|---:|---:|---:|
| Resident, graph OFF | 1.0000x | 0.9973x | 0.9931x |
| Resident, graph ON | 0.9952x | 0.9907x | 0.9990x |
| Pressure, graph ON | 0.9850x | 0.9921x | 1.0011x |

There is no monotonic page-size advantage for PF. Only the P32 pressure value
rounded above parity, and that +0.11% was not repeated across graph/stage
modes; P1 and P8 pressure were -1.50% and -0.79%.

Across individual paired repetitions, PF-static beat LF in 11/54 resident
pairs and 5/27 pressure pairs. The geometric means (0.9958x resident and
0.9924x pressure) agree with the arithmetic means, so the conclusion is not
driven by one high-throughput workload.

## Secondary result: unified allocator on the same PF layout

| Stage | Workload | PF-static tok/s | PF-unified tok/s | Unified/PF-static |
|---|---|---:|---:|---:|
| Resident, graph OFF | 3K | 10,817 | 10,381 | 0.9599x |
| Resident, graph OFF | 10K | 25,656 | 24,832 | 0.9680x |
| Resident, graph OFF | 50K | 28,205 | 27,741 | 0.9836x |
| Resident, graph ON | 3K | 27,098 | 26,431 | 0.9756x |
| Resident, graph ON | 10K | 43,112 | 42,775 | 0.9922x |
| Resident, graph ON | 50K | 29,312 | 29,064 | 0.9916x |
| Pressure, graph ON | 3K | 28,240 | 26,731 | 0.9465x |
| Pressure, graph ON | 10K | 45,223 | 44,655 | 0.9874x |
| Pressure, graph ON | 50K | 31,009 | 43,116 | 1.3913x |

All nine 50K pressure pairs favored unified memory. The per-run improvement
over PF-static ranged from 35.5% to 48.1%. At 50K, the shared arena reduced
mean eviction by 32.7% and retained 2.76x as many device tokens. Short and
middle workloads did not need that extra Full-side capacity, so allocator and
compaction overhead remained visible without a compensating retention gain.

This is why a direct `PF-unified / LF` number is not a valid answer to the
layout question: it combines PF addressing with dynamic capacity sharing.

## Profile evidence

The graph-OFF 10K profile is diagnostic and was kept separate from the clean
performance matrix. Mean values across three pages and three repetitions:

| Variant | tok/s | Forward s | Eviction s | Allocator CPU s | Compaction CPU s | Translation CPU s |
|---|---:|---:|---:|---:|---:|---:|
| LF static | 43,097 | 16.697 | 0.074 | 0.000 | 0.000 | 0.0013 |
| PF static | 42,489 | 16.958 | 0.072 | 0.000 | 0.000 | 0.0013 |
| PF unified | 42,856 | 16.250 | 0.552 | 4.483 | 1.655 | 0.2641 |

LF and PF-static show the same allocator/translation shape; the page-major
view adds no allocator or compaction machinery. Unified memory records the
expected allocation, virtual-to-physical translation, and compaction work.
CPU categories can overlap and allocator time includes compaction invoked
inside allocation, so these columns are diagnostic and must not be added to
end-to-end latency.

## Reproduction

```bash
source /home/sukwoo24/.venv_sglang/bin/activate

bash benchmark/hicache/run_qwen35_l1_layout_evaluation.sh

python benchmark/hicache/analyze_qwen35_l1_layout.py \
  --artifact-root artifacts/qwen35_l1_layout_743cae2 \
  --expected-repetitions 3
```

The raw artifacts and generated CSV/audit files are under
`artifacts/qwen35_l1_layout_743cae2`. They are intentionally gitignored. The
reproducible runner, analyzer, and this result report are tracked.
