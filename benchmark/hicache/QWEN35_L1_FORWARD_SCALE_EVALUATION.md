# Qwen3.5 4B/9B L1 forward-path scaling evaluation

Date: 2026-08-13 (KST)

## Conclusion

Changing only L1 storage from layer-first (LF) to page-first (PF) is
forward-performance neutral at 4B and 9B on the RTX 5090. The paired
PF-Triton/LF-Triton mean was 1.0000x for 4B prefill, 0.9974x for 4B decode,
1.0014x for 9B prefill, and 0.9984x for 9B decode. These are all within 0.27%
of parity, with no page-size trend favoring either layout.

Backend selection matters much more than layout for long prefill. LF-auto,
which resolved to FlashInfer/Triton/Triton, was 1.3792x faster than
LF-Triton for 4B prefill and 1.1863x faster for 9B prefill. Decode improved by
about 1% for both models. PF currently requires Triton, so LF-auto is a useful
production-performance control but is not a layout-only comparison.

All 156 logical runs and all 60 LF/PF pairs completed. The final configuration
and completeness audit passed with zero errors. This campaign intentionally
did not run accuracy or logprob parity tests; it isolates measured model
forward work after the earlier layout campaign established parity.

## Compared configurations

| ID | L1 layout | Requested backends | Resolved backends | Unified memory | HiCache |
|---|---|---|---|---:|---:|
| `l1-lf` | layer-first | Triton/Triton/Triton | Triton/Triton/Triton | off | off |
| `l1-pf-static` | page-first | Triton/Triton/Triton | Triton/Triton/Triton | off | off |
| `l1-lf-auto` | layer-first | no backend flags | FlashInfer/Triton/Triton | off | off |

`l1-pf-static / l1-lf` changes only the L1 layout and is the primary result.
`l1-lf-auto / l1-lf` holds LF fixed and measures the backend-selection effect.
A direct PF-Triton/LF-auto ratio would mix layout and backend effects and is
therefore not used to answer the LF/PF question.

## Experiment design

- Models: Qwen3.5-4B and Qwen3.5-9B, BF16.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 610.62.
- Pinned server source: `743cae224c5bc28687457558a074736776350392`.
- Page sizes: 1, 32, and 64.
- Prefill workload: 50,000 input / 1 output, 8 groups x 2 rounds,
  concurrency 2.
- Decode workload: 10,000 input / 512 output, 16 groups x 2 rounds,
  concurrency 8.
- LF-Triton/PF-Triton: five paired repetitions per point.
- LF-auto: three repetitions per point, paired with LF-Triton repetitions
  1-3.
- Fresh server for each run; model, page, workload, and variant orders were
  reversed or rotated between repetitions.
- `--max-total-tokens 120000`, `--chunked-prefill-size 4096`,
  `--context-length 65536`, and `--max-running-requests 8`.
- CUDA graph, radix cache, unified memory, and HiCache were disabled for every
  run. No L1 eviction occurred.

The primary rate is phase-specific GPU forward tokens/s, not request-wall
throughput. Prefill divides successful prompt tokens by the server's measured
`extend`/`mixed`/`split_prefill` GPU-forward seconds. Decode divides generated
decode positions (excluding the first sampled token) by measured `decode`
GPU-forward seconds. The campaign covered 3,744 successful requests,
87,360,000 prompt tokens, and 1,279,200 output tokens.

## Aggregate results

Ratios below are arithmetic means of paired run ratios over all three page
sizes. LF/PF uses 15 pairs per row; auto/Triton uses nine pairs per row.

| Model | Phase | LF-Triton tok/s | PF-Triton tok/s | PF/LF | LF-auto tok/s | Auto/Triton |
|---|---|---:|---:|---:|---:|---:|
| 4B | Prefill | 11,750 | 11,749 | 1.0000x | 16,357 | 1.3792x |
| 4B | Decode | 658.2 | 656.5 | 0.9974x | 670.9 | 1.0106x |
| 9B | Prefill | 8,232 | 8,243 | 1.0014x | 9,831 | 1.1863x |
| 9B | Decode | 475.9 | 475.1 | 0.9984x | 482.6 | 1.0100x |

LF/PF absolute rates use all 15 layout runs. LF-auto absolute rates use its
nine runs; paired ratios compare the matching repetition in every case.

## Page-size results

| Model | Page | Phase | PF-Triton/LF-Triton | LF-auto/LF-Triton |
|---|---:|---|---:|---:|
| 4B | 1 | Prefill | 1.0027x | 1.3673x |
| 4B | 32 | Prefill | 0.9992x | 1.3845x |
| 4B | 64 | Prefill | 0.9980x | 1.3857x |
| 4B | 1 | Decode | 0.9952x | 1.0026x |
| 4B | 32 | Decode | 0.9950x | 1.0206x |
| 4B | 64 | Decode | 1.0019x | 1.0085x |
| 9B | 1 | Prefill | 1.0024x | 1.1760x |
| 9B | 32 | Prefill | 1.0012x | 1.1903x |
| 9B | 64 | Prefill | 1.0006x | 1.1924x |
| 9B | 1 | Decode | 0.9982x | 1.0089x |
| 9B | 32 | Decode | 0.9978x | 1.0105x |
| 9B | 64 | Decode | 0.9991x | 1.0106x |

The prefill layout ratios are especially tight: individual paired runs range
from 0.9967x to 1.0036x at 4B and from 0.9999x to 1.0042x at 9B. Decode is
noisier, particularly at 4B, but its aggregate remains close to parity and
does not show a monotonic page-size effect.

## Audit and corrective reruns

| Gate | Result |
|---|---:|
| Logical runs | 156/156 |
| LF/PF layout pairs | 60/60 |
| Successful measured requests | 3,744/3,744 |
| Final audit | PASS, 0 errors |

The analyzer validates the resolved layout, all backend choices, the pinned
server SHA, 120K capacity, graph/radix/unified/HiCache-off settings, request
success, zero eviction, and zero HiCache activity for each selected artifact.
It selects the latest completed artifact for each logical identity.

Two environmental anomalies were removed transparently before the final
summary. The first global 4B/P1/prefill cell included a cold Triton disk-cache
compile, so its LF, PF, and auto variants were rerun consecutively after the
cache was warm. One 9B/P1/prefill LF/PF pair straddled a computer power-off,
so both variants were rerun consecutively after restart. Earlier raw artifacts
are retained, while the tracked summary points to the replacement completed
runs. No result was removed based on which variant won.

## Reproduction and tracked results

```bash
source /home/sukwoo24/.venv_sglang/bin/activate

python benchmark/hicache/run_qwen35_l1_forward_scale.py

python benchmark/hicache/analyze_qwen35_l1_forward_taste.py \
  --artifact-root artifacts/qwen35_l1_forward_scale_743cae2 \
  --run-prefix l1-forward-scale \
  --models 4b 9b \
  --triton-repetitions 5 \
  --auto-repetitions 3
```

The full server/client logs and raw JSON artifacts remain under
`artifacts/qwen35_l1_forward_scale_743cae2` and are intentionally gitignored.
The final audit and CSV summaries are tracked under
`benchmark/hicache/results/qwen35_l1_forward_scale_743cae2`.
