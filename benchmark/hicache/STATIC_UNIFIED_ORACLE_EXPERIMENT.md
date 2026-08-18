# Qwen3.5 static-oracle versus unified HiCache experiment

## Objective

Compare the best statically partitioned hybrid L1 against the unified L1
implementation at equal GPU and host-cache budgets.

- Source: `92eb0737857f4fef0ba46e19bed5cb9bc45816f9`
- Detached server worktree:
  `/home/sukwoo24/sglang-eval-worktrees/static-unified-oracle-92eb073`
- Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`
- L2 HiCache: 12 GB, write-back, page size 8
- Prefix reuse: 20%, 50%, 80%
- Repetitions for reported results: 3

## Variants

| ID | L1 | Attention / linear attention / Mamba |
|---|---|---|
| S-A | static | resolved defaults / resolved defaults / resolved defaults |
| S-T | static | Triton / Triton / Triton |
| U-T | unified | Triton / Triton / Triton |

S-A and S-T independently tune `--mamba-full-memory-ratio`. U-T uses 0.5 only
as a total-budget sizing input; it does not impose a runtime split on the
unified arena.

The fixed model budgets are:

| Model | `--mem-fraction-static` | `--max-total-tokens` |
|---|---:|---:|
| 0.8B | 0.27 | 120,000 |
| 4B | 0.55 | 120,000 |

## Homogeneous workloads

| Workload | Input / output | Groups x rounds | Client concurrency | Max running requests |
|---|---:|---:|---:|---:|
| Short | 3,000 / 256 | 120 x 2 | 32 | 32 |
| Middle | 10,000 / 256 | 40 x 2 | 8 | 8 |
| Long | 50,000 / 128 | 8 x 2 | 2 | 8 |

Every valid performance run must have successful requests, L1 eviction, L2
backup, L2 load-back, host-hit evidence, and zero dropped HiCache tokens.

## Static ratio selection

Static ratios are selected independently for each model, backend, and
homogeneous workload at 80% prefix reuse:

1. Screen 0.5, 0.4, and 0.6 once.
2. Continue by 0.1 only in the better direction.
3. Stop after the first non-improving point or the 0.1/0.9 boundary.
4. Repeat the best two screened ratios twice more.
5. Select the larger three-run median total-token throughput.

The selected 80% ratios are held fixed for the 20% and 50% controls. This
makes those two cases sensitivity controls, rather than separate static
oracles tuned on the evaluation condition.

## Length-interleaved mixed workload

The old mixed workload executed all short requests and then all long requests.
That measures a phase transition, not simultaneous competition between
different sequence lengths.  It is retained only as a historical artifact and
is not used in the new comparison.

The replacement uses two waves on one server.  Every wave contains all three
length classes in a deterministic shuffled order:

| Class | Input / output | Requests per wave | Input tokens per wave |
|---|---:|---:|---:|
| Short | 3,000 / 256 | 32 | 96,000 |
| Middle | 10,000 / 256 | 10 | 100,000 |
| Long | 50,000 / 128 | 2 | 100,000 |

Wave 1 creates the prefixes. Wave 2 reuses the same prefixes with fresh
suffixes. A barrier between waves guarantees that replay never races the first
use of its own prefix. Inside each wave, 16 client workers consume a shuffled
queue, so short, middle, and long requests overlap instead of arriving in
length order. The three classes contribute approximately equal input-token
pressure; request-count weighting would otherwise overrepresent short traffic.

Repetitions use seeds 7301, 7302, and 7303. Every variant receives the same
trace for a seed, and each result records a trace fingerprint and metrics split
by length class.

Static-auto and static-triton are each tuned again on this exact mixed workload
at 80% reuse. Screening starts at 0.5, 0.4, and 0.6 and continues by 0.1 only
in the improving direction. The best two candidates receive three repetitions
and the better median is selected. This mixed-specific ratio is fixed for the
20% and 50% controls. Unified-triton retains dynamic sharing.

## Run count

| Stage | Minimum | Expected | Maximum |
|---|---:|---:|---:|
| Correctness preflight | 6 | 6 | 6 |
| Static adaptive tuning at 80% | 84 | 108 | 120 |
| U-T homogeneous final at 80% | 18 | 18 | 18 |
| S-A/S-T/U-T controls at 20% and 50% | 108 | 108 | 108 |
| Interleaved-mixed static tuning at 80% | 28 | 36 | 40 |
| Interleaved-mixed finals and U-T 80% runs | 42 | 42 | 42 |
| Total | **286** | **318** | **334** |

One run means one server launch, one workload execution, artifact capture, and
server shutdown. Running only the redesigned `mixed` stage requires 70-82
runs (78 expected) and approximately 2-3.5 hours on the local RTX 5090. The
full matrix is expected to take about 10-14 hours including compilation,
thermal variation, and retries.

## Commands

Inspect the generated plan:

```bash
python benchmark/hicache/run_static_unified_oracle_matrix.py plan
```

Run stages independently and resumably:

```bash
python benchmark/hicache/run_static_unified_oracle_matrix.py preflight
python benchmark/hicache/run_static_unified_oracle_matrix.py tune
python benchmark/hicache/run_static_unified_oracle_matrix.py final
python benchmark/hicache/run_static_unified_oracle_matrix.py mixed
```

Or run/resume all remaining stages:

```bash
python benchmark/hicache/run_static_unified_oracle_matrix.py all
```

Artifacts are written under `artifacts/static_unified_oracle_92eb073`. Static
selections are checkpointed after every completed search in
`static_ratio_selection.json`.

Generate the final audit and summary:

```bash
python benchmark/hicache/analyze_static_unified_oracle_matrix.py \
  --artifact-root artifacts/static_unified_oracle_92eb073
```

The analyzer produces raw run CSV, homogeneous and mixed summaries,
U-T-versus-S-T and U-T-versus-static-ceiling comparisons, validation errors,
and a Markdown report under the artifact root's `summary` directory.

## Preparation status

- Model snapshots: present
- Pinned detached worktree: clean and at the expected SHA
- RTX 5090 and port 30000: available at preflight time
- Python environment: PyTorch 2.11.0+cu130, CUDA toolkit 13.3
- FlashInfer auto backend: JIT verified with the virtualenv Ninja executable
- Correctness preflight: 6/6 passed
- Historical phase-mixed smoke: 0.8B U-T, 80% reuse, repetition 1 passed
- Length-interleaved implementation: ready; new smoke run still required
