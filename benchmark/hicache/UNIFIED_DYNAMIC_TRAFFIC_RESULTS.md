# Unified HiCache dynamic-traffic discovery

## Scope

This experiment searches for dynamic mixed traffic where unified Triton
outperforms an adaptively tuned static Triton partition. The server uses
Qwen3.5-4B, a 12 GiB write-back HiCache, page size 8, decode-only CUDA graphs,
and `--mem-fraction-static 0.55`. Neither `--max-total-tokens` nor
`--max-mamba-cache-size` is specified, so both variants use automatic sizing.

The automatic-sizing preflight assigned both variants the same total GPU cache
budget of 8,448,638,976 bytes. Static partition tuning starts at 0.5, tests 0.4
and 0.6, then moves by 0.1 only in the improving direction.

## Reproducible positive case

The `ordered-long-late` profile keeps this 23-request order in each wave:

1. sixteen 3K-token short requests;
2. one 50K-token long request;
3. one 10K-token middle request;
4. one 50K-token long request;
5. four 10K-token middle requests.

Four waves alternate client concurrency 4, 16, 4, and 16. Prefix reuse is
80%. Static tuning selected `--mamba-full-memory-ratio 0.9`.

| Seed | Static Triton oracle | Unified Triton | Unified delta |
|---:|---:|---:|---:|
| 7701 | 10,126.5 tok/s | 10,789.1 tok/s | +6.54% |
| 7702 | 9,993.9 tok/s | 11,205.9 tok/s | +12.13% |
| 7703 | 10,120.1 tok/s | 11,208.4 tok/s | +10.75% |
| Mean | 10,080.2 tok/s | 11,067.8 tok/s | +9.81% |

All 92 requests completed in every reported run. The result supports the more
specific conclusion that unified sharing helps when short requests occupy the
initial admission window, long requests arrive near the back of the queue,
and a later high-concurrency wave reuses their prefixes. Merely mixing request
lengths is not sufficient.

## Counterexamples and controls

| Profile | Result | Interpretation |
|---|---:|---|
| `ordered-long-early` | -6.25% | Admitting long requests first favors static. |
| shuffled `concurrency-spike` | median -0.08% | The result changes with queue order. |
| `demand-shift` | -4.1% | Unified eviction overhead exceeds reuse benefit. |
| `residency-burst` | -19.1% | Repeated compaction and eviction dominate. |
| `reuse-shift` screen | -1.48% vs static 0.5 | Backed-up data receives almost no load-back reuse. |
| `heavy-tail` | mean +0.27% | +0.98%, -0.34%, and +0.15% are effectively a tie. |

The `heavy-tail` static search also selected ratio 0.9. Its three paired seeds
do not establish a meaningful unified advantage.

## Artifacts

Local raw artifacts are stored under
`artifacts/unified_dynamic_discovery`. The artifact directory is intentionally
git-ignored; manifests and result JSON files remain on the experiment host.
The workload implementation is `bench_phase_mixed.py`, and
`run_unified_ablation.py` launches the pinned static and unified variants.
