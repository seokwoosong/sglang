# Unified-memory HiCache trace scenario

This trace answers five implementation questions:

1. Does the L1 shared byte arena grow Mamba from the low end and KV from the high end?
2. Are L2 chunks dynamically typed as `KV`, `MAMBA`, or `FREE`, and pinned while copied?
3. Does a row-aware fence defer overlapping compaction while allowing disjoint compaction?
4. Can an evicted prefix load back from L2 with identical tokens and logprobs?
5. Which Mamba state is the next LRU eviction candidate, and why is an L2
   Mamba chunk not ranked yet?

The replay labels every L2 Mamba chunk explicitly. `L1 LOCK · backup` and
`L1 resident · backup` mean the L2 copy is not independently host-evictable
yet. Once its L1 state is evicted, the badge changes to an `E0..E100` Host-LRU
score; `E100` is the next unlocked eviction candidate.

## Required event coverage

| Phase | Required events | Evidence |
|---|---|---|
| L1 initialization/allocation | `l1_allocator_initialized`, `l1_allocator_state` | opposite growth directions, byte frontiers, watermarks |
| L1 page pressure | remaining frontier gap is smaller than either pool's next physical row, or `l1_watermark_pressure` | red attempted next-watermark is blocked before real frontiers cross |
| L2 reservation | `l2_arena_initialized`, `l2_chunk_owner_changed`, `l2_kv_allocated`, `l2_mamba_allocated` | typed chunk ownership and occupancy |
| Mamba LRU | `mamba_lru_state` | L1 row and L2 chunk eviction rank; score 100 is the next unlocked candidate |
| Async backup | `d2h_transfer_queued`, `l2_chunks_pinned`, `l2_transfer_pin_armed`, `d2h_transfer_completed` | scheduler-side enqueue and chunk lifetime |
| Eviction | `l1_node_evicted`, `l1_node_demoted` | FULL/Mamba device residency transitions |
| Load-back | `h2d_transfer_queued`, `h2d_transfer_completed`, `loadback_metadata_committed` | L2-to-L1 restoration |
| Overlapping fence | `l1_transfer_fence_checked` with a non-empty intersection, `l1_compaction_decision=deferred_row_fence` | protected row blocks relocation |
| Disjoint fence | `l1_transfer_fence_checked` with an empty intersection, allowed/completed compaction | unrelated compaction proceeds |

## Small serve configuration

The logical L1 capacity is fixed explicitly; a large `--mem-fraction-static` is
not required. This pressure run uses `--hicache-ratio 8` so several long
prefixes can remain in L2 while the small L1 arena is exercised.

```bash
source ~/.venv_sglang/bin/activate
export PYTHONPATH=$PWD/python:$PWD/test
export SGLANG_HICACHE_TRACE_PATH=/tmp/hicache-serve.jsonl

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 31000 \
  --enable-unified-memory \
  --enable-hierarchical-cache \
  --hicache-ratio 8 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --page-size 1 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton \
  --mamba-radix-cache-strategy extra_buffer \
  --max-total-tokens 8192 \
  --max-mamba-cache-size 12 \
  --mem-fraction-static 0.15 \
  --max-running-requests 1 \
  --chunked-prefill-size 256 \
  --context-length 8192 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --enable-metrics \
  --log-level error
```

Run four distinct 6000-token prefixes. They are submitted together, but
`--max-running-requests 1` still permits only one active request; prior cache
nodes remain resident and provide pressure. The long prefill makes KV visibly
grow toward Mamba, while Mamba allocation/free turnover leaves holes for an
urgent peer compaction. The oldest L2-resident prefix is replayed to verify
loadback:

```bash
python scripts/hicache_trace/run_scenario.py \
  --base-url http://127.0.0.1:31000 \
  --prompt-tokens 6000 \
  --pressure-requests 4 \
  --queue-pressure-requests \
  --request-timeout 600 \
  --expected-max-total-tokens 8192 \
  --expected-max-mamba-cache-size 12 \
  --output /tmp/hicache-scenario.json
```

The RTX 5090 reference arena was 319.6 MiB, about 2.06x the earlier 155.1 MiB
scenario. During one active long prefill, KV growth forced an urgent Mamba
compaction: Mamba rows `[14, 13, 12]` moved into holes `[5, 6, 7]`, its
watermark retreated from 15 to 12, and KV then grew from 3328 to 3584 live
pages. The same run produced D2H eviction and a verified H2D prefix loadback.

## Deterministic row-fence scenario

The serving run covers the real transfer lifecycle. Fence timing itself is
intentionally verified with pending-event test doubles: this makes both branch
decisions deterministic on H100, RTX 5090, and CPU CI instead of relying on a
GPU-clock-dependent `_sleep` window. The CUDA fence suite remains the separate
data-integrity check.

```bash
SGLANG_HICACHE_TRACE_PATH='/tmp/hicache-fence-{pid}.jsonl' \
python -m pytest \
  test/registered/unit/mem_cache/test_multi_ended_allocator.py::TestLazyCompaction::test_row_transfer_allows_unrelated_opportunistic_compaction \
  test/registered/unit/mem_cache/test_multi_ended_allocator.py::TestLazyCompaction::test_row_transfer_defers_overlapping_opportunistic_compaction \
  -v

python -m pytest \
  test/registered/kernels/ops/kvcache/test_unified_row_transfer_fence.py -v
```

## Validate and replay

```bash
python python/sglang/srt/mem_cache/hicache_trace_replay.py \
  /tmp/hicache-serve.jsonl /tmp/hicache-fence-*.jsonl \
  --validate --require-coverage --step 40 \
  --compact-repeated-allocator-states \
  --html /tmp/hicache-trace.html
```

Use `--play` for terminal animation. Compact HTML mode folds repeated scheduler
allocator snapshots but still applies them all to state; transfer, eviction,
fence, compaction, and loadback lifecycle events are never removed. The
standalone HTML file provides a
color-coded step player and can be copied off the GPU cluster together with the
JSONL trace. L1 ownership and temporary row state are different visual layers:

- blue/purple/gray bar segments are KV/Mamba/reserved ownership;
- a free hole is shown as an unlabeled black gap; `S` is a compaction source,
  `D` its destination, and `F` a row protected by an in-flight transfer;
- each L2 KV chunk fills horizontally by its token occupancy, while a Mamba
  state chunk is always shown as 100% occupied;
- with tracing enabled, lazy-free frames materialize the freed physical rows and
  immediately blank the exact freed row widths. Reuse and compaction then
  update those same rows step by step. This trace-only materialization can synchronize
  GPU state, so use the player for functional inspection rather than performance
  measurement; the normal tracing-disabled allocator path remains asynchronous.
