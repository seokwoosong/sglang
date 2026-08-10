# Unified-memory HiCache replay trace (2026-08-10)

This directory contains a validated unified-memory + HiCache typed-L2 serving
trace and deterministic row-fence traces collected on an NVIDIA GeForce RTX
5090. The source revision before adding these artifacts was
`fcdd52f4e1835bdb4996ac8c87c83d50c3fe55c2`.

## Results

- Four distinct 6,000-token prefixes were served with write-through HiCache.
- L1 eviction freed 6,016 KV tokens.
- Replaying prefix 0 loaded 5,954 tokens from L2 and reused 5,952 cached tokens.
- Output token IDs matched; output logprobs matched within `1e-2`.
- Replay validation reported zero errors and all 14 required coverage checks.
- The CUDA row-fence integrity suite passed all three D2H/H2D overlap cases.

The trace configuration intentionally uses a small logical L1 and disabled
CUDA graphs so allocator, compaction, transfer, and load-back transitions are
easy to inspect. It is a functional visualization workload, not a performance
benchmark.

## Files

- `server-trace.jsonl`: real serving timeline.
- `fence-overlap-disjoint.jsonl`: deterministic overlapping and disjoint
  row-fence decisions.
- `fence-cuda-integrity.jsonl`: CUDA D2H/H2D fence integrity trace.
- `scenario.json`: request outputs and eviction/load-back counters.
- `server-replay.html.gz`: standalone replay player, compressed to satisfy the
  repository's added-file size limit.

Open the replay with:

```bash
gzip -dk server-replay.html.gz
python -m http.server 8123 --bind 0.0.0.0
```

Then visit `http://localhost:8123/server-replay.html`.

## Serving configuration

```bash
source ~/.venv_sglang/bin/activate
export PYTHONPATH="$PWD/python:$PWD/test"
export SGLANG_HICACHE_TRACE_PATH="$PWD/server-trace.jsonl"

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
  --max-total-tokens 12288 \
  --max-mamba-cache-size 12 \
  --mem-fraction-static 0.15 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --context-length 8192 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --enable-metrics \
  --log-level error
```

Run the workload sequentially so each finished prefix can enter L2:

```bash
python scripts/hicache_trace/run_scenario.py \
  --base-url http://127.0.0.1:31000 \
  --prompt-tokens 6000 \
  --pressure-requests 4 \
  --request-timeout 600 \
  --expected-max-total-tokens 12288 \
  --expected-max-mamba-cache-size 12 \
  --output scenario.json
```

`--chunked-prefill-size 8192` is intentional. Intermediate chunked-prefill
nodes do not trigger write-through backup, so using 256 here produces L1
pressure but no long-prefix L2 load-back.

## Rebuild the replay

```bash
python python/sglang/srt/mem_cache/hicache_trace_replay.py \
  server-trace.jsonl \
  fence-overlap-disjoint.jsonl \
  fence-cuda-integrity.jsonl \
  --validate --require-coverage --step 40 \
  --compact-repeated-allocator-states \
  --html server-replay.html
```

Expected validation summary:

```text
14/14 coverage checks passed
Validation errors: 0
Replay steps: 401
```

## SHA-256

```text
5c277861c362316f0b0ec1e28a7f4bf8f1d91131f0478fbf18a104ce2e7580d4  fence-cuda-integrity.jsonl
b650fe6ac12bc43578b549cee6512623144389f2088d4c50251496004571c8bc  fence-overlap-disjoint.jsonl
d3abf33844dd0a8f14bf7b45069c32c5a587c2c9955c8779ca8b284307728a16  scenario.json
1bb9569db516e4e5f7a2860d9cf02af05d24c86a3a80c9b7a2a9dc015b08efa6  server-replay.html.gz
72c917d0840ded01cf06adedfb358fdf01ab30cc8db79d3c0e71aba25103605d  server-trace.jsonl
```
