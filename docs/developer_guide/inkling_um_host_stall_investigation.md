# Inkling Unified Memory host-stall investigation

## Status

This is an investigation note, not a production fix. It records a one-H100 reproduction, matched static controls, torch-profiler evidence, and scheduler phase attribution for the time-dependent Unified Memory (UM) throughput collapse.

No production source is changed by this branch. Device dependence remains unproven: the issue has been reproduced only on H100, but the evidence points to UM host-side scheduling rather than a GPU kernel slowdown. One H100 at TP1 is sufficient; multiple GPUs are not required.

## Reproduction workload

Each cycle runs three workloads in order:

| Shape | ISL | OSL | Concurrency | Requests |
|---|---:|---:|---:|---:|
| `cache_comp` | 2,000 | 256 | 32 | 100 |
| `kv_heavy` | 50,000 | 128 | 16 | 32 |
| `long_decode` | 32 | 1,024 | 16 | 32 |

The UM server uses `--enable-unified-memory` and greedy exact-token checks. A same-pod static server is an optional control. The co-located absolute throughput comparison is diagnostic; the decisive signal is each arm's own early-to-late trend.

The experiment root is `/group-volume/seokwoo/um-h100-fix-20261124`:

- UM driver: `eval/scripts/um4_inkling_soak_profile.py`
- Static driver: `eval/scripts/um4_inkling_soak_static.py`
- Shared launch/benchmark helper: `eval/scripts/um4_perf_mono.py`
- Instrumentation: `eval/perf_debug_site/sitecustomize.py`

Use a fresh attempt tag for every run so existing artifacts are not overwritten:

```bash
ROOT=/group-volume/seokwoo/um-h100-fix-20261124
PY=/opt/sglang/bin/python3   # use /opt/venv/bin/python3 when absent
ATTEMPT=inkphase2
cd /root

export PYTHONPATH=$ROOT/eval/perf_debug_site:$ROOT/sglang/python:$ROOT/eval/scripts
export UM4_PERF_DEBUG_DIR=$ROOT/eval/logs/perf_debug/$ATTEMPT
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export UM3_SOAK_SECONDS=3600
export UM3_MAX_CYCLES=110
export UM3_PROFILE_CYCLES=
export UM3_ATTEMPT_TAG=$ATTEMPT

"$PY" "$ROOT/eval/scripts/um4_inkling_soak_profile.py" \
  > "$ROOT/eval/logs/${ATTEMPT}_driver.log" 2>&1
```

Set `UM3_PROFILE_CYCLES=95,105` and a local `UM4_ADMIN_KEY` when matched torch traces are needed. Reaching 110 cycles before 3,600 seconds makes the harness duration gate report `FAIL`; this is not a correctness failure.

## Throughput evidence

### `inkprof1`

| Arm / shape | Early median tok/s | Late median tok/s |
|---|---:|---:|
| UM `cache_comp` | 7,569.16 | 2,015.26 |
| UM `kv_heavy` | 1,733.30 | 2,659.76 |
| UM `long_decode` | 5,685.14 | 960.24 |
| Static `long_decode` | 7,209.35 | 7,170.09 |

UM and static each passed all 330 measured rows with exact-token equality, zero request errors, and zero retractions.

### `inkphase1`

| Shape | Early median tok/s | Late median tok/s |
|---|---:|---:|
| `cache_comp` | 7,607.70 | 2,339.91 |
| `kv_heavy` | 2,861.12 | 2,837.50 |
| `long_decode` | 5,582.29 | 1,063.74 |

All 330 rows again passed with exact-token equality. The regression is shape-selective rather than a uniform server slowdown.

## Torch-profiler attribution

Matched cycle-95 and cycle-105 `cache_comp` traces showed nearly unchanged GPU kernel time but a large scheduler host-time increase:

| Measurement | Cycle 95 | Cycle 105 |
|---|---:|---:|
| Total `_grouped_gemm_kernel` GPU time | 11.965 ms | 12.048 ms |
| Total `scheduler.get_next_batch_to_run` | 25.153 ms | 594.375 ms |
| Longest single `get_next_batch_to_run` | — | 576.782 ms |

The traces do not support a general MoE or attention-kernel regression. They isolate the slowdown to host-side scheduling around UM cache and batch preparation.

## Targeted instrumentation

### `inkprune1`: PageUsage hypothesis

Instrumentation covered `PageUsage.prune`, `Band._read_ahead`, and `Scheduler.get_next_batch_to_run`:

- 812 `GET_NEXT` events
- Maximum `GET_NEXT`: 600.377 ms
- `PRUNE` events at or above 5 ms: 0
- `READ_AHEAD` events at or above 5 ms: 0

A single large `prune` or `read_ahead` call therefore does not explain the 600 ms stall. This does not exclude accumulation of smaller calls.

### `inkphase1`: scheduler phase attribution

Of 807 `GET_NEXT` events, 144 took at least 500 ms and none took one second.

The largest prefill-side interval was:

```text
GET_NEXT          609.726 ms
└─ GET_NEW_PREFILL 608.162 ms
```

The largest decode-side interval was:

```text
GET_NEXT          565.193 ms
└─ UPDATE_RUNNING 564.759 ms
   └─ PREPARE_DECODE 563.074 ms
```

`UPDATE_RUNNING` was the largest child for 92 of the 144 long intervals and `GET_NEW_PREFILL` for 52. No events above the 5 ms threshold were recorded for `HICACHE_EVENTS`, `ENSURE_CAPACITY`, `RETRACT_DECODE`, `FILTER_BATCH`, `PRUNE`, or `READ_AHEAD`.

This is still outer-function attribution. A patch must not be selected from these spans alone.

## Aggregating parent and child intervals

Instrumentation is emitted at event completion. Reconstruct each interval as `[ts - elapsed_ms/1000, ts]` before computing overlap:

```python
import glob
import json

paths = glob.glob(
    "/group-volume/seokwoo/um-h100-fix-20261124/"
    "eval/logs/perf_debug/phase1/*.jsonl"
)
events = [
    json.loads(line)
    for path in paths
    for line in open(path)
    if line.strip()
]
parents = sorted(
    (event for event in events if event["event"] == "GET_NEXT"),
    key=lambda event: event["elapsed_ms"],
    reverse=True,
)
children = [event for event in events if event["event"] != "GET_NEXT"]
for parent in parents[:20]:
    parent_end = parent["ts"]
    parent_start = parent_end - parent["elapsed_ms"] / 1000
    overlapping = []
    for child in children:
        child_end = child["ts"]
        child_start = child_end - child["elapsed_ms"] / 1000
        if child_start <= parent_end and child_end >= parent_start:
            overlapping.append(child)
    overlapping.sort(key=lambda event: event["elapsed_ms"], reverse=True)
    print("PARENT", parent["elapsed_ms"], parent["ts"])
    for child in overlapping[:5]:
        print("  CHILD", child["event"], child["elapsed_ms"], child["ts"])
```

## Next steps before a production patch

1. Split `get_new_batch_prefill` / `_get_new_batch_prefill_raw` into waiting-queue traversal, prefix match/request initialization, admission and budget accounting, allocation, and batch preparation.
2. Split `ScheduleBatch.prepare_for_decode` using its actual direct children, separating allocation, ID translation, and tensor construction/copies.
3. Record both inclusive and direct-child-subtracted exclusive time. Do not sum nested intervals.
4. Require a deterministic regression or repeatable RED harness before changing production behavior.
5. Require exact-token correctness, late-throughput recovery, and removal of the targeted host stall in a matched capture.

## Artifacts

Under `/group-volume/seokwoo/um-h100-fix-20261124`:

- `eval/PHASE_D_SOAK_inkprof1_DONE`
- `eval/PHASE_D_SOAK_inkstatic1_DONE`
- `eval/PHASE_D_SOAK_inkprune1_DONE`
- `eval/PHASE_D_SOAK_inkphase1_DONE`
- `eval/logs/perf_debug/prune1/1197.jsonl`
- `eval/logs/perf_debug/phase1/1103.jsonl`
- `eval/logs/phase_d_soak/profiles/inkprof1/`
- `eval/results/phase_d_soak/traffic_inkling06b_um_inkprof1.jsonl`
- `eval/results/phase_d_soak/traffic_inkling06b_um_inkprune1.jsonl`
- `eval/results/phase_d_soak/traffic_inkling06b_um_inkphase1.jsonl`
