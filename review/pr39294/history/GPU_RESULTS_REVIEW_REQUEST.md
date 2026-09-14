# GPU results and explicit overlap limitations

Candidate remains CPU-accepted tree `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`; no production or registered-test change after acceptance. RTX5090, driver610.62, torch2.13.0+cu130, CUDA13.0; exact Triton/import/device records in each provenance JSON. Hashes: `gpu-final-manifest.json`.

All runs were sequential, used at least4GiB free and external180-second timeouts, with no server/model/download/unrelated-process change. The largest recorded allocator workload peak was658944 bytes (2MiB reserved), below the1GiB limit. No CUDA error, timeout or payload/retention failure occurred in these revised runs.

## Final interpretation, without double-counting diagnostic repeats

| Scenario | Evidence | Status |
| --- | --- | --- |
| Tri conv controls: eager/lazy, states before demand4/90 and states after demand4 | `tri-gpu-v3-cases.json` six rows | PASS allocation and payload; state-before exact95/9, state-after94 is a control, not a95 assertion |
| Temporal eager writer + graph | `tri-gpu-v4-cases.json`, lazy=false | PASS: actual original wait_stream sees unfinished event at~1.7ms; one Mamba page and six SWA pages move;95 retained; fixed-address writer markers and same captured translating graph pass |
| Temporal lazy writer + graph | `tri-gpu-v4-cases.json`, lazy=true | Allocation/payload/95retention/state+FLOATmovement/samegraph pass; ordering INCONCLUSIVE because event completes before relevant wait |
| Two-pool eager/lazy, page sizes1/4, demands4/90 pages | first8 relevant rows in `gpu-extra-cases.json` | PASS exact95/9 prefix **records**, each one page, plus data/accounting |
| Paged tri, eager/lazy page_size4, request-owned temporal states | `gpu-extra-cases.json` |2PASS,316 retained KV tokens and7states; no paged Mamba checkpoint caching claim |
| FULL/Mamba/FLOAT gates | `gpu-gate-pending-v3-cases.json` |3PASS with corrected entire closed interval observation; own state allocation+marker for Mamba; zero forbidden copies; reopened movement/payload checks |
| Real pending reader reuse at page sizes1/4 | `gpu-gate-pending-v3-cases.json` |2INCONCLUSIVE: no pending set remains while event is unfinished, so exact pending-source reuse/newmarker/drain assertions are not reached |

Do not count v3's eight rows as all PASS. In the23 intended unique scenarios this amounts to20 conclusive scenario passes and3 inconclusive overlap scenarios; the lazy row nevertheless provides narrower data/graph evidence. Prefer the table's exact claims over a blanket count.

## Harness corrections and chronology

- Original `gpu_tri_validation.py` exposed the already resolved CPU retention regression; its delayed writer also had independent ownership/address defects. Those old logs are not validation of this candidate.
- Unexecuted v2 had a mechanical undefined-name error. v3 corrects both statements and passes `ruff --select F821`; fixed writer never translates virtual IDs, captured reader does. v3 kept original100Mcycle delay and settle observer, both event cases honestly INCONCLUSIVE.
- v4 diagnoses early completion with only the two event cases, actual `Stream.wait_stream`/`wait_event` pre-call observers and2Bcycle bounded delay. It makes eager ordering conclusive, lazy remains inconclusive. Original production waits are always called; no extra harness wait is inserted before recovery.
- Extra v1's allocation controls remain valid, but its gate interval and pending reuse checks were incomplete per GPU_JOINT_EXTRA_REVIEW.md. The corrected v3 gate/pending script extends the copy observer through allocation, allocates actual state under closed Mamba gate and tracks marker901, then observes reopened copies. Its pending branch records exact reader source membership, event state at urgent drain, actual source reuse and new marker997 before final sync. That branch is not claimed as executed: real event completion prevents obtaining a nonempty pending state.

## Bounded overlap diagnosis

Warmup plus a longer delay still makes pending tests INCONCLUSIVE. A separate tiny real-CUDA diagnostic on both default and separate scheduler streams observed that warmed `torch.tensor([1],device='cuda')` blocks~685–688ms until the other stream's prior2Bcycle event completes. Warmed GPU indexing returns~0.01–0.17ms with that event still pending. No CUDA_LAUNCH_BLOCKING or CUDA_DEVICE_MAX_CONNECTIONS override was set. Recorded observations are in `gpu-overlap-diagnosis.json`; this is an environment observation, not a universal PyTorch claim.

The current production `_commit_move_batch` constructs CUDA src/dst tensors from host lists before testing `latest_event.query()`. The observation explains why this queued-before-flush reader arrangement does not create a pending set here; increasing the delay again would not establish the desired coverage. I have stopped repeats rather than change production synchronization or fake a pending entry/event. CPU pending metadata tests remain valid CPU control-flow evidence only.

Please decide whether to accept this bounded allocator GPU checkpoint with explicit lazy-ordering/pending limitations, or specify the concrete additional environment/fixture needed before acceptance. No full-GPU-plan completion, model-output equivalence, serving speedup or merge readiness is claimed. No observer-free benchmark or serving run is proposed without a separate concrete plan.

## Reproduction

Each command ran from `/home/sukwoo24/sglang-eval-worktrees/review-tri-float-candidate`:

```bash
timeout --signal=TERM 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/gpu_tri_validation_v3.py
timeout --signal=TERM 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/gpu_tri_validation_v4.py
timeout --signal=TERM 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/gpu_joint_extra_validation.py
timeout --signal=TERM 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer /home/sukwoo24/.venv_sglang_upstream_full/bin/python -B /home/sukwoo24/sglang-eval-results/pr39294-review-20260914/gpu_gate_pending_validation_v3.py
```
