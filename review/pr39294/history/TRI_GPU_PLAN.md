# Proposed GPU validation after tri CPU acceptance

Request scope approval from the same session, contingent on acceptance of the separately hashed composed CPU candidate. No GPU workload has run during this review continuation. Read-only inventory found one local RTX 5090 (32,607 MiB total, approximately 6,378 MiB occupied at inspection); check free memory again before each launch and leave unrelated processes untouched.

## First GPU lane: real allocator data movement, no model server

Use the existing local Python environment and an isolated harness in this artifact directory, with PYTHONPATH pointing to the accepted composed candidate and CUDA_VISIBLE_DEVICES=0. Require at least 4 GiB free, bound the fixture budget below 1 GiB, one GPU process at a time, and a 180-second process timeout. On CUDA error collect the failure and stop that lane; do not retry blindly or change kernels.

1. Build production unified two-pool and tri-pool factories on CUDA with the same tiny page/layout fixtures used in CPU regression. Exercise eager and lazy allocation, FULL/SWA joint demand4 and90, and tri state-before/after-token layouts. Use actual binding/translation/compaction kernels (no CPU stand-ins).
2. Stamp distinct FULL K/V, SWA K/V, conv and nonempty temporal state payloads. Predetermine the retained cache key/virtual-ID set and request-owned states/KV before eviction. After helper plus actual allocation, assert mappings, payload, accounting and fixed retention. Do not infer output equivalence or model accuracy from allocator markers.
3. Use a separate CUDA forward stream to write updated live payloads at their original kernel locations. Record a real event, install it through existing allocator latest-forward/inflight hooks, and trigger permitted recovery on the scheduler stream. Validate after synchronization that moved live values contain the updated writes. Cover genuine pending-reuse creation through nonurgent compaction, then explicit urgent recovery. Preserve the current event/copy ownership; no custom synchronization inserted into production.
4. Capture a small graph that translates retained virtual IDs and gathers their K/V and state values into stable output buffers; replay after permitted compaction changes physical placement. Compare with the predetermined marker values. Capture only the read/translation workload, not host-driven eviction. This is focused pointer/translation/event evidence, not a full-model CUDA-graph certification.
5. Close FLOAT/END movement gates independently while immediate holes/gaps remain usable. Assert no forbidden movement at the real move callback boundary and retained payloads; reopen and verify permitted recovery. This simulates the gate contract, not a live network transfer.
6. Record exact GPU/torch/CUDA/Triton versions, source hashes, commands, counts and any NOT_RUN/FAIL. Run one candidate sweep; repeat only to investigate a new failure. CPU work counters remain separate from GPU timing.

## Timing and serving follow-up

Do not label fixture time as model throughput improvement. If the GPU lane passes, propose a small observer-free pressure benchmark on the exact compared source pair, with allocator initialization outside timing and identical live layouts. A baseline that fails allocation is a correctness comparison, not a valid latency speedup ratio.

Full serving/output behavior remains a separate concrete lane: first reuse the original PR's documented model/server/workload choices, inspect local model availability and current server ownership, then submit the exact selected command/model/workload/resource budget for approval. Do not automatically repeat the historical 80-run matrix or reuse its results for this candidate. No new model download, unrelated server shutdown, external GPU host use or public performance claim is included here.

## Integration and remote actions

The candidate is built on unmerged #36729; local CPU/GPU success cannot settle reviewer C3's request to coordinate with its author. Prepare exact proposed reviewer replies and an integration message describing accepted two-pool regressions plus the remaining tri delta, while acknowledging the other PR's allocator dispatch. Keep these local until publication is authorized. Recheck both heads before any authorized rebase/push. No automatic merge or claim of maintainer approval.
