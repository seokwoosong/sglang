# Final bounded GPU results acceptance

**APPROVED WITH EXPLICIT LIMITATIONS. The bounded local allocator GPU checkpoint is accepted. No additional GPU run, different environment, longer delay, production change or fixture workaround is required to complete this checkpoint.**

This final disposition confirms `TRI_GPU_ACCEPTANCE.md` against the consolidated submission. It applies to CPU-accepted tree `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`, full composed patch SHA256 `1ed1e2a56a4a637697e46fb0ed54484735e0ab6b6b60eb747e67eeca0f4b7e9b`.

## Verified submission

- `GPU_RESULTS_REVIEW_REQUEST.md`: SHA256 `fdbc17220efb668e20b4e249a1959150db58d25488e8bac62b0fc48ea50ccc9f`.
- `gpu-final-manifest.json`: SHA256 `3c1db493711240e63596872b2ea5683083f018bd7c035c99768a634b64259a9b`.
- All 13 manifest entries matched their files. The candidate index still matches the accepted tree, with no unstaged delta.

I independently recomputed the scenario aggregation from the JSON files: six non-event v3 rows, two v4 event rows, eight two-pool and two paged-tri rows from the extra controls, and five corrected gate/pending rows. This yields **23 unique intended scenarios: 20 PASS and 3 INCONCLUSIVE**, without counting earlier diagnostic versions again.

## Accepted evidence and remaining limits

| Evidence | Final disposition |
| --- | --- |
| Six non-event tri controls | Allocation/payload assertions accepted; state-before retention is exactly 95/9, while the state-after 94 result is its stated control |
| Eight two-pool controls, eager/lazy and page sizes 1/4 | Exact 95/9 retained prefix-page records, payload and accounting accepted |
| Two paged tri controls | Actual allocation, 316 retained KV tokens and seven request-owned states accepted; no paged Mamba-cache support claim |
| Three corrected independent gate controls | Allocation-inclusive zero closed-owner copies, Mamba-owned allocation/new marker and reopened movement/payload accepted |
| Eager v4 temporal writer/graph | Observed unfinished target dependency at the original wait, actual Mamba/SWA movement, 95 retention, protected marker values and same-graph replay accepted |
| Lazy v4 temporal writer/graph | Allocation, movement, 95 retention, payload and same-graph replay accepted; unfinished-writer ordering remains INCONCLUSIVE |
| Two corrected pending-reader cases | INCONCLUSIVE before pending creation; unfinished-event urgent drain, exact pending-source reuse and competing new-marker assertions were not executed |

The corrected gate rows replace the earlier incomplete gate observations. The corrected pending code contains the requested stronger checks, but code presence is not executed evidence. Preserve all three INCONCLUSIVE classifications in reports and reviewer replies; do not turn the aggregate into “all GPU tests passed.”

No CUDA error, timeout or retention/payload failure is reported in these revised runs. That statement does not include the preserved earlier candidate failure. Peak recorded allocator allocation of 658,944 bytes is within the fixture budget; PyTorch allocation counters are not total device/context memory.

## Overlap disposition

The reported warmed H2D/event measurements and the inspected source ordering are consistent with event completion before `_commit_move_batch` records pending reuse in this environment. They explain why further increases to a prequeued delay are not a justified validation strategy. They are not a universal PyTorch synchronization guarantee or a proof of lazy safety on every runtime.

The numerical diagnostic JSON records the named nondefault-stream probe; the report also describes a default-stream observation. Acceptance relies on the demonstrated local observation gap and source inspection, not on a claim of exhaustive stream/environment coverage or independent proof that this H2D call is the only serialization point.

The remaining lazy-ordering and pending-reuse gaps are explicitly accepted as limitations of this bounded checkpoint. Stop redundant overlap repeats. Do not change synchronization, inject production pending state, substitute fake GPU events or select arbitrary new hardware merely to obtain a PASS. A future change to these synchronization/H2D paths, a concrete asynchronous failure or a specifically identified deployment requirement would justify a separately scoped validation lane.

## Work remaining beyond this checkpoint

This closes the bounded local CPU/GPU evidence checkpoint, not the broader goal of addressing all reviewer comments and merging. Model-output accuracy, full serving/graph behavior, performance, real distributed/PD traffic, refreshed integration heads, C3 author/maintainer agreement and CI on the integrated branch remain outside the accepted evidence.

`REVIEW_REPLIES_FLOAT_DRAFT.md` remains a local draft. Before any separately authorized publication, describe the tested #36729 revision by its frozen SHA rather than an unverified “current head,” link the final candidate/results, and carry the overlap limitations into the evidence summary. This acceptance does not authorize posting messages, resolving remote threads, pushing or merging. No additional work is required solely to complete this bounded GPU checkpoint.

Only this acceptance artifact was created. The reviewer inspected and hashed submitted evidence; no source/harness was changed, and no experiment, GPU workload or remote action was performed.
