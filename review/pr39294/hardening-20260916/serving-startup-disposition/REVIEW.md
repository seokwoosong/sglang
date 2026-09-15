# PR39294 serving startup disposition — 2026-09-16

**ACCEPT the proposed closure of serving as BLOCKED by checkpoint/config-loader compatibility; inference and all downstream serving checks are NOT_RUN. APPROVED to proceed with the independent, already selected P2 CPU checks within their original120-second/8GiB/32MiB allowance.** No serving retry, dependency change, model substitution or budget restart is authorized.

The saved status contains only `0-B`, pinned to clean baseline HEAD `0b1065a618f09f7fe04d00ceaeeb73ef8067af0f`. Its server log fails during server-argument resolution, through `ModelConfig`/`AutoConfig.from_pretrained`, with `KeyError: 'granite_swa'` and the resulting unrecognized-model-type ValueError. This occurs before successful server readiness or workload execution. The harness records `STOPPED`, `server exited`, and `cleanup_errors=[]`; it records no client or candidate launch. The request reports no remaining listener on port31994; this reviewer did not perform a live port/process check.

Classify this as **the selected local checkpoint being incompatible with the active Transformers/config-registration path**, not evidence of an allocator regression. The failure is on baseline before the allocator workload, so it neither implicates the pending fix nor establishes successful candidate serving. The local SGLang `granite.py` does list `GraniteSWAForCausalLM` as an entry class, but that registration is distinct from HF `AutoConfig` model-type recognition.

Do not simplify the diagnosis to “Transformers is too old.” The installed source reports version5.12.1, while the local snapshot README states a requirement above5.8.0; the observed loader still rejects this model type. The generic exception's upgrade suggestion is not evidence that an upgrade alone fixes this checkpoint. No dependency installation, shim, config edit, backend change or alternative model is approved here. The earlier serving-plan review established geometry and harness checks, not successful checkpoint loading.

Serving disposition:

- Preserve the exact startup failure as one attempted baseline launch out of the six-launch maximum. It is not a successful model run, baseline/candidate pair or inference failure result.
- Close this lane without using its remaining attempts. Preserve original monotonic deadline `19905.492290348`; do not reset it, restart the batch initializer or transfer unused time to another phase.
- Candidate startup, inference, output parity, eviction pressure, cancellation/RID reuse, runtime pool readiness and joint recovery are **NOT_RUN**. Config-derived FULL7/SWA17/window128 does not become observed runtime topology. No accuracy, latency or allocator-serving conclusion follows from this startup attempt.

P2 disposition:

1. Run only `test/registered/unit/disaggregation/test_unified_memory_move_gate.py` and `test/registered/unit/server_args/test_unified_prefill_cuda_graph_gate.py`, on the accepted source: HEAD `8b15837f2e1d125081b427d1296a6f92fbecf670`, pending patch SHA256 `1e1afc2ff57e207a92f149c554edf5b4feb4805b5e75412be51ee0d79377b911`, tree `bbf48421d7d23dfd888b687a718e8714b39cea16`.
2. Keep the pinned interpreter, CPU-only environment, reviewed external guard, one owned process group,120 seconds total including startup,8GiB RSS and32MiB log cap. Preserve source pins and fresh results. No added selectors, retries or extensions. Stop and classify a new test/guard failure; unfinished checks remain NOT_RUN/PARTIAL.
3. This config-loader startup stop does not block those independent local interface/rejection checks. P2 completion will not establish real-model serving, distributed DCP/PD or GPU graph correctness. Submit its results and any local commit request separately.

The prior narrow pending-fix acceptance remains in force with its CPU/GPU and historical INCONCLUSIVE/PARTIAL limitations. No local pending commit, review-branch adoption, push, remote comment, CI acceptance or PR merge is authorized by this disposition.

Reviewed input hashes:

| Input | SHA256 |
| --- | --- |
| serving-startup-disposition/REQUEST.md | `20c010b0e9ca432102a86d09d590128cfc254ee37fc5dda19bfb5c27cd5bfb03` |
| serving-v4/serving-status.json | `bda025942446a4fadb8eea68077c4d923f29f598d85a665a88a9a41dc8723a70` |
| serving-v4/0-B-server.log | `a547ff888d5cc47b0f5478f08223a9c112db4029526787752a8275743856ce42` |
| /tmp/PR39294_PENDING_FIX_RESULTS_REVIEW.md | `2331759bf257ef8791957e732cd45975596ede920c438ca122e78ef96627defe` |

Static saved-record, local source/README and hash inspection only. No tests, server, telemetry probe or experiment was executed; no source or harness was edited. This review artifact is the sole output change.
