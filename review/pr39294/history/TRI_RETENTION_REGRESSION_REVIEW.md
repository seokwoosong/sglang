# Tri retention regression: checkpoint reopened

**CHANGES_REQUESTED for further acceptance/integration of the composed tri candidate. GPU validation remains stopped.** This decision supersedes reliance on the earlier CPU acceptance as permission to advance this candidate beyond diagnosis. The earlier report and accepted patch artifacts remain unchanged as historical evidence. The accepted two-pool result is not withdrawn by this tri counterexample.

Affected composed patch: `fb1484263d9f170ff26b87764af5bcb9987b9251c8948ea23637cb3580a144ec`; tree `0b1bd85dd84bad2fb6bc0d4b3c79429dc510304b`.

## Observed evidence

The submitted GPU log records three successful non-event eager tri controls, followed by an assertion failure in the temporal/event case at the retained-prefix check. It does not record a CUDA error, and graph replay is not reached. The failed JSON row alone does not serialize the retained set; the newly available CPU reproduction supplies the explicit cache-retention counts.

I read `tri_temporal_float_repro.py` and its completed submitted results. It builds the production CPU fixture with nonempty temporal state, allocates eight states before 96 paired tokens, inserts three legal single-state cache checkpoints, and requests four tokens. It contains no asynchronous writer or graph. Results:

| Submitted arm | Eager cached tokens | Lazy cached tokens | Allocation | Recovery ladders |
| --- | ---: | ---: | --- | --- |
| Composed tri candidate | 0 | 0 | succeeds in both | 0 in both |
| #39294 comparator | 95 | 95 | succeeds in both | 1 in both |

Both arms check predetermined outside states and byte accounting. The script checks matched prefix IDs but does not check retained KV payload in this reproduction (`slots[:0]` is passed to the payload helper). Thus it establishes a retention counterexample, not complete retained-KV payload validation. Arm identities are those of the submitted run artifacts; the reviewer did not rerun them.

This reproduces the retention failure without the earlier GPU writer-lifetime issue. It activates the acceptance caveat concerning material excess eviction in certificate-negative FLOAT layouts. The finite quota does not prevent the final indivisible 94-token leaf from overshooting a four-token request. The reported explanation—END-only sufficiency remains false until the walker has passed a recoverable FLOAT layout—is consistent with the submitted zero-ladder result and the reviewed phase policy. The supplemental ablation should establish the exact first-victim boundary and matching recovery target directly.

Evidence SHA256:

- `tri-gpu.log`: `56942d7b4a5c3c133df370c6e5a64d095c0b166f76b831d40f198181a40d840a`
- `tri-gpu-cases.json`: `da56b9e3bc6146af74dd223a28ecc8b4b02ff3bdde81d0e3ad95e167b4a19c85`
- `tri_temporal_float_repro.py`: `a9f76725fe15d56afb1a561f2497746792e147a60c7177f6dd7a10696fb08d0b`
- `tri-candidate-temporal-float-repro.json`: `fc33586052ff59267f5845c267d55eb0c0e2488ba33cf4789704a19d651d26dc`
- `tri-pr39294-temporal-float-repro.json`: `e9302ead811489ce6c0e72be86e7c6d83cecedc15bd5dfea6d57273ae55cb0b1`

## Next review requirements

Continue isolated CPU diagnosis, geometry/proof work and supplemental plan preparation. **The proposed O(1) FLOAT certificate is not yet approved for production implementation.** Its plan should specify:

1. The exact legal state after the first victim, including component reclaim counts, live bindings, reusable/pending holes and frontiers; demonstrate on a replay that explicit permitted recovery then allocates four and retains 95. Include both eager and lazy modes, exact source/import provenance and retained KV plus conv/temporal payload checks.
2. Integer page-grid equations for a sufficient certificate and its exact `make_room(side, min_bytes)` target. Reserve the far-side space needed to relocate live FLOAT pages while also funding subsequent FULL-before-SWA allocation. Do not count the same gap or hole toward both obligations. State precisely which END flushes or zero-copy boundary absorption occur before the target is computed/executed.
3. A proof under unknown hole positions: a host-count-only formula must conservatively cover every layout it accepts. Add equal-count/different-hole-position and asymmetric/page-size cases, transparent FLOAT, closed gates and pending reuse. A false result remains unknown. Demonstrate that positive certificates are realized by the specified executor without an unrelated fallback concealing a bad prediction.
4. Fresh state/gate validation at execution, no stored stale closure plan, no device hole-list read or mutation in the callback, and no per-victim recovery. Preserve the bounded preparation policy unless the new plan explicitly proposes and justifies a change.
5. CPU ablation on the frozen candidate, #39294 and the isolated proposed policy; restore the exact 95 target in this fixture without weakening the existing retention, impossible-demand, namespace, temporal, event and two-pool tests. Include legal Python/Rust cache fixtures before requesting renewed candidate acceptance.

Do not resume GPU execution or make production tri edits before the supplemental checkpoint decision. The GPU writer corrections in `TRI_GPU_HARNESS_REVIEW.md` still apply independently. No merge-ready claim, C3 resolution or remote action follows from the earlier partial GPU controls.

Only this status/review artifact was created. No source, existing approval artifact or frozen patch was changed; no experiment or GPU workload was run by the reviewer.
