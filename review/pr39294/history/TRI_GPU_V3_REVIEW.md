# GPU harness v3 execution approval

**APPROVED for the previously bounded local GPU validation lane.** This decision binds execution approval to `gpu_tri_validation_v3.py`, SHA256 `2643a3011021182cd58cb57b7a577d3939e7e84119297ec2b27239d945228bcc`, importing CPU-accepted tree `fe601ec6fc21ca0efc9c914385f9e3ffc77ccacf`.

This is verification of the exact corrections required by `TRI_FLOAT_CANDIDATE_ACCEPTANCE.md`, not a new production design review. I compared the complete v2-to-v3 diff. Changes are limited to:

- Restoring state translation inside `collect`, so the captured reader resolves virtual state IDs at replay.
- Removing translation from `update_forward`, so delayed state writes use its supplied original physical destinations.
- Renaming result/provenance/environment outputs from v2 to v3, preserving older artifacts.

The reviewed ownership locks, selective writer markers, original destination snapshots, actual state-movement assertion and same-graph replay remain unchanged. The reviewer independently ran `ruff check --select F821 gpu_tri_validation_v3.py`; it passed. The full composed production patch still hashes to `1ed1e2a56a4a637697e46fb0ed54484735e0ab6b6b60eb747e67eeca0f4b7e9b`.

The earlier conditional harness corrections are satisfied. **Execution of this v3 subset is authorized now; no further preliminary approval is required.** Keep the existing one-process RTX 5090 limit, at least 4 GiB free, below-1-GiB fixture allocation budget, external 180-second timeout, source/command provenance and stop-on-error/material-regression rules. Unobserved outstanding dependencies remain INCONCLUSIVE; eight subset cases cannot establish the entire GPU plan, model accuracy or merge readiness.

The rejected v2 hash is not authorized for execution. Remaining pending-reader, gate, two-pool and paged GPU cases retain their previously approved bounded harness scope. Production edits, serving, model downloads and remote actions remain outside this approval.

Only this verification artifact was created. No harness/source was changed, and no GPU workload was executed by the reviewer.
