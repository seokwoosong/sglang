# Reproduce and inspect the validation

The accompanying `evidence.tar.gz` preserves original scripts, manifests, JSON results and focused review artifacts byte for byte. `EVIDENCE_BUNDLE_MANIFEST.json` binds the compressed file; the archive contains `ARCHIVE_MANIFEST.json` with every member hash. Historical FAIL/STOPPED/PARTIAL records are preserved. GitHub reviewer comments are linked in the reply draft rather than copied into the archive.

Extract into a separate artifact directory. Read the manifests before running anything: original commands contain absolute local paths, fixed source pins, an environment fingerprint and finite per-ticket resource limits. Historical runners with an expired cutoff are preserved for provenance, not intended for blind replay. The continuation runner uses ticket-specific limits.

The tested sources are B=`a3bf25dc620f31fc672aeced1465d6fe7c81d28f`, A=`15ccf48036bf5de0ec0aa886dedaa74bbdea8123`, and C2=A plus the archived `C2.patch`. For equivalent replay, create clean B/A checkouts and an isolated C2 checkout with that exact uncommitted binary diff; the harness checks these identities. Map the original worktree/artifact/Python executable paths to your environment and explicitly regenerate the manifest bindings for that new environment. Do not relabel a replay as the original measured run. GPU confirmation requires a compatible CUDA environment; resource thresholds were specific to the recorded RTX 5090 host. No model download is required for the pool fixtures.

Key entries:

- `T14.json` and `correctness_entry.py`: exact Python/Rust test commands and backend provenance.
- `T16.json`, `cpu-c2-harness/`, `summarize_c2_cost.py`: fixed cost registry, raw 36-process data and analysis. The analyzer binds absolute original result paths; map them explicitly for a relocated read-only replay.
- `T17D.json`, `cuda-c2-regression-harness/`: synchronized ready/reclaim/gate checks.
- `T19S.json`/`T20S.json`, `async-c2-same-harness/`: same-event admission and confirmation.
- `T26V2.json`/`T27.json`, `async-immutable-decision-v2-harness/`: separately reviewed distinct historical-decision observer and confirmation.
- `ENVIRONMENT.json`, `CUDA_CONTEXT.json`, `TESTED_SOURCE_FILES.json`: environment and source identity.

The registered CPU regressions also run directly from a configured final checkout with `PYTHONPATH=python python -m pytest -q -p no:cacheprovider`, followed by the file list in T14. CPU/Rust selection and exclusions are recorded in that manifest. The GPU harness is supplementary validation, not a newly registered CI suite. Local focused-review approvals are not maintainer or CI approvals.
