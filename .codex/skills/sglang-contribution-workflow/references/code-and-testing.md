# SGLang Code and Validation

Use this reference while implementing or reviewing an SGLang change.

## Read the local authority

Start with the current checkout's relevant files:

- `docs/docs/developer_guide/contribution_guide.mdx`
- `test/README.md`
- `test/registered/README.md`
- `test/registered/unit/README.md`
- `docs/README.md` for documentation changes
- the closest existing source and test files

Also honor every applicable `AGENTS.md`. For kernel work, read the kernel-specific README and release guidance identified by the contribution guide; do not infer JIT, AOT, or separately released package behavior from ordinary Python paths.

## Code decisions

- Avoid duplication. If a block longer than roughly five lines recurs, consider a focused shared helper.
- SGLang is a latency-sensitive runtime. Avoid device synchronization such as unnecessary `tensor.item()` or `tensor.cpu()`, prefer vectorized work, and keep repeated checks out of model forward loops. Compute configuration-derived invariants once during initialization when possible.
- Prefer pure functions, immutable data, and explicit inputs over in-place argument mutation or passing a large state object.
- Keep functions around 100 lines or less when practical. Make orchestration read like high-level pseudocode by extracting cohesive details. Split files that grow beyond roughly 2,000 lines.
- Prefer composition or plain functions over new mixins. Use `msgspec.Struct` for new data containers rather than introducing `dataclasses.dataclass` or `attrs`.
- Prefer keyword arguments when a call has two or more arguments.
- Put core data structures near the top of a file and utilities near the bottom, following the surrounding module's established organization.
- Never deserialize untrusted or network-received data with `pickle.loads()`, `pickle.load()`, or `recv_pyobj()`. Use a safe format such as msgpack or JSON.
- For new hardware or feature support, minimize disruption to existing code, prefer a dedicated file for the new component, and keep the common existing path first in repeated branches.
- Preserve semantic contracts, not only types. Distinguish values such as immediate availability, schedulable capacity, allocation shortfall, and absolute quotas; audit all backends and wrappers when changing a shared API.

## Tests

Add a focused regression test when it protects a concrete behavior, invariant, or bookkeeping contract. For a change under `python/sglang/srt/`, first look for the mirrored module under `test/registered/unit/`.

Registered unit tests:

- Exercise component logic without launching a server or loading model weights.
- Use `CustomTestCase`, register at module scope with the appropriate `register_*_ci(...)`, and retain the standard `__main__` entry point expected by the CI runner.
- Keep registration values literal so `test/run_suite.py` can discover them by AST.
- Use the lightest runner that satisfies the test. Prefer CPU for hardware-independent behavior and a single small GPU when sufficient.
- Keep a test file below 500 seconds; split longer files. Reuse server launches across E2E methods. Keep CI jobs below 30 minutes.

Run the narrowest meaningful checks first. The command closest to registered CI for one file is direct execution:

```bash
python3 test/registered/unit/<module>/test_<name>.py
```

Use pytest for focused discovery and the suite runner when registration or CI placement matters:

```bash
pytest test/registered/unit/<module>/ -v
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

Do not claim GPU, multi-node, or backend-specific validation when the required hardware was unavailable. Record skips and environment limitations explicitly.

## Formatting, docs, accuracy, and speed

- Run pre-commit on changed files during iteration. Before push or PR handoff, follow the current contribution guide, which currently requests `pre-commit run --all-files`; rerun after hooks apply fixes.
- For documentation changes, use the current Mintlify version and commands in `docs/README.md` and the contribution guide. Check links, anchors, redirects, and visually preview layout when relevant.
- If model output can change, run an appropriate feature- or model-specific accuracy evaluation. Treat the documented GSM8K command as a sanity check, not a rigorous benchmark, and do not report its throughput as a speed result.
- If the critical path or inference speed can change, provide a reproducible benchmark. Record hardware, software versions, model, server arguments, workload, baseline and proposed commits, repetitions, summary statistic, correctness parity, and the metric directly tied to the intended optimization.
- For AOT `sglang-kernel` changes, normal PR CI can build a PR-local wheel, but selective rerun workflows do not. Validate the kernel and caller according to the kernel README and pinned-wheel compatibility rules.

## Before staging

- Review the complete diff and search for missed call sites, duplicate implementations, stale names, and accidental generated files.
- Run focused tests plus any broader suite justified by the change.
- Run `git diff --check`.
- Record exact commands, pass counts, skips, warnings that matter, and unrun hardware checks for the eventual commit or PR handoff.
