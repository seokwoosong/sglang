Title: fix(unified-memory): recover joint capacity before token allocation

## Motivation

With `--enable-unified-memory`, FULL and SWA availability can count the same shared bytes. Checking each component separately can therefore stop eviction even though the pending combined allocation cannot fit.

The regression test constructs a real shared pool with FULL availability **4**, SWA availability **4**, but joint availability **3** for a request of **4 tokens**:

| Before | This patch |
|---|---|
| Zero individual shortfall stops eviction, leaving reclaimable cache entries untouched and the allocation unable to fit. | Evicts one eligible cached page and successfully allocates all four tokens while preserving byte accounting. |

A related case requires eviction to continue beyond the initial component quotas.

This PR fixes allocation readiness for unified FULL+SWA and FULL+SWA+Mamba/state pools, including the final decode-capacity check.

## Modifications

Keeping readiness in the allocator lets the two- and three-pool FULL+SWA layouts share eviction control flow without duplicating layout rules in the cache. Checking the original demand gives eviction a stopping condition tied to actual allocation feasibility, rather than component quotas or a fixed retry count. Preparing recoverable space first can preserve cached prefixes, while the ready fast path and byte bound avoid unnecessary recovery work.

- Add the explicit allocator-owned `TokenAllocationRecovery` capability. Readiness checks FULL, SWA and joint capacity against the original pending demand.
- Carry that demand through allocation-driven eviction and streaming sessions. Preserve normal victim selection first, then continue through remaining eligible entries until the allocation is ready or the cache is exhausted. Explicit count-based eviction retains its quota contract.
- Flush deferred frees before layout preparation. An optimistic shared-byte lower bound skips recovery when more live bytes must first be freed; passing the bound still requires the complete readiness check.
- Distinguish occupied logical slots from recoverable band shortages. Use the existing recovery ladder and movement guards to relocate a SWA float when possible, preserving cached KV payloads before attempting eviction.
- Replace the tri-pool fixed four-round retry with the shared readiness path. Keep the sufficient-capacity fast path before preparation. No new CLI flags or model-kernel changes. The existing prohibition on combining unified memory with HiCache/LMCache remains; this PR does not enable host-tiered unified memory.
- Document shared-capacity accounting, allocation-driven eviction and existing HiCache/LMCache incompatibility in the radix eviction guide.

## Accuracy Tests

Validation focuses on allocation/eviction correctness, byte accounting and preserving cached KV payloads.

| Validation | Result |
|---|---|
| Allocator/cache CPU regressions | **162 tests /117 subtests passed** |
| CUDA float relocation; page interleave | **6 CUDA cases;35 CPU tests passed** |
| CPU/CUDA allocator matrices | **1,032 valid cases passed per device**;168 impossible initial setups excluded per device |
| Separate serving/accounting integration | **574 successes +482 expected queue-full responses**, zero unexpected errors;20 snapshots passed and all four empty baselines restored |

Some concurrent workloads still show unresolved output differences; these checks do not establish general task accuracy or output equivalence.

<details>
<summary>Test commands, options and tested revisions</summary>

**CPU regressions.** Run from the PR checkout using a configured SGLang Python environment. The commands below retain the executed options, with the local virtualenv executable replaced by `python`:

```bash
PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer \
  python -B -m pytest -q -p no:cacheprovider \
  test/registered/unit/mem_cache/test_multi_ended_allocator.py \
  test/registered/unit/mem_cache/test_unified_radix_allocation_eviction.py \
  test/registered/unit/mem_cache/test_unified_tri_pool.py \
  test/registered/unit/mem_cache/test_unified_joint_allocation_eviction.py

PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer \
  python -B -m pytest -x -q -p no:cacheprovider \
  test/registered/unit/mem_cache/test_page_interleave_shard.py
```

The new test is registered in `base-a-test-cpu`. It uses production pool factories and real cache entries to cover joint shortfall, continued eviction, deferred frees, locked/exhausted caches, decode readiness, tri-pool pressure and streaming-session forwarding.

**CUDA payload checks.** A local helper ran the committed float-recovery test with its pool device changed to CUDA in memory: eager/lazy compaction × preparation/eviction/decode, checking live KV payloads and cached-prefix preservation. The equivalent helper is included here with a checkout-relative path; it requires a CUDA-enabled SGLang environment and does not edit the test file:

```bash
PYTHONPATH=python FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer python -B - <<'PY'
from pathlib import Path
import unittest

path = Path("test/registered/unit/mem_cache/test_unified_joint_allocation_eviction.py")
source = path.read_text().replace('device="cpu"', 'device="cuda"')
namespace = {"__name__": "joint_review_cuda", "__file__": str(path)}
exec(compile(source, str(path), "exec"), namespace)
suite = unittest.TestSuite([
    namespace["TestUnifiedJointAllocationEviction"](
        "test_float_recovery_preserves_prefix_when_full_band_is_short"
    )
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

**Supplementary serving output checks.** Boundary diagnostics compared an intermediate implementation with the complete patch: GPT **7,331/7,331** and Inkling **11,365/11,365** paired output sequences equal. These are not whole-patch Before/This patch comparisons. Settings: unified memory, page1/concurrency1, one output token, lazy compaction ON, decode CUDA graphs OFF and overlap ON; two reversed-order pairs with60s warm-up and120s measurement including drain. Equality counts are not task-accuracy scores. Full allocator matrices and serving checks used supplementary local harnesses that are not committed CI tests; those harnesses still need to be attached for external reproduction.

**Measured revisions.** For the serving table, **Before** is `203d7e812c6c9cde8859499daf54686091714638` and **This patch** is `5a2ce77f634a5439340f4bf926340bbc08327a80`. Serving results and full matrices were collected before updating the branch to newer upstream; they have not been rerun on the submitted branch. The CPU suite, six CUDA cases and page-interleave tests were rerun at current HEAD `276771eefca951ae6d55608415f2d795cbfabdae` on upstream `14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6` (197 CPU tests /117 subtests and six CUDA cases passed). All six code/test/documentation patches are unchanged according to `git range-diff`; diff checks and all-files pre-commit passed. The intermediate diagnostic revision is `b4f486722b9ba2b1ba935c025c351829d623bed7`. Inkling used the same separate prerequisite patch in each comparison checkout; it is outside this PR.

</details>

## Speed Tests and Profiling

Compared **Before / This patch** on one RTX 5090 (WSL2, Python 3.12), using four models covering **2-pool and 3-pool unified memory**. Each model used a headroom workload with **64 input tokens, 32 output tokens and concurrency 4**: 60s warm-up, then 120s measurement, repeated as two reversed-order pairs (16 fresh-server runs total).

The speed workload leaves memory headroom so **Before can also complete**: all eight Before runs finished with zero measured failures/retractions. This compares serving overhead on a working workload; the allocation-failure condition is covered separately by the regression tests, not timed as a successful Before run.

Changes below are relative to Before: **positive throughput is faster; positive latency is slower**.

| Model | Unified-memory configuration | Throughput change | TTFT p95 change | TPOT p50 change | E2E p95 change |
|---|---|---:|---:|---:|---:|
| GPT-OSS-20B |2-pool: FULL KV + SWA KV|−0.94%|+1.37%|+0.38%|+1.43%|
| Qwen3.5-0.8B |2-pool: FULL KV + GDN state|+0.13%|−0.28%|−0.09%|−1.05%|
| Kimi Linear tiny random |2-pool: FULL KV + KDA state|−1.95%|+2.83%|+0.72%|+2.34%|
| Inkling |3-pool: FULL KV + SWA KV + Mamba state|+0.94%|−1.60%|−0.89%|−2.20%|

These are point estimates, not proof of performance noninferiority. Inkling latency under other workloads remains unresolved, and p99 validation is incomplete.

<details>
<summary>Traffic, server options and measurement procedure</summary>

**Traffic.** A local client streamed requests to `/generate`, keeping four requests in flight and submitting a replacement as each completed. It cycled through eight synthetic 64-token prompts; this is a cache-reuse/headroom workload, not a task-accuracy benchmark. The payload for request index `i` was:

```python
payload = {
    "input_ids": [1000 + i % 8] + [300 + j % 127 for j in range(63)],
    "stream": True,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 32,
        "ignore_eos": True,
    },
}
```

**Procedure.** Start a fresh server for each run, warm it for 60s, then collect for 120s. Stop submitting at the phase deadline and drain outstanding requests; elapsed measurement time includes that drain. Run Before then This patch for one pair, and reverse the order for the other. Server processes used CPU cores 0–7 and the client used cores 8–15. Matching physical pool bytes were checked between compared runs.

Throughput is completed output tokens divided by measured elapsed time. TTFT measures time to the first received output token; TPOT is `(request end − first token time) / (output tokens − 1)`; E2E is total request time. Table values are percentage changes from the geometric mean of the two paired ratios.

**Server options.** All four table configurations used page size 1, lazy compaction ON, CUDA graphs OFF, overlap ON, 4,096 max total tokens, context length 4,096 and chunked prefill size 512. The following is the recorded Qwen launch command with the Python environment and source path made checkout-relative; use the measured revisions listed in the Accuracy Tests details to reproduce the historical comparison:

```bash
PYTHONPATH=python SGLANG_DISABLE_LAZY_COMPACTION=0 \
FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 \
taskset -c 0-7 python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --enable-unified-memory --attention-backend triton \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --mem-fraction-static 0.65 --max-total-tokens 4096 \
  --swa-full-tokens-ratio 0.8 --context-length 4096 \
  --chunked-prefill-size 512 --max-running-requests 4 --page-size 1 \
  --max-mamba-cache-size 64 --mamba-backend triton --linear-attn-backend triton \
  --mamba-full-memory-ratio 0.9 --port 31293 --random-seed 42137 \
  --incremental-streaming-output --enable-metrics
```

The original launcher cleared inherited `SGLANG_*` settings before selecting lazy compaction. Offline mode requires the pinned model snapshot to be cached. Model-specific changes to the command above were:

| Model | Model revision | Changes from the Qwen command |
|---|---|---|
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | Static memory fraction 0.75; omit the four Mamba/linear-attention options; add `--moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune` |
| `yujiepan/kimi-linear-tiny-random` | `a8e383cb67f27aad25167d44e5c1824755ec7275` | Add `--skip-tokenizer-init` |
| `thinkingmachines/Inkling` | `85b071f87d9bf5ff16a213a2d825faeed3af2cbf` | Use the cached snapshot directory as model path instead of `--revision`; Mamba memory ratio 0.1; add `--moe-runner-backend triton`; apply the same separate Inkling prerequisite to both checkouts |

GDN/KDA use the runtime's Mamba-state pool infrastructure. Kimi is a tiny random checkpoint, not a production-model accuracy evaluation. The custom client, pinned environment and Inkling prerequisite still need to be packaged for external reproduction; the launch example and traffic description alone are not a complete reproduction bundle.

**Additional coverage and limits.** All 20 predefined configurations completed across 80 runs, including headroom/pressure/mixed workloads, page16/concurrency16 and single-token allocation boundaries: 192,042 measured successes, zero measured failures/retractions. Only the four headroom conditions in the main table compare the whole patch against upstream; the other conditions tested implementation stages and are not included in that table. This is the declared matrix, not every possible configuration.

All 116 applicable statistical comparisons were inconclusive; p99 cells lacked 10,000 valid requests in every independent run. The Inkling B/C comparison included in the original 80-run matrix (page16/concurrency16; B: intermediate implementation, C: this patch) showed TPOT p50 changes of +14.02%/−1.46% across its two pairs (approximately +6.0% geometric mean); the cause remains unresolved. Later stopped experiments did not complete the intended performance/stress validation. These observations do not establish a fixed whole-patch slowdown or close the output differences noted above.

</details>

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/docs/developer_guide/contribution_guide#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/docs/developer_guide/contribution_guide#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/docs/developer_guide/contribution_guide#write-documentation).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/docs/developer_guide/contribution_guide#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/docs/developer_guide/contribution_guide#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/docs/developer_guide/contribution_guide#code-style-guidance).

Checked items are backed by pre-commit and unit/CUDA checks, the committed registered regression, and the scoped accuracy/serving results reported above. The benchmark checkbox means results are provided; it does **not** assert output equivalence, resolved Inkling latency, performance noninferiority or p99 completion. Measured revisions are recorded in the test details.

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/docs/developer_guide/contribution_guide#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
