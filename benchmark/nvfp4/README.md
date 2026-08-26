# NVFP4 and FP4 Tensor Core Evaluation

This directory records the scope and completed reproducible measurements for
evaluating NVFP4 on consumer Blackwell. It deliberately separates
three concepts that are easy to conflate:

1. NVFP4 W4A4 linear GEMM;
2. an NVFP4 KV cache;
3. attention whose QK and PV matrix multiplications both execute with NVFP4
   Tensor Core MMA.

The measurements are based on SGLang commit `3d00762885` and the local environment
described below. Results are experimental and are not SGLang release claims.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-26 |
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| Compute capability | SM120 |
| NVIDIA driver | 610.62 |
| PyTorch | 2.13.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| FlashInfer | 0.6.15.post1 |
| SGLang package | 0.5.17 |
| SGLang source | `3d00762885` on `exp/nvfp4-core-evaluation` |

## Current support status

### NVFP4 W4A4 linear GEMM

SGLang can serve a serialized NVFP4 checkpoint with
`--quantization modelopt_fp4`. `--fp4-gemm-backend` selects the runner for
linear GEMMs; it does not select an attention backend or change attention
math. On SM120, the current `auto` policy resolves to the FlashInfer CUTLASS
runner. This is the serving path that uses FP4 Tensor Cores for ordinary
linear projections such as QKV, attention output, and MLP projections.

The current `nvfp4_online` mode is not a substitute for a serialized dense
NVFP4 checkpoint: it converts supported MoE expert weights at load time, while
dense linear layers retain their source precision.

Relevant implementation:

- [`fp4_utils.py`](../../python/sglang/srt/layers/quantization/fp4_utils.py)
- [`nvfp4_online.py`](../../python/sglang/srt/layers/quantization/nvfp4_online.py)
- [`modelopt_quant.py`](../../python/sglang/srt/layers/quantization/modelopt_quant.py)

### NVFP4 KV cache

SGLang supports `--kv-cache-dtype nvfp4` on SM100 and SM120. This reduces KV
storage and memory traffic, but the cache dtype alone does not specify the MMA
operand types used by attention.

For the hybrid configuration investigated here:

```bash
--kv-cache-dtype nvfp4 \
--prefill-attention-backend flashinfer \
--decode-attention-backend trtllm_mha
```

- FlashInfer prefill exposes a dequantized workspace to the attention kernel.
- TRTLLM MHA decode on SM120 dispatches to XQA. The query remains BF16, while
  packed NVFP4 K/V and their scales are passed to XQA.
- In the inspected FlashInfer 0.6.15.post1 XQA source, packed E2M1 K/V values
  are converted to BF16 and the attention MMA instruction is BF16-by-BF16.

Therefore, this path is a native packed FP4 *cache* path, not an attention path
whose QK and PV matrix multiplications both use FP4 MMA.

Relevant implementation:

- [`fp4_kv_cache_quant_method.py`](../../python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py)
- [`trtllm_mha_backend.py`](../../python/sglang/srt/layers/attention/trtllm_mha_backend.py)
- [FlashInfer XQA API](https://docs.flashinfer.ai/api/attention.html#xqa)

### FA4 on SM120

`fa4` means FlashAttention-4, not FP4 attention. The SGLang-owned SM120 FA4
implementation builds both QK and PV from `MmaF16BF16Op`; selecting
`--attention-backend fa4` does not by itself enable FP4 attention math.

Relevant implementation:

- [`fa4_sm120/flash_fwd.py`](../../python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py)
- [`fa4_sm120/flash_fwd_decode.py`](../../python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd_decode.py)

### Dense SM120 NVFP4 attention

FlashInfer separately provides `nvfp4_attention_sm120_quantize_qkv` and
`nvfp4_attention_sm120_fwd`. Its kernel traits select the SM120 block-scaled
NVFP4 MMA atom for both QK and PV. This is the path in this evaluation that
actually exercises FP4 Tensor Core attention.

It is not currently wired into the SGLang serving runtime. The installed API
has the following restrictions:

- dense self-attention with equal Q/K/V shapes;
- MHA rather than GQA or MQA;
- equal query and KV sequence lengths;
- head dimension 64 or 128;
- no paged KV cache, radix cache, or single-token decode path.

These restrictions make it a kernel microbenchmark and a potential dense
prefill ceiling, not a fourth SGLang serving configuration.

Upstream references:

- [FlashInfer SM120 NVFP4 attention API](https://docs.flashinfer.ai/api/attention.html#sm120-nvfp4-attention)
- [FlashInfer Python implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py)
- [FlashInfer NVFP4 QK/PV kernel traits](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/traits.h)

### Partial FP4 attention-related paths

SGLang also contains an experimental DeepSeek V4 FP4 indexer. It accelerates
the sparse-attention index selection component; it is not a general full
attention backend in which both the main QK and PV products use NVFP4.

## Capability matrix

| Path | Available in SGLang serving | Packed FP4 data | QK/PV use FP4 MMA | Scope |
| --- | --- | --- | --- | --- |
| NVFP4 W4A4 linear | Yes | Activations and weights | Not attention | Dense/MoE linear layers supported by the checkpoint and backend |
| NVFP4 KV + TRTLLM MHA on SM120 | Yes | K/V cache | No; SM120 XQA uses BF16 query/MMA | Paged decode |
| FA4 on SM120 | Yes | FP4 KV where supported | No; SM120 implementation uses FP16/BF16 MMA | Prefill and decode |
| DeepSeek V4 FP4 indexer | Experimental | Indexer Q/K | Partial indexer only | DeepSeek V4 sparse selection |
| FlashInfer dense NVFP4 attention | No runtime integration | Q/K/V; P is computed by FP32 softmax and then requantized for PV | Yes, QK and PV use FP4 MMA; softmax uses FP32 | Dense MHA self-attention microbenchmark |

## Recorded SM120 attention microbenchmark

The benchmark is
[`bench_nvfp4_attention_sm120.py`](../kernels/attention/bench_nvfp4_attention_sm120.py).
It compares:

1. PyTorch BF16 fused Flash SDPA;
2. FlashInfer NVFP4 attention with Q/K/V already quantized;
3. FlashInfer Q/K/V preprocessing and quantization plus NVFP4 attention.

The FP4 end-to-end path reuses preallocated output and LSE buffers but includes
the Q/K/V preprocessing and packed-tensor allocations performed by the
FlashInfer quantization API.

Reproduction command:

```bash
python3 benchmark/kernels/attention/bench_nvfp4_attention_sm120.py \
  --seq-lens 512,1024,2048 \
  --head-dims 64,128 \
  --num-heads 8 \
  --seeds 0,1,2 \
  --causal \
  --warmup-ms 10 \
  --measure-ms 30 \
  --output-json benchmark/nvfp4/results/2026-08-26-rtx5090/operator/attention_full_matrix.json
```

Configuration: `B=1`, `H=8`, BF16 input/output, causal attention, cold L2,
and seeds 0-2. Values below are the median across seeds; JIT compilation
completed before the timed samples.

| D | S | BF16 Flash SDPA (ms) | Prequantized NVFP4 attention (ms) | Kernel speedup | Quantize + NVFP4 attention (ms) | Operator speedup | Relative RMSE | Cosine similarity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 0.0143 | 0.0103 | 1.39x | 0.0869 | 0.16x | 0.5337 | 0.947845 |
| 64 | 1024 | 0.0226 | 0.0165 | 1.37x | 0.0921 | 0.25x | 0.5331 | 0.947971 |
| 64 | 2048 | 0.0635 | 0.0276 | 2.30x | 0.1650 | 0.38x | 0.5329 | 0.946054 |
| 128 | 512 | 0.0185 | 0.0144 | 1.28x | 0.0921 | 0.20x | 0.5325 | 0.947623 |
| 128 | 1024 | 0.0349 | 0.0206 | 1.70x | 0.1014 | 0.34x | 0.5326 | 0.951377 |
| 128 | 2048 | 0.0881 | 0.0348 | 2.53x | 0.1865 | 0.47x | 0.5286 | 0.956237 |

Interpretation:

- Prequantized FP4 attention becomes more favorable as sequence length grows.
- With FlashInfer 0.6.15.post1, quantizing BF16 Q/K/V for every call costs more
  than the attention kernel saves for these shapes.
- Accuracy numbers use synthetic Gaussian Q/K/V. They are operator-level
  numerical comparisons, not model-task accuracy results.
- These are repeated operator measurements, not SGLang serving measurements.
  The serving results in the next section use the actual runtime separately.

## Completed linear GEMM decomposition

[`bench_nvfp4_gemm_sm120.py`](../kernels/quantization/bench_nvfp4_gemm_sm120.py)
measured 120 cases: four Llama-3.1-8B projection shapes, five token counts,
Gaussian and synthetic-outlier inputs, and three seeds. Each production-layout
measurement separates:

1. BF16 GEMM;
2. BF16 activation quantization into the swizzled NVFP4 GEMM layout;
3. GEMM with activation and weight already packed as FP4;
4. activation quantization followed immediately by FP4 GEMM.

The following table shows the Gaussian `M=2048` medians across seeds.

| Projection | Shape `(M,N,K)` | BF16 GEMM (ms) | Activation quant (ms) | Prequantized FP4 GEMM (ms) | Quant + FP4 GEMM (ms) | Prequantized speedup | Operator speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QKV | `(2048,6144,4096)` | 0.5418 | 0.0184 | 0.1025 | 0.1168 | 5.29x | 4.64x |
| Attention output | `(2048,4096,4096)` | 0.3942 | 0.0184 | 0.0737 | 0.0900 | 5.35x | 4.39x |
| Gate/up | `(2048,28672,4096)` | 2.3497 | 0.0184 | 0.4354 | 0.4517 | 5.40x | 5.20x |
| Down | `(2048,4096,14336)` | 1.3650 | 0.0532 | 0.2418 | 0.2938 | 5.65x | 4.65x |

Quantization-inclusive Gaussian speedups remained above one for every tested
shape. They ranged from 1.69x to 3.46x at `M<=128`, and from 3.69x to 5.20x at
`M>=512`. Median output relative RMSE was approximately 0.134 and cosine
similarity approximately 0.991; all outputs had zero NaN/Inf values.

A targeted `gate_up`, `M=2048` rerun also controlled for the BF16 API path.
Across three seeds, PyTorch `torch.mm` BF16 and FlashInfer BF16 with the
cuBLASLt backend measured 2.3428 ms and 2.3435 ms respectively, with an exact
BF16 output match. Quantization plus the CUTLASS FP4 GEMM measured 0.4506 ms,
or 5.20x versus either BF16 path. FlashInfer's CUTLASS BF16 backend is not
available on SM120, so this validates the practical BF16 baseline rather than
claiming a same-CUTLASS datatype-only ratio.

### What “dequantization time” means here

The production FP4 GEMM does **not** launch a standalone dequantization kernel
after the matrix multiplication. Scaling and conversion to BF16 are fused into
the GEMM epilogue, so that cost is already included in the prequantized FP4
GEMM timing. A separate linear-layout KV-style quantize/dequantize round trip
was measured only as a diagnostic:

| Input shape | Linear quant (ms) | Linear dequant (ms) |
| --- | ---: | ---: |
| `(2048,4096)` | 0.0186-0.0205 | 0.0348 |
| `(2048,14336)` | 0.0588 | 0.1144 |

Those diagnostic values use a different scale layout and are not additive to
the GEMM pipeline. Likewise, independently cold-L2-timed components should not
be expected to sum exactly to the back-to-back quantize-plus-GEMM timing.

## Completed SGLang end-to-end evaluation

Three actual SGLang servers were compared on the same GPU:

| Config | Weights / linear path | KV cache | Attention path |
| --- | --- | --- | --- |
| `bf16` | BF16 | BF16 | FlashInfer prefill + TRTLLM MHA decode |
| `w4a4_bf16_kv` | Serialized NVFP4 W4A4 | BF16 | Same backends |
| `w4a4_nvfp4_kv` | Serialized NVFP4 W4A4 | NVFP4 | Same backends; SM120 XQA consumes packed KV |

Workloads were `(input, output) = (2048,128), (1024,1024), (128,2048)` at
concurrency 1, 8, and 32. Each condition used three repeats, fixed random
inputs, temperature zero, exact output lengths, and a cache flush after warmup.
All 81 results completed with exact requested input and output token counts.

### Output throughput

Values are median output tokens/s across three repeats. Parentheses show the
speedup over BF16.

| Workload | Concurrency | BF16 | W4A4 + BF16 KV | W4A4 + NVFP4 KV |
| --- | ---: | ---: | ---: | ---: |
| Prefill-heavy | 1 | 81.4 | 231.6 (2.85x) | 216.4 (2.66x) |
| Prefill-heavy | 8 | 330.4 | 1023.1 (3.10x) | 1030.4 (3.12x) |
| Prefill-heavy | 32 | 502.8 | 1650.1 (3.28x) | 1746.5 (3.47x) |
| Balanced | 1 | 108.5 | 248.0 (2.29x) | 227.5 (2.10x) |
| Balanced | 8 | 693.0 | 1572.7 (2.27x) | 1547.8 (2.23x) |
| Balanced | 32 | 2020.9 | 3804.5 (1.88x) | 4642.2 (2.30x) |
| Decode-heavy | 1 | 108.5 | 250.0 (2.30x) | 228.7 (2.11x) |
| Decode-heavy | 8 | 736.5 | 1727.0 (2.35x) | 1617.3 (2.20x) |
| Decode-heavy | 32 | 2503.8 | 4659.3 (1.86x) | 5333.2 (2.13x) |

W4A4 weights produced the dominant serving gain. Holding W4A4 weights fixed,
NVFP4 KV was 6-9% slower at concurrency 1, roughly neutral at concurrency 8,
and 5.8-22.0% faster at concurrency 32. The KV result is therefore a
memory-capacity/high-concurrency tradeoff rather than a universal latency win.

### GPU memory

`nvidia-smi` peak memory includes the host's approximately 2.3-2.5 GiB
display/driver baseline.

| Config | Median peak (MiB) | Maximum peak (MiB) | Change from BF16 median |
| --- | ---: | ---: | ---: |
| BF16 | 30,112 | 30,280 | baseline |
| W4A4 + BF16 KV | 20,990 | 21,030 | -30.3% |
| W4A4 + NVFP4 KV | 14,570 | 14,584 | -51.6% |

Server logs independently reported a 9.00-GiB BF16 KV allocation versus a
2.68-GiB NVFP4 KV allocation for the same 73,728-token capacity.

## Completed accuracy checks

Each server evaluated the same first 200 GSM8K test questions using five-shot
prompts, temperature zero, and a 512-token limit. Passkey retrieval used exact
token-ID prompts with 30 cases each at 4K and 8K context.

| Config | GSM8K | Change vs BF16 | Passkey 4K | Passkey 8K |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 154/200 (77.0%) | baseline | 30/30 | 30/30 |
| W4A4 + BF16 KV | 137/200 (68.5%) | -8.5 pp | 30/30 | 30/30 |
| W4A4 + NVFP4 KV | 129/200 (64.5%) | -12.5 pp | 30/30 | 30/30 |

The GSM8K drop is material in this 200-example slice, although paired
per-question significance was not retained and should not be inferred from the
aggregate alone. Perfect passkey results show no failure in this lightweight
4K/8K retrieval probe; with only 30 cases per length, the Wilson 95% lower
bound for a reported 100% rate is 88.6%, so this is not proof of general
long-context equivalence.

## Reproduction and raw data

- Operator raw data: `results/2026-08-26-rtx5090/operator/`
- Serving raw data and logs: `results/2026-08-26-rtx5090/serving/`
- Validated aggregate: `results/2026-08-26-rtx5090/summary.json`
- Aggregator: [`analyze_results.py`](analyze_results.py)
- Serving runner: [`run_serving_experiments.py`](run_serving_experiments.py)
- Passkey evaluator: [`eval_passkey_retrieval.py`](eval_passkey_retrieval.py)
- Presentation: [`presentation.html`](presentation.html)
- Targeted BF16-backend rerun: `results/2026-08-27-rtx5090/rerun/`

```bash
python3 benchmark/nvfp4/analyze_results.py \
  --results-dir benchmark/nvfp4/results/2026-08-26-rtx5090
```

## Main conclusion

The answer is now more specific than the initial kernel-only result:

- NVFP4 W4A4 linear GEMMs deliver real SGLang serving gains on this RTX 5090.
- NVFP4 KV cache cuts memory substantially and helps high-concurrency
  throughput, but adds overhead at low concurrency.
- The separate dense FP4 attention kernel still has no SGLang paged-serving
  integration, and its per-call Q/K/V quantization makes the tested operator
  slower than BF16 end to end.
- The observed GSM8K degradation means performance and memory wins should not
  be accepted without a model-specific accuracy gate.

## What full FP4-attention serving would require

Connecting the dense FlashInfer kernel to a restricted prefill-only path is
conceptually possible, with a BF16/XQA fallback for decode. General LLM
serving would additionally require:

1. GQA and MQA support;
2. ragged/variable-length prefill;
3. paged NVFP4 KV and radix-cache integration;
4. an efficient single-token decode kernel;
5. CUDA Graph support and stable workspace ownership;
6. backend capability checks and numerically equivalent fallbacks.

Until those pieces exist, the dense NVFP4 attention result should be treated
as an operator-level potential rather than an available SGLang serving mode.
