# NVFP4 and FP4 Tensor Core Evaluation

This directory records the scope, reproducible measurements, and follow-up
plan for evaluating NVFP4 on consumer Blackwell. It deliberately separates
three concepts that are easy to conflate:

1. NVFP4 W4A4 linear GEMM;
2. an NVFP4 KV cache;
3. attention whose QK and PV matrix multiplications both execute with NVFP4
   Tensor Core MMA.

The notes are based on SGLang commit `3c481b9421` and the local environment
described below. Results are experimental and are not SGLang release claims.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-25 |
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| Compute capability | SM120 |
| NVIDIA driver | 610.62 |
| PyTorch | 2.13.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| FlashInfer | 0.6.15.post1 |
| SGLang package | 0.5.17 |
| SGLang source | `3c481b9421` |

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
| FlashInfer dense NVFP4 attention | No runtime integration | Q/K/V and softmax P | Yes, both QK and PV | Dense MHA self-attention microbenchmark |

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
  --head-dims 128 \
  --num-heads 8 \
  --causal \
  --warmup-ms 10 \
  --measure-ms 30
```

Configuration: `B=1`, `H=8`, `D=128`, BF16 input/output, causal attention,
cold L2, seed 0. JIT compilation completed before the timed samples.

| S | BF16 Flash SDPA (ms) | NVFP4 attention-only (ms) | Kernel speedup | NVFP4 end-to-end (ms) | End-to-end speedup | Relative RMSE | Cosine similarity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.0184 | 0.0138 | 1.33x | 0.0921 | 0.20x | 0.5318 | 0.945350 |
| 1024 | 0.0348 | 0.0206 | 1.69x | 0.1076 | 0.32x | 0.5330 | 0.950340 |
| 2048 | 0.0882 | 0.0348 | 2.53x | 0.1895 | 0.47x | 0.5330 | 0.953663 |

Interpretation:

- Prequantized FP4 attention becomes more favorable as sequence length grows.
- With FlashInfer 0.6.15.post1, quantizing BF16 Q/K/V for every call costs more
  than the attention kernel saves for these shapes.
- Accuracy numbers use synthetic Gaussian Q/K/V. They are operator-level
  numerical comparisons, not model-task accuracy results.
- This is one local measurement series, not a statistically repeated serving
  benchmark. Results should not be extrapolated to SGLang throughput.

## Planned evaluation matrix

No additional experiments in this section have been run yet.

| ID | Evaluation | Weight/linear path | KV cache | Attention math | Accuracy signal |
| ---: | --- | --- | --- | --- | --- |
| 1 | BF16 serving baseline | BF16 | BF16 | BF16 | GSM8K subset and long-context retrieval |
| 2 | W4A4 serving | NVFP4 W4A4 | BF16 | BF16 | Difference from experiment 1 |
| 3 | W4A4 plus FP4 KV serving | NVFP4 W4A4 | NVFP4 | SM120 XQA BF16 MMA | Difference from experiment 2, especially at long context |
| 4 | Dense attention microbenchmark | N/A | Prequantized dense Q/K/V | QK and PV NVFP4 MMA | RMSE, cosine, SQNR, and p99 error versus BF16 |
| 5 | Linear GEMM microbenchmark | BF16 versus NVFP4 W4A4 | N/A | N/A | RMSE, cosine, SQNR, and p99 error versus BF16 GEMM |

Experiments 1-3 will use identical prompts, scheduling parameters, attention
backends, seeds, and checkpoint lineage. Experiment 2 isolates W4A4 linear
effects; experiment 3 changes only the KV cache dtype. Serving speed and task
accuracy will be measured separately.

Experiments 4-5 are operator microbenchmarks. Their timings estimate the FP4
Tensor Core ceiling and quantization overhead; their numerical errors do not
directly represent task accuracy.

### Lightweight accuracy plan

- Experiments 1-3: the same 200 GSM8K questions at temperature zero.
- Experiments 1-3: 30 exact-match passkey retrieval cases at 4K and 8K context
  to make KV-cache degradation more visible.
- Experiment 4: relative RMSE, cosine similarity, SQNR, p99 absolute error, and
  NaN/Inf checks across representative attention shapes and three seeds.
- Experiment 5: the same numerical metrics for representative QKV, output, and
  MLP projection shapes, separating prequantized GEMM from activation
  quantization plus GEMM.

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
