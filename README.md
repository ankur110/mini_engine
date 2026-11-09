# mini_engine

A from-scratch LLM inference engine for Qwen3, built to understand how vLLM-style serving works under the hood. The model, paged KV cache, continuous batching scheduler, prefix cache, and sampler are written in plain PyTorch, and the attention hot path runs on hand-written CUDA kernels. No vLLM or FlashAttention code is used.

On a single A100, mini_engine currently reaches **~53% of vLLM's output throughput** (1.9x gap). It started at 5.8x.

## Features

**Model**
- Qwen3 decoder implemented from scratch: grouped-query attention, RoPE, RMSNorm, SwiGLU, tied embeddings.
- QKV and gate/up projections packed into single GEMMs.
- Loads bf16 safetensors checkpoints directly.

**Serving**
- Paged KV cache with block tables and hash-based prefix caching.
- Continuous batching with FCFS admission and newest-first preemption under memory pressure.
- Chunked prefill bounded by `max_num_batched_tokens`; decode batches bounded by `max_num_seqs`.
- KV pool sized automatically from a measured warm-up peak and `gpu_memory_utilization`.
- Greedy decoding at temperature 0, exponential-race (Gumbel-style) sampling otherwise.

**Custom CUDA kernels** (bf16, sm_80, built as a PyTorch extension against the stable ABI)
- **KV-cache write**: scatters new K/V rows into their paged slots with 16-byte vectorized copies; skips padded slots.
- **Paged prefill attention**: FlashAttention-style tiled kernel on tensor cores (WMMA). Reads K/V straight from the paged cache through the block table, uses online softmax, and handles ragged batches and prefix-cache hits (queries attending to previously cached context) in one launch.
- **Paged decode attention, two variants**:
  - *v1*: one thread block per (sequence, KV head); each warp streams whole KV blocks, and both query heads in a GQA group share every K/V load.
  - *v2*: split-K. The context is cut into 256-token partitions processed in parallel, then merged by a small reduce kernel using the saved per-partition max and exp-sum.
  - A dispatcher picks v2 when the batch is too small to fill the GPU (few sequences or long contexts) and v1 when there is already enough parallelism.

The kernels are currently specialized to Qwen3-0.6B's shape: head dim 128, KV block size 16, GQA ratio 2.

## Results

Benchmark: one NVIDIA A100-SXM4-40GB, Qwen3-0.6B, 256 concurrent sequences, 142,827 input tokens and 133,966 output tokens. Prompts are random token IDs of 100–1024 tokens, `max_tokens` is 100–1024, `temperature=0.6`, `ignore_eos=True`, `max_model_len=2048`.

| Engine | Attention path | Wall time | Output throughput | vs. vLLM |
|---|---|---:|---:|---:|
| mini_engine | eager PyTorch (gather + SDPA) | 95.15 s | 1,408 tok/s | 0.18x |
| mini_engine | **custom CUDA kernels** | **32.42 s** | **4,133 tok/s** | **0.53x** |
| vLLM | FlashAttention + CUDA graphs | 17.14 s | 7,814 tok/s | 1.00x |

Swapping in the CUDA kernels made the whole engine **2.9x faster** end to end, without changing any model math.

> mini_engine and vLLM run in separate container images with different PyTorch/CUDA builds (2.15.0+cu132 vs 2.13.0+cu130), so the ratio is directional rather than a controlled kernel-for-kernel comparison.

### Correctness

Every kernel is checked against an fp32 reference that reads only valid tokens. The cache is pre-filled with NaN, so any read of an unowned or out-of-range slot shows up immediately. Cases covered: ragged prefill, prefix-cache hits, tile-boundary lengths, and decode at small, long, and mixed batch shapes. The KV-cache write is bit-exact; attention outputs stay within bf16 tolerance (max abs error under 0.009).

## Where the remaining gap comes from

Attention is no longer the main bottleneck. What's left is mostly kernel-launch and memory-traffic overhead from the rest of the layer:

- RMSNorm, residual add, RoPE, and `silu(gate) * up` are still separate eager PyTorch kernels.
- The sampler runs separate softmax and exponential-race kernels over the full vocabulary.
- Decode has no CUDA graph, so Python and launch overhead are paid on every token.

## Roadmap

- Fused RMSNorm + residual add
- Fused RoPE (folded into the QKV split / KV-cache write)
- Fused SwiGLU
- A cheaper fused sampler
- CUDA graphs for decode
- Generalizing the attention kernels beyond Qwen3-0.6B's head shape

The goal is to close the remaining 1.9x gap without changing model math, re-benchmarking after each step.

## Usage

```python
from llm import LLM
from sampling_params import SamplingParams

llm = LLM("~/models/Qwen3-0.6B", max_num_seqs=256, max_model_len=2048)
outputs = llm.generate(
    ["Explain paged attention in one paragraph."],
    SamplingParams(temperature=0.6, max_tokens=256),
)
print(outputs[0]["text"])
```

The CUDA extension is JIT-compiled on first import (requires `nvcc` and `ninja`). Set `MINI_ENGINE_OPS=0` to fall back to the pure-PyTorch attention path for comparison.

## Reproducing the benchmark

Locally, on a CUDA machine:

```bash
python bench.py --engine mini --model ~/models/Qwen3-0.6B
python bench.py --engine vllm --model ~/models/Qwen3-0.6B
```

Or on [Modal](https://modal.com), which downloads the model, runs the kernel checks, and benchmarks every configuration on an A100:

```bash
modal run modal_run.py              # everything
modal run modal_run.py --step mini  # one step: kernels | checks | mini | gather | vllm
```

Each benchmark run appends one JSON line with timing and environment info to the results log.
