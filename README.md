# mini_engine

A from-scratch LLM inference engine for Qwen3, built to understand how vLLM-style serving works under the hood. The model, paged KV cache, continuous batching scheduler, chunked prefill, prefix cache, and sampler are written in plain PyTorch, and the attention hot path runs on hand-written CUDA kernels. No vLLM or FlashAttention code is used.

On a single A100-80GB, mini_engine reaches **~41% of vLLM 0.32's offline throughput** (a 2.5x gap). The first version was 5.8x behind.

## Features

**Model**
- Qwen3 decoder implemented from scratch: grouped-query attention, RoPE, RMSNorm, SwiGLU, tied embeddings.
- QKV and gate/up projections packed into single GEMMs.
- Loads bf16 safetensors checkpoints directly.

**Serving**
- Paged KV cache with block tables and hash-based prefix caching. Blocks are hashed as soon as they fill, so prefixes are shared even across partially prefilled requests.
- Continuous batching with FCFS or priority scheduling and preemption under memory pressure (newest-first, or lowest-priority-first).
- **Chunked prefill**: each step has a token budget (`max_num_batched_tokens`). Running decodes are scheduled first, then partially prefilled prompts, then new prompts are admitted in chunks that fill whatever budget is left. Decode and prefill tokens share one forward pass.
- Optional per-request chunk cap (`long_prefill_token_threshold`) so a single long prompt can't take the whole budget.
- A prefill-first mode (`enable_chunked_prefill=False`) is kept as a baseline: whole prompts first, decode only when nothing is waiting.
- KV pool sized automatically from a measured warm-up peak and `gpu_memory_utilization`.
- Greedy decoding at temperature 0, exponential-race (Gumbel-style) sampling otherwise.

**Custom CUDA kernels** (bf16, sm_80, built as a PyTorch extension against the stable ABI)
- **KV-cache write**: scatters new K/V rows into their paged slots with 16-byte vectorized copies; skips padded slots.
- **Paged prefill attention**: FlashAttention-style tiled kernel on tensor cores (WMMA). Reads K/V straight from the paged cache through the block table, uses online softmax, and handles ragged batches, prefix-cache hits, and prefill chunks that attend to earlier chunks, all in one launch.
- **Paged decode attention, two variants**:
  - *v1*: one thread block per (sequence, KV head); each warp streams whole KV blocks, and both query heads in a GQA group share every K/V load.
  - *v2*: split-K. The context is cut into 256-token partitions processed in parallel, then merged by a small reduce kernel using the saved per-partition max and exp-sum.
  - A dispatcher picks v2 when the batch is too small to fill the GPU (few sequences or long contexts) and v1 when there is already enough parallelism.
- **Split attention for mixed batches**: in a step with both decodes and prefill chunks, decode tokens go to the decode kernel and prefill chunks go to the prefill kernel, both writing into one output tensor. A "unified" mode that sends everything through the prefill kernel is kept for comparison.

The kernels are currently specialized to Qwen3-0.6B's shape: head dim 128, KV block size 16, GQA ratio 2.

## Results

All results use Qwen3-0.6B on one NVIDIA A100-SXM4-80GB.

### Offline throughput

256 requests submitted at once: 142,827 input and 133,966 output tokens. Prompts are random token IDs of 100–1024 tokens, `max_tokens` is 100–1024, `temperature=0.6`, `ignore_eos=True`, `max_model_len=2048`. No preemptions occurred in any run.

| Engine | Configuration | Wall time | Output throughput | vs. vLLM |
|---|---|---:|---:|---:|
| mini_engine | prefill-first | 31.63 s | 4,235 tok/s | 0.41x |
| mini_engine | **chunked, 16k budget** (default) | **31.86 s** | **4,205 tok/s** | **0.41x** |
| mini_engine | chunked, 2k budget | 33.22 s | 4,032 tok/s | 0.39x |
| mini_engine | chunked, 16k, unified attention | 78.60 s | 1,705 tok/s | 0.16x |
| vLLM 0.32.0 | default (CUDA graphs) | 12.91 s | 10,377 tok/s | 1.00x |

vLLM's number is the mean of two runs (10,358 and 10,397 tok/s).

- Chunked prefill costs almost nothing in offline throughput (under 1% at a 16k budget, about 5% at 2k).
- Split attention matters: sending decode tokens through the prefill kernel ("unified") makes the engine 2.5x slower, because a decode query is a single row and wastes almost all of a 32-row prefill tile.

> mini_engine and vLLM run in separate container images with different PyTorch/CUDA builds (2.15.0+cu132 vs 2.13.0+cu130), so the ratio is directional rather than a controlled kernel-for-kernel comparison.

### Online latency: prefill-first vs. chunked prefill

240 requests arrive as a Poisson process. 90% are short (128–512 prompt tokens) and 10% are long (4,096–7,680 prompt tokens); each produces 128–256 output tokens. TTFT is time to first token; ITL is the gap between consecutive tokens of one request.

**8 requests/s:**

| Config | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | ITL max | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| prefill-first | 49 ms | 775 ms | 45 ms | 417 ms | 1,334 ms | 1,102 |
| chunked 512 | 1,404 ms | 8,628 ms | 46 ms | 112 ms | **123 ms** | 958 |
| chunked 1024 | 386 ms | 3,273 ms | 45 ms | 142 ms | 158 ms | 1,071 |
| chunked 2048 | 82 ms | 1,363 ms | 45 ms | 190 ms | 235 ms | 1,103 |
| chunked 4096 | 79 ms | 861 ms | 45 ms | 268 ms | 325 ms | 1,107 |
| chunked 8192 | 77 ms | **681 ms** | 44 ms | 345 ms | 434 ms | **1,108** |
| chunked 2048, unified attention | 200 ms | 8,545 ms | 97 ms | 235 ms | 248 ms | 804 |

**4 requests/s:**

| Config | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | ITL max |
|---|---:|---:|---:|---:|---:|
| prefill-first | 43 ms | 394 ms | 39 ms | 253 ms | 606 ms |
| chunked 512 | 72 ms | 1,462 ms | 39 ms | 104 ms | 125 ms |
| chunked 2048 | 56 ms | 612 ms | 38 ms | 153 ms | 206 ms |
| chunked 8192 | 58 ms | 433 ms | 38 ms | 204 ms | 423 ms |

What this shows:

- **Prefill-first stalls decoding.** When a 7k-token prompt arrives, every running request waits for it, so the worst gap between tokens hits 1.3 s at 8 req/s.
- **Chunking bounds those stalls.** A 512-token budget cuts the worst gap ~11x (1,334 → 123 ms), because no step can carry more than 512 prefill tokens.
- **Small budgets cost TTFT.** At 8 req/s a 512 budget can't absorb incoming prompts fast enough and the queue grows (p99 TTFT 8.6 s). At 4 req/s, where there's headroom, the same budget is cheap.
- **The budget sets the trade-off.** 2048–4096 is a good middle ground. At 8 req/s, 8192 beats prefill-first on *both* TTFT p99 and ITL tail, with the highest throughput.

### Earlier milestone: CUDA kernels vs. eager attention

On an A100-40GB with the offline workload above, replacing the eager PyTorch attention path (gather KV blocks + SDPA) with the custom kernels made the whole engine **2.9x faster**: 95.15 s → 32.42 s (1,408 → 4,133 tok/s), with no change to model math.

### Correctness

Every kernel is checked against an fp32 reference that reads only valid tokens. The cache is pre-filled with NaN, so any read of an unowned or out-of-range slot shows up immediately. Cases covered: ragged prefill, prefix-cache hits, tile-boundary lengths, and decode at small, long, and mixed batch shapes. The KV-cache write is bit-exact; attention outputs stay within bf16 tolerance (max abs error under 0.009).

## Where the remaining gap comes from

Attention is no longer the main bottleneck. What's left is mostly kernel-launch and memory-traffic overhead from the rest of the layer:

- RMSNorm, residual add, RoPE, and `silu(gate) * up` are still separate eager PyTorch kernels.
- The sampler runs separate softmax and exponential-race kernels over the full vocabulary.
- Decode has no CUDA graph, so Python and launch overhead are paid on every step. This is also why ITL p50 sits around 40 ms.

## Roadmap

- Fused RMSNorm + residual add
- Fused RoPE (folded into the QKV split / KV-cache write)
- Fused SwiGLU
- A cheaper fused sampler
- CUDA graphs for decode
- Generalizing the attention kernels beyond Qwen3-0.6B's head shape

The goal is to close the remaining 2.5x gap without changing model math, re-benchmarking after each step.

## Usage

```python
from llm import LLM
from sampling_params import SamplingParams

llm = LLM(
    "~/models/Qwen3-0.6B",
    max_num_seqs=256,
    max_model_len=2048,
    max_num_batched_tokens=4096,     # per-step token budget
    enable_chunked_prefill=True,     # False = prefill-first baseline
    scheduling_policy="fcfs",        # or "priority"
)
outputs = llm.generate(
    ["Explain paged attention in one paragraph."],
    SamplingParams(temperature=0.6, max_tokens=256),
)
print(outputs[0]["text"])
```

With `scheduling_policy="priority"`, pass `priorities=[...]` to `generate` (lower value = served first). For streaming-style use, call `add_request()` and `step()` directly.

The CUDA extension is JIT-compiled on first import (requires `nvcc` and `ninja`). Environment switches for comparisons:

- `MINI_ENGINE_OPS=0` uses the pure-PyTorch attention path instead of the CUDA kernels.
- `MINI_ENGINE_ATTN=unified` sends all tokens through the prefill kernel instead of splitting decode and prefill.

## Reproducing the benchmarks

Locally, on a CUDA machine:

```bash
# offline throughput
python bench.py --engine mini --model ~/models/Qwen3-0.6B                  # chunked, 16k budget
python bench.py --engine mini --model ~/models/Qwen3-0.6B --chunked 0      # prefill-first
python bench.py --engine mini --model ~/models/Qwen3-0.6B --budget 2048
python bench.py --engine mini --model ~/models/Qwen3-0.6B --attn unified
python bench.py --engine vllm --model ~/models/Qwen3-0.6B

# online latency sweep (Poisson arrivals, all configs, 4 and 8 req/s)
python bench_latency.py --model ~/models/Qwen3-0.6B
```

Or on [Modal](https://modal.com), which downloads the model and runs everything on an A100-80GB:

```bash
modal run modal_run.py                 # everything
modal run modal_run.py --step tput     # one step: kernels | checks | tput | vllm | latency
```

Each run appends one JSON line per configuration to the results logs, along with a GPU fingerprint (name, memory, clocks, power limit, driver).
