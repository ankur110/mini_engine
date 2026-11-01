# mini_engine

A from-scratch Qwen3 inference engine built to understand how vLLM-style serving works under the hood. The model, paged KV cache, continuous batching, scheduler, prefix cache, and sampler are all plain PyTorch — no vLLM kernels.

## What it does

- Implements the Qwen3 decoder from scratch: grouped-query attention, RoPE, RMSNorm, SwiGLU, and tied input/output embeddings.
- Packs QKV and gate/up projections into single GEMMs.
- Serves with a paged KV cache, block tables, and prefix caching.
- Continuous batching with FCFS admission and newest-first preemption.
- Chunked prefill under `max_num_batched_tokens`; decode bounded by `max_num_seqs`.
- Greedy sampling at temperature 0 and exponential-race sampling otherwise.
- Loads bf16 safetensors checkpoints, sizes the KV pool from a measured warm-up peak, and runs on CUDA.

## Results

Benchmark: one NVIDIA A100-SXM4-40GB, Qwen3-0.6B, 256 concurrent sequences, 142,827 input tokens, and 133,966 output tokens. All requests use `ignore_eos=True`, `max_tokens` between 100 and 1024, and `max_model_len=2048`.

| Engine | Wall time | Output throughput | Relative |
|---|---:|---:|---:|
| mini_engine | 86.78 s | 1,543.7 tok/s | 1.00x |
| vLLM | 14.98 s | 8,940.5 tok/s | 5.79x |

End-to-end (input + output) throughput is about 3,190 tok/s for mini_engine and 18,478 tok/s for vLLM. The current gap is **5.8x**.

> The two engines ran in separate Modal images, so their PyTorch/CUDA builds differ (2.15.0+cu132 vs 2.13.0+cu130). The ratio is directional, not a controlled kernel-for-kernel comparison.

## Where the gap comes from

The gap is mostly kernel launches and memory traffic, not model FLOPs. Each decoder layer currently runs a long sequence of small eager PyTorch ops:

- RMSNorm, residual add, RoPE, and `silu(gate) * up` are separate kernels.
- Attention materializes and gathers KV blocks in eager PyTorch.
- The sampler runs separate softmax and exponential-race kernels.
- Decode has no CUDA graph, so Python and launch overhead are paid every token.

At 256 concurrent sequences those overheads dominate. vLLM uses fused kernels, a tuned paged-attention kernel, and CUDA graphs.

## Next: fused kernels

The next milestone is replacing the eager layer path with fused kernels and re-benchmarking after each change:

- Fused RMSNorm + residual add.
- Fused RoPE.
- Fused SwiGLU (`silu(gate) * up`).
- Fused QKV split and KV-cache write.
- Cheaper sampling.
- CUDA graphs for decode.

Goal: reduce the 5.8x gap without changing model math.

## Reproduce

```bash
python bench.py --engine mini --model ~/models/Qwen3-0.6B
python bench.py --engine vllm --model ~/models/Qwen3-0.6B
```

Both append a JSON line to the results log.
