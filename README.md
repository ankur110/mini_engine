# mini_engine

A small, from-scratch LLM inference engine, built to understand how
vLLM-style serving actually works under the hood.


## What it does

- Implements the Qwen3 decoder architecture from scratch in plain PyTorch.
- Merges projections into single GEMMs (QKV and gate/up) the way serving engines do.
- Uses grouped-query attention with a causal mask over a full prompt.
- Applies RMSNorm with a fused residual add, SwiGLU activations, and rotary position
  embeddings.
- Supports tied input/output embeddings.
- Loads weights from a standard Hugging Face checkpoint format.

## Status

Work in progress. The model definition and core layers are in place; the runtime pieces
(paged KV cache, continuous batching, scheduling and sampling) are being built next.


