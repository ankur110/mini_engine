#pragma once
#include <torch/csrc/stable/tensor.h>

void store_kvcache(
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& value,
    torch::stable::Tensor& k_cache,
    torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& slot_mapping
);

void paged_decode(
    torch::stable::Tensor& out,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& block_tables,
    const torch::stable::Tensor& context_lens,
    double scale
);

void paged_decode_v2(
    torch::stable::Tensor& out,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& block_tables,
    const torch::stable::Tensor& context_lens,
    torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits,
    torch::stable::Tensor& exp_sums,
    double scale
);