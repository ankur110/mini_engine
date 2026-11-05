#pragma once
#include <torch/csrc/stable/tensor.h>

void store_kvcache(
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& value,
    torch::stable::Tensor& k_cache,
    torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& slot_mapping
);
