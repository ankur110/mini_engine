#include "ops.h"
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(mini_engine,m){
    m.def("store_kvcache(Tensor key,Tensor value,Tensor(a!) k_cache,Tensor(b!) v_cache,Tensor slot_mapping) -> ()");
    m.def("paged_decode(Tensor(a!) out,Tensor q,Tensor k_cache,Tensor v_cache,Tensor block_tables,Tensor context_lens,float scale) -> ()");
    m.def("paged_decode_v2(Tensor(a!) out,Tensor q,Tensor k_cache,Tensor v_cache,Tensor block_tables,Tensor context_lens,"
          "Tensor(b!) tmp_out,Tensor(c!) max_logits,Tensor(d!) exp_sums,float scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(mini_engine,CUDA,m){
    m.impl("store_kvcache",TORCH_BOX(&store_kvcache));
    m.impl("paged_decode",TORCH_BOX(&paged_decode));
    m.impl("paged_decode_v2",TORCH_BOX(&paged_decode_v2));
}