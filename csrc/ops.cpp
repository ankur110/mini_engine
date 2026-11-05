#include "ops.h"
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(mini_engine,m){
    m.def("store_kvcache(Tensor key,Tensor value,Tensor(a!) k_cache,Tensor(b!) v_cache,Tensor slot_mapping) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(mini_engine,CUDA,m){
    m.impl("store_kvcache",TORCH_BOX(&store_kvcache));
}