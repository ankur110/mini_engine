#include "ops.h"
#include <torch/headeronly/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/stable/macros.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

__global__ void store_kvcache_kernel(
    const uint4* __restrict__ key,
    const uint4* __restrict__ value,
    uint4* __restrict__ k_cache,
    uint4* __restrict__ v_cache,
    const int32_t* __restrict__ slot_mapping,
    int64_t key_stride,
    int64_t value_stride,
    int row_chunks
){
    int64_t token=blockIdx.x;
    int64_t slot=slot_mapping[token];
    if(slot<0) return;                       
    uint4* dst_k=k_cache+slot*row_chunks;
    uint4* dst_v=v_cache+slot*row_chunks;
    const uint4* src_k=key+token*key_stride;
    const uint4* src_v=value+token*value_stride;
    for(int i=threadIdx.x;i<row_chunks;i+=blockDim.x){
        dst_k[i]=src_k[i];
        dst_v[i]=src_v[i];
    }
}

void store_kvcache(
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& value,
    torch::stable::Tensor& k_cache,
    torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& slot_mapping
){
    const int64_t N=key.size(0);
    if(N==0) return;
    const int64_t row_elems=key.size(1)*key.size(2);

    // only the checks that prevent a silently wrong copy
    STD_TORCH_CHECK(key.stride(2)==1 && key.stride(1)==key.size(2) &&
                    value.stride(2)==1 && value.stride(1)==value.size(2),
                    "key/value: each token's heads must form one contiguous row");
    STD_TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),"caches must be contiguous");
    STD_TORCH_CHECK(row_elems%8==0 && key.stride(0)%8==0 && value.stride(0)%8==0,
                    "row size and token strides must be multiples of 8 (16-byte chunks)");

    const int row_chunks=static_cast<int>(row_elems/8);
    const int64_t key_stride=key.stride(0)/8;
    const int64_t value_stride=value.stride(0)/8;
    const unsigned int grid=static_cast<unsigned int>(N);
    const int block=std::min(row_chunks,1024);

    void* stream_ptr=nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(key.get_device_index(),&stream_ptr));
    cudaStream_t stream=static_cast<cudaStream_t>(stream_ptr);

    store_kvcache_kernel<<<grid,block,0,stream>>>(
        reinterpret_cast<const uint4*>(key.const_data_ptr()),
        reinterpret_cast<const uint4*>(value.const_data_ptr()),
        reinterpret_cast<uint4*>(k_cache.mutable_data_ptr()),
        reinterpret_cast<uint4*>(v_cache.mutable_data_ptr()),
        slot_mapping.const_data_ptr<int32_t>(),
        key_stride,value_stride,row_chunks
    );
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}