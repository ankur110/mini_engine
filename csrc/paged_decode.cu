#include "ops.h"
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/stable/macros.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

constexpr int kHeadDim=128;
constexpr int kKvBlockSize=16;
constexpr int kNumWarps=4;
constexpr int kGqa=2;
constexpr int kPartitionSize=256;   


__device__ __forceinline__ float warp_sum(float x){
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1){
        x+=__shfl_xor_sync(0xffffffff,x,offset);
    }
    return x;
}


__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p,float (&x)[4]){
    const uint2 packed=*reinterpret_cast<const uint2*>(p);
    const float2 a=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.x));
    const float2 b=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.y));
    x[0]=a.x;
    x[1]=a.y;
    x[2]=b.x;
    x[3]=b.y;
}


template<int HEAD_DIM,int KV_BLOCK_SIZE,int NUM_WARPS,int GQA>
__device__ __forceinline__ void decode_range(
    const __nv_bfloat16* k_cache,const __nv_bfloat16* v_cache,
    const int32_t* bt_row,int kv_head,int L,
    int start,int end,
    int64_t kv_block_stride,int64_t kv_row_stride,
    const float (&q)[GQA][4],float (&m)[GQA],float (&l)[GQA],float (&acc)[GQA][4]
){
    const int warp=threadIdx.x/32;
    const int lane=threadIdx.x%32;

    #pragma unroll
    for(int g=0;g<GQA;g++){
        m[g]=-FLT_MAX;
        l[g]=0.0f;
        #pragma unroll
        for(int d=0;d<4;d++){
            acc[g][d]=0.0f;
        }
    }

    for(int i=start+warp;i<end;i+=NUM_WARPS){
        const int64_t id=static_cast<int64_t>(bt_row[i]);
        const int64_t base=id*kv_block_stride+static_cast<int64_t>(kv_head)*HEAD_DIM+4*lane;
        const int t0=i*KV_BLOCK_SIZE;


        float kx[KV_BLOCK_SIZE][4];
        #pragma unroll
        for(int r=0;r<KV_BLOCK_SIZE;r++){
            load_bf16x4(k_cache+base+static_cast<int64_t>(r)*kv_row_stride,kx[r]);
        }


        float s[GQA][KV_BLOCK_SIZE];
        #pragma unroll
        for(int r=0;r<KV_BLOCK_SIZE;r++){
            #pragma unroll
            for(int g=0;g<GQA;g++){
                float partial=0.0f;
                #pragma unroll
                for(int d=0;d<4;d++){
                    partial=fmaf(q[g][d],kx[r][d],partial);
                }
                s[g][r]=warp_sum(partial);
                if(t0+r>=L){
                    s[g][r]=-INFINITY;
                }
            }
        }


        #pragma unroll
        for(int g=0;g<GQA;g++){
            float tile_max=-FLT_MAX;
            #pragma unroll
            for(int r=0;r<KV_BLOCK_SIZE;r++){
                tile_max=fmaxf(tile_max,s[g][r]);
            }
            const float m_new=fmaxf(m[g],tile_max);
            const float alpha=__expf(m[g]-m_new);
            float l_sum=0.0f;
            #pragma unroll
            for(int r=0;r<KV_BLOCK_SIZE;r++){
                s[g][r]=__expf(s[g][r]-m_new);  
                l_sum+=s[g][r];
            }
            l[g]=l[g]*alpha+l_sum;
            #pragma unroll
            for(int d=0;d<4;d++){
                acc[g][d]*=alpha;
            }
            m[g]=m_new;
        }

        #pragma unroll
        for(int r=0;r<KV_BLOCK_SIZE;r++){
            if(t0+r<L){
                float vx[4];
                load_bf16x4(v_cache+base+static_cast<int64_t>(r)*kv_row_stride,vx);
                #pragma unroll
                for(int g=0;g<GQA;g++){
                    #pragma unroll
                    for(int d=0;d<4;d++){
                        acc[g][d]=fmaf(s[g][r],vx[d],acc[g][d]);
                    }
                }
            }
        }
    }
}


template<int HEAD_DIM,int NUM_WARPS,int GQA>
__device__ __forceinline__ void merge_warps(
    const float (&m)[GQA],const float (&l)[GQA],const float (&acc)[GQA][4],
    float (&m_out)[GQA],float (&l_out)[GQA],float (&o_out)[GQA]
){
    static_assert(HEAD_DIM==NUM_WARPS*32,"merge_warps: needs one thread per head dim");

    __shared__ float sm_acc[NUM_WARPS][GQA][HEAD_DIM];
    __shared__ float sm_m[NUM_WARPS][GQA];
    __shared__ float sm_l[NUM_WARPS][GQA];

    const int warp=threadIdx.x/32;
    const int lane=threadIdx.x%32;
    #pragma unroll
    for(int g=0;g<GQA;g++){
        #pragma unroll
        for(int d=0;d<4;d++){
            sm_acc[warp][g][4*lane+d]=acc[g][d];
        }
        if(lane==0){
            sm_m[warp][g]=m[g];
            sm_l[warp][g]=l[g];
        }
    }

    __syncthreads();

    const int t=threadIdx.x;
    #pragma unroll
    for(int g=0;g<GQA;g++){
        float M=-FLT_MAX;
        #pragma unroll
        for(int w=0;w<NUM_WARPS;w++){
            M=fmaxf(M,sm_m[w][g]);
        }
        float l_sum=0.0f;
        float o_sum=0.0f;
        #pragma unroll
        for(int w=0;w<NUM_WARPS;w++){
            const float f=__expf(sm_m[w][g]-M);
            l_sum+=f*sm_l[w][g];
            o_sum+=f*sm_acc[w][g][t];
        }
        m_out[g]=M;
        l_out[g]=l_sum;
        o_out[g]=o_sum;
    }
}


template<int GQA>
__device__ __forceinline__ void load_q_scaled(
    const __nv_bfloat16* q,int b,int j,
    int64_t q_stride_seq,int64_t q_stride_head,float scale,
    float (&qf)[GQA][4]
){
    const int lane=threadIdx.x%32;
    #pragma unroll
    for(int g=0;g<GQA;g++){
        const int head=j*GQA+g;
        const int64_t q_off=static_cast<int64_t>(b)*q_stride_seq+static_cast<int64_t>(head)*q_stride_head+4*lane;
        load_bf16x4(q+q_off,qf[g]);
        #pragma unroll
        for(int d=0;d<4;d++){
            qf[g][d]*=scale;
        }
    }
}


template<int HEAD_DIM,int KV_BLOCK_SIZE,int NUM_WARPS,int GQA>
__global__ void __launch_bounds__(NUM_WARPS*32,4) paged_decode_v1_kernel(
    __nv_bfloat16* __restrict__ out,const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,const int32_t* __restrict__ context_lens,
    int64_t q_stride_seq,int64_t q_stride_head,
    int64_t out_stride_seq,int64_t out_stride_head,
    int64_t kv_block_stride,int64_t kv_row_stride,
    int max_kv_blocks,float scale
){
    static_assert(HEAD_DIM==32*4,"each lane owns 4 dims: HEAD_DIM must be 128");

    const int b=blockIdx.x;
    const int j=blockIdx.y;

    const int L=context_lens[b];
    const int n=(L+KV_BLOCK_SIZE-1)/KV_BLOCK_SIZE;
    const int32_t* bt_row=block_tables+static_cast<int64_t>(b)*max_kv_blocks;

    float qf[GQA][4];
    load_q_scaled<GQA>(q,b,j,q_stride_seq,q_stride_head,scale,qf);

    float m[GQA];
    float l[GQA];
    float acc[GQA][4];
    decode_range<HEAD_DIM,KV_BLOCK_SIZE,NUM_WARPS,GQA>(
        k_cache,v_cache,bt_row,j,L,0,n,kv_block_stride,kv_row_stride,qf,m,l,acc
    );

    float m_out[GQA];
    float l_out[GQA];
    float o_out[GQA];
    merge_warps<HEAD_DIM,NUM_WARPS,GQA>(m,l,acc,m_out,l_out,o_out);

    const int t=threadIdx.x;
    #pragma unroll
    for(int g=0;g<GQA;g++){
        const int head=j*GQA+g;
        const float val=l_out[g]>0.0f?o_out[g]/l_out[g]:0.0f;   
        const int64_t out_off=static_cast<int64_t>(b)*out_stride_seq+static_cast<int64_t>(head)*out_stride_head+t;
        out[out_off]=__float2bfloat16(val);
    }
}


template<int HEAD_DIM,int KV_BLOCK_SIZE,int NUM_WARPS,int GQA,int PARTITION_SIZE>
__global__ void __launch_bounds__(NUM_WARPS*32,4) paged_decode_v2_partition_kernel(
    float* __restrict__ tmp_out,float* __restrict__ max_logits,float* __restrict__ exp_sums,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,const int32_t* __restrict__ context_lens,
    int64_t q_stride_seq,int64_t q_stride_head,
    int64_t kv_block_stride,int64_t kv_row_stride,
    int max_kv_blocks,float scale
){
    constexpr int KV_BLOCKS_PER_PART=PARTITION_SIZE/KV_BLOCK_SIZE;
    static_assert(PARTITION_SIZE%KV_BLOCK_SIZE==0,"partition must be a whole number of kv_blocks");
    static_assert(HEAD_DIM==32*4,"each lane owns 4 dims: HEAD_DIM must be 128");

    const int b=blockIdx.x;
    const int j=blockIdx.y;
    const int p=blockIdx.z;

    const int L=context_lens[b];
    const int n=(L+KV_BLOCK_SIZE-1)/KV_BLOCK_SIZE;
    if(p*KV_BLOCKS_PER_PART>=n) return;
    const int start=p*KV_BLOCKS_PER_PART;
    const int end=min(start+KV_BLOCKS_PER_PART,n);
    const int32_t* bt_row=block_tables+static_cast<int64_t>(b)*max_kv_blocks;

    float qf[GQA][4];
    load_q_scaled<GQA>(q,b,j,q_stride_seq,q_stride_head,scale,qf);

    float m[GQA];
    float l[GQA];
    float acc[GQA][4];
    decode_range<HEAD_DIM,KV_BLOCK_SIZE,NUM_WARPS,GQA>(
        k_cache,v_cache,bt_row,j,L,start,end,kv_block_stride,kv_row_stride,qf,m,l,acc
    );

    float m_out[GQA];
    float l_out[GQA];
    float o_out[GQA];
    merge_warps<HEAD_DIM,NUM_WARPS,GQA>(m,l,acc,m_out,l_out,o_out);

    const int num_heads=gridDim.y*GQA;
    const int max_parts=gridDim.z;
    const int t=threadIdx.x;
    #pragma unroll
    for(int g=0;g<GQA;g++){
        const int head=j*GQA+g;
        const int64_t row=(static_cast<int64_t>(b)*num_heads+head)*max_parts+p;
        tmp_out[row*HEAD_DIM+t]=o_out[g];   
        if(t==0){
            max_logits[row]=m_out[g];
            exp_sums[row]=l_out[g];
        }
    }
}


template<int HEAD_DIM,int PARTITION_SIZE>
__global__ void __launch_bounds__(HEAD_DIM) paged_decode_v2_reduce_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ tmp_out,const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums,const int32_t* __restrict__ context_lens,
    int64_t out_stride_seq,int64_t out_stride_head,
    int max_parts
){
    const int b=blockIdx.x;
    const int h=blockIdx.y;
    const int t=threadIdx.x;
    const int num_heads=gridDim.y;


    const int P=(context_lens[b]+PARTITION_SIZE-1)/PARTITION_SIZE;
    const int64_t row0=(static_cast<int64_t>(b)*num_heads+h)*max_parts;

    float M=-FLT_MAX;
    for(int p=0;p<P;p++){
        M=fmaxf(M,max_logits[row0+p]);   
    }
    float l_tot=0.0f;
    float o=0.0f;
    for(int p=0;p<P;p++){
        const float f=__expf(max_logits[row0+p]-M);
        l_tot+=f*exp_sums[row0+p];
        o+=f*tmp_out[(row0+p)*HEAD_DIM+t];
    }

    const float val=l_tot>0.0f?o/l_tot:0.0f;   
    out[static_cast<int64_t>(b)*out_stride_seq+static_cast<int64_t>(h)*out_stride_head+t]=__float2bfloat16(val);
}


static void check_decode_args(
    const torch::stable::Tensor& out,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& block_tables,
    const torch::stable::Tensor& context_lens
){
    using torch::headeronly::ScalarType;
    const int64_t num_seqs=q.size(0);
    const int64_t num_heads=q.size(1);
    const int64_t head_dim=q.size(2);
    const int64_t num_kv_heads=k_cache.size(2);

    STD_TORCH_CHECK(q.scalar_type()==ScalarType::BFloat16 && out.scalar_type()==ScalarType::BFloat16 &&
                    k_cache.scalar_type()==ScalarType::BFloat16 && v_cache.scalar_type()==ScalarType::BFloat16,
                    "paged_decode: q, out, k_cache, v_cache must be bfloat16");
    STD_TORCH_CHECK(block_tables.scalar_type()==ScalarType::Int && context_lens.scalar_type()==ScalarType::Int,
                    "paged_decode: block_tables and context_lens must be int32");
    STD_TORCH_CHECK(head_dim==kHeadDim && k_cache.size(3)==kHeadDim,"paged_decode: head_dim must be 128");
    STD_TORCH_CHECK(k_cache.size(1)==kKvBlockSize,"paged_decode: kv_block size must be 16");
    STD_TORCH_CHECK(num_heads==num_kv_heads*kGqa,"paged_decode: num_heads must be 2 x num_kv_heads");
    STD_TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),"paged_decode: caches must be contiguous");
    STD_TORCH_CHECK(v_cache.stride(0)==k_cache.stride(0) && v_cache.stride(1)==k_cache.stride(1),
                    "paged_decode: k_cache and v_cache must have the same layout");
    STD_TORCH_CHECK(q.stride(2)==1 && out.stride(2)==1,"paged_decode: last dim of q/out must be contiguous");
    STD_TORCH_CHECK(q.stride(0)%4==0 && q.stride(1)%4==0 &&
                    reinterpret_cast<uintptr_t>(q.const_data_ptr())%8==0,
                    "paged_decode: q must allow 8-byte loads (strides multiple of 4, 8-byte aligned)");
    STD_TORCH_CHECK(block_tables.is_contiguous(),"paged_decode: block_tables must be contiguous");
    STD_TORCH_CHECK(block_tables.size(0)==num_seqs && context_lens.size(0)==num_seqs,
                    "paged_decode: block_tables/context_lens must have num_seqs rows");
    STD_TORCH_CHECK(out.size(0)==num_seqs && out.size(1)==num_heads && out.size(2)==head_dim,
                    "paged_decode: out must have q's shape");
}


void paged_decode(
    torch::stable::Tensor& out,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& block_tables,
    const torch::stable::Tensor& context_lens,
    double scale
){
    const int64_t num_seqs=q.size(0);
    if(num_seqs==0) return;
    check_decode_args(out,q,k_cache,v_cache,block_tables,context_lens);
    const int64_t num_kv_heads=k_cache.size(2);
    const int64_t max_kv_blocks=block_tables.size(1);

    void* stream_ptr=nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(q.get_device_index(),&stream_ptr));
    cudaStream_t stream=static_cast<cudaStream_t>(stream_ptr);

    const dim3 grid(static_cast<unsigned int>(num_seqs),static_cast<unsigned int>(num_kv_heads));
    const int threads=kNumWarps*32;
    paged_decode_v1_kernel<kHeadDim,kKvBlockSize,kNumWarps,kGqa><<<grid,threads,0,stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.const_data_ptr()),
        block_tables.const_data_ptr<int32_t>(),
        context_lens.const_data_ptr<int32_t>(),
        q.stride(0),q.stride(1),
        out.stride(0),out.stride(1),
        k_cache.stride(0),k_cache.stride(1),
        static_cast<int>(max_kv_blocks),
        static_cast<float>(scale)
    );
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}


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
){
    const int64_t num_seqs=q.size(0);
    if(num_seqs==0) return;
    check_decode_args(out,q,k_cache,v_cache,block_tables,context_lens);
    const int64_t num_heads=q.size(1);
    const int64_t head_dim=q.size(2);
    const int64_t num_kv_heads=k_cache.size(2);
    const int64_t max_kv_blocks=block_tables.size(1);


    using torch::headeronly::ScalarType;
    STD_TORCH_CHECK(tmp_out.scalar_type()==ScalarType::Float && max_logits.scalar_type()==ScalarType::Float &&
                    exp_sums.scalar_type()==ScalarType::Float,
                    "paged_decode_v2: scratch tensors must be float32");
    STD_TORCH_CHECK(tmp_out.is_contiguous() && max_logits.is_contiguous() && exp_sums.is_contiguous(),
                    "paged_decode_v2: scratch tensors must be contiguous");
    STD_TORCH_CHECK(tmp_out.dim()==4 && max_logits.dim()==3 && exp_sums.dim()==3,
                    "paged_decode_v2: tmp_out is 4-D, max_logits/exp_sums are 3-D");
    const int64_t max_parts=tmp_out.size(2);
    STD_TORCH_CHECK(max_parts>=1 && max_parts<=65535,"paged_decode_v2: max_parts must be in [1, 65535]");
    STD_TORCH_CHECK(tmp_out.size(0)==num_seqs && tmp_out.size(1)==num_heads && tmp_out.size(3)==head_dim,
                    "paged_decode_v2: tmp_out must be [num_seqs, num_heads, max_parts, head_dim]");
    STD_TORCH_CHECK(max_logits.size(0)==num_seqs && max_logits.size(1)==num_heads && max_logits.size(2)==max_parts &&
                    exp_sums.size(0)==num_seqs && exp_sums.size(1)==num_heads && exp_sums.size(2)==max_parts,
                    "paged_decode_v2: max_logits/exp_sums must be [num_seqs, num_heads, max_parts]");

    void* stream_ptr=nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(q.get_device_index(),&stream_ptr));
    cudaStream_t stream=static_cast<cudaStream_t>(stream_ptr);


    const dim3 grid1(static_cast<unsigned int>(num_seqs),static_cast<unsigned int>(num_kv_heads),
                     static_cast<unsigned int>(max_parts));
    paged_decode_v2_partition_kernel<kHeadDim,kKvBlockSize,kNumWarps,kGqa,kPartitionSize><<<grid1,kNumWarps*32,0,stream>>>(
        tmp_out.mutable_data_ptr<float>(),
        max_logits.mutable_data_ptr<float>(),
        exp_sums.mutable_data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.const_data_ptr()),
        block_tables.const_data_ptr<int32_t>(),
        context_lens.const_data_ptr<int32_t>(),
        q.stride(0),q.stride(1),
        k_cache.stride(0),k_cache.stride(1),
        static_cast<int>(max_kv_blocks),
        static_cast<float>(scale)
    );
    STD_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 grid2(static_cast<unsigned int>(num_seqs),static_cast<unsigned int>(num_heads));
    paged_decode_v2_reduce_kernel<kHeadDim,kPartitionSize><<<grid2,kHeadDim,0,stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        tmp_out.const_data_ptr<float>(),
        max_logits.const_data_ptr<float>(),
        exp_sums.const_data_ptr<float>(),
        context_lens.const_data_ptr<int32_t>(),
        out.stride(0),out.stride(1),
        static_cast<int>(max_parts)
    );
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}
