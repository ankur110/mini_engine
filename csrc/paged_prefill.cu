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
#include <mma.h>
using namespace nvcuda;

constexpr int kHeadDim=128;
constexpr int kKvBlockSize=16;
constexpr int kGqa=2;
constexpr int kNumWarps=4;
constexpr int kTileQ=32;
constexpr int kRows=kTileQ*kGqa;
constexpr int kTileK=64;
constexpr int kPadH=kHeadDim+8;
constexpr int kPadS=kTileK+4;
constexpr int kPadP=kTileK+8;
constexpr int kPadO=kHeadDim+4;
constexpr int kSmemBytes=3*kRows*kPadH*2+kRows*kPadS*4+kRows*kPadP*2+kRows*kPadO*4;

__device__ __forceinline__ int find_seq(const int32_t* cu,int num_steps,int g){
    int lo=0;
    int hi=num_steps;
    while(lo<hi){
        int mid=lo+(hi-lo)/2;
        if((cu[mid]/kTileQ)+mid<=g)lo=mid+1;
        else hi=mid;
    }
    return lo-1;
}

__device__ __forceinline__ void load_q_tile(
    __nv_bfloat16* sQ,const __nv_bfloat16* q, int q_start,int q_len,
    int i0,int j,int64_t q_stride_tok, int64_t q_stride_head
){
    const int t=threadIdx.x;
    const int chunk=t%16;
    #pragma unroll
    for(int p=0;p<8;p++){
        const int r=p*8+t/16;
        const int h=r/kTileQ;
        const int i=i0+r%kTileQ;
        uint4 val=make_uint4(0,0,0,0);
        if(i<q_len){
            const __nv_bfloat16* src=q+static_cast<int64_t>(q_start+i)*q_stride_tok+static_cast<int64_t>(2*j+h)*q_stride_head+chunk*8;
            val=*reinterpret_cast<const uint4*>(src);
        }
        *reinterpret_cast<uint4*>(sQ+r*kPadH+chunk*8)=val;

    }
}

__device__ __forceinline__ void load_kv_tile(
    __nv_bfloat16* dst,const __nv_bfloat16* cache,const int32_t* bt_row,
    int n,int L,int j,int64_t kv_block_stride,int64_t kv_row_stride
){
    const int t=threadIdx.x;
    const int chunk=t%16;
    #pragma unroll
    for(int pass=0;pass<8;pass++){
        const int row=pass*8+t/16;
        const int p=n*kTileK+row;
        uint4 val=make_uint4(0,0,0,0);
        if(p<L){
            const int64_t phy=static_cast<int64_t>(bt_row[p/kKvBlockSize]);
            const int slot=p%kKvBlockSize;
            const __nv_bfloat16* src=cache+phy*kv_block_stride+static_cast<int64_t>(slot)*kv_row_stride+static_cast<int64_t>(j)*kHeadDim+chunk*8;
            val=*reinterpret_cast<const uint4*>(src);
        }
        *reinterpret_cast<uint4*>(dst+row*kPadH+chunk*8)=val;
    }
}

__device__ __forceinline__ void compute_scores(
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> (&qf)[kHeadDim/16],
    const __nv_bfloat16* sK,float* sS,int warp
){
    #pragma unroll
    for(int f=0;f<4;f++){
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.0f);
        #pragma unroll
        for(int k=0;k<8;k++){
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>kf;
            wmma::load_matrix_sync(kf,sK+(16*f)*kPadH+16*k,kPadH);
            wmma::mma_sync(c,qf[k],kf,c);
        }
        wmma::store_matrix_sync(sS+(16*warp)*kPadS+16*f,c,kPadS,wmma::mem_row_major);
    }
}

__device__ __forceinline__ void online_softmax(
    const float* sS,__nv_bfloat16* sP,float* sO,int r,int half,
    int i,int n,int start,int L,float scale,float& m,float& l
){
    float s[32];
    float mx=-FLT_MAX;
    #pragma unroll
    for(int c=0;c<32;c++){
        const int p=n*kTileK+32*half+c;
        float x=sS[r*kPadS+32*half+c]*scale;
        if (p>start+i || p>=L) x=-INFINITY;
        s[c]=x;
        mx=fmaxf(mx,x); 

    }
    mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
    const float m_new=fmaxf(m,mx);
    const float alpha=__expf(m-m_new);
    float sum=0.0f;
    #pragma unroll
    for(int c=0;c<32;c++){
        const float e=__expf(s[c]-m_new);
        sum+=e;
        sP[r*kPadP+32*half+c]=__float2bfloat16(e);
    }
    sum+=__shfl_xor_sync(0xffffffff,sum,1);
    l=l*alpha+sum;
    m=m_new;
    #pragma unroll
    for(int c=0;c<64;c++){
        sO[r*kPadO+64*half+c]*=alpha;            
    }
}

__device__ __forceinline__ void accumulate_pv(
    const __nv_bfloat16* sP,const __nv_bfloat16* sV,float* sO,int warp
){
    #pragma unroll
    for(int f=0;f<8;f++){
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::load_matrix_sync(c,sO+(16*warp)*kPadO+16*f,kPadO,wmma::mem_row_major);
        #pragma unroll
        for(int k=0;k<4;k++){                   
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
            wmma::load_matrix_sync(a,sP+(16*warp)*kPadP+16*k,kPadP);
            wmma::load_matrix_sync(b,sV+(16*k)*kPadH+16*f,kPadH);
            wmma::mma_sync(c,a,b,c);
        }

        wmma::store_matrix_sync(sO+(16*warp)*kPadO+16*f,c,kPadO,wmma::mem_row_major);


    }
}

__device__ __forceinline__ void store_output(
    __nv_bfloat16* out,const float* sO,int r,int half,float l,int q_start,
    int i,int q_len,int head,int64_t out_stride_tok,int64_t out_stride_head
){
    if(i>=q_len) return;

    const float inv=1.0f/l;
    __nv_bfloat16* dst=out
        +static_cast<int64_t>(q_start+i)*out_stride_tok
        +static_cast<int64_t>(head)*out_stride_head
        +64*half;

    #pragma unroll
    for(int c=0;c<64;c+=8){
        __align__(16) __nv_bfloat16 v8[8];
        #pragma unroll
        for(int e=0;e<8;e++){
            v8[e]=__float2bfloat16(sO[r*kPadO+64*half+c+e]*inv);
        }
        *reinterpret_cast<uint4*>(dst+c)=*reinterpret_cast<const uint4*>(v8);
    }
}

__global__ void __launch_bounds__(kNumWarps*32) paged_prefill_kernel(
    __nv_bfloat16* __restrict__ out,const __nv_bfloat16* __restrict__ q,const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ context_lens,const int32_t* __restrict__ block_tables,
    int64_t q_stride_tok,int64_t q_stride_head,int64_t out_stride_tok,int64_t out_stride_head,
    int64_t kv_block_stride,int64_t kv_row_stride,int num_seqs,int max_kv_blocks,float scale
){
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* sQ=reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK=sQ+kRows*kPadH;
    __nv_bfloat16* sV=sK+kRows*kPadH;
    float* sS=reinterpret_cast<float*>(sV+kRows*kPadH);
    __nv_bfloat16* sP=reinterpret_cast<__nv_bfloat16*>(sS+kRows*kPadS);
    float* sO=reinterpret_cast<float*>(sP+kRows*kPadP);

    const int g=blockIdx.x;
    const int j=blockIdx.y;
    const int b=find_seq(cu_seqlens_q,num_seqs,g);
    const int q_start=cu_seqlens_q[b];
    const int q_len=cu_seqlens_q[b+1]-q_start;
    const int i0=(g-(q_start/kTileQ+b))*kTileQ;
    if(i0>=q_len) return;  

    const int L=context_lens[b];
    const int start=L-q_len;
    const int i1=(i0+kTileQ<q_len?i0+kTileQ:q_len)-1;
    const int num_tiles=(start+i1+1+kTileK-1)/kTileK;
    const int32_t* bt_row=block_tables+static_cast<int64_t>(b)*max_kv_blocks;
    const int t=threadIdx.x;
    const int warp=t/32;
    const int lane=t%32;
    const int r=16*warp+lane/2;
    const int half=lane%2;
    const int i=i0+r%kTileQ;
    const int head=j*kGqa+r/kTileQ;

    load_q_tile(sQ,q,q_start,q_len,i0,j,q_stride_tok,q_stride_head);

    float m=-FLT_MAX;
    float l=0.0f;
    #pragma unroll
    for(int c=0;c<64;c++){
        sO[r*kPadO+64*half+c]=0.0f;
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qf[kHeadDim/16];
    #pragma unroll
    for(int k=0;k<kHeadDim/16;k++){
        wmma::load_matrix_sync(qf[k],sQ+(16*warp)*kPadH+16*k,kPadH);
    }

    for(int n=0;n<num_tiles;n++){
        __syncthreads();                        
        load_kv_tile(sK,k_cache,bt_row,n,L,j,kv_block_stride,kv_row_stride);
        load_kv_tile(sV,v_cache,bt_row,n,L,j,kv_block_stride,kv_row_stride);
        __syncthreads();                        

        compute_scores(qf,sK,sS,warp);
        __syncwarp();
        online_softmax(sS,sP,sO,r,half,i,n,start,L,scale,m,l);
        __syncwarp();
        accumulate_pv(sP,sV,sO,warp);
        __syncwarp();
    }

    store_output(out,sO,r,half,l,q_start,i,q_len,head,out_stride_tok,out_stride_head);


}

void paged_prefill(
    torch::stable::Tensor& out,const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache,const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& cu_seqlens_q,const torch::stable::Tensor& context_lens,
    const torch::stable::Tensor& block_tables,double scale
){
    const int64_t T=q.size(0);
    if(T==0) return;
    const int64_t B=context_lens.size(0);

    using torch::headeronly::ScalarType;
    STD_TORCH_CHECK(out.scalar_type()==ScalarType::BFloat16 && q.scalar_type()==ScalarType::BFloat16 &&
                    k_cache.scalar_type()==ScalarType::BFloat16 && v_cache.scalar_type()==ScalarType::BFloat16,
                    "paged_prefill: out, q, k_cache, v_cache must be bfloat16");
    STD_TORCH_CHECK(cu_seqlens_q.scalar_type()==ScalarType::Int &&
                    context_lens.scalar_type()==ScalarType::Int &&
                    block_tables.scalar_type()==ScalarType::Int,
                    "paged_prefill: cu_seqlens_q, context_lens, block_tables must be int32");
    STD_TORCH_CHECK(q.size(2)==kHeadDim && k_cache.size(3)==kHeadDim,
                    "paged_prefill: head_dim must be 128");
    STD_TORCH_CHECK(k_cache.size(1)==kKvBlockSize,
                    "paged_prefill: kv_block size must be 16");
    STD_TORCH_CHECK(q.size(1)==k_cache.size(2)*kGqa,
                    "paged_prefill: num_heads must be 2 x num_kv_heads");
    STD_TORCH_CHECK(q.stride(2)==1 && out.stride(2)==1,
                    "paged_prefill: last dim of q/out must be contiguous");
    STD_TORCH_CHECK(q.stride(0)%8==0 && q.stride(1)%8==0,
                    "paged_prefill: q strides must allow 16-byte loads");
    STD_TORCH_CHECK(out.stride(0)%8==0 && out.stride(1)%8==0,
                    "paged_prefill: out strides must allow 16-byte stores");
    STD_TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() &&
                    block_tables.is_contiguous(),
                    "paged_prefill: caches and block_tables must be contiguous");
    STD_TORCH_CHECK(cu_seqlens_q.size(0)==B+1,
                    "paged_prefill: cu_seqlens_q must have B+1 entries");

    void* stream_ptr=nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(q.get_device_index(),&stream_ptr));
    cudaStream_t stream=static_cast<cudaStream_t>(stream_ptr);

    static bool smem_set=false;
    if(!smem_set){
        STD_CUDA_CHECK(cudaFuncSetAttribute(
            paged_prefill_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmemBytes));
        smem_set=true;
    }

    const dim3 grid(static_cast<unsigned int>(T/kTileQ+B),
                    static_cast<unsigned int>(k_cache.size(2)));

    paged_prefill_kernel<<<grid,kNumWarps*32,kSmemBytes,stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.const_data_ptr()),
        cu_seqlens_q.const_data_ptr<int32_t>(),
        context_lens.const_data_ptr<int32_t>(),
        block_tables.const_data_ptr<int32_t>(),
        q.stride(0),q.stride(1),
        out.stride(0),out.stride(1),
        k_cache.stride(0),k_cache.stride(1),
        static_cast<int>(B),
        static_cast<int>(block_tables.size(1)),
        static_cast<float>(scale)
    );
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}