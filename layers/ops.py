import os
import torch
from torch.utils.cpp_extension import load

CSRC=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"csrc")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST","8.0")   
STABLE="-DTORCH_TARGET_VERSION=0x020a000000000000"     
USE_CUDA="-DUSE_CUDA"   


load(
    name="mini_engine_ops",
    sources=[os.path.join(CSRC,f) for f in ("ops.cpp","kv_cache.cu","paged_decode.cu","paged_prefill.cu")],
    extra_cflags=["-O3",STABLE,USE_CUDA],
    extra_cuda_cflags=["-O3","-lineinfo","-Xptxas=-v",STABLE,USE_CUDA],   
    is_python_module=False,   
    verbose=True,
)

PARTITION_SIZE=256   
NUM_SMS=torch.cuda.get_device_properties(0).multi_processor_count   

def store_kvcache(key,value,k_cache,v_cache,slot_mapping):
    torch.ops.mini_engine.store_kvcache(key,value,k_cache,v_cache,slot_mapping)

def paged_decode_v1(q,k_cache,v_cache,block_tables,context_lens,scale,out=None):
    if out is None:
        out=torch.empty(q.shape,dtype=q.dtype,device=q.device)
    torch.ops.mini_engine.paged_decode(out,q,k_cache,v_cache,block_tables,context_lens,float(scale))
    return out

def paged_decode_v2(q,k_cache,v_cache,block_tables,context_lens,scale,max_seqlen_k,out=None):
    num_seqs,num_heads,head_dim=q.shape
    max_parts=(max_seqlen_k+PARTITION_SIZE-1)//PARTITION_SIZE
    if out is None:
        out=torch.empty(q.shape,dtype=q.dtype,device=q.device)
    tmp_out=torch.empty(num_seqs,num_heads,max_parts,head_dim,dtype=torch.float32,device=q.device)
    max_logits=torch.empty(num_seqs,num_heads,max_parts,dtype=torch.float32,device=q.device)
    exp_sums=torch.empty_like(max_logits)
    torch.ops.mini_engine.paged_decode_v2(out,q,k_cache,v_cache,block_tables,context_lens,
                                          tmp_out,max_logits,exp_sums,float(scale))
    return out

def paged_decode(q,k_cache,v_cache,block_tables,context_lens,scale,max_seqlen_k,out=None):
    num_seqs=q.shape[0]
    num_kv_heads=k_cache.shape[2]
    max_parts=(max_seqlen_k+PARTITION_SIZE-1)//PARTITION_SIZE
    use_v1=max_seqlen_k<=8192 and (max_parts==1 or num_seqs*num_kv_heads>=2*NUM_SMS)
    if use_v1:
        return paged_decode_v1(q,k_cache,v_cache,block_tables,context_lens,scale,out)
    return paged_decode_v2(q,k_cache,v_cache,block_tables,context_lens,scale,max_seqlen_k,out)

def paged_prefill(q,k_cache,v_cache,cu_seqlens_q,context_lens,block_tables,scale,out=None):
    if out is None:
        out=torch.empty(q.shape,dtype=q.dtype,device=q.device)
    torch.ops.mini_engine.paged_prefill(out,q,k_cache,v_cache,cu_seqlens_q,context_lens,block_tables,float(scale))
    return out