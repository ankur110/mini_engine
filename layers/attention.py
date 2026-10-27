import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.context import get_context

def store_kvcache(k,v,k_cache,v_cache,slot_mapping):
    kc=k_cache.view(-1,k_cache.shape[-2],k_cache.shape[-1])
    vc=v_cache.view(-1,v_cache.shape[-2],v_cache.shape[-1])
    keep=slot_mapping>=0
    kc[slot_mapping[keep]]=k[keep]
    vc[slot_mapping[keep]]=v[keep]




def paged_attention(q,k_cache,v_cache,context,scale):
    bt=context.block_tables.clamp(min=0).long()
    k=k_cache[bt].flatten(1,2)[:,:context.max_seqlen_k]
    v=v_cache[bt].flatten(1,2)[:,:context.max_seqlen_k]
    cu=context.cu_seqlens_q
    q_lens=cu[1:]-cu[:-1]
    B=context.block_tables.shape[0]
    seq_idx=torch.repeat_interleave(torch.arange(B,device=q.device),q_lens,output_size=q.shape[0])
    local=torch.arange(q.shape[0],device=q.device)-cu[seq_idx]
    q_pad=q.new_zeros(B,context.max_seqlen_q,q.shape[1],q.shape[2])
    q_pad[seq_idx,local]=q  

    i=torch.arange(context.max_seqlen_q, device=q.device)[None, :, None]
    j=torch.arange(context.max_seqlen_k, device=q.device)[None, None, :]
    start=(context.context_lens - q_lens)[:, None, None]        # tokens cached before this step
    mask = (j <= start + i) & (j < context.context_lens[:, None, None])
    o=F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask[:, None], scale=scale, enable_gqa=True,
    )
    return o.transpose(1, 2)[seq_idx, local]   

class Attention(nn.Module):
    def __init__(self,num_heads,head_dim,scale,num_kv_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.scale=scale
        self.num_kv_heads=num_kv_heads
        self.k_cache=self.v_cache=torch.tensor([])


    def forward(self,q,k,v):
        context=get_context()
        if self.k_cache.numel() == 0 or context.slot_mapping is None:
            q=q.transpose(0,1).unsqueeze(0)
            k=k.transpose(0,1).unsqueeze(0)
            v=v.transpose(0,1).unsqueeze(0)
            o=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.scale,enable_gqa=True)
            return o.squeeze(0).transpose(0,1)
        store_kvcache(k,v,self.k_cache,self.v_cache,context.slot_mapping)
        return paged_attention(q,self.k_cache,self.v_cache,context,self.scale)