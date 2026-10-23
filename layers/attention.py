import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,num_heads,head_dim,scale,num_kv_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.scale=scale
        self.num_kv_heads=num_kv_heads


    def forward(self,q,k,v):
        q=q.transpose(0,1).unsqueeze(0)
        k=k.transpose(0,1).unsqueeze(0)
        v=v.transpose(0,1).unsqueeze(0)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.scale,enable_gqa=True)
        return o.squeeze(0).transpose(0,1)