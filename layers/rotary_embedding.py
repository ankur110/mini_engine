import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1=x[...,:x.shape[-1]//2]
    x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim,max_position,base):
        super().__init__()
        inv_freq=1.0/(base**(torch.arange(0,head_dim,2).float()/head_dim))
        t=torch.arange(max_position).float()
        freqs=torch.outer(t,inv_freq)
        emb=torch.cat((freqs,freqs),dim=-1)
        self.register_buffer("cos",emb.cos(),persistent=False)
        self.register_buffer("sin",emb.sin(),persistent=False)

    def forward(self,positions,q,k):
        cos=self.cos[positions].unsqueeze(1).to(q.dtype)
        sin=self.sin[positions].unsqueeze(1).to(q.dtype)
        q_out=q*cos+rotate_half(q)*sin
        k_out=k*cos+rotate_half(k)*sin
        return q_out,k_out
    
    
