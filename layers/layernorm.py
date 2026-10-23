import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(hidden_size))

    def rms(self,x):
        dtype=x.dtype
        x=x.float()
        var=x.pow(2).mean(-1,keepdim=True)
        x=x*torch.rsqrt(var+self.eps)
        return self.weight*x.to(dtype)
    

    def forward(self,x,residual=None):

        if residual is None:
            return self.rms(x)
        residual=x+residual
        return self.rms(residual),residual