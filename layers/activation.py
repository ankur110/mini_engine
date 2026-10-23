import torch
import torch.nn.functional as F
import torch.nn as nn
class SiluAndMul(nn.Module):
    def forward(self,x):
        gate,up=x.chunk(2,dim=-1)
        return F.silu(gate)*up
