import torch
from torch import nn
import torch.nn.functional as F
class QKVLinear(nn.Module):
    def __init__(self,hidden_size,head_dim,num_heads,num_kv_heads):
        super().__init__()
        self.q_size=num_heads*head_dim
        self.kv_size=num_kv_heads*head_dim
        rows=self.q_size+2*self.kv_size
        self.weight=nn.Parameter(torch.empty(rows,hidden_size))
        self.weight.weight_loader=self.weight_loader

    def weight_loader(self,param,loaded_weight,shard_id):
        if shard_id=="q":
            start,size=0,self.q_size
        elif shard_id=="k":
            start,size=self.q_size,self.kv_size
        elif shard_id=="v":
            start,size=self.q_size+self.kv_size,self.kv_size
        rows=param.data.narrow(0,start,size)
        assert rows.shape==loaded_weight.shape
        rows.copy_(loaded_weight)


    def forward(self,x):
        return F.linear(x,self.weight)


class MergedLinear(nn.Module):
    def __init__(self,input_size,output_sizes):
        super().__init__()
        self.output_sizes=output_sizes
        rows=sum(self.output_sizes)
        self.weight=nn.Parameter(torch.empty(rows,input_size))
        self.weight.weight_loader=self.weight_loader

    def weight_loader(self,param,loaded_weight,shard_id):
        start=sum(self.output_sizes[:shard_id])
        size=self.output_sizes[shard_id]
        rows=param.data.narrow(0,start,size)
        assert rows.shape==loaded_weight.shape
        rows.copy_(loaded_weight)

    def forward(self,x):
        return F.linear(x,self.weight)