import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.context import get_context
class VocabEmbedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim))

    def forward(self,input_ids):
        return F.embedding(input_ids,self.weight)


class LMHead(nn.Module):
    def __init__(self,num_embeddings,embedding_dim):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim))

    def forward(self,hidden_states):
        context=get_context()
        if context.is_prefill:
            hidden_states=hidden_states[context.cu_seqlens_q[1:].long()-1]
        return F.linear(hidden_states,self.weight)

