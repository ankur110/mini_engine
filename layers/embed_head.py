import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return F.linear(hidden_states,self.weight)