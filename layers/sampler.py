import torch
import torch.nn as nn

class Sampler(nn.Module):
    def forward(self,logits:torch.Tensor,temperatures:torch.Tensor)->torch.Tensor:
        logits=logits.float()
        greedy=logits.argmax(dim=-1)
        t=torch.where(temperatures==0,1.0,temperatures)
        probs=torch.softmax(logits/t.unsqueeze(1),dim=-1)
        sampled=(probs/torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        return torch.where(temperatures==0,greedy,sampled)

