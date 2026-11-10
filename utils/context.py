import torch
from dataclasses import dataclass
@dataclass
class Context:
    cu_seqlens_q:torch.Tensor|None=None          
    max_seqlen_q:int=0
    max_seqlen_k:int=0
    slot_mapping:torch.Tensor|None=None          
    context_lens:torch.Tensor|None=None          
    block_tables:torch.Tensor|None=None          
    num_decodes:int=0                            
    decode_max_seqlen_k:int=0                   
    prefill_cu_seqlens_q:torch.Tensor|None=None  
    logits_indices:torch.Tensor|None=None        

_CONTEXT=Context()

def get_context():
    return _CONTEXT

def set_context(**kwargs):
    global _CONTEXT
    _CONTEXT=Context(**kwargs)

def reset_context():
    global _CONTEXT
    _CONTEXT=Context()
    return _CONTEXT