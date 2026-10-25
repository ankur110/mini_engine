# miniengine/utils/context.py
# §9 theory: one global with this step's batch metadata. Runner sets it, 28 layers read it.
import torch
from dataclasses import dataclass
@dataclass
class Context:
    is_prefill:bool=False
    cu_seqlens_q:torch.Tensor|None=None   
    max_seqlen_q:int=0                    
    max_seqlen_k:int=0                    
    slot_mapping:torch.Tensor|None=None   
    context_lens:torch.Tensor|None=None   
    block_tables:torch.Tensor|None=None   

_CONTEXT=Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill,cu_seqlens_q=None,max_seqlen_q=0,max_seqlen_k=0,
                slot_mapping=None,context_lens=None,block_tables=None):
    global _CONTEXT
    _CONTEXT=Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables
    )

def reset_context():
    global _CONTEXT
    _CONTEXT=Context()
    return _CONTEXT