import os
from dataclasses import dataclass
from transformers import AutoConfig,PretrainedConfig
@dataclass
class Config:
    model:str
    kvcache_block_size:int=16
    hf_config:PretrainedConfig|None=None
    num_kvcache_blocks:int=-1
    max_num_seqs:int=512
    max_num_batched_tokens:int=16384             
    eos:int=-1
    max_model_len:int=4096
    gpu_memory_utilization:float=0.9
    enable_chunked_prefill:bool=True             
    long_prefill_token_threshold:int=0           
    scheduling_policy:str="fcfs"                 

    def __post_init__(self):
        self.model=os.path.expanduser(self.model)
        assert os.path.isdir(self.model)
        assert self.max_num_batched_tokens>=self.max_num_seqs   
        assert self.scheduling_policy in ("fcfs","priority")
        assert self.long_prefill_token_threshold>=0
        self.hf_config=AutoConfig.from_pretrained(self.model)
        self.max_model_len=min(self.max_model_len,self.hf_config.max_position_embeddings)
        if not self.enable_chunked_prefill:                    
            assert self.max_num_batched_tokens>=self.max_model_len
        assert 0<self.gpu_memory_utilization<=1