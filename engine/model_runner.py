import torch
from config import Config
from engine.sequence import Sequence
from layers.sampler import Sampler
from models.qwen3 import Qwen3ForCausalLM
from utils.context import set_context,reset_context
from utils.loader import load_model

def allocate_kv_cache(model,num_blocks,block_size,num_kv_heads,head_dim,dtype,device):
    mods=[m for m in model.modules() if hasattr(m,"k_cache")]
    num_layers=len(getattr(model,"model",model).layers)
    assert len(mods)==num_layers
    kv=torch.zeros(2,len(mods),num_blocks,block_size,num_kv_heads,head_dim,dtype=dtype,device=device)
    for i,m in enumerate(mods):
        m.k_cache=kv[0,i]
        m.v_cache=kv[1,i]
    return kv


def to_gpu(x,dtype):
    return torch.tensor(x,dtype=dtype,pin_memory=True).cuda(non_blocking=True)


class ModelRunner:
    def __init__(self,config:Config):
        self.config=config
        hf=config.hf_config
        self.block_size=config.kvcache_block_size
        assert Sequence.block_size==self.block_size,(
            f"Sequence.block_size={Sequence.block_size} != config.kvcache_block_size={self.block_size}")
        self.dtype=torch.bfloat16

        old=torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        with torch.device("cuda"):
            self.model=Qwen3ForCausalLM(hf)
        torch.set_default_dtype(old)

        load_model(self.model,config.model)
        self.model.eval()
        self.sampler=Sampler()

        allocate_kv_cache(self.model,1,self.block_size,hf.num_key_value_heads,hf.head_dim,self.dtype,"cuda")
        self.warmup_model()

        num_blocks=self.size_kv_pool()
        assert num_blocks>0,"warm-up peak ate the budget: lower max_num_seqs / max_num_batched_tokens"
        config.num_kvcache_blocks=num_blocks
        self.kv_cache=allocate_kv_cache(self.model,num_blocks,self.block_size,
                                        hf.num_key_value_heads,hf.head_dim,self.dtype,"cuda")

    def _dummy(self,num_tokens:int,num_computed:int,num_scheduled:int)->Sequence:
        seq=Sequence([0]*num_tokens)
        seq.num_computed_tokens=num_computed
        seq.num_scheduled_tokens=num_scheduled
        seq.block_table=[0]*seq.num_blocks          
        return seq

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        L=self.config.max_model_len
        m=min(L,self.config.max_num_batched_tokens)
        n=max(1,min(self.config.max_num_batched_tokens//m,self.config.max_num_seqs))
        self.run([self._dummy(m,0,m) for _ in range(n)])                              
        self.run([self._dummy(L,L-1,1) for _ in range(self.config.max_num_seqs)])     
        torch.cuda.empty_cache()

    def size_kv_pool(self)->int:
        free,total=torch.cuda.mem_get_info()
        used=total-free
        stats=torch.cuda.memory_stats()
        peak=stats["allocated_bytes.all.peak"]
        current=stats["allocated_bytes.all.current"]
        hf=self.config.hf_config
        block_bytes=2*hf.num_hidden_layers*self.block_size*hf.num_key_value_heads*hf.head_dim*self.dtype.itemsize
        num_blocks=int(total*self.config.gpu_memory_utilization-used-peak+current)//block_bytes
        print(f"total={total} used={used} current={current} peak={peak} "
              f"num_blocks={num_blocks} pool_tokens={num_blocks*self.block_size}")
        return num_blocks

    def prepare_block_tables(self,seqs:list[Sequence])->torch.Tensor:
        max_len=max(len(seq.block_table) for seq in seqs)
        rows=[seq.block_table+[-1]*(max_len-len(seq.block_table)) for seq in seqs]
        return to_gpu(rows,torch.int32)

    def prepare(self,seqs:list[Sequence]):
        decodes=[s for s in seqs if s.num_scheduled_tokens==1]
        order=decodes+[s for s in seqs if s.num_scheduled_tokens>1]   
        bs=self.block_size
        input_ids=[]
        positions=[]
        slot_mapping=[]
        context_lens=[]
        cu=[0]
        logits_indices=[]
        max_q=0
        max_k=0
        for seq in order:
            start=seq.num_computed_tokens
            end=start+seq.num_scheduled_tokens
            input_ids.extend(seq[start:end])
            positions.extend(range(start,end))
            bt=seq.block_table
            slot_mapping.extend(bt[p//bs]*bs+p%bs for p in range(start,end))
            context_lens.append(end)
            cu.append(cu[-1]+end-start)
            if seq.samples_this_step:
                logits_indices.append(cu[-1]-1)
            max_q=max(max_q,end-start)
            max_k=max(max_k,end)
        D=len(decodes)
        assert len(slot_mapping)==len(input_ids)
        set_context(
            cu_seqlens_q=to_gpu(cu,torch.int32),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            slot_mapping=to_gpu(slot_mapping,torch.int32),
            context_lens=to_gpu(context_lens,torch.int32),
            block_tables=self.prepare_block_tables(order),
            num_decodes=D,
            decode_max_seqlen_k=max(context_lens[:D],default=0),
            prefill_cu_seqlens_q=to_gpu([x-D for x in cu[D:]],torch.int32) if D<len(order) else None,
            logits_indices=to_gpu(logits_indices,torch.int64),
        )
        return to_gpu(input_ids,torch.int64),to_gpu(positions,torch.int64),order

    def prepare_sample(self,seqs:list[Sequence])->torch.Tensor:
        temperatures=[seq.sampling_params.temperature for seq in seqs]
        return to_gpu(temperatures,torch.float32)

    @torch.inference_mode()
    def run_model(self,input_ids:torch.Tensor,positions:torch.Tensor,need_logits:bool):
        hidden=self.model(input_ids,positions)
        return self.model.compute_logits(hidden) if need_logits else None

    def run(self,seqs:list[Sequence])->list[int|None]:
        input_ids,positions,order=self.prepare(seqs)
        samplers=[s for s in order if s.samples_this_step]
        logits=self.run_model(input_ids,positions,bool(samplers))
        sampled={}
        if samplers:
            token_ids=self.sampler(logits,self.prepare_sample(samplers)).tolist()
            sampled={s.seq_id:t for s,t in zip(samplers,token_ids)}
        reset_context()
        return [sampled.get(s.seq_id) for s in seqs]