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

class ModelRunner:
    def __init__(self,config:Config):
        self.config=config
        hf=config.hf_config

        self.block_size=config.kvcache_block_size
        assert Sequence.block_size==self.block_size,(
            f"Sequence.block_size={Sequence.block_size} != "
            f"config.kvcache_block_size={self.block_size}"
        )
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
        assert num_blocks>0,"warm-up peak ate the budget: lower max_num_seqs / max_model_len"
        config.num_kvcache_blocks=num_blocks

        self.kv_cache=allocate_kv_cache(self.model,num_blocks,self.block_size,
                                        hf.num_key_value_heads,hf.head_dim,self.dtype,"cuda")

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        L=self.config.max_model_len

        def make_seqs(n):
            seqs=[Sequence([0]*L) for _ in range(n)]
            for seq in seqs:
                seq.block_table=[0]*seq.num_blocks
            return seqs

        n=max(1,min(self.config.max_num_batched_tokens//L,self.config.max_num_seqs))
        self.run(make_seqs(n),True)

        self.run(make_seqs(self.config.max_num_seqs),False)

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

        print(
            f"total={total} used={used} current={current} peak={peak} "
            f"num_blocks={num_blocks} "
            f"pool_tokens={num_blocks*self.block_size}"
        )
        return num_blocks

    def prepare_block_tables(self,seqs:list[Sequence])->torch.Tensor:
        max_len=max(len(seq.block_table) for seq in seqs)
        rows=[seq.block_table+[-1]*(max_len-len(seq.block_table)) for seq in seqs]
        return torch.tensor(rows,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self,seqs:list[Sequence])->tuple[torch.Tensor,torch.Tensor]:
        input_ids=[]
        positions=[]
        cu_seqlens_q=[0]
        slot_mapping=[]
        context_lens=[]
        max_seqlen_q=0
        max_seqlen_k=0
        for seq in seqs:
            cached=seq.num_cached_tokens
            n=len(seq)
            q_len=n-cached
            input_ids.extend(seq[cached:])
            positions.extend(range(cached,n))
            cu_seqlens_q.append(cu_seqlens_q[-1]+q_len)
            max_seqlen_q=max(max_seqlen_q,q_len)
            context_lens.append(n)
            max_seqlen_k=max(max_seqlen_k,n)
            for i in range(seq.num_cached_blocks,seq.num_blocks):
                start=seq.block_table[i]*self.block_size
                if(i!=seq.num_blocks-1):
                    end=start+self.block_size
                else:
                    end=start+seq.last_block_num_tokens
                slot_mapping.extend(range(start,end))
        assert len(slot_mapping)==len(input_ids)
        input_ids=torch.tensor(input_ids,dtype=torch.int64,pin_memory=True).cuda(non_blocking=True)
        positions=torch.tensor(positions,dtype=torch.int64,pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q=torch.tensor(cu_seqlens_q,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)
        slot_mapping=torch.tensor(slot_mapping,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)
        context_lens=torch.tensor(context_lens,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)

        block_tables=self.prepare_block_tables(seqs)
        set_context(True,cu_seqlens_q,max_seqlen_q,max_seqlen_k,slot_mapping,context_lens,block_tables)
        return input_ids,positions

    def prepare_decode(self,seqs:list[Sequence])->tuple[torch.Tensor,torch.Tensor]:
        input_ids=[]
        positions=[]
        context_lens=[]
        slot_mapping=[]
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1]*self.block_size+seq.last_block_num_tokens-1)
        cu_seqlens_q=list(range(len(seqs)+1))
        max_seqlen_q=1
        max_seqlen_k=max(context_lens)
        input_ids=torch.tensor(input_ids,dtype=torch.int64,pin_memory=True).cuda(non_blocking=True)
        positions=torch.tensor(positions,dtype=torch.int64,pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q=torch.tensor(cu_seqlens_q,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)
        slot_mapping=torch.tensor(slot_mapping,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)
        context_lens=torch.tensor(context_lens,dtype=torch.int32,pin_memory=True).cuda(non_blocking=True)

        block_tables=self.prepare_block_tables(seqs)
        set_context(False,cu_seqlens_q,max_seqlen_q,max_seqlen_k,slot_mapping,context_lens,block_tables)
        return input_ids,positions

    def prepare_sample(self,seqs:list[Sequence])->torch.Tensor:
        temperatures=[seq.sampling_params.temperature for seq in seqs]
        return torch.tensor(temperatures,dtype=torch.float32,pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self,input_ids:torch.Tensor,positions:torch.Tensor)->torch.Tensor:
        hidden=self.model(input_ids,positions)
        return self.model.compute_logits(hidden)

    def run(self,seqs:list[Sequence],is_prefill:bool)->list[int]:
        if is_prefill:
            input_ids,positions=self.prepare_prefill(seqs)
        else:
            input_ids,positions=self.prepare_decode(seqs)
        temperatures=self.prepare_sample(seqs)
        logits=self.run_model(input_ids,positions)
        token_ids=self.sampler(logits,temperatures).tolist()
        reset_context()
        return token_ids