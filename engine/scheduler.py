from collections import deque
from config import Config
from engine.sequence import Sequence,SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:

    def __init__(self,config:Config):
        self.max_num_seqs=config.max_num_seqs
        self.max_num_batched_tokens=config.max_num_batched_tokens
        self.long_prefill_token_threshold=config.long_prefill_token_threshold
        self.enable_chunked_prefill=config.enable_chunked_prefill
        self.policy=config.scheduling_policy
        self.eos=config.eos
        self.block_size=config.kvcache_block_size
        self.max_model_len=config.max_model_len
        self.block_manager=BlockManager(config.num_kvcache_blocks,config.kvcache_block_size)
        self.waiting:deque[Sequence]=deque()
        self.running:list[Sequence]=[]          
        self.num_preemptions=0

    def is_finished(self)->bool:
        return not self.waiting and not self.running

    def add(self,seq:Sequence):
        total=seq.num_prompt_tokens+seq.sampling_params.max_tokens
        if total>self.max_model_len:
            raise ValueError(f"prompt+max_tokens={total} exceeds max_model_len={self.max_model_len}")
        num_blocks=(total+self.block_size-1)//self.block_size
        if num_blocks>len(self.block_manager.blocks):
            raise ValueError(f"needs {num_blocks} KV blocks, pool has {len(self.block_manager.blocks)}")
        self._push_waiting(seq)


    @staticmethod
    def _key(seq:Sequence):
        return (seq.priority,seq.arrival_time)

    def _push_waiting(self,seq:Sequence,front:bool=False):
        if self.policy=="priority":
            i=0
            while i<len(self.waiting) and self._key(self.waiting[i])<=self._key(seq):
                i+=1
            self.waiting.insert(i,seq)
        elif front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def _cap(self,owed:int,budget:int)->int:
        n=owed
        if 0<self.long_prefill_token_threshold<n:
            n=self.long_prefill_token_threshold
        return min(n,budget)

    def _pick_victim(self,scheduled_ids:set[int])->Sequence:

        cands=[s for s in self.running if s.seq_id not in scheduled_ids]
        if self.policy=="priority":
            return max(cands,key=self._key)
        return cands[-1]                         

    def _preempt(self,seq:Sequence):
        self.running.remove(seq)
        seq.status=SequenceStatus.WAITING
        seq.num_scheduled_tokens=0
        self.block_manager.deallocate(seq)      
        self._push_waiting(seq,front=True)
        self.num_preemptions+=1

    def _allocate_or_preempt(self,seq:Sequence,n:int,scheduled_ids:set[int])->bool:
        while not self.block_manager.allocate_slots(seq,n):
            victim=self._pick_victim(scheduled_ids)
            self._preempt(victim)
            if victim is seq:
                return False
        return True

    def _start(self,seq:Sequence,n:int,scheduled:list[Sequence],ids:set[int]):
        self.waiting.popleft()
        seq.status=SequenceStatus.RUNNING
        seq.num_scheduled_tokens=n
        self.running.append(seq)
        scheduled.append(seq)
        ids.add(seq.seq_id)


    def schedule(self)->list[Sequence]:
        if self.enable_chunked_prefill:
            scheduled=self._schedule_chunked()
        else:
            scheduled=self._schedule_prefill_first()
        assert scheduled,"nothing schedulable: KV pool or token budget too small"
        return scheduled

    def _schedule_chunked(self)->list[Sequence]:
        budget=self.max_num_batched_tokens
        scheduled=[]
        ids=set()
        preempted=False

        decodes=[s for s in self.running if s.num_uncomputed_tokens==1]
        partials=[s for s in self.running if s.num_uncomputed_tokens>1]
        for seq in decodes+partials:
            if budget==0:
                break
            if seq.status!=SequenceStatus.RUNNING:   
                continue
            n=self._cap(seq.num_uncomputed_tokens,budget)
            before=self.num_preemptions
            ok=self._allocate_or_preempt(seq,n,ids)
            preempted|=self.num_preemptions>before
            if not ok:
                break
            seq.num_scheduled_tokens=n
            budget-=n
            scheduled.append(seq)
            ids.add(seq.seq_id)
        if preempted:                                
            return scheduled

        while self.waiting and budget>0 and len(self.running)<self.max_num_seqs:
            seq=self.waiting[0]
            hit=self.block_manager.find_prefix_hit(seq)
            n=self._cap(seq.num_tokens-len(hit)*self.block_size,budget)
            if not self.block_manager.allocate_slots(seq,n,hit):
                break
            self._start(seq,n,scheduled,ids)
            budget-=n
        return scheduled

    def _schedule_prefill_first(self)->list[Sequence]:
        budget=self.max_num_batched_tokens
        scheduled=[]
        ids=set()
        while self.waiting and len(self.running)<self.max_num_seqs:
            seq=self.waiting[0]
            hit=self.block_manager.find_prefix_hit(seq)
            n=seq.num_tokens-len(hit)*self.block_size
            if n>budget or not self.block_manager.allocate_slots(seq,n,hit):
                break
            self._start(seq,n,scheduled,ids)
            budget-=n
        if scheduled:
            return scheduled
        for seq in list(self.running):
            if seq.status!=SequenceStatus.RUNNING:
                continue
            if not self._allocate_or_preempt(seq,1,ids):
                break
            seq.num_scheduled_tokens=1
            scheduled.append(seq)
            ids.add(seq.seq_id)
        return scheduled

    def postprocess(self,seqs:list[Sequence],token_ids:list[int|None])->list[Sequence]:
        emitted=[]
        for seq,token_id in zip(seqs,token_ids):
            sampled=seq.samples_this_step                
            seq.num_computed_tokens+=seq.num_scheduled_tokens
            seq.num_scheduled_tokens=0
            self.block_manager.cache_full_blocks(seq)    
            if not sampled:                              
                continue
            seq.append_token(token_id)                   
            emitted.append(seq)
            done=(token_id==self.eos and not seq.sampling_params.ignore_eos) or \
                 seq.num_completion_tokens==seq.sampling_params.max_tokens
            if done:
                seq.status=SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return emitted