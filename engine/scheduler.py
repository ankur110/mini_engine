from collections import deque
from config import Config
from engine.sequence import SequenceStatus,Sequence
from engine.block_manager import BlockManager


class Scheduler:

    def __init__(self,config:Config):
        self.max_num_seqs=config.max_num_seqs
        self.max_num_batched_tokens=config.max_num_batched_tokens
        self.eos=config.eos
        self.block_size=config.kvcache_block_size

        self.block_manager=BlockManager(config.num_kvcache_blocks,config.kvcache_block_size)
        self.waiting:deque[Sequence]=deque()
        self.running:deque[Sequence]=deque()


    def is_finished(self)->bool:
        return not self.waiting and not self.running

    def add(self,seq:Sequence):
        total=seq.num_prompt_tokens+seq.sampling_params.max_tokens
        if total>self.max_num_batched_tokens:
            raise ValueError(f"total tokens should be less than max batched tokens")
        num_blocks=(total+self.block_size-1)//self.block_size
        if num_blocks>len(self.block_manager.blocks):
            raise ValueError(f"needs {num_blocks} KV blocks, pool has {len(self.block_manager.blocks)}")
        self.waiting.append(seq)

    def schedule(self)->tuple[list[Sequence],bool]:
        scheduled=[]
        num_batched_tokens=0
        while self.waiting and len(self.running)<self.max_num_seqs:
            seq=self.waiting[0]
            if(num_batched_tokens+len(seq)>self.max_num_batched_tokens or not self.block_manager.can_allocate(seq)):
                break
            self.block_manager.allocate(seq)
            num_batched_tokens+=(len(seq)-seq.num_cached_tokens)
            seq.status=SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)

        if scheduled:
            return scheduled,True 

        while self.running:
            seq=self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        assert scheduled
        self.running.extendleft(reversed(scheduled))
        return scheduled,False

    def preempt(self,seq:Sequence):
        seq.status=SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self,seqs:list[Sequence],token_ids:list[int]):
        for seq,token_id in zip(seqs,token_ids):
            seq.append_token(token_id)
            finished =(token_id==self.eos and not seq.sampling_params.ignore_eos) or seq.num_completion_tokens==seq.sampling_params.max_tokens
            if finished:
                seq.status=SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)