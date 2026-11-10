from collections import deque
import xxhash
import numpy as np
from engine.sequence import Sequence

class Block:

    def __init__(self,block_id:int):
        self.block_id=block_id
        self.ref_count=0
        self.hash=-1
        self.token_ids=[]

    def update(self,hash:int,token_ids:list[int]):
        self.hash=hash
        self.token_ids=token_ids

    def reset(self):
        self.ref_count=1
        self.hash=-1
        self.token_ids=[]


class BlockManager:

    def __init__(self,num_blocks:int,block_size:int):
        self.block_size=block_size
        self.blocks=[Block(i) for i in range(num_blocks)]
        self.hash_to_block_id={}
        self.free_block_ids=deque(range(num_blocks))    
        self.used_block_ids=set()

    @classmethod
    def compute_hash(cls,token_ids:list[int],prefix:int=-1)->int:
        h=xxhash.xxh64()
        if prefix!=-1:
            h.update(prefix.to_bytes(8,"little"))
        h.update(np.array(token_ids,dtype=np.int64).tobytes())
        return h.intdigest()

    @property
    def num_free_blocks(self)->int:
        return len(self.free_block_ids)

    def _allocate_block(self,block_id:int)->Block:
        block=self.blocks[block_id]
        assert block.ref_count==0
        if block.hash!=-1 and self.hash_to_block_id.get(block.hash)==block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self,block_id:int):
        block=self.blocks[block_id]
        assert block.ref_count==0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _touch(self,block_id:int):
        block=self.blocks[block_id]
        if block.ref_count==0:
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        block.ref_count+=1

    def find_prefix_hit(self,seq:Sequence)->list[int]:
        hit=[]
        h=-1
        for i in range((seq.num_tokens-1)//self.block_size):
            token_ids=seq.block(i)
            h=self.compute_hash(token_ids,h)
            block_id=self.hash_to_block_id.get(h,-1)
            if block_id==-1 or self.blocks[block_id].token_ids!=token_ids:
                break
            hit.append(block_id)
        return hit

    def allocate_slots(self,seq:Sequence,num_new_tokens:int,hit_blocks:list[int]|None=None)->bool:
        hit_blocks=hit_blocks or []
        assert not hit_blocks or not seq.block_table
        bs=self.block_size
        computed=seq.num_computed_tokens+len(hit_blocks)*bs
        total=(computed+num_new_tokens+bs-1)//bs
        num_fresh=max(0,total-len(seq.block_table)-len(hit_blocks))
        num_free_hits=sum(1 for b in hit_blocks if self.blocks[b].ref_count==0)
        if num_fresh+num_free_hits>len(self.free_block_ids):
            return False                                   
        for block_id in hit_blocks:                        
            self._touch(block_id)
            seq.block_table.append(block_id)
        seq.num_computed_tokens=computed
        seq.num_cached_tokens+=len(hit_blocks)*bs
        seq.num_hashed_blocks+=len(hit_blocks)
        for _ in range(num_fresh):
            block_id=self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        return True

    def cache_full_blocks(self,seq:Sequence):
        bs=self.block_size
        end=seq.num_computed_tokens//bs
        for i in range(seq.num_hashed_blocks,end):
            block=self.blocks[seq.block_table[i]]
            prefix=self.blocks[seq.block_table[i-1]].hash if i>0 else -1
            assert i==0 or prefix!=-1
            token_ids=seq.block(i)
            h=self.compute_hash(token_ids,prefix)
            block.update(h,token_ids)
            self.hash_to_block_id[h]=block.block_id
        seq.num_hashed_blocks=max(seq.num_hashed_blocks,end)

    def deallocate(self,seq:Sequence):
        for block_id in reversed(seq.block_table):
            block=self.blocks[block_id]
            block.ref_count-=1
            if block.ref_count==0:
                self._deallocate_block(block_id)
        seq.block_table.clear()
        seq.num_computed_tokens=0
        seq.num_cached_tokens=0
        seq.num_hashed_blocks=0