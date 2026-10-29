from enum import Enum,auto
import copy
from itertools import count
from sampling_params import SamplingParams
class SequenceStatus(Enum):
    WAITING=1
    RUNNING=2
    FINISHED=3


class Sequence:
    block_size=16          
    counter=count()        

    def __init__(self,token_ids:list[int],sampling_params=None):
        self.seq_id=next(Sequence.counter)
        self.status=SequenceStatus.WAITING
        self.token_ids=copy.deepcopy(token_ids)
        self.last_token=self.token_ids[-1]
        self.num_tokens=len(self.token_ids)
        self.num_prompt_tokens=len(self.token_ids)
        self.num_cached_tokens=0
        self.block_table=[]
        self.sampling_params=sampling_params if sampling_params is not None else SamplingParams()
    def __len__(self):
        return self.num_tokens

    def __getitem__(self,key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status==SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens-self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens//self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens+self.block_size-1)//self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens-(self.num_blocks-1)*self.block_size

    def block(self,i:int)->list[int]:
        assert 0<=i<self.num_blocks
        return self.token_ids[i*self.block_size:(i+1)*self.block_size]

    def append_token(self,token_id:int):
        self.token_ids.append(token_id)
        self.last_token=token_id
        self.num_tokens+=1