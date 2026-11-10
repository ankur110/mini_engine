from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from config import Config
from sampling_params import SamplingParams
from engine.sequence import Sequence
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner

class LLMEngine:

    def __init__(self,model:str,**kwargs):
        names={f.name for f in fields(Config)}
        config_kwargs={k:v for k,v in kwargs.items() if k in names}
        config=Config(model,**config_kwargs)
        self.model_runner=ModelRunner(config)
        self.tokenizer=AutoTokenizer.from_pretrained(config.model,use_fast=True)
        config.eos=self.tokenizer.eos_token_id
        self.scheduler=Scheduler(config)

    def add_request(self,prompt:str|list[int],sampling_params:SamplingParams,priority:int=0)->int:
        if isinstance(prompt,str):
            prompt=self.tokenizer.encode(prompt)
        seq=Sequence(prompt,sampling_params,priority)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self)->tuple[list[tuple[int,list[int]]],int,int,list[int]]:
        seqs=self.scheduler.schedule()
        num_decode=sum(1 for s in seqs if s.num_scheduled_tokens==1)
        num_prefill=sum(s.num_scheduled_tokens for s in seqs if s.num_scheduled_tokens>1)
        token_ids=self.model_runner.run(seqs)
        emitted=self.scheduler.postprocess(seqs,token_ids)
        outputs=[(s.seq_id,s.completion_token_ids) for s in emitted if s.is_finished]
        return outputs,num_prefill,num_decode,[s.seq_id for s in emitted]

    def is_finished(self)->bool:
        return self.scheduler.is_finished()

    def generate(self,prompts:list[str]|list[list[int]],
                 sampling_params:SamplingParams|list[SamplingParams],
                 use_tqdm:bool=True,priorities:list[int]|None=None)->list[dict]:
        if not isinstance(sampling_params,list):
            sampling_params=[sampling_params]*len(prompts)
        priorities=priorities or [0]*len(prompts)
        for prompt,params,prio in zip(prompts,sampling_params,priorities):
            self.add_request(prompt,params,prio)

        pbar=tqdm(total=len(prompts),desc="Generating",dynamic_ncols=True,disable=not use_tqdm)
        outputs={}
        while not self.is_finished():
            t=perf_counter()
            out,num_prefill,num_decode,_=self.step()
            dt=perf_counter()-t
            pbar.set_postfix({"prefill":num_prefill,"decode":num_decode,
                              "tok/s":f"{(num_prefill+num_decode)/dt:.0f}"})
            for seq_id,ids in out:
                outputs[seq_id]=ids
                pbar.update(1)
        pbar.close()

        ordered=[outputs[seq_id] for seq_id in sorted(outputs)]
        return [{"text":self.tokenizer.decode(ids,skip_special_tokens=True),"token_ids":ids} for ids in ordered]