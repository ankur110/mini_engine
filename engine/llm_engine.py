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

    def add_request(self,prompt:str|list[int],sampling_params:SamplingParams):
        if isinstance(prompt,str):
            prompt=self.tokenizer.encode(prompt)
        seq=Sequence(prompt,sampling_params)
        self.scheduler.add(seq)

    def step(self)->tuple[list[tuple[int,list[int]]],int,bool]:
        seqs,is_prefill=self.scheduler.schedule()
        num_tokens=sum(len(seq)-seq.num_cached_tokens for seq in seqs) if is_prefill else len(seqs)
        token_ids=self.model_runner.run(seqs,is_prefill)
        self.scheduler.postprocess(seqs,token_ids)
        outputs=[(seq.seq_id,seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs,num_tokens,is_prefill

    def is_finished(self)->bool:
        return self.scheduler.is_finished()

    def generate(self,prompts:list[str]|list[list[int]],
                 sampling_params:SamplingParams|list[SamplingParams],
                 use_tqdm:bool=True)->list[dict]:
        if not isinstance(sampling_params,list):
            sampling_params=[sampling_params]*len(prompts)
        for prompt,params in zip(prompts,sampling_params):
            self.add_request(prompt,params)

        pbar=tqdm(total=len(prompts),desc="Generating",dynamic_ncols=True,disable=not use_tqdm)
        outputs={}
        prefill_tput=decode_tput=0.0
        while not self.is_finished():
            t=perf_counter()
            out,n,is_prefill=self.step()
            dt=perf_counter()-t
            if is_prefill:
                prefill_tput=n/dt
            else:
                decode_tput=n/dt
            pbar.set_postfix({"prefill":f"{prefill_tput:.1f} tok/s","decode":f"{decode_tput:.1f} tok/s"})
            for seq_id,ids in out:
                outputs[seq_id]=ids
                pbar.update(1)
        pbar.close()

        ordered=[outputs[seq_id] for seq_id in sorted(outputs)]
        return [{"text":self.tokenizer.decode(ids,skip_special_tokens=True),"token_ids":ids} for ids in ordered]