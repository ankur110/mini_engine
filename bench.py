import os
import json
import time
import argparse
from datetime import datetime
from random import randint,seed
import torch

p=argparse.ArgumentParser()
p.add_argument("--engine",choices=["mini","vllm","nano"],default="mini")
p.add_argument("--model",default="~/models/Qwen3-0.6B")
p.add_argument("--num-seqs",type=int,default=256)
p.add_argument("--out",default="results/bench.jsonl")
p.add_argument("--label",default="")   
args=p.parse_args()
path=os.path.expanduser(args.model)

seed(0)
N=args.num_seqs
prompt_token_ids=[[randint(0,10000) for _ in range(randint(100,1024))] for _ in range(N)]
max_tokens=[randint(100,1024) for _ in range(N)]

if args.engine=="mini":
    from llm import LLM
    from sampling_params import SamplingParams
    llm=LLM(path,max_num_seqs=256,max_model_len=2048)
    prompts=prompt_token_ids
elif args.engine=="nano":
    from nanovllm import LLM,SamplingParams
    llm=LLM(path,enforce_eager=False,max_num_seqs=256,max_model_len=2048)
    prompts=prompt_token_ids
else:
    from vllm import LLM,SamplingParams
    llm=LLM(path,enforce_eager=False,max_num_seqs=256,max_model_len=2048)
    prompts=[dict(prompt_token_ids=x) for x in prompt_token_ids]

sps=[SamplingParams(temperature=0.6,ignore_eos=True,max_tokens=m) for m in max_tokens]

llm.generate(["Benchmark: "],SamplingParams(),use_tqdm=False)   

t=time.time()
llm.generate(prompts,sps,use_tqdm=False)
t=time.time()-t

total=sum(max_tokens)
row={
    "engine":args.engine,
    "label":args.label,
    "gpu":torch.cuda.get_device_name(0),
    "torch":torch.__version__,
    "num_seqs":N,
    "input_tokens":sum(len(x) for x in prompt_token_ids),
    "output_tokens":total,
    "time_s":round(t,2),
    "throughput_tok_s":round(total/t,1),
    "time":datetime.now().isoformat(timespec="seconds"),
}
print(json.dumps(row))
os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
with open(args.out,"a") as f:
    f.write(json.dumps(row)+"\n")