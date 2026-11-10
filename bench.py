import os
import json
import time
import argparse
from datetime import datetime
from random import randint,seed

p=argparse.ArgumentParser()
p.add_argument("--engine",choices=["mini","vllm","nano"],default="mini")
p.add_argument("--model",default="~/models/Qwen3-0.6B")
p.add_argument("--num-seqs",type=int,default=256)
p.add_argument("--out",default="results/bench.jsonl")
p.add_argument("--label",default="")
p.add_argument("--chunked",type=int,default=1)          
p.add_argument("--budget",type=int,default=16384)       
p.add_argument("--threshold",type=int,default=0)        
p.add_argument("--attn",choices=["split","unified"],default="split")
p.add_argument("--expect-gpu",default="")    
args=p.parse_args()
os.environ["MINI_ENGINE_ATTN"]=args.attn                
import torch
path=os.path.expanduser(args.model)
import subprocess
gpu_name=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                        capture_output=True,text=True).stdout.strip().splitlines()[0]
assert args.expect_gpu in gpu_name,f"wrong GPU: got {gpu_name}, expected *{args.expect_gpu}*"

def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":",1)[1].strip()
    except OSError:
        pass
    return "unknown"

seed(0)
N=args.num_seqs
prompt_token_ids=[[randint(0,10000) for _ in range(randint(100,1024))] for _ in range(N)]
max_tokens=[randint(100,1024) for _ in range(N)]

extra={}
if args.engine=="mini":
    from llm import LLM
    from sampling_params import SamplingParams
    llm=LLM(path,max_num_seqs=256,max_model_len=2048,max_num_batched_tokens=args.budget,
            enable_chunked_prefill=bool(args.chunked),long_prefill_token_threshold=args.threshold)
    prompts=prompt_token_ids
    extra=dict(chunked=bool(args.chunked),budget=args.budget,threshold=args.threshold,attn=args.attn)
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

if args.engine=="mini":
    extra["preemptions"]=llm.scheduler.num_preemptions

total=sum(max_tokens)
row={
    "engine":args.engine,
    "label":args.label,
    "gpu":gpu_name,
    "cpu":cpu_model(),
    "vllm":__import__("vllm").__version__ if args.engine=="vllm" else None,
    "torch":torch.__version__,
    "num_seqs":N,
    "input_tokens":sum(len(x) for x in prompt_token_ids),
    "output_tokens":total,
    "time_s":round(t,2),
    "throughput_tok_s":round(total/t,1),
    **extra,
    "time":datetime.now().isoformat(timespec="seconds"),
}
print(json.dumps(row))
os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
with open(args.out,"a") as f:
    f.write(json.dumps(row)+"\n")