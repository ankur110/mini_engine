import os
import json
import time
import argparse
from datetime import datetime
from random import Random

p=argparse.ArgumentParser()
p.add_argument("--model",default="~/models/Qwen3-0.6B")
p.add_argument("--num-reqs",type=int,default=240)
p.add_argument("--rates",default="4,8")                              
p.add_argument("--configs",default="pf,c512,c1024,c2048,c4096,c8192")  
p.add_argument("--attn",choices=["split","unified"],default="split")
p.add_argument("--long-frac",type=float,default=0.1)
p.add_argument("--out",default="results/latency.jsonl")
p.add_argument("--label",default="")
args=p.parse_args()
os.environ["MINI_ENGINE_ATTN"]=args.attn
import numpy as np
import torch
from llm import LLM
from sampling_params import SamplingParams
from engine.block_manager import BlockManager

path=os.path.expanduser(args.model)


def workload(n,rate):
    rng=Random(int(rate*1000))                     
    t=0.0
    reqs=[]
    for _ in range(n):
        t+=rng.expovariate(rate)
        long=rng.random()<args.long_frac
        plen=rng.randint(4096,7680) if long else rng.randint(128,512)
        reqs.append(dict(t=t,prompt=[rng.randint(0,10000) for _ in range(plen)],
                         out=rng.randint(128,256),long=long))
    return reqs


def set_config(llm,name):
    sch=llm.scheduler
    if name=="pf":
        sch.enable_chunked_prefill=False
        sch.max_num_batched_tokens=16384
    else:
        sch.enable_chunked_prefill=True
        sch.max_num_batched_tokens=int(name[1:])
    sch.block_manager=BlockManager(len(sch.block_manager.blocks),sch.block_size)   


def run(llm,reqs):
    sch=llm.scheduler
    p0=sch.num_preemptions
    arrive={}
    times={}
    is_long={}
    i=0
    steps=0
    start=time.perf_counter()
    while i<len(reqs) or not llm.is_finished():
        now=time.perf_counter()-start
        while i<len(reqs) and reqs[i]["t"]<=now:
            r=reqs[i]
            sid=llm.add_request(r["prompt"],SamplingParams(temperature=0.6,ignore_eos=True,max_tokens=r["out"]))
            arrive[sid]=start+r["t"]
            times[sid]=[]
            is_long[sid]=r["long"]
            i+=1
        if llm.is_finished():                       
            time.sleep(max(0.0,reqs[i]["t"]-(time.perf_counter()-start)))
            continue
        _,_,_,emitted=llm.step()
        steps+=1
        t=time.perf_counter()
        for sid in emitted:
            times[sid].append(t)
    wall=time.perf_counter()-start
    ms=lambda x:round(float(x)*1000,1)
    pct=lambda a,q:ms(np.percentile(a,q)) if len(a) else None
    ttft=[times[s][0]-arrive[s] for s in times]
    ttft_long=[times[s][0]-arrive[s] for s in times if is_long[s]]
    ttft_short=[times[s][0]-arrive[s] for s in times if not is_long[s]]
    itl=np.concatenate([np.diff(times[s]) for s in times])
    tpot=[(times[s][-1]-times[s][0])/(len(times[s])-1) for s in times if len(times[s])>1]
    out_tok=sum(len(v) for v in times.values())
    return dict(
        ttft_p50_ms=pct(ttft,50),ttft_p99_ms=pct(ttft,99),
        ttft_short_p99_ms=pct(ttft_short,99),
        ttft_long_p50_ms=pct(ttft_long,50),ttft_long_p99_ms=pct(ttft_long,99),
        itl_p50_ms=pct(itl,50),itl_p99_ms=pct(itl,99),itl_p999_ms=pct(itl,99.9),itl_max_ms=ms(itl.max()),
        tpot_mean_ms=ms(np.mean(tpot)),
        out_tok_s=round(out_tok/wall,1),wall_s=round(wall,1),steps=steps,
        preemptions=sch.num_preemptions-p0,num_long=len(ttft_long),
    )


llm=LLM(path,max_num_seqs=128,max_model_len=8192,max_num_batched_tokens=16384)   
set_config(llm,"c2048")
run(llm,workload(20,20.0))                          

os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
for rate in [float(r) for r in args.rates.split(",")]:
    reqs=workload(args.num_reqs,rate)
    for name in args.configs.split(","):
        set_config(llm,name)
        res=run(llm,reqs)
        row=dict(engine="mini",label=args.label,config=name,attn=args.attn,rate=rate,
                 num_reqs=args.num_reqs,long_frac=args.long_frac,**res,
                 gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                 time=datetime.now().isoformat(timespec="seconds"))
        print(json.dumps(row),flush=True)
        with open(args.out,"a") as f:
            f.write(json.dumps(row)+"\n")