import os
import json
import subprocess
from datetime import datetime
import modal

app=modal.App("mini-engine-m5")

GPU="A100-80GB"
EXPECT="80GB"
MODEL="/models/Qwen3-0.6B"
REPO="/root/mini_engine"
IGNORE=[".git",".venv","venv","__pycache__","results","*.ipynb"]

models=modal.Volume.from_name("qwen3-models",create_if_missing=True)
results=modal.Volume.from_name("mini-engine-results",create_if_missing=True)

cuda_image=(
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04",add_python="3.12")
    .entrypoint([])
    .pip_install("torch==2.15.0","ninja","transformers","safetensors","xxhash","tqdm","numpy","huggingface_hub")
    .add_local_dir(".",remote_path=REPO,ignore=IGNORE)
)
vllm_image=(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0")
    .env({"VLLM_USE_FLASHINFER_SAMPLER":"0"})
    .add_local_dir(".",remote_path=REPO,ignore=IGNORE)
)


def fingerprint()->dict:
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version",
                        "--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    cpu="unknown"
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu=line.split(":",1)[1].strip()
                break
    return {"fingerprint":gpu,"cpu":cpu,"time":datetime.now().isoformat(timespec="seconds")}


@app.function(image=cuda_image,volumes={"/models":models},timeout=1800)
def download():
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{MODEL}/config.json"):
        snapshot_download("Qwen/Qwen3-0.6B",local_dir=MODEL)
        models.commit()
    print("model ready:",sorted(os.listdir(MODEL)))


@app.function(image=cuda_image,gpu=GPU,volumes={"/results":results},timeout=1800)
def kernels()->str:
    subprocess.run(["nvidia-smi"],check=True)
    results.reload()
    r=subprocess.run(["python","kernel_checks.py","--out","/results/kernels.jsonl"],cwd=REPO)
    results.commit()
    if r.returncode!=0:
        raise RuntimeError("kernel_checks.py failed: read the rows with ok=false above")
    with open("/results/kernels.jsonl") as f:
        return f.read()


@app.function(image=cuda_image,gpu=GPU,volumes={"/models":models},timeout=1800)
def checks():
    subprocess.run(["python","checks.py",MODEL],cwd=REPO,check=True)


def _bench(engine:str,label:str,extra:list[str]|None=None)->str:
    results.reload()
    subprocess.run(["python","bench.py","--engine",engine,"--model",MODEL,"--label",label,
                    "--out","/results/bench.jsonl",*(extra or [])],cwd=REPO,check=True)
    results.commit()
    with open("/results/bench.jsonl") as f:
        return f.read()


@app.function(image=cuda_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=3600)
def tput()->str:
    results.reload()
    with open("/results/bench.jsonl","a") as f:
        f.write(json.dumps(fingerprint())+"\n")
    g=["--expect-gpu",EXPECT]
    _bench("mini","m5-chunked-16k",g)
    _bench("mini","m5-prefill-first",["--chunked","0",*g])
    _bench("mini","m5-chunked-2k",["--budget","2048",*g])
    return _bench("mini","m5-unified-16k",["--attn","unified",*g])


@app.function(image=vllm_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def vllm_bench()->str:
    results.reload()
    with open("/results/bench.jsonl","a") as f:
        f.write(json.dumps(fingerprint())+"\n")
    _bench("vllm","vllm-m5",["--expect-gpu",EXPECT])
    return _bench("vllm","vllm-m5",["--expect-gpu",EXPECT])          # twice: spread within the container


@app.function(image=cuda_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=3600)
def latency()->str:
    results.reload()
    with open("/results/latency.jsonl","a") as f:
        f.write(json.dumps(fingerprint())+"\n")
    base=["python","bench_latency.py","--model",MODEL,"--out","/results/latency.jsonl"]
    subprocess.run(base+["--label","m5-e1"],cwd=REPO,check=True)
    results.commit()
    subprocess.run(base+["--label","m5-e3-unified","--configs","c2048","--rates","8",
                         "--attn","unified"],cwd=REPO,check=True)
    results.commit()
    with open("/results/latency.jsonl") as f:
        return f.read()


def _save(name:str,text:str):
    os.makedirs("results",exist_ok=True)
    with open(f"results/{name}","w") as f:
        f.write(text)
    print(f"\nresults/{name}:\n"+text)


@app.local_entrypoint()
def main(step:str="all"):
    if step in ("all","kernels"):
        _save("kernels.jsonl",kernels.remote())
    if step not in ("all","checks","tput","vllm","latency"):
        return
    download.remote()
    if step in ("all","checks"):
        checks.remote()
    if step in ("all","tput"):
        _save("bench.jsonl",tput.remote())
    if step in ("all","vllm"):
        _save("bench.jsonl",vllm_bench.remote())
    if step in ("all","latency"):
        _save("latency.jsonl",latency.remote())