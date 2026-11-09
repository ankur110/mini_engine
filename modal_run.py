import os
import subprocess
import modal

app=modal.App("mini-engine-m4")

GPU="A100-40GB"
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
    .pip_install("vllm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER":"0"})
    .add_local_dir(".",remote_path=REPO,ignore=IGNORE)
)


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
    subprocess.run(["nvcc","--version"],check=True)
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


def _bench(engine:str,label:str)->str:
    results.reload()
    subprocess.run(["python","bench.py","--engine",engine,"--model",MODEL,"--label",label,
                    "--out","/results/bench.jsonl"],cwd=REPO,check=True)
    results.commit()
    with open("/results/bench.jsonl") as f:
        return f.read()


@app.function(image=cuda_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def bench_mini()->str:
    return _bench("mini","m4-cuda-ops")

@app.function(image=cuda_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def bench_gather()->str:
    os.environ["MINI_ENGINE_OPS"]="0"            
    return _bench("mini","m3-gather")

@app.function(image=vllm_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def bench_vllm()->str:
    return _bench("vllm","vllm")


def _save(name:str,text:str):
    os.makedirs("results",exist_ok=True)
    with open(f"results/{name}","w") as f:
        f.write(text)
    print(f"\nresults/{name}:\n"+text)


@app.local_entrypoint()
def main(step:str="all"):
    if step in ("all","kernels"):
        _save("kernels.jsonl",kernels.remote())
    if step not in ("all","checks","mini","gather","vllm"):
        return
    download.remote()
    if step in ("all","checks"):
        checks.remote()
    jsonl=None
    if step in ("all","mini"):
        jsonl=bench_mini.remote()

    
    if step in ("all","gather"):
        jsonl=bench_gather.remote()
    
    if step in ("all","vllm"):
        jsonl=bench_vllm.remote()
    if jsonl is not None:
        _save("bench.jsonl",jsonl)