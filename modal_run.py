import os
import subprocess
import modal

app=modal.App("mini-engine-m3")

GPU="A100-40GB"                       
MODEL="/models/Qwen3-0.6B"
REPO="/root/mini_engine"
IGNORE=[".git",".venv","venv","__pycache__","results","*.ipynb"]

models=modal.Volume.from_name("qwen3-models",create_if_missing=True)          
results=modal.Volume.from_name("mini-engine-results",create_if_missing=True)  

base_image=(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch","transformers","safetensors","xxhash","tqdm","numpy","huggingface_hub")
    .add_local_dir(".",remote_path=REPO,ignore=IGNORE)
)
vllm_image=(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER":"0"})   
    .add_local_dir(".",remote_path=REPO,ignore=IGNORE)
)


@app.function(image=base_image,volumes={"/models":models},timeout=1800)   
def download():
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{MODEL}/config.json"):
        snapshot_download("Qwen/Qwen3-0.6B",local_dir=MODEL)
        models.commit()
    print("model ready:",sorted(os.listdir(MODEL)))


@app.function(image=base_image,gpu=GPU,volumes={"/models":models},timeout=1800)
def checks():
    subprocess.run(["nvidia-smi"],check=True)
    subprocess.run(["python","checks.py",MODEL],cwd=REPO,check=True)


def _bench(engine:str)->str:
    results.reload()                      
    subprocess.run(["python","bench.py","--engine",engine,"--model",MODEL,
                    "--out","/results/bench.jsonl"],cwd=REPO,check=True)
    results.commit()
    with open("/results/bench.jsonl") as f:
        return f.read()


@app.function(image=base_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def bench_mini()->str:
    return _bench("mini")


@app.function(image=vllm_image,gpu=GPU,volumes={"/models":models,"/results":results},timeout=1800)
def bench_vllm()->str:
    return _bench("vllm")


@app.local_entrypoint()
def main(step:str="all"):
    download.remote()
    jsonl=None
    if step in ("all","checks"):
        checks.remote()
    if step in ("all","mini"):
        jsonl=bench_mini.remote()
    if step in ("all","vllm"):
        jsonl=bench_vllm.remote()
    if jsonl is not None:
        os.makedirs("results",exist_ok=True)
        with open("results/bench.jsonl","w") as f:      
            f.write(jsonl)
        print("\nresults/bench.jsonl:\n"+jsonl)