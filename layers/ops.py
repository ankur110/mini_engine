import os
import torch
from torch.utils.cpp_extension import load

CSRC=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"csrc")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST","8.0")
STABLE="-DTORCH_TARGET_VERSION=0x020a000000000000"

load(
    name="mini_engine_ops",
    sources=[os.path.join(CSRC,"ops.cpp"),os.path.join(CSRC,"kv_cache.cu")],
    extra_cflags=["-O3",STABLE],
    extra_cuda_cflags=["-O3","-lineinfo",STABLE],
    is_python_module=False,
)

def store_kvcache(key,value,k_cache,v_cache,slot_mapping):
    torch.ops.mini_engine.store_kvcache(key,value,k_cache,v_cache,slot_mapping)