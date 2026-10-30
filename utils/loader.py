import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param:nn.Parameter,loaded_weight:torch.Tensor):
    assert param.shape==loaded_weight.shape,(tuple(param.shape),tuple(loaded_weight.shape))
    param.data.copy_(loaded_weight)


def load_model(model:nn.Module,path:str):
    packed=getattr(model,"packed_modules_mapping",{})
    tied=model.lm_head.weight is model.model.embed_tokens.weight
    loaded=set()
    files=sorted(glob(os.path.join(path,"*.safetensors")))
    assert files,f"no .safetensors files in {path}"

    for file in files:
        with safe_open(file,"pt",device="cpu") as f:
            for name in f.keys():
                if name=="lm_head.weight" and tied:
                    continue
                for disk_part,(merged,shard_id) in packed.items():
                    if disk_part in name:
                        param_name=name.replace(disk_part,merged)
                        param=model.get_parameter(param_name)
                        param.weight_loader(param,f.get_tensor(name),shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param=model.get_parameter(name)
                    loader=getattr(param,"weight_loader",default_weight_loader)
                    loader(param,f.get_tensor(name))
                    loaded.add(name)

    missing={n for n,_ in model.named_parameters()}-loaded
    assert not missing,f"weights not loaded: {sorted(missing)}"