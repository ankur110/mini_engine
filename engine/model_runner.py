import torch

def allocate_kv_cache(model,num_blocks,block_size,num_kv_heads,head_dim,dtype,device):
    mods=[m for m in model.modules() if hasattr(m,"k_cache")]
    num_layers= len(getattr(model,"model",model).layers)
    assert len(mods)==num_layers

    kv= torch.zeros(2,len(mods),num_blocks,block_size,num_kv_heads,head_dim,dtype=dtype,device=device)
    for i,m in enumerate(mods):
        m.k_cache=kv[0,i]
        m.v_cache=kv[1,i]
    return kv