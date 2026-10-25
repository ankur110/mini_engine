import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layernorm import RMSNorm
from layers.activation import SiluAndMul
from layers.attention import Attention
from layers.embed_head import VocabEmbedding,LMHead
from layers.linear import QKVLinear,MergedLinear
from layers.rotary_embedding import RotaryEmbedding


class Qwen3Attention(nn.Module):
    def __init__(self,config,rotary_emb):
        super().__init__()
        self.num_heads=config.num_attention_heads
        self.num_kv_heads=config.num_key_value_heads
        self.head_dim=config.head_dim
        self.q_size=self.num_heads*self.head_dim
        self.kv_size=self.num_kv_heads*self.head_dim
        self.qkv_proj=QKVLinear(config.hidden_size,self.head_dim,self.num_heads,self.num_kv_heads)
        self.o_proj=nn.Linear(self.q_size,config.hidden_size,bias=False)
        self.q_norm=RMSNorm(config.head_dim,config.rms_norm_eps)
        self.k_norm=RMSNorm(config.head_dim,config.rms_norm_eps)
        self.rotary_emb=rotary_emb
        self.attn=Attention(self.num_heads,self.head_dim,self.head_dim**-0.5,self.num_kv_heads)


    def forward(self,positions,hidden_states):
        qkv=self.qkv_proj(hidden_states)
        q,k,v=qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
        q=q.view(-1,self.num_heads,self.head_dim)
        k=k.view(-1,self.num_kv_heads,self.head_dim)
        v=v.view(-1,self.num_kv_heads,self.head_dim)
        q=self.q_norm(q)
        k=self.k_norm(k)
        q,k=self.rotary_emb(positions,q,k)
        o=self.attn(q,k,v)
        return self.o_proj(o.reshape(-1, self.q_size))
    


class Qwen3MLP(nn.Module):
    def __init__(self,hidden_size,intermediate_size):
        super().__init__()
        self.gate_up_proj=MergedLinear(hidden_size,[intermediate_size]*2)
        self.act_fn=SiluAndMul()
        self.down_proj=nn.Linear(intermediate_size,hidden_size,bias=False)


    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

    


class Qwen3DecoderLayer(nn.Module):
    def __init__(self,config,rotary_emb):
        super().__init__()
        self.self_attn=Qwen3Attention(config,rotary_emb)
        self.mlp=Qwen3MLP(config.hidden_size,config.intermediate_size)
        self.input_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)


    def forward(self,positions,hidden_states,residual):
        if residual is None:
            residual=hidden_states
            hidden_states=self.input_layernorm(hidden_states)
        else:
            hidden_states,residual=self.input_layernorm(hidden_states,residual)
        hidden_states=self.self_attn(positions,hidden_states)
        hidden_states,residual=self.post_attention_layernorm(hidden_states,residual)
        hidden_states=self.mlp(hidden_states)
        return hidden_states,residual
    


class Qwen3Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        rotary=RotaryEmbedding(config.head_dim,config.max_position_embeddings,config.rope_parameters["rope_theta"])
        self.embed_tokens=VocabEmbedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([Qwen3DecoderLayer(config,rotary) for _ in range(28)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,input_ids,positions):
        h=self.embed_tokens(input_ids)
        residual=None
        for layer in self.layers:
            h,residual=layer(positions,h,residual)
        h,_=self.norm(h,residual)
        return h


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping={
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self,config):
        super().__init__()
        self.model=Qwen3Model(config)
        self.lm_head=LMHead(config.vocab_size,config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight=self.model.embed_tokens.weight



    def forward(self,input_ids,positions):
        return self.model(input_ids,positions)

    def compute_logits(self,hidden_states):
        return self.lm_head(hidden_states)