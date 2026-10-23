# miniengine/models/qwen3.py
# What: the Qwen3 model built from our layers. Names match the HF checkpoint (Concept 5).

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
        # read from config: num_attention_heads (16), num_key_value_heads (8),
        #   head_dim (128 — from config, NEVER hidden/heads), hidden_size, rms_norm_eps
        # self.q_size=16*128; self.kv_size=8*128
        # self.qkv_proj=QKVLinear(...)
        # self.o_proj=nn.Linear(q_size,hidden_size,bias=False)       2048 → 1024
        # self.q_norm=RMSNorm(head_dim,eps); self.k_norm=RMSNorm(head_dim,eps)
        # self.rotary_emb=rotary_emb                                 (shared object)
        # self.attn=Attention(num_heads,head_dim,head_dim**-0.5,num_kv_heads)
        super().__init__()
        self.num_heads=config.num_attention_heads
        self.num_kv_heads=config.num_key_value_heads
        self.head_dim=self.head_dim
        self.q_size=self.num_heads*self.head_dim
        self.kv_size=self.num_kv_heads*self.head_dim
        self.qkv_proj=QKVLinear(config.hidden_size,self.num_heads,self.num_kv_heads)
        self.o_proj=nn.Linear(self.q_size,config.hidden_size,bias=False)
        self.q_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.k_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.rotary_emb=rotary_emb
        self.attn=Attention(self.num_heads,self.head_dim,self.head_dim**-0.5,self.num_kv_heads)


    def forward(self,positions,hidden_states):
        # 1. qkv=self.qkv_proj(hidden_states) → split [q_size,kv_size,kv_size] on dim -1
        # 2. into heads: hint .view(-1,num_heads,head_dim) → q [T,16,128], k/v [T,8,128]
        # 3. q=self.q_norm(q); k=self.k_norm(k)        (V not normed)
        # 4. q,k=self.rotary_emb(positions,q,k)        (AFTER the norm)
        # 5. o=self.attn(q,k,v) → [T,16,128] → [T,2048]   hint: .flatten(1)
        # 6. return self.o_proj(o)                      → [T,1024]
        qkv=self.qkv_proj(hidden_states)
        q,k,v=qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
        q=q.view(-1,self.num_heads,self.head_dim)
        k=k.view(-1,self.num_kv_heads,self.head_dim)
        v=v.view(-1,self.num_kv_heads,self.head_dim)
        q=self.q_norm(q)
        k=self.k_norm(k)
        q,k=self.rotary_emb(positions,q,k)
        o=self.attn(q,k,v)
        return self.o_proj(o)
    


class Qwen3MLP(nn.Module):
    def __init__(self,hidden_size,intermediate_size):
        # self.gate_up_proj=MergedLinear(hidden_size,[intermediate_size]*2)
        # self.act_fn=SiluAndMul()
        # self.down_proj=nn.Linear(intermediate_size,hidden_size,bias=False)
        super().__init__()
        self.gate_up_proj=MergedLinear(hidden_size,[intermediate_size]*2)
        self.act_fn=SiluAndMul()
        self.down_proj=nn.Linear(intermediate_size,hidden_size,bias=False)


    def forward(self,x):
        # gate_up → act_fn → down_proj
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

    


class Qwen3DecoderLayer(nn.Module):
    def __init__(self,config,rotary_emb):
        # self.self_attn, self.mlp, self.input_layernorm, self.post_attention_layernorm
        super().__init__()
        self.self_attn=Qwen3Attention(config,rotary_emb)
        self.mlp=Qwen3MLP(config.hidden_size,config.intermediate_size)
        self.input_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)


    def forward(self,positions,hidden_states,residual):
        # follow Concept 6 exactly; the "residual is None" branch is the first layer
        # return hidden_states,residual
        if residual is None:
            hidden_states,residual=self.input_layernorm(hidden_states),residual
        else:
            hidden_states,residual=self.input_layernorm(hidden_states,residual)
        hidden_states=self.self_attn(positions,hidden_states)
        hidden_states=self.post_attention_layernorm(hidden_states,residual)
        hidden_states=self.mlp(hidden_states)
        return hidden_states,residual
    


class Qwen3Model(nn.Module):
    def __init__(self,config):
        # theta: config.rope_theta (older transformers) or config.rope_parameters["rope_theta"] (newer)
        # rotary=RotaryEmbedding(head_dim,config.max_position_embeddings,theta)   ← built ONCE
        # self.embed_tokens=VocabEmbedding(vocab_size,hidden_size)
        # self.layers=nn.ModuleList([Qwen3DecoderLayer(config,rotary) for _ in range(28)])
        # self.norm=RMSNorm(hidden_size,eps)
        super().__init_()
        theta=config.rope_parameters["rope_theta"]
        rotary=RotaryEmbedding(config.head_dim,config.max_positions)
        self.embed_tokens=VocabEmbedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([Qwen3DecoderLayer(config,rotary) for _ in range(28)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,input_ids,positions):
        # h=embed; residual=None; loop layers; h,_=self.norm(h,residual); return h  → [T,1024]
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
        # self.model=Qwen3Model(config)
        # self.lm_head=LMHead(vocab_size,hidden_size)
        # if config.tie_word_embeddings: self.lm_head.weight=self.model.embed_tokens.weight
        #   (same Parameter OBJECT — like your Llama)
        self.model=Qwen3Model(config)
        self.lm_head=LMHead(config.vocab_size,config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight=self.model.embed_tokens.weight



    def forward(self,input_ids,positions):
        # return self.model(input_ids,positions)   → hidden states, NOT logits
        return self.model(input_ids,positions)

    def compute_logits(self,hidden_states):
        # return self.lm_head(hidden_states)
        return self.lm_head(hidden_states)