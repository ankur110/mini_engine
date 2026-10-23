# miniengine/config.py
# Purpose: one dataclass holding engine settings + the model's HF config.
# Grows step by step (step 2 adds block_size, step 4 max_num_seqs, ...).
import os
from dataclasses import dataclass
from transformers import AutoConfig,PretrainedConfig
@dataclass
class Config:
    model:str
    # local folder, e.g. "~/models/Qwen3-0.6B" — must contain config.json + *.safetensors

    hf_config:PretrainedConfig|None=None
    # never passed by hand; filled in __post_init__

    def __post_init__(self):
        # 1. expand "~" (os.path.expanduser) and store it back in self.model
        # 2. assert the folder exists (os.path.isdir) — fail early with a clear message
        # 3. self.hf_config=AutoConfig.from_pretrained(self.model)
        # 4. (one-time, then delete) print(self.hf_config) and check where rope_theta lives:
        #    older transformers → hf_config.rope_theta
        #    newer transformers → hf_config.rope_parameters["rope_theta"]
        #    you'll need it in 1B
        self.model=os.path.expanduser(self.model)
        assert os.path.isdir(self.model)
        self.hf_config=AutoConfig.from_pretrained(self.model)
