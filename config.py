import os
from dataclasses import dataclass
from transformers import AutoConfig,PretrainedConfig
@dataclass
class Config:
    model:str

    hf_config:PretrainedConfig|None=None

    def __post_init__(self):
        self.model=os.path.expanduser(self.model)
        assert os.path.isdir(self.model)
        self.hf_config=AutoConfig.from_pretrained(self.model)
