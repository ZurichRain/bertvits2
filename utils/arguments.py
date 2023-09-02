from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers

# @dataclass
# class ModelArguments:
#     modelname: Optional[str] = field(default="wf")
#     model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
#     use_cache: bool = field(default=False)
#     vision_tower: Optional[str] = field(default="~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/")
#     freeze_vision_tower: bool = field(default=False)
#     freeze_lm_model: bool = field(default=False)
#     pretrained_stage1_model: Optional[str] = field(default=None) # mlp &/ vision tower
#     vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
#     use_im_start_end: bool = field(default=False)


# @dataclass
# class DataArguments:
#     datasets: str = field(default=None, metadata={"help": "combinations of the training data."})
#     interleave_datasets: str = field(default=None)
#     conversation_datasets: str = field(default=None)
#     sep_image_conv_front: bool = False
#     use_eos_for_each_turn: bool = False
#     image_token_len: int = 256
#     image_aspect_ratio: str = 'square'
#     conversation_version: str = 'v0'
#     box_limit: int = 0
#     merge_round: int = 0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"