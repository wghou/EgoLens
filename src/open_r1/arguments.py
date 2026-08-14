from trl import ScriptArguments
from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class SFTScriptArguments:
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "whether to skip special tokens, use when rec task."}
    )
    image_dir: str = field(
        default="",
        metadata={"help": "the directory of image in the dataset"}
    )
    rec_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "the ratio of rec loss to the total loss"}
    )
    lm_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "the ratio of rec loss to the total loss"}
    )
    res_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "the ratio of res loss to the total loss (unused)"}
    )
    res_loss_mode: str = field(
        default="legacy",
        metadata={"help": "how to aggregate the mask losses, see "
                          "Qwen2VLSFTTrainer.compute_loss. 'legacy' reproduces the "
                          "upstream formula, which weights the empty-mask BCE about "
                          "3*batch_size higher than the non-empty BCE+Dice and puts "
                          "the non-empty terms at 1/3 of lm_loss. 'norm' removes the "
                          "empty-term over-weighting only, 'mean' removes both.",
                  "choices": ["legacy", "norm", "mean"]}
    )
    if_detach_res_loss: bool = field(
        default=False,
        metadata={"help": "whether to detach res loss"}
    )
    if_freeze_llm: bool = field(
        default=False,
        metadata={"help": "whether to freeze llm"}
    )
    num_of_query: int = field(
        default=64,
        metadata={"help": "the number of query"}
    )
    if_use_qwen_connector: bool = field(
        default=True,
        metadata={"help": "whether to use qwen connector"}
    )
    if_include_sam: bool = field(
        default=True,
        metadata={"help": "whether to include sam"}
    )
    prompt_difficulty: str = field(
        default="easy",
        metadata={"help": "the difficulty of the input-output prompts for stage 1",
                  "choices": "hard, easy_cot"}
    )
    dataset_size: int = field(
        default=2000,
        metadata={"help": "the scene number to use for training."}
    )

@dataclass
class SFTScriptArguments(SFTScriptArguments):
    """
    Script arguments for the SFT training script.
    """
    pass