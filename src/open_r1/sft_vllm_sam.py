import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import TrainingArguments
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
from trl import ModelConfig, TrlParser

from src.open_r1.dataset import EgoAffordTrainDataset
from src.open_r1.trainer.trainer_sft import Qwen2VLSFTTrainer
from src.open_r1.utils import save_args_to_txt
from src.open_r1.arguments import SFTScriptArguments

def custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
        cos = cos.to(torch.float)
        sin = sin.to(torch.float)
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


def main(script_args, training_args, model_args):
    print("Training arguments:", training_args)
    print("Script arguments:", script_args)
    print("Model arguments:", model_args)
    save_dir = os.path.join(training_args.output_dir, 'config')
    os.makedirs(save_dir, exist_ok=True)
    save_args_to_txt(script_args, os.path.join(save_dir, 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(save_dir, 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(save_dir, 'model_args.txt'))

    dataset = EgoAffordTrainDataset(script_args)
    eval_dataset = EgoAffordTrainDataset(script_args, split='val')

    trainer = Qwen2VLSFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        script_args=script_args,
        rec_loss_ratio=script_args.rec_loss_ratio,
        lm_loss_ratio=script_args.lm_loss_ratio,
        if_detach_res_loss=script_args.if_detach_res_loss,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if 'num_of_query' in script_args.__dict__:
        training_args.model_init_kwargs = {'num_of_query': script_args.num_of_query}

    if 'if_use_qwen_connector' in script_args.__dict__:
        if training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {'if_use_qwen_connector': script_args.if_use_qwen_connector}
        else:
            training_args.model_init_kwargs['if_use_qwen_connector'] = script_args.if_use_qwen_connector
    
    if 'if_include_sam' in script_args.__dict__:
        if training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {'if_include_sam': script_args.if_include_sam}
        else:
            training_args.model_init_kwargs['if_include_sam'] = script_args.if_include_sam
            
    training_args.remove_unused_columns = False
    
    training_args.save_strategy = "epoch"
    
    main(script_args, training_args, model_args)
