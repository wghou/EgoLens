import json
import os
import torch
import transformers
from collections import defaultdict
from typing import Optional, Union, Any, Dict, List
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import Dataset, IterableDataset
from trl.trainer.grpo_config import GRPOConfig
from src.open_r1.arguments import SFTScriptArguments
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
from src.segment_anything_2.sam2.build_sam import build_sam2
import torch.nn.functional as F
from src.open_r1.trainer.sam_loss import dice_loss, sigmoid_bce_loss

# optimizer about
import logging
logger = logging.getLogger(__name__)

local_rank = int(os.environ.get("LOCAL_RANK", -1))

def find_subseq_1d(seq_1d: torch.Tensor, subseq_1d: torch.Tensor) -> int:
    """Return first index of subseq in seq, else -1. Both must be 1D long tensors on same device."""
    if seq_1d.dtype != torch.long or subseq_1d.dtype != torch.long:
        raise TypeError("find_subseq_1d expects torch.long tensors.")
    n = subseq_1d.numel()
    if n == 0 or seq_1d.numel() < n:
        return -1
    for idx in range(seq_1d.numel() - n + 1):
        if torch.equal(seq_1d[idx:idx + n], subseq_1d):
            return idx
    return -1

def build_answer_only_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    answer_start_strs=("<answer>\n", "<answer>"),
    answer_end_str="</answer>",
) -> torch.Tensor:
    """
    input_ids: (B, L) long
    attention_mask: (B, L) long/bool
    returns labels: (B, L) long, only answer span supervised, else -100
    """
    device = input_ids.device
    labels = torch.full_like(input_ids, -100)

    end_ids = torch.tensor(
        tokenizer.encode(answer_end_str, add_special_tokens=False),
        device=device, dtype=torch.long,
    )

    start_ids_list = [
        torch.tensor(tokenizer.encode(s, add_special_tokens=False), device=device, dtype=torch.long)
        for s in answer_start_strs
    ]

    B, L = input_ids.shape
    for i in range(B):
        ids = input_ids[i]

        # pick first start that matches
        start = -1
        start_len = 0
        for start_ids in start_ids_list:
            s = find_subseq_1d(ids, start_ids)
            if s != -1:
                start = s
                start_len = start_ids.numel()
                break

        end = find_subseq_1d(ids, end_ids) if start != -1 else -1

        if start != -1 and end != -1:
            ans_begin = start + start_len
            ans_end = end
            if ans_begin < ans_end:
                # labels[i, ans_begin:ans_end] = ids[ans_begin:ans_end]
                labels[i, ans_begin:] = ids[ans_begin:]
        elif start == -1 and end == -1:
            labels[i, :] = ids

    labels[attention_mask == 0] = -100
    return labels


class Qwen2VLSFTTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
            script_args: SFTScriptArguments = None, 

            rec_loss_ratio: float = 1.0,
            lm_loss_ratio: float = 1.0,
            if_detach_res_loss: bool = False,
            if_freeze_llm: bool = False
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.rec_loss_ratio = rec_loss_ratio
        self.lm_loss_ratio = lm_loss_ratio
        self.if_detach_res_loss = if_detach_res_loss
        # script_args may be None when the trainer is constructed directly.
        self.res_loss_mode = getattr(script_args, "res_loss_mode", None) or "legacy"
        if self.res_loss_mode not in ("legacy", "norm", "mean"):
            raise ValueError(f"unknown res_loss_mode: {self.res_loss_mode!r}")
        logger.info("mask loss aggregation mode: %s", self.res_loss_mode)

        self.bce_loss_weight = 1.0
        self.dice_loss_weight = 1.0
        self.if_freeze_llm = if_freeze_llm
        self.script_args = script_args
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            
            if "Qwen2.5-VL" in model_id or "2p5" in model_id:
                model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(model, **model_init_kwargs)
                print(f"Qwen initialization arguments: {model_init_kwargs}")
                model.sam = build_sam2("sam2_hiera_l.yaml", "./pretrained/sam2_hiera_large.pt")
                model.sam.requires_grad_(False)
                model.sam.sam_prompt_encoder.requires_grad_(True)
                model.sam.sam_mask_decoder.requires_grad_(True)  
                if self.if_freeze_llm:
                    model.visual.requires_grad = False
                    model.model.requires_grad = False
                    model.lm_head.requires_grad = False
                model_init_kwargs = {k:v for k,v in model_init_kwargs.items() if k=="attn_implementation"}
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # set extra training config to model
        model.set_if_detach_res_loss(self.if_detach_res_loss)

        # meta query
        self.keep_query_grounding = False
        self.if_meta_query = True

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "qwen2p5" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        
        # Data collator
        def sft_data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch_messages = []
            for f in features:
                message = f['prompt'].copy()
                message.append({"role": "assistant", "content": f['solution']})
                batch_messages.append(message)
                
            batch_images = [f["image"] for f in features]

            all_texts = [
                processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in batch_messages
            ]
                
            # build labels
            batch_enc = processing_class(
                text=all_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            
            input_ids = batch_enc["input_ids"]
            attention_mask = batch_enc["attention_mask"]
            pixel_values = batch_enc.get("pixel_values")
            image_grid_thw = batch_enc.get("image_grid_thw")

            # supervise only <answer>...</answer>
            labels = build_answer_only_labels(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=processing_class.tokenizer,
                answer_start_strs=("<answer>\n", "<answer>"),
                answer_end_str="</answer>",
            )

            sam_images = torch.stack([f["sam_image"] for f in features])   # (B, 3, H, W)
            masks = torch.stack([f["mask"] for f in features])  # (B, 3, H, W)
            ref_metas = [f["ref_meta"] for f in features]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "sam_images": sam_images,
                "gt_masks": masks,
                "ref_metas": ref_metas,
            }

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=sft_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        
        batch_size = inputs['gt_masks'].shape[0]
        
        model_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            sam_images=inputs['sam_images'],
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            use_learnable_query=True,
            use_cache=False,
        )
        if getattr(self, "if_meta_query", False) and hasattr(model, "learnable_query"):
            model_kwargs["use_learnable_query"] = True

        model_output, low_res_masks = model(**model_kwargs)

        logits = model_output.logits[:, :-self.model.config.num_of_query, :]                 # (B, Lm, V)
        labels = inputs["labels"]                    # (B, Ll)

        Lm = logits.size(1)
        Ll = labels.size(1)
        L = min(Lm, Ll)
        if Lm != Ll:
            logger.warning("Logit and label lengths differ; cropping to the shorter length.")
            logits = logits[:, :L, :].contiguous()
            labels = labels[:, :L].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        mask_empty_loss = 0.0

        non_empty_count = torch.zeros((), device=low_res_masks[0].device, dtype=low_res_masks[0].dtype)
        empty_count = torch.zeros((), device=low_res_masks[0].device, dtype=low_res_masks[0].dtype)
        for i in range(batch_size):
            for j in range(3):
                pred_mask = low_res_masks[i * 3 + j]
                pred_mask = F.interpolate(pred_mask.float(), size=inputs['gt_masks'][i, j].shape[-2:], mode='bilinear', align_corners=False)
                
                gt = inputs['gt_masks'][i, j].to(pred_mask.device)                # (H,W) or same
                gt = gt.unsqueeze(0).unsqueeze(0).float()                     # (1,1,H,W)

                valid = (gt.sum() > 0).to(pred_mask.dtype)                    # scalar tensor (float)
                non_empty_count = non_empty_count + valid
                
                empty = (gt.sum() == 0).to(pred_mask.dtype)                    # scalar tensor (float)
                empty_count = empty_count + empty

                loss_multimask = sigmoid_bce_loss(pred_mask, gt, 1., loss_on_multimask=True)  # shape (B, K) or (B,1)
                loss_multidice = dice_loss(pred_mask, gt, 1., True)
                loss_emptymask = sigmoid_bce_loss(pred_mask, gt, 1., loss_on_multimask=True)

                loss_multimask = loss_multimask * valid
                loss_multidice = loss_multidice * valid
                loss_emptymask = loss_emptymask * empty     

                if loss_multimask.size(1) > 1:
                    # take the mask indices with the smallest focal + dice loss for back propagation
                    loss_combo = self.bce_loss_weight * loss_multimask + self.dice_loss_weight * loss_multidice
                    best_loss_inds = torch.argmin(loss_combo, dim=-1)
                    batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)

                    loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
                    loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
                else:
                    loss_mask = loss_multimask
                    loss_dice = loss_multidice

                # loss_empty is not part of the multimask selection: with
                # multimask_output=False there is a single mask slot, and the
                # empty-mask term is plain BCE against an all-zero target.
                loss_empty = loss_emptymask

                mask_bce_loss += loss_mask.sum()
                mask_dice_loss += loss_dice.sum()
                mask_empty_loss += loss_empty.sum()

        # Sums so far: mask_bce_loss and mask_dice_loss over the non-empty slots,
        # mask_empty_loss over the empty ones (the * valid / * empty factors above
        # zero out the other side).
        num_elements = batch_size * 3           # 3 role slots per sample
        n_non_empty = non_empty_count.clamp(min=1.0)
        n_empty = empty_count.clamp(min=1.0)

        # Upstream divides bce/dice by num_elements but not the empty term, then
        # scales both by batch_size. Written out, that is
        #
        #   res_loss = mean_non_empty(BCE + Dice) / 3  +  B * mean_empty(BCE)
        #
        # so the empty-mask BCE carries roughly 3*B (48 at B=16) times the weight
        # of the terms that actually govern mask quality, and the non-empty terms
        # additionally sit at 1/3 of lm_loss. Two separable defects, hence three
        # modes rather than one "fix":
        #
        #   legacy  upstream formula, reproduces the published behaviour
        #   norm    add the missing /num_elements, removing only the empty-term
        #           over-weighting and keeping the extra 1/3
        #   mean    plain means, removing both
        if self.res_loss_mode == "legacy":
            res_loss = (mask_bce_loss + mask_dice_loss) / num_elements / n_non_empty * batch_size \
                       + mask_empty_loss / n_empty * batch_size
        elif self.res_loss_mode == "norm":
            res_loss = (mask_bce_loss + mask_dice_loss) / num_elements / n_non_empty * batch_size \
                       + mask_empty_loss / num_elements / n_empty * batch_size
        elif self.res_loss_mode == "mean":
            res_loss = (mask_bce_loss + mask_dice_loss) / n_non_empty \
                       + mask_empty_loss / n_empty
        else:
            raise ValueError(f"unknown res_loss_mode: {self.res_loss_mode!r}")

        # Logged with the upstream definition (sum over non-empty slots divided by
        # num_elements) in every mode, so the curves stay comparable across modes
        # and against the released checkpoint's trainer_state.json.
        mask_bce_loss = mask_bce_loss / num_elements
        mask_dice_loss = mask_dice_loss / num_elements

        total_loss = self.lm_loss_ratio * lm_loss + self.rec_loss_ratio * res_loss

        self._metrics["lm_loss"].append(lm_loss.item())
        self._metrics["mask_bce_loss"].append(mask_bce_loss.item())
        self._metrics["mask_dice_loss"].append(mask_dice_loss.item())
        self._metrics["mask_empty_loss"].append((mask_empty_loss / n_empty).item())
        self._metrics["res_loss"].append(res_loss.item())
        self._metrics["frac_empty"].append((empty_count / num_elements).item())

        return total_loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
