from transformers import AutoProcessor
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
import re
import torch
import json
import csv
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from PIL import Image
from eval.utils import AverageMeter, Summary
import cv2
from qwen_vl_utils import smart_resize

from metrics import compute_text_planning_score as compute_text_planning_score_
from metrics import compute_step_cand_similarity


COMPONENT_NAMES = ("direct_object", "instrument", "destination")


def resize_longest(image: Image.Image, longest_side_length):
    """
    Resize an image so that its longest side matches `longest_side_length`,
    keeping aspect ratio.
    """
    original_width, original_height = image.size

    if original_width > original_height:
        scale_factor = longest_side_length / original_width
    else:
        scale_factor = longest_side_length / original_height

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    return image.resize((new_width, new_height))

def overlay_mask(image, mask, color, alpha=0.4):
    mask = mask.astype(bool)
    
    overlay = image.copy()
    overlay[mask] = color
    
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def overlay_three_masks_additive(image: np.ndarray,
                                 masks: np.ndarray,
                                 alpha: float = 0.45):
    if masks.shape[0] != 3:
        raise ValueError(f"Expected masks shape (3,H,W), got {masks.shape}")

    img = image

    masks = masks.astype(bool)
    h, w = img.shape[:2]

    if masks.shape[1] != h or masks.shape[2] != w:
        raise ValueError(
            f"Mask size {masks.shape[2]}x{masks.shape[1]} does not match image size {w}x{h}"
        )

    colors = np.array([
        [255, 0, 0],    # Direct Object, red
        [0, 255, 0],    # Instrument, green
        [0, 0, 255],    # Destination, blue
    ], dtype=np.float32)

    color_layer = np.zeros_like(img, dtype=np.float32)

    for i in range(3):
        color_layer += masks[i][..., None].astype(np.float32) * colors[i]

    # Keep additive colors within the valid range.
    color_layer = np.clip(color_layer, 0, 255)

    union_mask = np.any(masks, axis=0)[..., None].astype(np.float32)

    out = img * (1.0 - alpha * union_mask) + color_layer * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


class EgoAffordDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(self, base_dir, split='val'):
        self.base_dir = base_dir
        self.system_prompt_template = "You are a helpful assistant that can see images and perform reasoning segmentation."
        self.split = split
        
        self.question_template = (
            "Goal: '{main_task}'\n"
            "Please analyze the scene and infer the remaining action steps to accomplish the goal based on the image.\n"
            "Then segment the following three functional components by bounding boxes for the next action step:\n"
            "- The Direct Object (the object being manipulated).\n"
            "- The Instrument (the tool used to perform the action).\n"
            "- The Destination (the target location or container).\n"
            "A component could be absent (by putting 'None') if not used (e.g., no instrument for hand actions, or no destination for in-place actions).\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- You MUST infer actions from the image. Do NOT reuse any example or template content.\n"
            "- Analyze all objects in the image carefully against the target description in <think>.\n"
            "- Only output the NEXT action step's components.\n"
            "- Output components' name and their bounding boxes inside <answer> tags. Use [0, 0, 0, 0] if None.\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "<think>"
            "[Your step-by-step analysis and reasoning]"
            "</think>\n"
            "<answer>\nAction steps:\n"
            "1. <step description based on image>\n"
            "2. <optional more steps>\n"
            "Components:\n"
            "direct object: <object>, [x1, y1, x2, y2]\ninstrument: <tool or None>, [x1, y1, x2, y2]\n destination: <location or None>, [x1, y1, x2, y2]\n"
            "</answer>\n"
            "\n"
            "DO NOT COPY ANY EXAMPLE TEXT. GENERATE ALL CONTENT FROM THE IMAGE ONLY."
        )

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        if not os.path.exists(self.base_dir):
            print(f"Warning: dataset directory not found: {self.base_dir}")
            return samples

        scene_dirs = sorted([d for d in os.listdir(self.base_dir)
        if os.path.isdir(os.path.join(self.base_dir, d))])
        for sceneId in scene_dirs:
            scene_idx = int(sceneId.split('_')[-1])
            if self.split == 'train':
                if scene_idx <= 100:
                    continue
            else:
                if scene_idx > 100:
                    continue
            scene_path = os.path.join(self.base_dir, sceneId)
            task_json = os.path.join(scene_path, 'task.json')
            mask_npz = os.path.join(scene_path, 'masks.npz')
            # multi-gt files (optional; only required for multi-mask eval)
            alt_json = os.path.join(scene_path, 'alternatives.json')
            multi_mask_npz = os.path.join(scene_path, 'masks_multi.npz')
            multi_index_json = os.path.join(scene_path, 'masks_multi_index.json')
            constraints_json = os.path.join(scene_path, 'step_constraints.json')

            if not (os.path.exists(task_json) and (os.path.exists(mask_npz) or os.path.exists(multi_mask_npz))):
                continue

            with open(task_json, 'r', encoding='utf-8') as f:
                steps = json.load(f).get('steps', [])
                
            with open(constraints_json, 'r', encoding='utf-8') as f:
                constraints = json.load(f).get('constraints', [])

            for step_idx in range(len(steps)):
                samples.append({
                    'scene_id': sceneId,
                    'scene_path': scene_path,
                    'step_idx': step_idx,
                    'main_task': None,
                    'current_step': steps[step_idx],
                    'constraints': constraints,
                    'alt_json': alt_json,
                    'multi_mask_npz': multi_mask_npz,
                    'multi_index_json': multi_index_json,
                })
                
        return samples
    
    def _load_masks_sparse(self, path):
        data = np.load(path, allow_pickle=True)
        shape = tuple(data['shape'])
        masks = np.zeros(shape, dtype=bool)
        if 'nonempty_idx' in data and data['nonempty_idx'].size > 0:
            idx = data['nonempty_idx']
            nonempty_shape = data['nonempty_shape']
            packed = data['packed']
            total_bits = int(np.prod(nonempty_shape))
            unpacked = np.unpackbits(packed)[:total_bits]
            nonempty_masks = unpacked.reshape(nonempty_shape).astype(bool)
            masks[idx[:, 0], idx[:, 1]] = nonempty_masks
        return masks
    

    def _load_multi_gt_masks(self, scene_path: str, step_idx: int, item: dict):
        alt_path = item.get('alt_json', os.path.join(scene_path, 'alternatives.json'))
        idx_path = item.get('multi_index_json', os.path.join(scene_path, 'masks_multi_index.json'))
        mm_path = item.get('multi_mask_npz', os.path.join(scene_path, 'masks_multi.npz'))
        def _fallback_single_gt():
            mask_data = self._load_masks_sparse(os.path.join(scene_path, 'masks.npz'))
            gt = torch.from_numpy(mask_data[step_idx]).float()
            if gt.max() > 1.0:
                gt /= 255.0
            return [gt]
        if not (os.path.exists(alt_path) and os.path.exists(idx_path) and os.path.exists(mm_path)):
            return _fallback_single_gt()
        alt = json.load(open(alt_path, 'r', encoding='utf-8'))
        idx_obj = json.load(open(idx_path, 'r', encoding='utf-8'))
        entries = idx_obj.get('entries', [])
        multi = self._load_masks_sparse(mm_path)  # (M,3,H,W)
        gt_list = []
        task_steps = item.get("task_steps", None)
        if task_steps is None:
            task_path = os.path.join(scene_path, "task.json")
            if os.path.exists(task_path):
                try:
                    task = json.load(open(task_path, "r", encoding="utf-8"))
                    task_steps = task.get("steps", [])
                except Exception:
                    task_steps = []
            else:
                task_steps = []
        base_text = task_steps[step_idx] if 0 <= step_idx < len(task_steps) else None
        cand_names_map = alt.get("candidate_names", {})
        cand_keys = cand_names_map.get(base_text, None) if base_text is not None else None
        if not cand_keys:
            # If candidate_names missing for this step, treat as mapping missing -> fallback later
            cand_keys = []
        key_to_idx = {(int(e["state"]), str(e["cand_key"])): int(e["mask_index"]) for e in entries if "cand_key" in e}
        for ck in cand_keys:
            mi = key_to_idx.get((int(step_idx), str(ck)), None)
            if mi is None:
                continue
            gt = torch.from_numpy(multi[mi]).float()
            if gt.max() > 1.0:
                gt /= 255.0
            gt_list.append(gt)
        # If candidates list empty or mapping missing, fallback to old single-gt
        if len(gt_list) == 0:
            return _fallback_single_gt()
        return gt_list
    
    def _load_multi_gt_candidate_texts(self, scene_path: str, step_idx: int, item: dict):
        alt_path = item.get('alt_json', os.path.join(scene_path, 'alternatives.json'))
        if not os.path.exists(alt_path):
            return []

        # task steps (for base_text)
        task_path = os.path.join(scene_path, "task.json")
        if not os.path.exists(task_path):
            return []
        task = json.load(open(task_path, "r", encoding="utf-8"))
        steps = task.get("steps", [])
        base_text = steps[step_idx] if 0 <= step_idx < len(steps) else None
        if base_text is None:
            return []

        alt = json.load(open(alt_path, "r", encoding="utf-8"))
        cand_names_map = alt.get("candidate_names", {})
        cand_keys = cand_names_map.get(base_text, []) or []
        # Deduplicate while preserving order.
        seen = set()
        out = []
        for x in cand_keys:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        scene_path = item['scene_path']
        step_idx = item['step_idx']

        # Load task metadata.
        with open(os.path.join(scene_path, 'task.json'), 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        main_task = task_data['main_task']
        remaining_steps = task_data['steps'][step_idx:]
        past_steps = task_data['steps'][:step_idx]
        # constraints
        global_constraints = item['constraints']
        local_constraints = []
        offset = item['step_idx'] + 1
        for a, b in global_constraints:
            try:
                la, lb = a - offset, b - offset
            except Exception as exc:
                print(f"Warning: invalid step constraint in {scene_path}: {exc}")
                raise
            if 0 <= la < len(remaining_steps) and 0 <= lb < len(remaining_steps):
                local_constraints.append([la, lb])
        remaining_steps_text = ""
        for step_i, step in enumerate(remaining_steps):
            remaining_steps_text += f"{step_i + 1}. {step}\n"
        past_steps_text = ""
        for step_i, step in enumerate(past_steps):
            past_steps_text += f"{step_i + 1}. {step}\n"

        # Prepare image inputs.
        suffix = ['png', 'jpg', 'JPG']
        for suf in suffix:
            img_path = os.path.join(scene_path, f"step_{step_idx}.{suf}")
            if os.path.exists(img_path):
                break
        image_pil = Image.open(img_path).convert("RGB")
        width, height = image_pil.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            28,
            max_pixels=1000000
        )
        llm_image = image_pil.resize((resized_width, resized_height))

        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(image_cv, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image.float() - self.pixel_mean) / self.pixel_std

        gt_masks_candidates = self._load_multi_gt_masks(scene_path, step_idx, item)  # List[(3,H,W)]
        gt_masks = gt_masks_candidates[0]
        gt_candidate_texts = self._load_multi_gt_candidate_texts(scene_path, step_idx, item)

        prompt = self.question_template.format(
            main_task=main_task
        )

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ],
            "image": llm_image,
            "scene_id": str(item["scene_id"]),
            "step_idx": int(step_idx),
            "image_path": img_path,
            "sam_image": sam_image,
            "mask": gt_masks, 
            "gt_masks_candidates": gt_masks_candidates,
            "gt_candidate_texts": gt_candidate_texts,
            "gt_steps": remaining_steps,
            "constraints_list": local_constraints,
        }


class EgoAffEvaluator:
    """Evaluator class for ReasonSeg dataset"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        self.model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
            args.model_path, torch_dtype=self.dtype, attn_implementation="flash_attention_2"
        ).cuda()

        self.model.eval()
        
        self.processing_class = AutoProcessor.from_pretrained(args.model_path)
        pad_token_id = self.processing_class.tokenizer.pad_token_id
        self.processing_class.pad_token_id = pad_token_id
        self.processing_class.eos_token_id = self.processing_class.tokenizer.eos_token_id

        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]
    
    @staticmethod
    def get_answer_part(text):
        """Get the answer block from the string."""
        matches = list(re.finditer(r'<answer>.*?</answer>\n?', text, flags=re.DOTALL))
        if not matches:
            match = re.search(r'(?i)assistant\n.*', text, flags=re.DOTALL)
            if match:
                return match.group(0)
            else:
                return text
        start, end = matches[-1].span()
        return text[start:end]
        
    def compute_text_planning_score(
        self,
        pred_text,
        gt_steps,
        constraints=None,
        threshold=0.6
    ):

        return compute_text_planning_score_(EgoAffEvaluator.get_answer_part(pred_text), gt_steps, constraints, threshold)

    @torch.no_grad()
    def evaluate_single(self, input_data, data_idx):
        if isinstance(input_data, list):
            inputs = input_data
            meta = inputs[0] if len(inputs) > 0 else {}
        elif isinstance(input_data, dict):
            if "inputs" in input_data and isinstance(input_data["inputs"], list):
                inputs = input_data["inputs"]
            else:
                inputs = [input_data]
            meta = input_data
        else:
            raise TypeError(f"Unsupported input_data type: {type(input_data)}")

        texts = [
            self.processing_class.apply_chat_template(
                x["prompt"], tokenize=False, add_generation_prompt=True
            )
            for x in inputs
        ]
        image_inputs = [x["image"] for x in inputs]

        orig_mask_hw = inputs[0]["mask"][0].shape[-2:] if isinstance(inputs[0]["mask"], (list, tuple)) else inputs[0]["mask"].shape[-2:]
        sam_image_1 = inputs[0]["sam_image"]

        gen_inputs = self.processing_class(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device="cuda", dtype=getattr(self, "dtype", None) or torch.float16)

        # generate completion
        llm_out = self.model.generate(
            max_length=2048,
            use_cache=True,
            do_sample=False,
            **gen_inputs
        )
        completion_text = self.processing_class.batch_decode(
            llm_out, skip_special_tokens=True
        )

        # build new attention mask for generated sequence
        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pad_id = self.processing_class.tokenizer.pad_token_id
        pos = torch.where(llm_out == pad_id)
        new_attention_mask[pos] = 0

        # feed generated tokens back into model forward for mask prediction
        gen_inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
        gen_inputs.update(
            {
                "sam_images": sam_image_1.unsqueeze(0).repeat(len(texts), 1, 1, 1).to(
                    device="cuda", dtype=getattr(self, "dtype", None) or torch.float16
                )
            }
        )

        try:
            output, low_res_masks = self.model(
                output_hidden_states=True,
                use_learnable_query=True,
                **gen_inputs
            )

            # pred_masks: [N_query, H, W] (int 0/1)
            pred_masks_list = []
            for idx, mask in enumerate(low_res_masks):
                pred_masks_list.append(
                    self.model.postprocess_masks(mask, orig_hw=orig_mask_hw)
                )
        except Exception as e:
            completion_text = self.processing_class.batch_decode(
                llm_out, skip_special_tokens=True
            )
            print(f"Warning: mask prediction failed for sample {data_idx}: {e}")
            print("Model completion:", completion_text)
            pred_masks_list = [torch.zeros([1, 1, 768, 1024], device=llm_out.device, dtype=llm_out.dtype) for _ in range(3)]

        pred_masks = torch.cat(pred_masks_list, dim=0)
        pred_masks = (pred_masks[:, 0] > 0).int()

        if isinstance(meta["mask"], torch.Tensor) and meta["mask"].dim() == 3:
            masks_list = [meta["mask"][k].bool() for k in range(meta["mask"].shape[0])]
        else:
            masks_list = [m.bool() for m in meta["mask"]]

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        index = 0

        remaining_steps_gt = meta["gt_steps"]
        constraints_list = meta["constraints_list"]

        completion_text = completion_text if len(completion_text) > 0 else [""]

        text_results = self.compute_text_planning_score(
            EgoAffEvaluator.get_answer_part(completion_text[index]),
            remaining_steps_gt,
            constraints_list
        )

        ROLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # meta may provide multiple GT candidates for this state.
        # Each candidate is a (3,H,W) tensor.
        gt_candidates = meta.get("gt_masks_candidates", None)
        if gt_candidates is None:
            # backward-compatible: treat masks_list as the only GT
            gt_candidates = [torch.stack([m.cpu() if hasattr(m, "cpu") else torch.from_numpy(m) for m in masks_list], dim=0)]

        # Ensure pred_masks is list of 3 (H,W) tensors on GPU/CPU as original code expects.
        # We'll compute score for each candidate and pick the best one.
        best_intersection = None
        best_union = None
        best_acc_iou = None
        best_score = None
        best_gt_for_vis = None
        best_cand_idx = None
        best_role_intersections = None
        best_role_unions = None
        best_role_ious = None
        best_role_gt_present = None
        best_role_pred_present = None

        # pred_masks is list length=3, each is (H,W) tensor (likely on cuda)
        for cand_i, cand_gt in enumerate(gt_candidates):
            # cand_gt: (3,H,W)
            if isinstance(cand_gt, list):
                cand_gt = torch.stack(cand_gt, dim=0)
            # build gt list per channel to reuse existing per-channel logic
            cand_masks_list = [cand_gt[i] for i in range(3)]

            cand_intersection = 0.0
            cand_union = 0.0
            cand_acc_iou = 0.0
            cand_role_intersections = []
            cand_role_unions = []
            cand_role_ious = []
            cand_role_gt_present = []
            cand_role_pred_present = []

            for ridx, output_i in enumerate(pred_masks):
                pred_b = output_i.cuda().bool()
                gt_b = cand_masks_list[ridx].cuda().bool()

                inter = (pred_b & gt_b).sum().float()
                uni = (pred_b | gt_b).sum().float()
                iou = 1.0 if uni.item() == 0 else (inter / (uni + 1e-6))

                cand_intersection += inter
                cand_union += uni
                cand_acc_iou += iou
                cand_role_intersections.append(inter)
                cand_role_unions.append(uni)
                cand_role_ious.append(
                    torch.as_tensor(iou, device=inter.device, dtype=torch.float32)
                )
                cand_role_gt_present.append(gt_b.any())
                cand_role_pred_present.append(pred_b.any())

            score = cand_acc_iou / float(len(pred_masks))  # mean IoU over roles
            score_val = float(score.item()) if hasattr(score, "item") else float(score)


            if (best_score is None) or (score_val > best_score):
                best_score = score_val
                best_intersection = cand_intersection
                best_union = cand_union
                best_acc_iou = cand_acc_iou
                best_gt_for_vis = cand_masks_list
                best_cand_idx = cand_i
                best_role_intersections = torch.stack(cand_role_intersections)
                best_role_unions = torch.stack(cand_role_unions)
                best_role_ious = torch.stack(cand_role_ious)
                best_role_gt_present = torch.stack(cand_role_gt_present)
                best_role_pred_present = torch.stack(cand_role_pred_present)

        # Use best candidate's stats for aggregation
        intersection = best_intersection
        union = best_union
        acc_iou = best_acc_iou
        masks_list = best_gt_for_vis  # for visualization only (no candidate id returned)

        # Component-wise statistics use exactly the same admissible GT candidate
        # selected for the original aggregate gIoU/cIoU.  Keep raw sums/counts
        # here; main() performs distributed reduction before computing metrics.
        role_gt_present = best_role_gt_present.to(torch.float64)
        role_pred_present = best_role_pred_present.to(torch.float64)
        role_stats = {
            "intersection": best_role_intersections.detach().to(torch.float64).cpu().numpy(),
            "union": best_role_unions.detach().to(torch.float64).cpu().numpy(),
            # Includes empty-empty pairs as IoU=1, matching the paper's gIoU.
            "giou_sum": best_role_ious.detach().to(torch.float64).cpu().numpy(),
            # Only GT-present samples contribute to non-empty mIoU. A missed
            # component therefore contributes IoU=0 rather than disappearing.
            "nonempty_iou_sum": (
                best_role_ious.to(torch.float64) * role_gt_present
            ).detach().cpu().numpy(),
            "gt_present": role_gt_present.detach().cpu().numpy(),
            "presence_tp": (role_gt_present * role_pred_present).detach().cpu().numpy(),
            "presence_fp": ((1.0 - role_gt_present) * role_pred_present).detach().cpu().numpy(),
            "presence_fn": (role_gt_present * (1.0 - role_pred_present)).detach().cpu().numpy(),
            "presence_tn": ((1.0 - role_gt_present) * (1.0 - role_pred_present)).detach().cpu().numpy(),
            "num_samples": 1.0,
            "scene_id": str(meta.get("scene_id", "unknown_scene")),
            "step_idx": int(meta.get("step_idx", data_idx)),
        }
        
        try:
            pred_steps = re.findall(r'(?:^\s*\d+\.\s*|^\s*-\s*)(.*)', EgoAffEvaluator.get_answer_part(completion_text[index]), flags=re.MULTILINE)
            pred_steps = [s.strip() for s in pred_steps if s.strip()]
            pred_first_step = pred_steps[0] if len(pred_steps) > 0 else ""
            gt_cand_texts = meta.get("gt_candidate_texts", [])
            best_cand_key = gt_cand_texts[best_cand_idx] if (best_cand_idx is not None and best_cand_idx < len(gt_cand_texts)) else ""
            first_step_cand_match_sim = compute_step_cand_similarity(pred_first_step, best_cand_key)
        except Exception:
            first_step_cand_match_sim = 0.0

        if getattr(self.args, "vis", False) and (data_idx % getattr(self.args, "vis_freq", 1) == 0):
            if 'real' in self.args.image_dir:
                output_dir = f"evaluations_multi_real"
            else:
                output_dir = f"evaluations_multi"
            good_dir = os.path.join(self.args.model_path, output_dir, "egoaff_vis")
            os.makedirs(good_dir, exist_ok=True)

            base_image_src = Image.open(meta['image_path'])
            base_img = np.array(resize_longest(base_image_src, 1024))
            if base_img.max() <= 1.0:
                base_img = (base_img * 255).astype(np.uint8)

            concat_rows = []
            for gt_mask, pred_mask, color in zip(masks_list, pred_masks, ROLE_COLORS):
                gt_mask_np = gt_mask.cpu().numpy() if hasattr(gt_mask, "cpu") else gt_mask
                pd_mask_np = pred_mask.cpu().numpy() if hasattr(pred_mask, "cpu") else pred_mask

                vis_gt = overlay_mask(base_img, gt_mask_np, color, alpha=0.4)
                vis_pd = overlay_mask(base_img, pd_mask_np, color, alpha=0.4)
                row = np.hstack([vis_gt, vis_pd])
                concat_rows.append(row)
                
            mul_vis = overlay_three_masks_additive(image=base_img, masks=pred_masks.cpu().numpy())
            mul_vis.save(os.path.join(good_dir, f"{data_idx}th_mul.png"), format="PNG", optimize=True)
            mul_gt = overlay_three_masks_additive(image=base_img, masks=np.array(masks_list))
            mul_gt.save(os.path.join(good_dir, f"{data_idx}th_gt.png"), format="PNG", optimize=True)

            final_vis = np.vstack(concat_rows) if len(concat_rows) > 0 else base_img
            save_path = os.path.join(good_dir, f"{data_idx}th.png")
            Image.fromarray(final_vis).save(save_path)

            with open(os.path.join(good_dir, f"{data_idx}th.txt"), "w") as f:
                f.write(f"{completion_text[0]}\n\n{remaining_steps_gt}\n\n")
                f.write(f"First step score: {first_step_cand_match_sim}\nGT Key:{best_cand_key}\nPred:{pred_first_step}\n")
                f.write(f"inter: {intersection} union: {union} iou: {acc_iou / 3}")

        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou / len(masks_list)

        return (
            intersection,
            union,
            acc_iou,
            first_step_cand_match_sim,
            text_results["semantic_f1"],
            text_results["order_score"],
            text_results["coverage_score"],
            text_results["constraint_satisfaction_ratio"],
            text_results["hard_constraint_success"],
            text_results["dag_edge_f1"],
            len(masks_list),
            role_stats,
        )


def main(args):
    os.makedirs(f"{args.model_path}/evaluations_multi", exist_ok=True)

    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = EgoAffEvaluator(args)
    p = next(evaluator.model.sam.parameters())
    print("SAM parameter statistics:", p.float().mean().item(), p.float().std().item())
    dataset = EgoAffordDataset(base_dir=args.image_dir)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank)
    dataloader = DataLoader(dataset, 1, False, sampler=sampler, collate_fn=lambda batch: list(batch))

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    sim_meter = AverageMeter("sim_score", ":6.3f", Summary.SUM)
    order_meter = AverageMeter("order_score", ":6.3f", Summary.SUM)
    coverage_meter = AverageMeter("coverage_score", ":6.3f", Summary.SUM)
    constraint_meter = AverageMeter("constraint_ratio", ":6.3f", Summary.SUM)
    hard_constraint_meter = AverageMeter("hard_constraint_ratio", ":6.3f", Summary.SUM)
    dag_f1_meter = AverageMeter("dag_f1", ":6.3f", Summary.SUM)
    first_step_sim_meter = AverageMeter("first_step_cand_match_sim", ":.4f", Summary.AVERAGE)

    # Raw component-wise accumulators. They are reduced across workers after
    # evaluation, avoiding assumptions about AverageMeter supporting vectors.
    role_stat_keys = (
        "intersection",
        "union",
        "giou_sum",
        "nonempty_iou_sum",
        "gt_present",
        "presence_tp",
        "presence_fp",
        "presence_fn",
        "presence_tn",
    )
    role_totals = {
        key: np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
        for key in role_stat_keys
    }
    role_num_samples = 0.0
    # Keep per-sample raw statistics for exact per-scene aggregation. These
    # records are gathered and deduplicated across distributed workers below.
    local_scene_records = []

    for idx, batch_data in enumerate(tqdm(dataloader)):
        assert len(batch_data) == 1, "Only batch_size=1 is supported"
        (
            intersection,
            union,
            acc_iou,
            first_step_sim,
            sim_score,
            order_score,
            coverage_score,
            constraint,
            hard_constraint,
            dag_f1,
            num_mask,
            role_stats,
        ) = evaluator.evaluate_single(batch_data[0], idx)
        intersection_meter.update(intersection, n=num_mask)
        union_meter.update(union, n=num_mask)
        acc_iou_meter.update(acc_iou, n=num_mask)
        sim_meter.update(sim_score, n=1)
        order_meter.update(order_score, n=1)
        coverage_meter.update(coverage_score, n=1)
        constraint_meter.update(constraint, n=1)
        hard_constraint_meter.update(hard_constraint, n=1)
        dag_f1_meter.update(dag_f1, n=1)
        first_step_sim_meter.update(first_step_sim, n=1)
        for key in role_stat_keys:
            role_totals[key] += np.asarray(role_stats[key], dtype=np.float64)
        role_num_samples += float(role_stats["num_samples"])
        local_scene_records.append({
            "sample_key": f"{role_stats['scene_id']}::step_{role_stats['step_idx']}",
            "scene_id": role_stats["scene_id"],
            "step_idx": int(role_stats["step_idx"]),
            "intersection": np.asarray(role_stats["intersection"], dtype=np.float64).tolist(),
            "union": np.asarray(role_stats["union"], dtype=np.float64).tolist(),
            "giou_sum": np.asarray(role_stats["giou_sum"], dtype=np.float64).tolist(),
            "nonempty_iou_sum": np.asarray(role_stats["nonempty_iou_sum"], dtype=np.float64).tolist(),
            "gt_present": np.asarray(role_stats["gt_present"], dtype=np.float64).tolist(),
            "presence_tp": np.asarray(role_stats["presence_tp"], dtype=np.float64).tolist(),
            "presence_fp": np.asarray(role_stats["presence_fp"], dtype=np.float64).tolist(),
            "presence_fn": np.asarray(role_stats["presence_fn"], dtype=np.float64).tolist(),
            "presence_tn": np.asarray(role_stats["presence_tn"], dtype=np.float64).tolist(),
            "first_step_similarity": float(first_step_sim),
            "semantic_f1": float(sim_score),
            "order_score": float(order_score),
            "coverage_score": float(coverage_score),
            "constraint_ratio": float(constraint),
            "hard_constraint_success": float(hard_constraint),
            "dag_edge_f1": float(dag_f1),
        })

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    sim_meter.all_reduce()
    order_meter.all_reduce()
    coverage_meter.all_reduce()
    constraint_meter.all_reduce()
    hard_constraint_meter.all_reduce()
    dag_f1_meter.all_reduce()
    first_step_sim_meter.all_reduce()

    # One all-reduce for all component statistics. float64 keeps large pixel
    # intersection/union accumulators exact enough for dataset-scale eval.
    packed_role_stats = np.concatenate(
        [role_totals[key] for key in role_stat_keys]
        + [np.asarray([role_num_samples], dtype=np.float64)]
    )
    packed_role_stats = torch.as_tensor(
        packed_role_stats, dtype=torch.float64, device=torch.cuda.current_device()
    )
    dist.all_reduce(packed_role_stats, op=dist.ReduceOp.SUM)
    packed_role_stats = packed_role_stats.cpu().numpy()

    # DistributedSampler pads with repeated indices when the dataset size is
    # not divisible by world size. Gather per-sample records and deduplicate by
    # scene/step before computing scene-level results.
    gathered_scene_records = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_scene_records, local_scene_records)

    offset = 0
    for key in role_stat_keys:
        role_totals[key] = packed_role_stats[offset:offset + len(COMPONENT_NAMES)]
        offset += len(COMPONENT_NAMES)
    role_num_samples = float(packed_role_stats[offset])
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-8)
    ciou = iou_class
    giou = acc_iou_meter.avg
    sim = sim_meter.avg
    order = order_meter.avg
    coverage = coverage_meter.avg
    constraint_ratio = constraint_meter.avg
    hard_constraint_success = hard_constraint_meter.avg
    dag_f1_score = dag_f1_meter.avg
    first_step_sim_score = first_step_sim_meter.avg

    def safe_divide(numerator, denominator):
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        result = np.full_like(numerator, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    role_giou = safe_divide(role_totals["giou_sum"], role_num_samples)
    role_ciou = safe_divide(role_totals["intersection"], role_totals["union"])
    role_nonempty_miou = safe_divide(
        role_totals["nonempty_iou_sum"], role_totals["gt_present"]
    )
    role_presence_rate = safe_divide(role_totals["gt_present"], role_num_samples)
    role_presence_precision = safe_divide(
        role_totals["presence_tp"],
        role_totals["presence_tp"] + role_totals["presence_fp"],
    )
    role_presence_recall = safe_divide(
        role_totals["presence_tp"],
        role_totals["presence_tp"] + role_totals["presence_fn"],
    )
    # Compute F1 directly from counts so that a model predicting no positives
    # receives F1=0 when positives exist, even though precision is undefined.
    role_presence_f1 = safe_divide(
        2.0 * role_totals["presence_tp"],
        2.0 * role_totals["presence_tp"]
        + role_totals["presence_fp"]
        + role_totals["presence_fn"],
    )

    component_metrics = {}
    for role_idx, role_name in enumerate(COMPONENT_NAMES):
        component_metrics[role_name] = {
            "gIoU": float(role_giou[role_idx]),
            "cIoU": float(role_ciou[role_idx]),
            "non_empty_mIoU": float(role_nonempty_miou[role_idx]),
            "presence_rate": float(role_presence_rate[role_idx]),
            "presence_precision": float(role_presence_precision[role_idx]),
            "presence_recall": float(role_presence_recall[role_idx]),
            "presence_f1": float(role_presence_f1[role_idx]),
            "gt_present_count": int(round(role_totals["gt_present"][role_idx])),
            "presence_tp": int(round(role_totals["presence_tp"][role_idx])),
            "presence_fp": int(round(role_totals["presence_fp"][role_idx])),
            "presence_fn": int(round(role_totals["presence_fn"][role_idx])),
            "presence_tn": int(round(role_totals["presence_tn"][role_idx])),
        }

    scene_metrics = {}
    num_unique_scene_samples = 0
    if rank == 0:
        unique_records = {}
        for worker_records in gathered_scene_records:
            for record in worker_records:
                unique_records.setdefault(record["sample_key"], record)
        num_unique_scene_samples = len(unique_records)

        records_by_scene = {}
        for record in unique_records.values():
            records_by_scene.setdefault(record["scene_id"], []).append(record)

        planning_metric_names = (
            "first_step_similarity",
            "semantic_f1",
            "order_score",
            "coverage_score",
            "constraint_ratio",
            "hard_constraint_success",
            "dag_edge_f1",
        )
        scene_role_stat_keys = (
            "intersection",
            "union",
            "giou_sum",
            "nonempty_iou_sum",
            "gt_present",
            "presence_tp",
            "presence_fp",
            "presence_fn",
            "presence_tn",
        )

        for scene_id, records in sorted(records_by_scene.items()):
            records = sorted(records, key=lambda x: x["step_idx"])
            scene_count = len(records)
            scene_role_totals = {
                key: np.sum(
                    [np.asarray(record[key], dtype=np.float64) for record in records],
                    axis=0,
                )
                for key in scene_role_stat_keys
            }

            scene_role_giou = safe_divide(scene_role_totals["giou_sum"], scene_count)
            scene_role_ciou = safe_divide(
                scene_role_totals["intersection"], scene_role_totals["union"]
            )
            scene_role_nonempty_miou = safe_divide(
                scene_role_totals["nonempty_iou_sum"],
                scene_role_totals["gt_present"],
            )
            scene_role_presence_rate = safe_divide(
                scene_role_totals["gt_present"], scene_count
            )
            scene_role_presence_precision = safe_divide(
                scene_role_totals["presence_tp"],
                scene_role_totals["presence_tp"] + scene_role_totals["presence_fp"],
            )
            scene_role_presence_recall = safe_divide(
                scene_role_totals["presence_tp"],
                scene_role_totals["presence_tp"] + scene_role_totals["presence_fn"],
            )
            scene_role_presence_f1 = safe_divide(
                2.0 * scene_role_totals["presence_tp"],
                2.0 * scene_role_totals["presence_tp"]
                + scene_role_totals["presence_fp"]
                + scene_role_totals["presence_fn"],
            )

            per_component = {}
            for role_idx, role_name in enumerate(COMPONENT_NAMES):
                per_component[role_name] = {
                    "gIoU": float(scene_role_giou[role_idx]),
                    "cIoU": float(scene_role_ciou[role_idx]),
                    "non_empty_mIoU": float(scene_role_nonempty_miou[role_idx]),
                    "presence_rate": float(scene_role_presence_rate[role_idx]),
                    "presence_precision": float(scene_role_presence_precision[role_idx]),
                    "presence_recall": float(scene_role_presence_recall[role_idx]),
                    "presence_f1": float(scene_role_presence_f1[role_idx]),
                    "gt_present_count": int(round(scene_role_totals["gt_present"][role_idx])),
                }

            scene_metrics[scene_id] = {
                "num_samples": scene_count,
                "step_indices": [record["step_idx"] for record in records],
                "gIoU": float(scene_role_totals["giou_sum"].sum() / (scene_count * len(COMPONENT_NAMES))),
                "cIoU": float(
                    scene_role_totals["intersection"].sum()
                    / (scene_role_totals["union"].sum() + 1e-8)
                ),
                **{
                    metric_name: float(np.mean([record[metric_name] for record in records]))
                    for metric_name in planning_metric_names
                },
                "component_wise": per_component,
            }

    if rank == 0:
        summary_line = (
            f"evaluation on EgoAfford: giou={giou}, ciou={ciou}, "
            f"first_step_sim={first_step_sim_score}, sim={sim}, order={order}, "
            f"coverage={coverage}, constraint={constraint_ratio}, "
            f"hard_constraint={hard_constraint_success}, dag_f1={dag_f1_score}"
        )
        print(f"Model path: {args.model_path}")
        print(summary_line)
        print("component-wise metrics:")
        for role_name, metrics in component_metrics.items():
            print(
                f"  {role_name}: gIoU={metrics['gIoU']:.4f}, "
                f"cIoU={metrics['cIoU']:.4f}, "
                f"non_empty_mIoU={metrics['non_empty_mIoU']:.4f}, "
                f"presence_rate={metrics['presence_rate']:.4f}, "
                f"presence_F1={metrics['presence_f1']:.4f}"
            )

        if 'real' in args.image_dir:
            output_dir = f"{args.model_path}/evaluations_multi_real"
        else:
            output_dir = f"{args.model_path}/evaluations_multi"
            
        os.makedirs(output_dir, exist_ok=True)
        text_path = os.path.join(output_dir, "egoaff.txt")
        json_path = os.path.join(output_dir, "egoaff_metrics.json")
        scene_json_path = os.path.join(output_dir, "egoaff_metrics_by_scene.json")
        scene_csv_path = os.path.join(output_dir, "egoaff_metrics_by_scene.csv")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"Model path: {args.model_path}\n{summary_line}\n")
            f.write("component-wise metrics:\n")
            for role_name, metrics in component_metrics.items():
                f.write(f"  {role_name}: {json.dumps(metrics, ensure_ascii=False)}\n")

        output_metrics = {
            "model_path": args.model_path,
            "num_samples": int(round(role_num_samples)),
            "overall": {
                "gIoU": float(giou),
                "cIoU": float(ciou),
                "first_step_similarity": float(first_step_sim_score),
                "semantic_f1": float(sim),
                "order_score": float(order),
                "coverage_score": float(coverage),
                "constraint_ratio": float(constraint_ratio),
                "hard_constraint_success": float(hard_constraint_success),
                "dag_edge_f1": float(dag_f1_score),
            },
            "component_wise": component_metrics,
            "metric_definitions": {
                "gIoU": "Mean per-sample IoU; empty prediction and empty GT score 1.",
                "cIoU": "Globally accumulated pixel intersection divided by union.",
                "non_empty_mIoU": "Mean IoU over samples where this GT component is present.",
                "presence": "A component is present iff its binary mask contains at least one positive pixel.",
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_metrics, f, ensure_ascii=False, indent=2, allow_nan=True)

        scene_output = {
            "model_path": args.model_path,
            "num_scenes": len(scene_metrics),
            "num_unique_samples": num_unique_scene_samples,
            "scene_metrics": scene_metrics,
        }
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_output, f, ensure_ascii=False, indent=2, allow_nan=True)

        csv_fields = [
            "scene_id",
            "num_samples",
            "step_indices",
            "gIoU",
            "cIoU",
            "first_step_similarity",
            "semantic_f1",
            "order_score",
            "coverage_score",
            "constraint_ratio",
            "hard_constraint_success",
            "dag_edge_f1",
        ]
        for role_name in COMPONENT_NAMES:
            csv_fields.extend([
                f"{role_name}_gIoU",
                f"{role_name}_cIoU",
                f"{role_name}_non_empty_mIoU",
                f"{role_name}_presence_rate",
                f"{role_name}_presence_precision",
                f"{role_name}_presence_recall",
                f"{role_name}_presence_f1",
                f"{role_name}_gt_present_count",
            ])

        with open(scene_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for scene_id, metrics in scene_metrics.items():
                row = {
                    "scene_id": scene_id,
                    "num_samples": metrics["num_samples"],
                    "step_indices": ";".join(map(str, metrics["step_indices"])),
                    **{
                        key: metrics[key]
                        for key in (
                            "gIoU",
                            "cIoU",
                            "first_step_similarity",
                            "semantic_f1",
                            "order_score",
                            "coverage_score",
                            "constraint_ratio",
                            "hard_constraint_success",
                            "dag_edge_f1",
                        )
                        if key in metrics
                    },
                }
                for role_name in COMPONENT_NAMES:
                    role_metrics = metrics["component_wise"][role_name]
                    row.update({
                        f"{role_name}_gIoU": role_metrics["gIoU"],
                        f"{role_name}_cIoU": role_metrics["cIoU"],
                        f"{role_name}_non_empty_mIoU": role_metrics["non_empty_mIoU"],
                        f"{role_name}_presence_rate": role_metrics["presence_rate"],
                        f"{role_name}_presence_precision": role_metrics["presence_precision"],
                        f"{role_name}_presence_recall": role_metrics["presence_recall"],
                        f"{role_name}_presence_f1": role_metrics["presence_f1"],
                        f"{role_name}_gt_present_count": role_metrics["gt_present_count"],
                    })
                writer.writerow(row)

        print(f"Per-scene metrics saved to: {scene_json_path}")
        print(f"Per-scene CSV saved to: {scene_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--image_dir", type=str, help="Path to data dir")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    parser.add_argument("--vis_freq", type=int, default=50, help="Visualize frequency")
    args = parser.parse_args()
    print("Arguments:", args)
    main(args)
