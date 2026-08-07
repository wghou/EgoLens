from transformers import AutoProcessor
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
import torch
import json
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from PIL import Image
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
import cv2
from qwen_vl_utils import smart_resize


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


class EgoAffordDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(self, base_dir, split='val', llm_type='gt'):
        self.base_dir = base_dir
        self.llm_type = llm_type
        self.system_prompt_template = "You are a helpful assistant that can see images and perform reasoning segmentation."
        self.split = split

        self.question_template = (
            "Action: '{next_step}'\n"
            "Segment the following three functional components by bounding boxes for the action step:\n"
            "- The Direct Object (the object being manipulated).\n"
            "- The Instrument (the tool used to perform the action, absent if performed by hands).\n"
            "- The Destination (the target location or container of transfer, absent if the action means no transfer).\n"
            "A component could be absent (by putting 'None') if not used.\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- egment the functional part, not the whole object.\n"
            "- Analyze all objects in the image carefully against the target description in <think>.\n"
            "- Output components' bounding boxes inside <answer> tags. Use [0, 0, 0, 0] if None.\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "<think>"
            "[Your step-by-step analysis and reasoning]"
            "</think>\n"
            "<answer>\n"
            "Components:\n"
            "direct object: [x1, y1, x2, y2]\ninstrument: [x1, y1, x2, y2]\n destination: [x1, y1, x2, y2]\n"
            "</answer>"
        )

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        if not os.path.exists(self.base_dir):
            print(f"Warning: dataset directory not found: {self.base_dir}")
            return samples

        scene_dirs = sorted([d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))], key=lambda x: int(x.split('_')[-1]))
        
        if self.llm_type != 'gt':
            if 'real' in self.base_dir:
                with open(os.path.join("llm/vlm_output_real", f"{self.llm_type}.json"), 'r') as f:
                    llm_output = json.load(f)
            else:
                with open(os.path.join("llm/vlm_output", f"{self.llm_type}.json"), 'r') as f:
                    llm_output = json.load(f)
        
        img_count = 0
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
                    'current_step': steps[step_idx] if self.llm_type == 'gt' else llm_output[str(img_count)]['steps'][0],
                    'constraints': constraints,
                    'alt_json': alt_json,
                    'multi_mask_npz': multi_mask_npz,
                    'multi_index_json': multi_index_json,
                })
                img_count += 1
                
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
            try:
                gt = torch.from_numpy(multi[mi]).float()
            except Exception as exc:
                print(f"Warning: failed to load ground-truth mask for {scene_path}, step {step_idx}, index {mi}: {exc}")
                raise
            if gt.max() > 1.0:
                gt /= 255.0
            gt_list.append(gt)
        # If candidates list empty or mapping missing, fallback to old single-gt
        if len(gt_list) == 0:
            return _fallback_single_gt()
        return gt_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        scene_path = item['scene_path']
        step_idx = item['step_idx']

        with open(os.path.join(scene_path, 'task.json'), 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        remaining_steps = task_data['steps'][step_idx:]
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
        
        prompt = self.question_template.format(
            next_step=item['current_step']
        )

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ],
            "image": llm_image,
            "image_path": img_path,
            "sam_image": sam_image,
            "mask": gt_masks, 
            "gt_masks_candidates": gt_masks_candidates,
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
        
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        pad_token_id = self.processor.tokenizer.pad_token_id
        self.processor.pad_token_id = pad_token_id
        self.processor.eos_token_id = self.processor.tokenizer.eos_token_id
        
        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]

    @torch.no_grad()
    def evaluate_single(self, input_data, data_idx):
        """Evaluate segmentation for a single image."""
        messages = [input_data["prompt"]]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        image_inputs = [input_data["image"]] * len(texts)
        inputs = self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(
            device="cuda", dtype=self.dtype
        )

        llm_out = self.model.generate(max_length=2048, use_cache=True, do_sample=False, **inputs)
        completion_text = self.processor.batch_decode(llm_out, skip_special_tokens=True)

        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pos = torch.where(llm_out == self.processor.tokenizer.pad_token_id)
        new_attention_mask[pos] = 0
        inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
        inputs.update({"sam_images": input_data["sam_image"].unsqueeze(0).repeat(len(texts), 1, 1, 1)})

        output, low_res_masks = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)

        pred_masks = []
        for idx, mask in enumerate(low_res_masks):
            pred_masks.append(self.model.postprocess_masks(mask, orig_hw=input_data["mask"].shape[-2:]))
        pred_masks = torch.cat(pred_masks, dim=0)
        pred_masks = (pred_masks[:, 0] > 0).int()

        masks_list = input_data["mask"].int().cuda() # (3, H, W)
        
        intersection, union, acc_iou = 0.0, 0.0, 0.0

        ROLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # meta may provide multiple GT candidates for this state.
        # Each candidate is a (3,H,W) tensor.
        gt_candidates = input_data.get("gt_masks_candidates", None)
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

        # pred_masks is list length=3, each is (H,W) tensor (likely on cuda)
        for cand_gt in gt_candidates:
            # cand_gt: (3,H,W)
            if isinstance(cand_gt, list):
                cand_gt = torch.stack(cand_gt, dim=0)
            # build gt list per channel to reuse existing per-channel logic
            cand_masks_list = [cand_gt[i] for i in range(3)]

            cand_intersection = 0.0
            cand_union = 0.0
            cand_acc_iou = 0.0

            for idx, output_i in enumerate(pred_masks):
                mask_i = cand_masks_list[idx]
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone().int(),
                    mask_i.contiguous().cuda().int(),
                    2,
                    ignore_index=255
                )
                cand_intersection += intersection_i
                cand_union += union_i

                iou = intersection_i / (union_i + 1e-5)
                cand_acc_iou += iou
                cand_acc_iou[union_i == 0] += 1.0  # same rule as before

            # choose best by mean foreground IoU over 3 channels (class 1)
            # (keep it simple; no need to report which candidate matched)
            score = (cand_acc_iou[1] / 3.0).detach().float()
            score_val = score.item() if hasattr(score, "item") else float(score)

            if (best_score is None) or (score_val > best_score):
                best_score = score_val
                best_intersection = cand_intersection
                best_union = cand_union
                best_acc_iou = cand_acc_iou
                best_gt_for_vis = cand_masks_list

        # Use best candidate's stats for aggregation
        intersection = best_intersection
        union = best_union
        acc_iou = best_acc_iou
        masks_list = best_gt_for_vis  # for visualization only (no candidate id returned)
        
        if self.args.vis and data_idx % args.vis_freq == 0:
            good_dir = os.path.join(self.args.model_path, f'evaluations_multi/egoaff_noplan_{self.args.type}_vis')
            os.makedirs(good_dir, exist_ok=True)
            base_image_src = Image.open(input_data['image_path'])
            base_img = np.array(resize_longest(base_image_src, 1024))
            if base_img.max() <= 1.0: base_img = (base_img * 255).astype(np.uint8)
            
            h, w = base_img.shape[:2]
            concat_rows = []
            for gt_mask, pred_mask, color in zip(masks_list, pred_masks, ROLE_COLORS):
                gt_mask_np = gt_mask.cpu().numpy() if hasattr(gt_mask, 'cpu') else gt_mask
                pd_mask_np = pred_mask.cpu().numpy() if hasattr(pred_mask, 'cpu') else pred_mask
                
                vis_gt = overlay_mask(base_img, gt_mask_np, color, alpha=0.4)
                vis_pd = overlay_mask(base_img, pd_mask_np, color, alpha=0.4)
                
                row = np.hstack([vis_gt, vis_pd])
                concat_rows.append(row)
                
            final_vis = np.vstack(concat_rows)
            save_path = os.path.join(good_dir, f"{data_idx}th.png")
            Image.fromarray(final_vis).save(save_path)
            with open(os.path.join(good_dir, f"{data_idx}th.txt"), "w") as f:
                    f.write(f'{completion_text[0]}')

        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        return intersection, union, acc_iou, len(masks_list)


def main(args):
    os.makedirs(f"{args.model_path}/evaluations_multi", exist_ok=True)

    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = EgoAffEvaluator(args)

    dataset = EgoAffordDataset(base_dir=args.image_dir, llm_type=args.type)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank)
    dataloader = DataLoader(dataset, 1, False, sampler=sampler, collate_fn=lambda batch: list(batch))

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    for idx, batch_data in enumerate(tqdm(dataloader)):
        assert len(batch_data) == 1, "Only batch_size=1 is supported"
        intersection, union, acc_iou, num_mask = evaluator.evaluate_single(batch_data[0], idx)
        intersection_meter.update(intersection, n=num_mask)
        union_meter.update(union, n=num_mask)
        acc_iou_meter.update(acc_iou, n=num_mask)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-8)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    print(f"Model path: {args.model_path}")
    print(f"Evaluation complete: gIoU={giou}, cIoU={ciou}")
    with open(f"{args.model_path}/evaluations_multi/egoaff_noplan_{args.type}.txt", "w") as f:
        f.write(f"Model path: {args.model_path}; gIoU={giou}, cIoU={ciou}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--image_dir", type=str, help="Path to data dir")
    parser.add_argument("--type", type=str, help="Backbone LLM")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    parser.add_argument("--vis_freq", type=int, default=50, help="Visualize frequency")
    args = parser.parse_args()
    print("Arguments:", args)
    main(args)
