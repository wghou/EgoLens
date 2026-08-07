import json
import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import smart_resize
import numpy as np
import cv2

from torchvision.ops import masks_to_boxes
from collections import defaultdict

local_rank = int(os.environ.get("LOCAL_RANK", -1))

class EgoAffordTrainDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(self, script_args, split='train'):
        self.base_dir = script_args.image_dir
        self.system_prompt_template = "You are a helpful assistant that can see images and perform reasoning segmentation."
        self.prompt_difficulty = script_args.prompt_difficulty
        self.max_scene = script_args.dataset_size
        self.split = split

        if script_args.prompt_difficulty == "hard":
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
        elif script_args.prompt_difficulty == "easy_cot":
            self.question_template = (
                "Action: '{next_step}'\n"
                "Segment the following three functional components by bounding boxes for the action step:\n"
                "- The Direct Object (the object being manipulated).\n"
                "- The Instrument (the tool used to perform the action, absent if performed by hands).\n"
                "- The Destination (the target location or container of transfer, absent if the action means no transfer).\n"
                "A component could be absent (by putting 'None') if not used.\n"
                "\n"
                "IMPORTANT RULES:\n"
                "- segment the functional part, not the whole object.\n"
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
        
        with open(os.path.join(self.base_dir, 'invalid.json'), 'r', encoding='utf-8') as f:
            invalid = json.load(f)

        scene_dirs = sorted([d for d in os.listdir(self.base_dir)
        if os.path.isdir(os.path.join(self.base_dir, d))])
        for sceneId in scene_dirs:
            scene_idx = int(sceneId.split('_')[-1])
            if self.split == 'train':
                if scene_idx <= 100 or scene_idx > self.max_scene:
                    continue
            else:
                if scene_idx > 100:
                    continue
            scene_path = os.path.join(self.base_dir, sceneId)
            task_json = os.path.join(scene_path, 'task.json')
            mask_npz = os.path.join(scene_path, 'masks.npz')
            constraints_json = os.path.join(scene_path, 'step_constraints.json')

            if not (os.path.exists(task_json) and os.path.exists(mask_npz)):
                continue

            with open(task_json, 'r', encoding='utf-8') as f:
                steps = json.load(f).get('steps', [])
                
            with open(constraints_json, 'r', encoding='utf-8') as f:
                constraints = json.load(f).get('constraints', [])

            for step_idx in range(len(steps)):
                if sceneId in invalid:
                    if f"step_{step_idx}" in invalid[sceneId]:
                        continue
                samples.append({
                    'scene_id': sceneId,
                    'scene_path': scene_path,
                    'step_idx': step_idx,
                    'main_task': None,
                    'current_step': steps[step_idx],
                    'constraints': constraints
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
    
    def _mask_to_bbox(self, masks):
        masks = torch.from_numpy(masks).to(torch.bool)
        shape = masks.shape
        N = shape[0] * shape[1]
        masks = masks.reshape(N, shape[2], shape[3])
        
        bboxes = torch.zeros((N, 4), dtype=torch.float32)
        
        is_non_empty = masks.any(dim=-1).any(dim=-1) # (N,)
        
        if is_non_empty.any():
            non_empty_masks = masks[is_non_empty]
            computed_boxes = masks_to_boxes(non_empty_masks)
            bboxes[is_non_empty] = computed_boxes
        
        bboxes = bboxes.reshape(shape[0], shape[1], 4).cpu().numpy()
        
        return bboxes
    
    def _bbox_to_text(self, bbox):
        text = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        scene_path = item['scene_path']
        step_idx = item['step_idx']

        # task
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

        # images
        img_path = os.path.join(scene_path, f"step_{step_idx}.png")
        image_pil = Image.open(img_path).convert("RGB")
        width, height = image_pil.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            28,
            max_pixels=1000000
        )
        llm_image = image_pil.resize((resized_width, resized_height))

        # high-res images for SAM2
        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(image_cv, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image.float() - self.pixel_mean) / self.pixel_std

        # masks
        mask_data = self._load_masks_sparse(os.path.join(scene_path, 'masks.npz'))
        gt_masks = torch.from_numpy(mask_data[step_idx]).float() # (3, H, W)
        if gt_masks.max() > 1.0: gt_masks /= 255.0  # Normalize to [0, 1].
        
        # forcing text
        with open(os.path.join(scene_path, 'obj_meta.json'), 'r', encoding='utf-8') as f:
            step_meta = json.load(f)['object_meta']['steps']
        try:
            current_meta = step_meta[step_idx]
        except Exception as exc:
            print(f"Warning: failed to load object metadata for {scene_path}, step {step_idx}: {exc}")
            raise
        bboxes = self._mask_to_bbox(mask_data)[step_idx] # (3, 4)
        if self.prompt_difficulty in ["hard"]:
            prompt = self.question_template.format(
                main_task=main_task
            )
            solution = (
                "<think>\n"
                f"{remaining_steps_text}"
                "</think>\n"
                "<answer>\nAction steps:\n"
                f"{remaining_steps_text}"
                "Components:\n"
                f"direct object: {current_meta['direct_object']}, {self._bbox_to_text(bboxes[0])}\ninstrument: {current_meta['instrument']}, {self._bbox_to_text(bboxes[1])}\n destination: {current_meta['destination']}, {self._bbox_to_text(bboxes[2])}\n"
                "</answer>"
            )
        elif self.prompt_difficulty == "easy_cot":
            prompt = self.question_template.format(
                next_step=remaining_steps[0]
            )
            solution = (
                "<think>\n"
                f"{remaining_steps_text}"
                "</think>\n"
                "<answer>\n"
                "Components:\n"
                f"direct object: {current_meta['direct_object']}, {self._bbox_to_text(bboxes[0])}\ninstrument: {current_meta['instrument']}, {self._bbox_to_text(bboxes[1])}\n destination: {current_meta['destination']}, {self._bbox_to_text(bboxes[2])}\n"
                "</answer>"
            )

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ],
            "image": llm_image,
            "problem": main_task,
            "solution": solution,
            "image_path": img_path,
            "sam_image": sam_image,
            "mask": gt_masks, 
            "ref_meta": {
                "ref_id": f"scene_{item['scene_id']}_step_{step_idx}"
            },
            
            # RL keys
            "gt_text": current_meta,
            "gt_bbox": bboxes,
            "gt_steps": remaining_steps,
            "constraints_list": local_constraints,
        }
