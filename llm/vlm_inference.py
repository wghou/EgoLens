import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
import re
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from llm import LLM
from typing import Optional

def load_masks_sparse(path):
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

def robust_json_loads(text: str) -> Optional[dict]:
    """Extract a JSON object from an LLM response."""
    if not text or not text.strip():
        print("Warning: empty LLM response.")
        return None

    # Parse the complete response first.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Fall back to JSON code blocks.
    json_code_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_code_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Finally, extract the outermost object.
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    print("Warning: no valid JSON found in LLM response.")
    return None

class EgoAffordDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024
    llm_img_size = 224
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.question_template = (
            "Goal: '{main_task}'\n"
            "Please analyze the scene and infer the remaining action steps to accomplish the goal based on the image ({size}*{size}).\n"
            "Then segment the following three functional components by a central point and a bounding box for the next action step:\n"
            "1. The Direct Object (the object being manipulated).\n"
            "2. The Instrument (the tool used to perform the action).\n"
            "3. The Destination (the target location or container).\n"
            "A component could be absent (by putting 'None') if not used (e.g., no instrument for hand actions, or no destination for in-place actions).\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- Actions are so defined: having exactly one main verb, one singular direct object, at most one instrument (using 'with'), and at most one destination/surface (using 'on', 'into', etc.).\n"
            "- You MUST infer actions from the image. Do NOT reuse any example or template content.\n"
            "- When segmenting objects, do not just segement the whole object but its functional part, i.e. the spout of a pot. Use both point and bbox to segment.\n"
            "- Only output the NEXT action step's components.\n"
            "- If a component does not exist, output 'none' instead of the point.\n"
            "\n"
            """
            OUTPUT FORMAT (JSON only, no extra text):
            {{
            "steps": ["<step 1>", "<step 2>", ...],
            "components": {{
                "direct_object": {{ "text": "<object name>", "point": [x, y], "bbox": [xmin, ymin, xmax, ymax] }},
                "instrument": {{ ... }} or null,
                "destination": {{ ... }} or null
            }}
            }}
            - "steps": one or more elements, depends on the number of inferred action steps.
            - "point": [x, y] coordinates of the functional part you are segmenting.
            - "bbox": [xmin, ymin, xmax, ymax] of the same functional part.
            - If a component does not exist, output null for that entire component object.
            """
        )

        self.samples = self._build_samples()
        
    def _build_samples(self):
        samples = []
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(self.base_dir)
                if d.startswith("scene_")
                and os.path.isdir(os.path.join(self.base_dir, d))
            ],
            key=lambda x: int(x.split("_")[-1]),
        )
        for sceneId in scene_dirs:
            scene_idx = int(sceneId.split("_")[-1])
            if scene_idx > 100:
                continue

            scene_path = os.path.join(self.base_dir, sceneId)
            task_json = os.path.join(scene_path, 'task.json')
            if not os.path.exists(task_json): continue
            
            with open(task_json, 'r') as f:
                steps = json.load(f).get('steps', [])
            
            # Treat each step as an independent sample.
            for step_idx in range(len(steps)):
                samples.append({
                    'scene_id': sceneId,
                    'scene_path': scene_path,
                    'step_idx': step_idx
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        scene_path = item['scene_path']
        step_idx = item['step_idx']

        with open(os.path.join(scene_path, 'task.json'), 'r') as f:
            task_data = json.load(f)
        main_task = task_data['main_task']
        remaining_steps = task_data['steps'][step_idx:]
        
        img_path = os.path.join(scene_path, f"step_{step_idx}.png")

        image_pil = Image.open(img_path).convert("RGB").resize((self.llm_img_size, self.llm_img_size), Image.LANCZOS)
        
        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(image_cv, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image.float() - self.pixel_mean) / self.pixel_std
        
        # Load ground-truth masks.
        mask_data = load_masks_sparse(os.path.join(scene_path, 'masks.npz')) # (S, 3, H, W)
        # Select the current step's three channels.
        gt_masks = mask_data[step_idx] # (3, H, W)
        gt_masks = torch.from_numpy(gt_masks).float() 
        
        # Build the model prompt.
        prompt_text = self.question_template.format(
            main_task=main_task,
            size=self.llm_img_size
        )

        return {
            'image': image_pil,       # Input for the processor.
            "sam_image": sam_image,   # Input for SAM.
            "mask": gt_masks,         # Shape: (3, H, W).
            "problems": [prompt_text],
            "remaining_steps_gt": remaining_steps,
            "image_path": img_path,
            "step_idx": step_idx
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--type", type=str, default="gpt")
    args = parser.parse_args()
    
    output_dir = "./vlm_output"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = EgoAffordDataset("/Path/to/EgoAfford")
    dataloader = DataLoader(dataset, 1, False, collate_fn=lambda batch: list(batch))
    
    llm_client = LLM(model=args.type, resume=False)
    
    if os.path.exists(os.path.join(output_dir, f"{args.type}.json")):
        with open(os.path.join(output_dir, f"{args.type}.json"), 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    for idx, batch_data in enumerate(tqdm(dataloader)):
        if (str(idx) in all_results.keys()) and (all_results[str(idx)] is not None):
            continue
        assert len(batch_data) == 1, "Only batch_size=1 is supported"
        sample = batch_data[0]
        
        sys_prompt = (f"You are a Action Analysis Agent. Your goal is to find the specific regions of an object where an action occurs. ")
        
        res = llm_client.chat(user_text=sample['problems'][0], image_path=sample['image'], system_text=sys_prompt)
        
        result = robust_json_loads(res)
        
        all_results[idx] = result
        
    with open(os.path.join(output_dir, f"{args.type}.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
