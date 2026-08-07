import os
import json
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

from sentence_transformers import SentenceTransformer

from ..eval.metrics import compute_text_planning_score


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
            "- When segmenting objects, do not just segement the whole object but its functional part, i.e. the spout of a pot. Use both point and bbox to segment."
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
        scene_dirs = sorted([d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))])
        for sceneId in scene_dirs:
            scene_path = os.path.join(self.base_dir, sceneId)
            task_json = os.path.join(scene_path, 'task.json')
            if not os.path.exists(task_json): continue
            
            with open(task_json, 'r') as f:
                steps = json.load(f).get('steps', [])
            
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
        remaining_steps = task_data['steps'][step_idx:]
        
        constraints_json = os.path.join(scene_path, 'step_constraints.json')
        with open(constraints_json, 'r') as f:
            constraints = json.load(f).get('constraints', [])
        
        local_constraints = []
        offset = item['step_idx'] + 1
        for a, b in constraints:
            la, lb = a - offset, b - offset
            if 0 <= la < len(remaining_steps) and 0 <= lb < len(remaining_steps):
                local_constraints.append([la, lb])

        return {
            "remaining_steps_gt": remaining_steps,
            "step_idx": step_idx,
            "constraints": local_constraints
        }
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--type", type=str, default="gpt")
    args = parser.parse_args()
    
    output_dir = "./text_output"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = EgoAffordDataset("/Path/to/EgoAfford")
    dataloader = DataLoader(dataset, 1, False, collate_fn=lambda batch: list(batch))
    
    with open(os.path.join("vlm_output", f"{args.type}.json"), 'r') as f:
        all_results = json.load(f)
    
    semantic_f1, order_score, coverage_score = 0, 0, 0
    constraint_ratio, hard_constraint_success, dag_edge_f1 = 0, 0, 0
    for idx, batch_data in enumerate(tqdm(dataloader)):
        sample = batch_data[0]
        meta = all_results[str(idx)]['steps']
        
        text_scores = compute_text_planning_score(meta, sample['remaining_steps_gt'], sample['constraints'])
        
        semantic_f1 += text_scores['semantic_f1']
        order_score += text_scores['order_score']
        coverage_score += text_scores['coverage_score']
        constraint_ratio += text_scores['constraint_satisfaction_ratio']
        hard_constraint_success += text_scores['hard_constraint_success']
        dag_edge_f1 += text_scores['dag_edge_f1']
            
    semantic_f1 = semantic_f1 / len(dataloader)
    order_score = order_score / len(dataloader)
    coverage_score = coverage_score / len(dataloader)
    constraint_ratio = constraint_ratio / len(dataloader)
    hard_constraint_success = hard_constraint_success / len(dataloader)
    dag_edge_f1 = dag_edge_f1 / len(dataloader)
    
    result = {"semantic_f1": float(semantic_f1),
        "order_score": float(order_score),
        "coverage_score": float(coverage_score),
        "constraint_ratio": float(constraint_ratio),
        "hard_constraint_success": float(hard_constraint_success),
        "dag_edge_f1": float(dag_edge_f1)
    }
    
    with open(os.path.join(output_dir, f"{args.type}.json"), 'w') as f:
        json.dump(result, f, indent=2)