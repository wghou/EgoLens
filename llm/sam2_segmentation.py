import os
import json
import argparse
import re
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from metrics import compute_step_cand_similarity

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

def overlay_three_masks_additive(image: np.ndarray,
                                 masks: list,
                                 alpha: float = 0.45):
    if len(masks) != 3:
        raise ValueError(f"Expected masks shape (3,H,W), got {masks.shape}")

    img = image

    h, w = img.shape[:2]

    masks = np.array(masks)
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

def load_multi_gt_masks(scene_path: str, step_idx: int) -> List[torch.Tensor]:
    """
    Load multi-candidate GT masks for a given state(step_idx).
    Requires:
      - alternatives.json
      - masks_multi_index.json
      - masks_multi.npz
    Fallback to old masks.npz if multi files missing.

    Returns: list of torch.FloatTensor, each (3,H,W).
    """
    alt_path = os.path.join(scene_path, "alternatives.json")
    mm_idx_path = os.path.join(scene_path, "masks_multi_index.json")
    mm_path = os.path.join(scene_path, "masks_multi.npz")

    def _fallback_single_gt():
        mask_data = load_masks_sparse(os.path.join(scene_path, 'masks.npz'))
        gt = torch.from_numpy(mask_data[step_idx]).float()
        if gt.max() > 1.0:
            gt /= 255.0
        return [gt]
    if not (os.path.exists(alt_path) and os.path.exists(mm_idx_path) and os.path.exists(mm_path)):
        return _fallback_single_gt()
    alt = json.load(open(alt_path, 'r', encoding='utf-8'))
    idx_obj = json.load(open(mm_idx_path, 'r', encoding='utf-8'))
    entries = idx_obj.get('entries', [])
    multi = load_masks_sparse(mm_path)  # (M,3,H,W)
    gt_list = []
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

def load_multi_gt_candidate_texts(scene_path: str, step_idx: int, item: dict):
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
        # Iterate over scenes.
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
        next_step = task_data['steps'][step_idx]
        remaining_steps = task_data['steps'][step_idx:]
        
        img_path = os.path.join(scene_path, f"step_{step_idx}.png")

        image_pil = Image.open(img_path).convert("RGB").resize((self.llm_img_size, self.llm_img_size), Image.LANCZOS)
        
        # Prepare the normalized SAM input.
        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(image_cv, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image.float() - self.pixel_mean) / self.pixel_std
        
        # Load multi-candidate ground-truth masks.
        gt_masks_candidates = load_multi_gt_masks(scene_path, step_idx)  # List[(3,H,W)]
        gt_masks = gt_masks_candidates[0]  # keep old key 'mask' for backward compatibility
        
        # Build the model prompt.
        prompt_text = self.question_template.format(
            main_task=main_task,
            size=self.llm_img_size
        )
        
        gt_candidate_texts = load_multi_gt_candidate_texts(scene_path, step_idx, item)

        return {
            'image': image_pil,
            "sam_image": sam_image,
            "mask": gt_masks,             # (3, H, W)
            "gt_masks_candidates": gt_masks_candidates,  # multi GT
            "problems": [prompt_text],
            "remaining_steps_gt": remaining_steps,
            "gt_candidate_texts": gt_candidate_texts,
            "image_path": img_path,
            "step_idx": step_idx
        }


class CoordinateConverter:
    """
    Maps coordinates from LLM input space to the original image space.
    Assumes the LLM input is a direct resize of the original image.
    """
    def __init__(self, llm_size=(224, 224), original_size=(1024, 768)):
        self.llm_w, self.llm_h = llm_size
        self.orig_w, self.orig_h = original_size

    def _convert_point(self, point_xy):
        """Map an LLM-space point to the original image."""
        x, y = point_xy
        new_x = x * (self.orig_w / self.llm_w)
        new_y = y * (self.orig_h / self.llm_h)
        return (new_x, new_y)

    def _convert_bbox(self, bbox_xyxy):
        """Map an LLM-space bounding box to the original image."""
        xmin, ymin, xmax, ymax = bbox_xyxy
        scale_x = self.orig_w / self.llm_w
        scale_y = self.orig_h / self.llm_h
        return [
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y
        ]

    def convert_component(self, comp_data):
        """
        Map a component's point and bounding box to the original image.
        """
        if comp_data is None:
            return None
        new_comp = {}
        if "point" in comp_data and comp_data["point"] is not None:
            new_comp["point"] = self._convert_point(comp_data["point"])
        if "bbox" in comp_data and comp_data["bbox"] is not None:
            new_comp["bbox"] = self._convert_bbox(comp_data["bbox"])
        return new_comp if new_comp else None


class SAM2MaskGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        llm_size=(224, 224),
        original_size=(1024, 768)
    ):
        self.device = device
        self.predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint_path, device=device)
        )
        self.converter = CoordinateConverter(llm_size, original_size)

    def _prepare_image(self, image_source):
        if isinstance(image_source, str):
            image = np.array(Image.open(image_source).convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            image = image_source
        else:
            raise TypeError("image_source must be path string or np.ndarray")
        self.predictor.set_image(image)
        return image.shape[:2]   # H, W

    def _points_to_input(self, point_xy):
        points = np.array([point_xy], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        return points, labels

    def _bbox_to_input(self, bbox_xyxy):
        return np.array([bbox_xyxy], dtype=np.float32)

    def generate_mask_with_point(self, image_source, point_xy_llm):
        """
        Generate a mask from a point prompt.
        """
        point_xy_orig = self.converter._convert_point(point_xy_llm)
        H, W = self._prepare_image(image_source)
        points, labels = self._points_to_input(point_xy_orig)

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def generate_mask_with_point_and_bbox(self, image_source, point_xy_llm, bbox_xyxy_llm):
        """
        Generate a mask from point and box prompts.
        """
        point_xy_orig = self.converter._convert_point(point_xy_llm)
        bbox_xyxy_orig = self.converter._convert_bbox(bbox_xyxy_llm)
        H, W = self._prepare_image(image_source)
        points, labels = self._points_to_input(point_xy_orig)
        bbox = self._bbox_to_input(bbox_xyxy_orig)

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def generate_masks_both_modes(self, image_source, component_llm):
        """
        Generate point-only and point-plus-box masks for one component.
        """
        results = {"point_only": None, "point_bbox": None}
        if component_llm is None:
            return results

        point = component_llm.get("point")
        bbox = component_llm.get("bbox")

        if point is not None:
            results["point_only"] = self.generate_mask_with_point(image_source, point)

        if point is not None and bbox is not None:
            results["point_bbox"] = self.generate_mask_with_point_and_bbox(
                image_source, point, bbox
            )
        elif point is not None:
            # Fall back to the point prompt when no box is available.
            results["point_bbox"] = results["point_only"]
        return results

def compute_iou(pred_mask, gt_mask):
    count = 0
    p, g = pred_mask.bool(), gt_mask.bool()
    inter = (p & g).sum().item()
    union = (p | g).sum().item()
    if union > 0:
        iou = inter / (union + 1e-6)  
    elif union == 0:
        iou = 1.0
        count = 1
    else:
        iou = 0.0
    
    return iou, inter, union, count

def overlay_mask(image, mask, color, alpha=0.4):
    """Overlay a binary mask on an image."""
    # Convert the mask to a boolean array.
    mask = mask.astype(bool)
    
    # Create the color overlay.
    overlay = image.copy()
    overlay[mask] = color
    
    # Blend the image and overlay.
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--type", type=str, default="gpt")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    parser.add_argument("--vis_freq", type=int, default=50, help="Visualize frequency")
    args = parser.parse_args()
    
    output_dir = "./seg_output"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = EgoAffordDataset("/Path/to/EgoAfford")
    dataloader = DataLoader(dataset, 1, False, collate_fn=lambda batch: list(batch))
    
    if args.task == 'all':
        prefix = args.type
    elif args.task == 'mask':
        prefix = f'{args.type}_mask'
    elif args.task == 'hard':
        prefix = f'{args.type}_mask_hard'
    
    with open(os.path.join("vlm_output", f"{prefix}.json"), 'r') as f:
        all_results = json.load(f)

    # Initialize the generator with input and original dimensions.
    generator = SAM2MaskGenerator(
        checkpoint_path="/Path/to/sam2_hiera_large.pt",
        model_cfg="sam2_hiera_l.yaml",
        llm_size=(224, 224),
        original_size=(1024, 768)
    )
    
    KEYS = ['direct_object', 'instrument', 'destination']
    
    intersection, union, all_iou, empty = 0, 0, 0, 0
    intersection_box, union_box, all_iou_box, empty_box = 0, 0, 0, 0
    first_step_sim = 0
    for idx, batch_data in enumerate(tqdm(dataloader)):
        sample = batch_data[0]
        meta = all_results[str(idx)]['components']
        gt_candidates = sample.get("gt_masks_candidates", None)
        if gt_candidates is None:
            gt_candidates = [sample["mask"]]  # fallback single GT (3,H,W)
        
        point_only_masks_list = []
        point_bbox_masks_list = []
        
        for role_idx, key in enumerate(KEYS):
            if meta[key] is None:
                masks = {"point_only": torch.zeros(768, 1024), "point_bbox": torch.zeros(768, 1024)}
            else:
                comp_llm = {
                    "point": meta[key]['point'],
                    "bbox": meta[key]['bbox']
                }
                masks = generator.generate_masks_both_modes(sample['image_path'], comp_llm)
                if masks['point_only'] is None:
                    masks['point_only'] = torch.zeros(768, 1024)
                if masks['point_bbox'] is None:
                    masks['point_bbox'] = torch.zeros(768, 1024)
                
            point_only_masks_list.append(masks['point_only'])
            point_bbox_masks_list.append(masks['point_bbox'])
            
        # -------- oracle over GT candidates (whole 3-channel set) --------
        # We pick the best-matched GT candidate for point_only and point_bbox separately.
        best_gt_point_only = None
        best_score_point_only = None
        best_stats_point_only = None  # (inter_sum, union_sum, iou_sum, empty_sum)

        best_gt_point_bbox = None
        best_score_point_bbox = None
        best_stats_point_bbox = None
        
        best_cand_idx = None

        for cand_i, cand in enumerate(gt_candidates):
            # cand: (3,H,W) tensor
            if isinstance(cand, np.ndarray):
                cand = torch.from_numpy(cand).float()

            # point_only stats across 3 roles
            inter_s = uni_s = iou_s = empty_s = 0.0
            for role_idx in range(3):
                gt_mask = cand[role_idx]
                iou, inter, uni, count = compute_iou(torch.Tensor(point_only_masks_list[role_idx]),
                                                    torch.Tensor(gt_mask))
                inter_s += inter
                uni_s += uni
                iou_s += iou
                empty_s += count
            score = float(iou_s) / 3.0
            if (best_score_point_only is None) or (score > best_score_point_only):
                best_score_point_only = score
                best_gt_point_only = cand
                best_stats_point_only = (inter_s, uni_s, iou_s, empty_s)

            # point_bbox stats across 3 roles
            inter_s = uni_s = iou_s = empty_s = 0.0
            for role_idx in range(3):
                gt_mask = cand[role_idx]
                iou, inter, uni, count = compute_iou(torch.Tensor(point_bbox_masks_list[role_idx]),
                                                    torch.Tensor(gt_mask))
                inter_s += inter
                uni_s += uni
                iou_s += iou
                empty_s += count
            score = float(iou_s) / 3.0
            if (best_score_point_bbox is None) or (score > best_score_point_bbox):
                best_score_point_bbox = score
                best_gt_point_bbox = cand
                best_stats_point_bbox = (inter_s, uni_s, iou_s, empty_s)
                best_cand_idx = cand_i

        # accumulate best stats
        inter_s, uni_s, iou_s, empty_s = best_stats_point_only
        intersection += inter_s
        union += uni_s
        all_iou += iou_s
        empty += empty_s

        inter_s, uni_s, iou_s, empty_s = best_stats_point_bbox
        intersection_box += inter_s
        union_box += uni_s
        all_iou_box += iou_s
        empty_box += empty_s
        
        if args.task == 'all':
            pred_first_step = all_results[str(idx)]['steps'][0]
            gt_cand_texts = sample.get("gt_candidate_texts", [])
            best_cand_key = gt_cand_texts[best_cand_idx] if (best_cand_idx is not None and best_cand_idx < len(gt_cand_texts)) else ""
            first_step_cand_match_sim = compute_step_cand_similarity(pred_first_step, best_cand_key)
            first_step_sim += first_step_cand_match_sim
        else:
            first_step_sim += 0.0

        # for visualization: use best-matched GTs
        gt_masks_list_point_only = [best_gt_point_only[i] for i in range(3)]
        gt_masks_list_point_bbox = [best_gt_point_bbox[i] for i in range(3)]

        if args.vis and idx % args.vis_freq == 0:
            good_dir = os.path.join(output_dir, f'{prefix}')
            os.makedirs(good_dir, exist_ok=True)
            base_img = np.array(Image.open(sample['image_path']))
            if base_img.max() <= 1.0: base_img = (base_img * 255).astype(np.uint8)
            
            mul_vis = overlay_three_masks_additive(image=base_img, masks=point_bbox_masks_list)
            mul_vis.save(os.path.join(good_dir, f"{idx}th_mul.png"), format="PNG", optimize=True)
            mul_gt = overlay_three_masks_additive(image=base_img, masks=gt_masks_list_point_bbox)
            mul_gt.save(os.path.join(good_dir, f"{idx}th_gt.png"), format="PNG", optimize=True)
            
            def create_grid_vis(preds_list, suffix):
                rows_gt = []
                rows_pd = []
                ROLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                
                # choose GT list depending on suffix
                gt_list = gt_masks_list_point_only if suffix == "point_only" else gt_masks_list_point_bbox
                for g_m, p_m, color in zip(gt_list, preds_list, ROLE_COLORS):
                    g_m_np = g_m.cpu().numpy() if hasattr(g_m, 'cpu') else np.array(g_m)
                    p_m_np = p_m.cpu().numpy() if hasattr(p_m, 'cpu') else np.array(p_m)
                    
                    vis_gt = overlay_mask(base_img.copy(), g_m_np, color, alpha=0.4)
                    vis_pd = overlay_mask(base_img.copy(), p_m_np, color, alpha=0.4)
                    
                    rows_gt.append(vis_gt)
                    rows_pd.append(vis_pd)
                
                top_row = np.hstack(rows_gt)
                bottom_row = np.hstack(rows_pd)

                final_grid = np.vstack([top_row, bottom_row])
                
                save_path = os.path.join(good_dir, f"batch_{idx}_{suffix}.png")
                Image.fromarray(final_grid).save(save_path)

            create_grid_vis(point_only_masks_list, "point_only")
            create_grid_vis(point_bbox_masks_list, "point_bbox")
            
    giou = all_iou / len(dataloader) / 3
    ciou = intersection / (union + 1e-10)
    matched_empty = empty / len(dataloader) / 3
    
    giou_box = all_iou_box / len(dataloader) / 3
    ciou_box = intersection_box / (union_box + 1e-10)
    matched_empty_box = empty_box / len(dataloader) / 3
    
    first_step_sim = first_step_sim / len(dataloader)
    
    result = {'point_only': {'giou': float(giou), 'ciou': float(ciou), 'empty_matched': float(matched_empty)},
              'point_bbox': {'giou': float(giou_box), 'ciou': float(ciou_box), 'empty_matched': float(matched_empty_box), 'first_step': float(first_step_sim)}}
    
    with open(os.path.join(output_dir, f"{prefix}.json"), 'w') as f:
        json.dump(result, f, indent=2)
