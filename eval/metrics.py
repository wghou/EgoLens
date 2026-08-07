import re
import torch
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer, util, CrossEncoder


_CE_MODEL = None

def get_ce_model():
    global _CE_MODEL
    if _CE_MODEL is None:
        _CE_MODEL = CrossEncoder("sentence-transformers/stsb-roberta-base", device='cuda')
        _CE_MODEL.eval()
        for param in _CE_MODEL.parameters():
            param.requires_grad = False
    return _CE_MODEL

def compute_best_mask_iou(pred_masks, gt_masks, num_slots=3):
    num_preds = len(pred_masks)
    num_gts = num_slots
    
    # Initialize aggregate statistics.
    sample_ious = []
    total_inter = 0.0
    total_union = 0.0
    
    gt_areas = [gt_masks[j].float().sum().item() for j in range(num_gts)]
    if num_preds == 0:
        for j in range(num_gts):
            if gt_areas[j] == 0:
                sample_ious.append(1.0)  # Correct empty prediction.
            else:
                sample_ious.append(0.0)  # Missed ground-truth mask.
                total_union += gt_areas[j]  # Include missed area in the union.
        return sum(sample_ious)/num_gts, 0.0, total_union
    iou_matrix = np.zeros((num_preds, num_gts))
    inter_matrix = np.zeros((num_preds, num_gts))
    union_matrix = np.zeros((num_preds, num_gts))
    
    for i in range(num_preds):
        for j in range(num_gts):
            p = pred_masks[i].bool()
            g = gt_masks[j].bool()
            inter = (p & g).float().sum().item()
            union = (p | g).float().sum().item()
            
            inter_matrix[i, j] = inter
            union_matrix[i, j] = union
            
            if union > 0:
                iou_matrix[i, j] = inter / union
            else:
                iou_matrix[i, j] = 1.0 if (not p.any() and not g.any()) else 0.0
    # Find the globally optimal assignment.
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_gts = set()
    matched_preds = set()
    for p_idx, g_idx in zip(row_ind, col_ind):
        iou = iou_matrix[p_idx, g_idx]
        sample_ious.append(iou)
        total_inter += inter_matrix[p_idx, g_idx]
        total_union += union_matrix[p_idx, g_idx]
        matched_gts.add(g_idx)
        matched_preds.add(p_idx)

    for j in range(num_gts):
        if j not in matched_gts:
            if gt_areas[j] == 0:
                sample_ious.append(1.0)
            else:
                sample_ious.append(0.0)
                total_union += gt_areas[j]
    # Account for unmatched predictions.
    for i in range(num_preds):
        if i not in matched_preds:
            p_area = pred_masks[i].float().sum().item()
            total_union += p_area
    avg_iou = sum(sample_ious) / num_gts
    return avg_iou, total_inter, total_union, {p: g for p, g in zip(row_ind, col_ind)}

def compute_best_mask_iou_multi(pred_masks, gt_masks_candidates, num_slots=3):
    """
    Wrapper for multi-GT evaluation.

    pred_masks: (P,H,W) tensor/int/bool; P can be < num_slots (or even 0).
    gt_masks_candidates: list of candidates; each candidate is (num_slots,H,W) tensor
                         OR list/tuple of length num_slots.

    Returns: (best_avg_iou, best_inter, best_union, best_pred_to_gt, best_gt_masks)
      - best_gt_masks: tensor (num_slots,H,W) on cuda (for visualization/consistency)
    """
    # fallback to single gt if needed
    if gt_masks_candidates is None or len(gt_masks_candidates) == 0:
        avg_iou, inter, uni, pred_to_gt = compute_best_mask_iou(pred_masks, gt_masks_candidates, num_slots=num_slots)
        return avg_iou, inter, uni, pred_to_gt, gt_masks_candidates

    best = None  # tuple(score, avg_iou, inter, uni, pred_to_gt, gt_tensor)
    for cand_i, cand in enumerate(gt_masks_candidates):
        if cand is None:
            continue
        # normalize candidate format to tensor (num_slots,H,W)
        if isinstance(cand, (list, tuple)):
            cand = torch.stack([c if torch.is_tensor(c) else torch.as_tensor(c) for c in cand], dim=0)
        elif not torch.is_tensor(cand):
            cand = torch.as_tensor(cand)

        # ensure shape (num_slots,H,W)
        if cand.dim() == 4 and cand.shape[0] == 1:
            cand = cand[0]
        if cand.dim() != 3:
            raise ValueError(f"gt candidate must be (S,H,W), got {tuple(cand.shape)}")

        cand = cand.int().cuda()
        avg_iou, inter, uni, pred_to_gt = compute_best_mask_iou(pred_masks, cand.to(pred_masks.device), num_slots=num_slots)

        # choose by avg_iou (giou definition). tie-breaker: larger inter/union ratio
        score = float(avg_iou)
        if best is None:
            best = (score, avg_iou, inter, uni, pred_to_gt, cand, cand_i)
        else:
            if score > best[0]:
                best = (score, avg_iou, inter, uni, pred_to_gt, cand, cand_i)
    if best is None:
        # extreme fallback: treat as empty gt
        empty_gt = torch.zeros((num_slots, 1, 1), dtype=torch.int32).cuda()
        avg_iou, inter, uni, pred_to_gt = compute_best_mask_iou(pred_masks, empty_gt, num_slots=num_slots)
        return avg_iou, inter, uni, pred_to_gt, empty_gt
    return best[1], best[2], best[3], best[4], best[5], best[6]

def compute_text_planning_score(
    pred_text,
    gt_steps,
    constraints=None,
    threshold=0.6
):
    """
    Compute semantic + coverage + partial order metrics for text planning.
    Uses CrossEncoder + Hungarian one-to-one matching (aligned with reward).

    Args:
        pred_text (str or List[str]): predicted text steps
        gt_steps (List[str]): remaining GT steps
        constraints (List[List[int]] or None): list of [before_idx, after_idx] constraints,
            indices are relative to gt_steps
        threshold (float): similarity threshold for matching

    Returns:
        Dict[str,float]: all metrics
    """
    # --- 1. parse predicted steps ---
    if type(pred_text) == list:
        pred_steps = pred_text
    else:
        pred_steps = re.findall(
            r'(?:^\s*\d+\.\s*|^\s*-\s*)(.*)',
            pred_text,
            flags=re.MULTILINE
        )

    pred_steps = [s.strip() for s in pred_steps if s.strip()]
    gt_steps = [s.strip() for s in gt_steps if s.strip()]

    if len(pred_steps) == 0 or len(gt_steps) == 0:
        return {
            "semantic_f1": 0.0,
            "coverage_score": 0.0,
            "order_score": 0.0,
            "constraint_satisfaction_ratio": 0.0,
            "hard_constraint_success": 0.0,
            "dag_edge_f1": 0.0,
        }

    # --- 2. Cross-encoder pairwise similarity (aligned with reward) ---
    ce_model = get_ce_model()
    n_gen = len(pred_steps)
    n_gt = len(gt_steps)

    pairs = [(pred_steps[i], gt_steps[j]) for i in range(n_gen) for j in range(n_gt)]

    raw = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)

    sim_matrix = np.asarray(raw, dtype=np.float32).reshape(n_gen, n_gt)

    # --- 3. Hungarian one-to-one matching ---
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    K = len(row_ind)

    # matched similarities
    if K > 0:
        matched_sims = sim_matrix[row_ind, col_ind]
    else:
        matched_sims = np.zeros((0,), dtype=np.float32)

    # --- 4. Semantic precision/recall with thresholding (penalize unmatched with 0) ---
    # precision over generated steps (size n_gen)
    gen_scores = np.zeros((n_gen,), dtype=np.float32)
    for r, s in zip(row_ind.tolist(), matched_sims.tolist()):
        gen_scores[r] = s if s > threshold else 0.0
    precision = float(gen_scores.mean()) if n_gen > 0 else 0.0

    # recall/coverage over gt steps (size n_gt)
    gt_scores = np.zeros((n_gt,), dtype=np.float32)
    for c, s in zip(col_ind.tolist(), matched_sims.tolist()):
        gt_scores[c] = s if s > threshold else 0.0
    recall = float(gt_scores.mean()) if n_gt > 0 else 0.0

    semantic_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 1e-8 else 0.0
    coverage_score = recall

    # --- 5. order score (Kendall tau on matched pairs) ---
    if len(col_ind) <= 1:
        order_score = 1.0
    else:
        tau, _ = kendalltau(np.arange(len(col_ind)), col_ind)
        order_score = (tau + 1.0) / 2.0 if not np.isnan(tau) else 1.0

    # --- 6. constraint-based metrics (use gt_idx -> gen_idx mapping) ---
    # CSR is computed over every valid GT precedence edge.  A constraint
    # counts as satisfied only when both endpoint steps are matched above the
    # semantic threshold and their predicted positions respect the edge
    # direction.  Missing endpoints therefore count as unsatisfied rather
    # than disappearing from the denominator.
    constraint_satisfaction_ratio = 1.0
    hard_constraint_success = 1.0
    dag_edge_f1 = 1.0

    if constraints:
        # Build gt_idx -> gen_idx map using thresholded matched pairs
        gt_to_gen = {}
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if sim_matrix[r, c] > threshold:
                gt_to_gen[c] = r

        satisfied = 0

        pred_edges = set()
        gt_edges = set()

        for before_idx, after_idx in constraints:
            if before_idx >= n_gt or after_idx >= n_gt:
                continue

            gt_edges.add((before_idx, after_idx))

            if before_idx in gt_to_gen and after_idx in gt_to_gen:
                if gt_to_gen[before_idx] < gt_to_gen[after_idx]:
                    satisfied += 1
                    pred_edges.add((before_idx, after_idx))
                else:
                    pred_edges.add((before_idx, after_idx))

        # When valid constraints exist, unmatched endpoints and reversed
        # matched endpoints both remain in gt_edges but not in `satisfied`.
        # An empty constraint set is vacuously satisfied (the initialized 1.0).
        if len(gt_edges) > 0:
            constraint_satisfaction_ratio = satisfied / len(gt_edges)
            hard_constraint_success = 1.0 if satisfied == len(gt_edges) else 0.0

        true_positives = len(pred_edges & gt_edges)
        p = true_positives / len(pred_edges) if len(pred_edges) > 0 else 0.0
        r = true_positives / len(gt_edges) if len(gt_edges) > 0 else 0.0
        dag_edge_f1 = (2 * p * r / (p + r)) if (p + r) > 1e-8 else 0.0

    return {
        "semantic_f1": float(semantic_f1),
        "coverage_score": float(coverage_score),
        "order_score": float(order_score),
        "constraint_satisfaction_ratio": float(constraint_satisfaction_ratio),
        "hard_constraint_success": float(hard_constraint_success),
        "dag_edge_f1": float(dag_edge_f1),
    }

def compute_step_cand_similarity(pred_step: str, cand_key: str) -> float:
    pred_step = (pred_step or "").strip()
    cand_key = (cand_key or "").strip()
    if not pred_step or not cand_key:
        return 0.0
    ce_model = get_ce_model()
    s = float(ce_model.predict([(pred_step, cand_key)], show_progress_bar=False)[0])
    return s
