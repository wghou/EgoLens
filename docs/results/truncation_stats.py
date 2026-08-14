#!/usr/bin/env python3
"""Measure how many training samples get truncated at max_length=2048, and which
branch of build_answer_only_labels they land in.

Replicates the collator inside Qwen2VLSFTTrainer.__init__ (which is a closure and
therefore not importable) and reuses build_answer_only_labels unchanged. The
tokenization runs inside the DataLoader workers, so the large sam_image / mask
tensors never cross a process boundary.

Three outcomes matter, see build_answer_only_labels:

  ok        <answer> and </answer> both survive -> supervised from <answer> onward
  no_super  <answer> survives, </answer> truncated away -> neither branch fires,
            labels stay -100 everywhere, the sample contributes no language loss
            (and a micro-batch made entirely of these yields nan)
  full      <answer> itself truncated away -> labels[:] = ids, the sample is
            trained to reproduce the prompt template

Usage: python docs/results/truncation_stats.py [--max-length 2048] [--workers 16]
"""
import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, os.environ.get("REPO_ROOT", "/workspace/EgoLens"))
from src.open_r1.dataset import EgoAffordTrainDataset          # noqa: E402
from src.open_r1.trainer.trainer_sft import (                  # noqa: E402
    build_answer_only_labels,
    find_subseq_1d,
)

ARGS = None
_PROC = None


def processor():
    """One processor per worker process, configured like the trainer does."""
    global _PROC
    if _PROC is None:
        p = AutoProcessor.from_pretrained(ARGS.model_path)
        p.image_processor.max_pixels = ARGS.max_pixels
        p.image_processor.min_pixels = ARGS.min_pixels
        p.pad_token_id = p.tokenizer.pad_token_id
        _PROC = p
    return _PROC


def collate(features):
    pc = processor()
    tok = pc.tokenizer

    batch_messages, batch_images = [], []
    for f in features:
        msg = f["prompt"].copy()
        msg.append({"role": "assistant", "content": f["solution"]})
        batch_messages.append(msg)
        batch_images.append(f["image"])

    texts = [pc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in batch_messages]

    # Qwen2_5_VLProcessor.__call__ expands the image placeholders in place, so each
    # call needs its own copy of the list.
    enc = pc(text=list(texts), images=batch_images, return_tensors="pt",
             padding=True, truncation=True, max_length=ARGS.max_length)
    ids, am = enc["input_ids"], enc["attention_mask"]
    labels = build_answer_only_labels(
        input_ids=ids, attention_mask=am, tokenizer=tok,
        answer_start_strs=("<answer>\n", "<answer>"), answer_end_str="</answer>",
    )

    # Untruncated length, tokenized without the length cap.
    full = pc(text=list(texts), images=batch_images, return_tensors="pt", padding=True)
    full_len = full["attention_mask"].sum(1)

    start_ids = [torch.tensor(tok.encode(s, add_special_tokens=False), dtype=torch.long)
                 for s in ("<answer>\n", "<answer>")]
    end_ids = torch.tensor(tok.encode("</answer>", add_special_tokens=False),
                           dtype=torch.long)

    rows = []
    for i in range(ids.size(0)):
        seq = ids[i][am[i].bool()]
        has_start = any(find_subseq_1d(seq, s) != -1 for s in start_ids)
        has_end = find_subseq_1d(seq, end_ids) != -1
        n_sup = int((labels[i] != -100).sum())

        if has_start and has_end:
            branch = "ok"
        elif not has_start:
            branch = "full"          # labels[:] = ids, fits the prompt template
        else:
            branch = "no_super"      # all -100, no language loss at all
        rows.append({
            "full_len": int(full_len[i]),
            "kept_len": int(am[i].sum()),
            "truncated": int(full_len[i]) > ARGS.max_length,
            "branch": branch,
            "n_supervised": n_sup,
        })
    return rows


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="data/EgoAfford")
    ap.add_argument("--model-path", default="./pretrained/Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-pixels", type=int, default=1000000)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--prompt-difficulty", default="hard")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    a = ap.parse_args()
    ARGS = SimpleNamespace(model_path=a.model_path, max_length=a.max_length,
                           max_pixels=a.max_pixels, min_pixels=a.min_pixels)

    ds_args = SimpleNamespace(image_dir=a.image_dir, prompt_difficulty=a.prompt_difficulty,
                              dataset_size=2000, max_pixels=a.max_pixels,
                              min_pixels=a.min_pixels)
    ds = EgoAffordTrainDataset(ds_args, split="train")
    if a.limit:
        ds = torch.utils.data.Subset(ds, range(min(a.limit, len(ds))))
    print(f"samples          {len(ds)}")
    print(f"max_length       {a.max_length}")
    print(f"prompt_difficulty {a.prompt_difficulty}\n", flush=True)

    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                    collate_fn=collate, shuffle=False)

    lens, branches, trunc, sup_zero = [], Counter(), 0, 0
    seen = 0
    for rows in dl:
        for r in rows:
            lens.append(r["full_len"])
            branches[r["branch"]] += 1
            trunc += r["truncated"]
            sup_zero += (r["n_supervised"] == 0)
        seen += len(rows)
        if seen % 2000 < a.batch_size:
            print(f"  ... {seen}/{len(ds)}", flush=True)

    n = len(lens)
    t = torch.tensor(lens, dtype=torch.float)
    print("\n=== token length (untruncated) ===")
    for q in (0.5, 0.9, 0.95, 0.99, 1.0):
        print(f"  p{int(q*100):<3} {int(t.quantile(q))}")
    print(f"  mean {t.mean():.0f}  max {int(t.max())}")
    for thr in (1024, 1536, 2048, 2560, 3072):
        c = int((t > thr).sum())
        print(f"  > {thr:<5} {c:6d}  ({100*c/n:.2f}%)")

    print(f"\n=== truncated at {ARGS.max_length} ===")
    print(f"  {trunc}/{n}  ({100*trunc/n:.2f}%)")

    print("\n=== build_answer_only_labels branch ===")
    for k in ("ok", "no_super", "full"):
        c = branches[k]
        print(f"  {k:<9} {c:6d}  ({100*c/n:.2f}%)")
    print(f"\n  samples with zero supervised tokens: {sup_zero} ({100*sup_zero/n:.2f}%)")


if __name__ == "__main__":
    main()
