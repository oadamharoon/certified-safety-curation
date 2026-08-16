"""Diagnostic for the rating-based VLM strategy.

Mirrors debug_vlm.py but for the single-segment Likert rating path:
  * rates N safe segments (cost == 0) individually
  * rates N unsafe segments (cost >= 5) individually
  * reports the rating distribution per cohort and the implied pair
    preference accuracy when we cross-pair safe vs unsafe

If the rating signal is real, safe segments should score noticeably higher
than unsafe ones, and the implied preference accuracy should sit well above
the 50% chance line.
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed
from utils.vlm_backends._common import parse_rating


def _to_pil(items):
    out = []
    for x in items:
        if isinstance(x, (bytes, bytearray)):
            out.append(Image.open(BytesIO(x)).convert("RGB"))
        else:
            out.append(x.convert("RGB"))
    return out


def _build_messages_single(frames, prompt):
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "--- Frames in temporal order ---"},
    ]
    content.extend({"type": "image"} for _ in frames)
    return [{"role": "user", "content": content}]


def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    scale = int(cfg.get("rating_scale", 5))

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "segment_frames.pkl"), "rb") as f:
        frames = pickle.load(f)

    with open(cfg["prompt_template"]) as f:
        prompt = f.read()
    print(f"[debug] strategy=rating | scale={scale} | prompt={cfg['prompt_template']}")

    safe_pool   = [i for i, s in enumerate(active) if s["total_cost"] == 0   and s["seg_id"] in frames]
    unsafe_pool = [i for i, s in enumerate(active) if s["total_cost"] >= 5.0 and s["seg_id"] in frames]
    rng = np.random.default_rng(0)
    n_per_cohort = 10
    safe_idxs   = rng.choice(safe_pool,   size=n_per_cohort, replace=False)
    unsafe_idxs = rng.choice(unsafe_pool, size=n_per_cohort, replace=False)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = cfg["vlm_model"]
    print(f"[debug] loading {model_id} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def rate(seg_idx: int) -> tuple[int, str]:
        seg = active[int(seg_idx)]
        f = _to_pil(frames[seg["seg_id"]])
        messages = _build_messages_single(f, prompt)
        text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_in], images=list(f),
                           return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        new_tokens = gen[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return parse_rating(raw, scale), raw

    print(f"\n========== rating {n_per_cohort} SAFE segments (cost == 0) ==========")
    safe_ratings = []
    for k, idx in enumerate(safe_idxs):
        r, raw = rate(idx)
        seg = active[int(idx)]
        safe_ratings.append(r)
        shown = raw.replace("\n", " ")[:60]
        print(f"  seg {int(idx):>5} (cost={seg['total_cost']:.0f}): rating={r}  raw={shown!r}")

    print(f"\n========== rating {n_per_cohort} UNSAFE segments (cost >= 5) ==========")
    unsafe_ratings = []
    for k, idx in enumerate(unsafe_idxs):
        r, raw = rate(idx)
        seg = active[int(idx)]
        unsafe_ratings.append(r)
        shown = raw.replace("\n", " ")[:60]
        print(f"  seg {int(idx):>5} (cost={seg['total_cost']:.0f}): rating={r}  raw={shown!r}")

    print("\n========== distributions ==========")
    print(f"  SAFE   counts: {dict(sorted(Counter(safe_ratings).items()))}")
    print(f"  UNSAFE counts: {dict(sorted(Counter(unsafe_ratings).items()))}")
    safe_valid   = [r for r in safe_ratings   if r >= 0]
    unsafe_valid = [r for r in unsafe_ratings if r >= 0]
    parse_fail = (len(safe_ratings) - len(safe_valid)) + (len(unsafe_ratings) - len(unsafe_valid))
    if safe_valid:
        print(f"  SAFE   mean rating: {np.mean(safe_valid):.2f}  (n_valid={len(safe_valid)})")
    if unsafe_valid:
        print(f"  UNSAFE mean rating: {np.mean(unsafe_valid):.2f}  (n_valid={len(unsafe_valid)})")
    print(f"  parse failures: {parse_fail}/{len(safe_ratings) + len(unsafe_ratings)}")

    # implied preference accuracy: cross-pair every safe vs every unsafe,
    # derive preference = "safe is better" iff safe_rating > unsafe_rating
    print("\n========== implied pair-preference accuracy ==========")
    correct = tie = total = 0
    for r_s in safe_valid:
        for r_u in unsafe_valid:
            total += 1
            if r_s > r_u:
                correct += 1
            elif r_s == r_u:
                tie += 1
    if total:
        print(f"  safe > unsafe: {correct}/{total} = {correct/total:.1%}  "
              f"(ties: {tie}/{total} = {tie/total:.1%})")
        print(f"  ⇒ would yield ~{correct} valid preferences vs ~{tie} dropped ties on this cohort")


if __name__ == "__main__":
    main()
