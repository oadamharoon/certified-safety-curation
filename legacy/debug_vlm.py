"""Diagnostic: test different prompts on the same Qwen2.5-VL model + same pairs
to find whether prompt structure is causing the -1 dominance.
"""
from __future__ import annotations

import os
import pickle
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed
from utils.vlm_backends._common import parse_label


def _to_pil(items):
    out = []
    for x in items:
        if isinstance(x, (bytes, bytearray)):
            out.append(Image.open(BytesIO(x)).convert("RGB"))
        else:
            out.append(x.convert("RGB"))
    return out


def _build_messages(frames1, frames2, prompt):
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "--- Sequence 1 frames ---"},
    ]
    content.extend({"type": "image"} for _ in frames1)
    content.append({"type": "text", "text": "--- Sequence 2 frames ---"})
    content.extend({"type": "image"} for _ in frames2)
    return [{"role": "user", "content": content}]


# Three prompt variants are built per-task from the task's actual prompt
# template (cfg["prompt_template"]) so the debug harness asks Qwen about the
# right env, not whatever was hardcoded once. Built lazily inside main().
PROMPTS: dict = {}


def _build_prompt_variants(original_prompt: str) -> dict:
    """Derive the three variants from the task's production prompt.

    * original   : the production prompt verbatim
    * binary     : production prompt with an explicit "must pick 0 or 1, no tie" footer
    * chain-of-thought : production prompt with a "describe then ANSWER:" footer
    """
    # Strip any trailing "Your answer:" so the footers attach cleanly.
    base = original_prompt.rstrip()
    if base.endswith("Your answer:"):
        base = base[: -len("Your answer:")].rstrip()
    return {
        "original (as used in production)": original_prompt,
        "binary (force 0 or 1, no tie)":
            base + "\n\nYou MUST choose. Reply with EXACTLY '0' or '1' and nothing else. "
                   "Do not refuse, do not output -1, do not write anything else.\n\nYour answer:",
        "chain-of-thought (describe first, then answer)":
            base + "\n\nBefore answering, briefly describe what the agent does in each "
                   "sequence (one sentence each), then conclude with a single line of the form:\n"
                   "  ANSWER: 0   (if Sequence 1 is better)\n"
                   "  ANSWER: 1   (if Sequence 2 is better)\n"
                   "You must conclude with exactly one such ANSWER line.\n",
    }


def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "segment_frames.pkl"), "rb") as f:
        frames = pickle.load(f)

    # Read the production prompt and derive all three variants from it so the
    # diagnostic asks Qwen about the *task's* env, not a hardcoded one.
    with open(cfg["prompt_template"]) as f:
        original_prompt = f.read()
    PROMPTS.update(_build_prompt_variants(original_prompt))

    # Threshold tuned per env: safety-gym segments commonly accumulate cost > 5
    # over 30 steps; metadrive's per-step cost is ~binary (1 at terminal event)
    # so an unsafe segment has cost == 1. Use the task's cost_safe_threshold +
    # cost_contrast_min from config to pick a value that works in both regimes.
    cost_unsafe_min = float(cfg.get("cost_safe_threshold", 0.0)) + float(cfg.get("cost_contrast_min", 5.0))
    safe_pool   = [i for i, s in enumerate(active) if s["total_cost"] <= cfg.get("cost_safe_threshold", 0.0) and s["seg_id"] in frames]
    unsafe_pool = [i for i, s in enumerate(active) if s["total_cost"] >= cost_unsafe_min and s["seg_id"] in frames]
    if not unsafe_pool:
        raise RuntimeError(
            f"No unsafe segments with total_cost >= {cost_unsafe_min}. "
            f"Max segment cost in this dataset is "
            f"{max(s['total_cost'] for s in active):.1f}. "
            f"Adjust cost_safe_threshold / cost_contrast_min in the task config."
        )
    rng = np.random.default_rng(0)
    n_pairs = 5
    safe_idxs   = rng.choice(safe_pool,   size=n_pairs, replace=False)
    unsafe_idxs = rng.choice(unsafe_pool, size=n_pairs, replace=False)

    # Sample frames saved earlier — make sure user has them
    _to_pil(frames[active[int(safe_idxs[0])]["seg_id"]])[0].save("/tmp/safety_safe.png")
    _to_pil(frames[active[int(unsafe_idxs[0])]["seg_id"]])[0].save("/tmp/safety_unsafe.png")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = cfg["vlm_model"]
    print(f"[debug] loading {model_id} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for name, prompt in PROMPTS.items():
        print(f"\n========== prompt: {name} ==========")
        n_correct, n_total, n_minus_one = 0, 0, 0
        for k, (i_safe, i_unsafe) in enumerate(zip(safe_idxs, unsafe_idxs)):
            # Test BOTH orderings: safe-in-slot-A (expected label 0) and
            # safe-in-slot-B (expected label 1). If Qwen has positional bias,
            # one ordering will always be "right" and the other always "wrong".
            for swap in (False, True):
                if not swap:
                    seg_a, seg_b, expected = active[int(i_safe)], active[int(i_unsafe)], 0
                else:
                    seg_a, seg_b, expected = active[int(i_unsafe)], active[int(i_safe)], 1
                f_a = _to_pil(frames[seg_a["seg_id"]])
                f_b = _to_pil(frames[seg_b["seg_id"]])
                messages = _build_messages(f_a, f_b, prompt)
                text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[text_in], images=list(f_a) + list(f_b),
                                   return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    max_new = 512 if "chain" in name else 32
                    gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
                new_tokens = gen[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
                label = parse_label(raw)
                ok = (label == expected)
                if ok: n_correct += 1
                if label == -1: n_minus_one += 1
                n_total += 1
                shown = raw[-160:].replace("\n", " ") if "chain" in name else raw
                tag = "✓" if ok else "✗"
                slot = "A=safe" if not swap else "B=safe"
                print(f"  pair {k} [{slot}]: cost_A={seg_a['total_cost']:.0f}  cost_B={seg_b['total_cost']:.0f}  "
                      f"label={label} expected={expected} {tag}  raw=...{shown!r}")
        print(f"  → accuracy: {n_correct}/{n_total}={n_correct/n_total:.0%}, "
              f"-1 rate: {n_minus_one}/{n_total}={n_minus_one/n_total:.0%}")


if __name__ == "__main__":
    main()
