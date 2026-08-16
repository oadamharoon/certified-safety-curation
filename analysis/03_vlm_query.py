"""Step 03: sample preference pairs and query the VLM for labels."""
from __future__ import annotations

import json
import os
import pickle
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from utils.segment_utils import sample_pair_indices  # noqa: E402
from utils.vlm_utils import build_prompt_from_template, query_vlm  # noqa: E402
from utils.vlm_backends import query_vlm_rating  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()  # allow --task to pass through
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    run = init_wandb(cfg, job_type="vlm_query")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "segment_frames.pkl"), "rb") as f:
        frames = pickle.load(f)

    prompt = build_prompt_from_template(cfg["prompt_template"])
    backend = cfg.get("vlm_backend", "openai")
    strategy = cfg.get("vlm_strategy", "pairwise").lower()
    rating_scale = int(cfg.get("rating_scale", 5))
    print(f"Sampling {cfg['num_pairs']} pairs and querying VLM "
          f"[backend={backend} | model={cfg['vlm_model']} | strategy={strategy}] ...")

    # Per-backend auth + headers
    if backend == "openrouter":
        api_key  = cfg.get("openrouter_api_key") or None
        base_url = cfg.get("openrouter_base_url") or "https://openrouter.ai/api/v1"
        referer   = cfg.get("openrouter_referer") or None
        app_title = cfg.get("openrouter_app_title") or None
    else:
        api_key = cfg.get("openai_api_key") or None
        base_url  = None
        referer   = None
        app_title = None

    # frames are stored as JPEG bytes -- decode to PIL on demand
    def _to_pil(items):
        out = []
        for x in items:
            if isinstance(x, (bytes, bytearray)):
                out.append(Image.open(BytesIO(x)).convert("RGB"))
            else:
                out.append(x)
        return out

    # cache distinct pairs to avoid duplicate VLM calls
    seen = set()
    pref_data = []
    n_ties = 0
    rating_cache: dict = {}  # seg_id -> rating (only used in rating strategy)

    def _get_rating(seg) -> int:
        sid = seg["seg_id"]
        if sid in rating_cache:
            return rating_cache[sid]
        r = query_vlm_rating(
            _to_pil(frames[sid]),
            prompt,
            scale=rating_scale,
            backend=backend,
            model=cfg["vlm_model"],
            api_key=api_key,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            retries=cfg["vlm_retries"],
        )
        rating_cache[sid] = r
        return r

    for _ in tqdm(range(cfg["num_pairs"])):
        i, j = sample_pair_indices(
            active,
            cost_safe_threshold=cfg["cost_safe_threshold"],
            cost_contrast_min=cfg["cost_contrast_min"],
            rng=rng,
        )
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)

        seg_a, seg_b = active[i], active[j]
        if seg_a["seg_id"] not in frames or seg_b["seg_id"] not in frames:
            continue

        if strategy == "rating":
            r_a = _get_rating(seg_a)
            r_b = _get_rating(seg_b)
            if r_a < 0 or r_b < 0 or r_a == r_b:
                # parse failure on either segment, or rated equally -> tie
                n_ties += 1
                continue
            label = 0 if r_a > r_b else 1
            entry_extra = {"rating_A": int(r_a), "rating_B": int(r_b)}
        else:
            frames_a = _to_pil(frames[seg_a["seg_id"]])
            frames_b = _to_pil(frames[seg_b["seg_id"]])
            label = query_vlm(
                frames_a, frames_b,
                prompt,
                backend=backend,
                model=cfg["vlm_model"],
                api_key=api_key,
                base_url=base_url,
                referer=referer,
                app_title=app_title,
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                retries=cfg["vlm_retries"],
            )
            if label not in (0, 1):
                n_ties += 1
                continue
            entry_extra = {}

        pref_data.append({
            "seg_A_idx": int(i),
            "seg_B_idx": int(j),
            "preference": int(label),
            "cost_A": seg_a["total_cost"],
            "cost_B": seg_b["total_cost"],
            "reward_A": seg_a["total_reward"],
            "reward_B": seg_b["total_reward"],
            **entry_extra,
        })

    out_path = os.path.join(cfg["output_dir"], "vlm_labels.json")
    with open(out_path, "w") as f:
        json.dump(pref_data, f)

    # quick agreement metric: how often the lower-cost segment wins
    agree = np.mean([
        (p["preference"] == 0 and p["cost_A"] < p["cost_B"]) or
        (p["preference"] == 1 and p["cost_B"] < p["cost_A"])
        for p in pref_data
    ]) if pref_data else 0.0
    print(f"Saved {len(pref_data)} preferences (ties/parse-fail: {n_ties}) -> {out_path}")
    print(f"VLM-vs-cost agreement: {agree:.2%}")
    if strategy == "rating":
        print(f"Rating cache: {len(rating_cache)} unique segments rated")

    if run is not None:
        import wandb
        wandb.log({
            "vlm/n_preferences": len(pref_data),
            "vlm/n_ties": n_ties,
            "vlm/cost_agreement": agree,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
