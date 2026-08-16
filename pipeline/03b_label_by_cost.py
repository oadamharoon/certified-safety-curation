"""Step 03b: build preference labels directly from ground-truth segment
costs (no VLM call).

Drops a file with the same schema as ``vlm_labels.json`` so downstream
training scripts (``04c_train_cpl_gt.py``) can consume it interchangeably.

Use this for the *BC + Ground-Truth CPL* baseline — an upper bound on
the CPL component because preferences are oracle-correct.

Preference rule:
    cost_A < cost_B  →  preference = 0  (segment A safer; A wins)
    cost_A > cost_B  →  preference = 1
    cost_A == cost_B →  skipped (ties carry no signal)
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402
from utils.segment_utils import sample_pair_indices  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    if "NUM_PAIRS" in os.environ:
        cfg["num_pairs"] = int(os.environ["NUM_PAIRS"])

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)

    print(f"Sampling {cfg['num_pairs']} pairs and labeling from ground-truth cost ...")

    seen = set()
    pref_data = []
    n_ties = 0

    for _ in range(cfg["num_pairs"]):
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
        c_a, c_b = seg_a["total_cost"], seg_b["total_cost"]
        if c_a == c_b:
            n_ties += 1
            continue
        label = 0 if c_a < c_b else 1

        pref_data.append({
            "seg_A_idx":  int(i),
            "seg_B_idx":  int(j),
            "preference": int(label),
            "cost_A":     c_a,
            "cost_B":     c_b,
            "reward_A":   seg_a["total_reward"],
            "reward_B":   seg_b["total_reward"],
        })

    out_path = os.path.join(cfg["output_dir"], "gt_labels.json")
    with open(out_path, "w") as f:
        json.dump(pref_data, f)

    # Sanity: cost-rule agreement is 1.0 by construction
    print(f"Saved {len(pref_data)} preferences (ties skipped: {n_ties}) -> {out_path}")


if __name__ == "__main__":
    main()
