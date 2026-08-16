"""Convert the raw DSRL .npz dataset into a list-of-trajectories pickle.

Reads ``cfg['raw_npz']`` and produces ``cfg['data_pickle']``.
Each trajectory is a dict with keys: observations, actions, rewards, costs.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

# bootstrap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    set_seed(cfg["seed"])

    npz_path = cfg["raw_npz"]
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(cfg["_repo_root"], npz_path)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Raw npz not found at {npz_path}. Run data_collect.ipynb first."
        )

    print(f"Loading {npz_path} ...")
    d = np.load(npz_path)
    obs = d["observations"]
    act = d["actions"]
    rew = d["rewards"]
    cost = d["costs"]
    term = d["terminals"]
    tout = d["timeouts"]
    end = np.logical_or(term, tout)

    # split by episode end
    ends = np.where(end)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])

    trajectories = []
    for s, e in zip(starts, ends):
        sl = slice(s, e + 1)
        if e + 1 - s < 2:
            continue
        trajectories.append({
            "observations": obs[sl].astype(np.float32),
            "actions": act[sl].astype(np.float32),
            "rewards": rew[sl].astype(np.float32),
            "costs": cost[sl].astype(np.float32),
        })

    print(f"Built {len(trajectories)} trajectories "
          f"(avg length {np.mean([len(t['observations']) for t in trajectories]):.1f})")

    out_path = cfg["data_pickle"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
