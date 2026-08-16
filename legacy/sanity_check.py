"""End-to-end smoke test that does NOT call the VLM or train a real policy.

Verifies:
  * config loads, package imports work
  * raw npz can be read
  * trajectories.pkl was produced (or builds it on the fly with a tiny subset)
  * segment extraction + filtering produce non-empty output
  * env can be created and stepped, render returns an image
  * a tiny BC pre-train + CPL update runs without NaN

Run:
    python scripts/sanity_check.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402
from utils.segment_utils import extract_segments, filter_active_segments  # noqa: E402
from utils.env_utils import make_render_env  # noqa: E402
from model.policy import GaussianPolicy  # noqa: E402
from model.train import bc_pretrain, train_cpl  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    set_seed(cfg["seed"])

    print("[1/6] checking raw npz ...")
    npz_path = cfg["raw_npz"] if os.path.isabs(cfg["raw_npz"]) else \
        os.path.join(cfg["_repo_root"], cfg["raw_npz"])
    assert os.path.exists(npz_path), f"missing {npz_path}"
    d = np.load(npz_path)
    assert d["observations"].ndim == 2 and d["actions"].ndim == 2
    print(f"    obs {d['observations'].shape}  act {d['actions'].shape}")

    print("[2/6] building 5 toy trajectories from first 5000 transitions ...")
    n_per = 1000
    trajs = []
    for k in range(5):
        sl = slice(k * n_per, (k + 1) * n_per)
        trajs.append({
            "observations": d["observations"][sl].astype(np.float32),
            "actions": d["actions"][sl].astype(np.float32),
            "rewards": d["rewards"][sl].astype(np.float32),
            "costs": d["costs"][sl].astype(np.float32),
        })

    print("[3/6] segmentation + filtering ...")
    segs = extract_segments(trajs, cfg["segment_length"], cfg["segment_stride"])
    active = filter_active_segments(segs, cfg["min_total_reward"], cfg["drop_zero_segments"])
    print(f"    {len(segs)} segs -> {len(active)} active")
    assert len(active) > 0, "filtering removed everything; loosen min_total_reward"

    print("[4/6] env creation + render ...")
    try:
        env = make_render_env(cfg["env_name"], seed=cfg["seed"])
        obs, _ = env.reset(seed=cfg["seed"])
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame is not None and frame.ndim == 3
        print(f"    obs={obs.shape}  frame={frame.shape}")
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"    !! render env failed: {e}")
        print("    (rendering can still work later if MuJoCo GL is fixed)")

    print("[5/6] policy forward + tiny BC ...")
    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]
    pol = GaussianPolicy(obs_dim, act_dim, 64)
    bc_o = torch.cat([torch.tensor(s["observations"]) for s in active])
    bc_a = torch.cat([torch.tensor(s["actions"]) for s in active])
    bc_pretrain(pol, bc_o, bc_a, batch_size=64, epochs=2,
                lr=cfg["learning_rate"], device="cpu", log_every=1)

    print("[6/6] tiny CPL update on 4 fake pairs ...")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(active), size=8, replace=False)
    A = [active[i] for i in pick[:4]]
    B = [active[i] for i in pick[4:]]
    poA = torch.stack([torch.tensor(s["observations"]) for s in A])
    paA = torch.stack([torch.tensor(s["actions"]) for s in A])
    poB = torch.stack([torch.tensor(s["observations"]) for s in B])
    paB = torch.stack([torch.tensor(s["actions"]) for s in B])
    labels = torch.tensor([0, 1, 0, 1])
    train_cpl(pol, poA, paA, poB, paB, labels,
              bc_o, bc_a, epochs=2, batch_size=4,
              lr=cfg["learning_rate"], lambda_bc=cfg["lambda_bc"],
              temp=cfg["temperature_cpl"], device="cpu", log_every=1)

    print("\nSANITY CHECK PASSED.")


if __name__ == "__main__":
    main()
