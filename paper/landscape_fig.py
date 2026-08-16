"""Safety-landscape heatmap: learned V(s) swept over the workspace plane.

Teleports the agent over an (x, y) grid in a fixed episode layout, rebuilds
the observation at each pose (averaged over 4 headings to remove egocentric
lidar orientation), evaluates the preference-learned V-ensemble, and plots
the resulting field against the ground-truth hazard/goal geometry the model
was never given. CPU-only; no training, no GPU.

Usage:
  SAFETY_VLM_TASK=pointgoal1_dsrl python landscape_fig.py [--seed-layout 7]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

NEW_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "vlm-with-cpl", "new_data"))
sys.path.insert(0, NEW_DATA)

import torch  # noqa: E402
from utils.common import load_cfg  # noqa: E402
from model.policy import VEnsemble  # noqa: E402


def load_ensemble(cfg, ckpt_name, device):
    p = os.path.join(NEW_DATA, cfg["output_dir"], ckpt_name)
    ck = torch.load(p, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        obs_dim, hid, K = ck["obs_dim"], ck["hidden_dim"], ck["K"]
        sd = ck["state_dict"]
    else:
        sd = ck
        obs_dim = cfg.get("obs_dim")
        hid, K = cfg["hidden_dim"], int(cfg.get("v_ensemble_k", 3))
    ens = VEnsemble(obs_dim, hid, K=K)
    ens.load_state_dict(sd)
    ens.eval()
    return ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-layout", type=int, default=7)
    ap.add_argument("--ckpt", default="v_ensemble_pess_seed0.pt")
    ap.add_argument("--n", type=int, default=81, help="grid points per axis")
    ap.add_argument("--extent", type=float, default=1.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    task_name = os.environ.get("SAFETY_VLM_TASK", "default")
    device = torch.device("cpu")
    ens = load_ensemble(cfg, args.ckpt, device)

    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass
    import mujoco
    env = gym.make(cfg["env_name"])
    obs0, _ = env.reset(seed=args.seed_layout)
    task = env.unwrapped.task
    model, data = task.model, task.data

    hazards = np.array([p[:2] for p in task.hazards.pos])
    hazard_r = float(task.hazards.size)
    goal = np.array(task.goal.pos[:2])
    goal_r = float(task.goal.size)
    vases = (np.array([p[:2] for p in task.vases.pos])
             if hasattr(task, "vases") and len(task.vases.pos) else None)

    xs = np.linspace(-args.extent, args.extent, args.n)
    ys = np.linspace(-args.extent, args.extent, args.n)
    headings = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    obs_batch = []
    for y in ys:
        for x in xs:
            for th in headings:
                data.qpos[0], data.qpos[1], data.qpos[2] = x, y, th
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                obs_batch.append(task.obs())
    obs_batch = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32)

    with torch.no_grad():
        v_all = []
        for i in range(0, len(obs_batch), 8192):
            v_all.append(ens.forward_all(obs_batch[i:i + 8192]).mean(0))
        v = torch.cat(v_all).numpy()
    grid = v.reshape(args.n, args.n, len(headings)).mean(-1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.pcolormesh(xs, ys, grid, shading="auto", cmap="RdYlGn",
                       rasterized=True)
    ax.contour(xs, ys, grid, levels=8, colors="k", linewidths=0.3, alpha=0.35)
    for h in hazards:
        ax.add_patch(plt.Circle(h, hazard_r, fill=False, color="black",
                                lw=1.6, ls="--"))
    ax.add_patch(plt.Circle(goal, goal_r, fill=False, color="tab:blue", lw=1.8))
    ax.annotate("goal", goal, color="tab:blue", fontsize=8,
                ha="center", va="center")
    if vases is not None:
        for vp in vases:
            ax.plot(*vp, marker="s", ms=6, mec="black", mfc="none", mew=1.4)
    ax.set_xlim(-args.extent, args.extent)
    ax.set_ylim(-args.extent, args.extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Learned $V(s)$ over workspace ({task_name}, layout seed "
                 f"{args.seed_layout})", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("mean ensemble $V(s)$", fontsize=8)
    fig.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "figures",
        f"landscape_{task_name}_seed{args.seed_layout}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    print(f"saved -> {out}.png / .pdf")

    # quantitative sanity: correlation between V and distance to nearest hazard
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    dmin = np.min(np.linalg.norm(pts[:, None] - hazards[None], axis=2), 1)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(dmin, grid.ravel())
    print(f"Spearman rho( dist-to-nearest-hazard , V ) = {rho:.3f}")


if __name__ == "__main__":
    main()
