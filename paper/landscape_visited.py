"""Binned Spearman between V and distance-to-nearest-hazard on the VISITED
distribution (real velocities from policy rollouts), the complement to the
frozen zero-velocity teleport probe in landscape_fig.py.
"""
import argparse
import json
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
from model.policy import VEnsemble, GaussianPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=os.environ.get("SAFETY_VLM_TASK",
                                                     "pointgoal1_dsrl"))
    ap.add_argument("--policy", default="bc_calfilt_ltt_seed0_policy.pt")
    ap.add_argument("--ckpt", default="v_ensemble_pess_seed0.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()

    cfg = load_cfg()  # resolves --task from argv, as elsewhere in the repo
    outdir = os.path.join(NEW_DATA, cfg["output_dir"])

    vck = torch.load(os.path.join(outdir, args.ckpt), map_location="cpu",
                     weights_only=False)
    ens = VEnsemble(vck["obs_dim"], vck["hidden_dim"], K=vck["K"])
    ens.load_state_dict(vck["state_dict"])
    ens.eval()

    pck = torch.load(os.path.join(outdir, args.policy), map_location="cpu",
                     weights_only=False)
    pol = GaussianPolicy(pck["obs_dim"], pck["act_dim"], pck["hidden_dim"])
    pol.load_state_dict(pck["state_dict"])
    pol.eval()

    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass
    env = gym.make(cfg["env_name"])

    dists, vals = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        task = env.unwrapped.task
        hazards = np.array([p[:2] for p in task.hazards.pos])
        done = False
        while not done:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32)[None]
                a = pol(ot)
                a = a[0] if isinstance(a, tuple) else a
                a = a.squeeze(0).numpy()
                v = float(ens.forward_all(ot).mean().item())
            xy = np.array(task.data.qpos[:2])
            dists.append(float(np.min(np.linalg.norm(hazards - xy, axis=1))))
            vals.append(v)
            obs, _, term, trunc, _ = env.step(np.clip(a, -1, 1))
            done = term or trunc

    dists, vals = np.asarray(dists), np.asarray(vals)
    edges = np.quantile(dists, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(dists, edges[1:-1]), 0, args.bins - 1)
    bd = np.array([dists[idx == b].mean() for b in range(args.bins)
                   if (idx == b).any()])
    bv = np.array([vals[idx == b].mean() for b in range(args.bins)
                   if (idx == b).any()])

    from scipy.stats import spearmanr
    rho_b, p_b = spearmanr(bd, bv)
    rho_r, _ = spearmanr(dists, vals)
    out = {"task": args.task, "policy": args.policy, "episodes": args.episodes,
           "n_states": int(len(dists)), "bins": int(len(bd)),
           "rho_binned": float(rho_b), "p_binned": float(p_b),
           "rho_raw": float(rho_r)}
    print(f"  visited-distribution: n={out['n_states']} states, "
          f"binned rho={rho_b:.3f} (p={p_b:.4f}), raw rho={rho_r:.3f}")
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "landscape_visited.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  saved -> data/landscape_visited.json")


if __name__ == "__main__":
    main()
