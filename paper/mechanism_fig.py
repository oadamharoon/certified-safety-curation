"""Exploratory: does the clone's occupancy avoid the value's low basins?

For one task and layout, roll BC-All (unfiltered) and the deployed calibrated
clone, bin both by (x, y), and plot where the clone goes relative to where the
unfiltered policy goes, against the hazard field. If curation works through the
value, the clone should vacate exactly the cells the value scores low.
CPU-only. Built to be judged, not assumed useful.
"""
import argparse, json, os, sys
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.abspath(os.path.join(BASE, "..", "vlm-with-cpl", "new_data"))
sys.path.insert(0, NEW_DATA)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from utils.common import load_cfg
from model.policy import VEnsemble, GaussianPolicy

NBIN, EXT, EPISODES = 26, 1.8, 40


def rollout(env, pol, ens, seed, n_ep, jitter=0.10):
    xs, ys, vs = [], [], []
    haz = hr = goal = None
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed)
        task = env.unwrapped.task
        if haz is None:
            haz = np.array([p[:2] for p in task.hazards.pos])
            hr = float(task.hazards.size); goal = np.array(task.goal.pos[:2])
        done, buf = False, []
        while not done:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32)[None]
                a = pol(ot); a = a[0] if isinstance(a, tuple) else a
            p = np.array(task.data.qpos[:2])
            xs.append(p[0]); ys.append(p[1]); buf.append(obs)
            act = a.squeeze(0).numpy() + jitter * np.random.randn(a.shape[-1])
            obs, _, term, trunc, _ = env.step(np.clip(act, -1, 1))
            done = term or trunc
        with torch.no_grad():
            ob = torch.as_tensor(np.asarray(buf), dtype=torch.float32)
            vs.extend(ens.forward_all(ob).mean(0).numpy().tolist())
    return np.asarray(xs), np.asarray(ys), np.asarray(vs), haz, hr, goal


def occupancy(xs, ys, edges):
    ix = np.clip(np.digitize(xs, edges[1:-1]), 0, NBIN - 1)
    iy = np.clip(np.digitize(ys, edges[1:-1]), 0, NBIN - 1)
    o = np.zeros((NBIN, NBIN))
    for a, b in zip(ix, iy):
        o[b, a] += 1
    return o / max(o.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed-layout", type=int, default=7)
    args, _ = ap.parse_known_args()
    os.chdir(NEW_DATA)
    cfg = load_cfg()
    out = os.path.join(NEW_DATA, cfg["output_dir"])
    vck = torch.load(os.path.join(out, "v_ensemble_pess_seed0.pt"),
                     map_location="cpu", weights_only=False)
    ens = VEnsemble(vck["obs_dim"], vck["hidden_dim"], K=vck["K"])
    ens.load_state_dict(vck["state_dict"]); ens.eval()

    def load(fn):
        ck = torch.load(os.path.join(out, fn), map_location="cpu", weights_only=False)
        m = GaussianPolicy(ck["obs_dim"], ck["act_dim"], ck["hidden_dim"])
        m.load_state_dict(ck["state_dict"]); m.eval(); return m

    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"): dsrl.register_envs()
    except Exception:
        pass
    env = gym.make(cfg["env_name"])
    res = {}
    for name, fn in (("BC-All", "bc_bc_policy.pt"),
                     ("calibrated", "bc_calfilt_csf_seed0_policy.pt")):
        xs, ys, vs, haz, hr, goal = rollout(env, load(fn), ens, args.seed_layout, EPISODES)
        d = np.min(np.linalg.norm(np.stack([xs, ys], 1)[:, None] - haz[None], axis=2), 1)
        res[name] = dict(xs=xs, ys=ys, vs=vs, d=d)
        print(f"  {args.task} {name:11s} mean V {vs.mean():+.3f} | "
              f"mean dist-to-hazard {d.mean():.3f} | frac within hazard radius "
              f"{(d < hr).mean():.4f}", flush=True)
    edges = np.linspace(-EXT, EXT, NBIN + 1)
    oa = occupancy(res["BC-All"]["xs"], res["BC-All"]["ys"], edges)
    ob = occupancy(res["calibrated"]["xs"], res["calibrated"]["ys"], edges)
    diff = ob - oa
    lim = np.abs(diff).max()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(diff, origin="lower", extent=[-EXT, EXT, -EXT, EXT],
                   cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for h in haz:
        ax.add_patch(plt.Circle(h, hr, fill=False, color="k", ls="--", lw=1.0))
    ax.add_patch(plt.Circle(goal, 0.16, fill=False, color="k", lw=1.4))
    ax.set_title(f"{args.task}: clone minus BC-All occupancy", fontsize=9)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("occupancy difference", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    dst = os.path.join(BASE, "figures", f"mechanism_{args.task}")
    fig.savefig(dst + ".png", dpi=200, bbox_inches="tight")
    json.dump({k: {"mean_V": float(v["vs"].mean()),
                   "mean_dist": float(v["d"].mean()),
                   "frac_in_hazard": float((v["d"] < hr).mean())} for k, v in res.items()},
              open(os.path.join(BASE, "data", f"mechanism_{args.task}.json"), "w"), indent=1)
    print(f"  saved figures/mechanism_{args.task}.png")


if __name__ == "__main__":
    main()
