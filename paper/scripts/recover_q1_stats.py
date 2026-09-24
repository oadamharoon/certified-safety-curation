"""Recover (kept_frac, kept_unsafe) at the first grid threshold for the cells
guarantee_stats.json cannot supply.

cert_rate_theory.py needs u1 and f1 at q = 0.85, the first threshold of the
fixed sequence. guarantee_stats records those only for the quantile a draw
actually SELECTED, so two pipeline seeds (walker2d seed 1, hopper seed 0) have no
q = 0.85 record at any budget and were excluded from the validation. Excluding
cells on a criterion correlated with the outcome is exactly the bias the pooled
lookup was introduced to remove, so the last 8 cells are recovered here directly.

This replays 04q's scoring path exactly: same VEnsemble(K=3), same checkpoint,
same mean-over-observations trajectory score, same np.quantile grid. The
checkpoints predate guarantee_stats.json (Jul 11 vs Jul 17) and were untouched by
the August ensemble rewrite, so they are the weights that produced the recorded
certification rates.

Writes data/q1_recovered.json.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(os.path.dirname(BASE), "datasets")
sys.path.insert(0, REPO)
from model.policy import VEnsemble  # noqa: E402

Q1 = 0.85
MISSING = [("walker2d_velocity", 1), ("hopper_velocity", 0)]


def score_trajectories(trajectories, ens, device):
    """Byte-for-byte the rule in 04q_calibrated_vfilter.score_trajectories."""
    scores = np.zeros(len(trajectories))
    with torch.no_grad():
        for i, traj in enumerate(trajectories):
            o = torch.as_tensor(traj["observations"], dtype=torch.float32).to(device)
            vs = [ens(o[j:j + 8192]).cpu() for j in range(0, len(o), 8192)]
            scores[i] = torch.cat(vs).mean().item()
    return scores


def main():
    cfg_all = yaml.safe_load(open(os.path.join(REPO, "config.yaml")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for task, seed in MISSING:
        tcfg = cfg_all["tasks"][task]
        limit = float(tcfg.get("cost_limit", cfg_all["cost_limit"]))
        odir = os.path.join(REPO, tcfg["output_dir"])
        with open(os.path.join(REPO, tcfg["data_pickle"]), "rb") as f:
            trajs = pickle.load(f)
        obs_dim = trajs[0]["observations"].shape[1]
        ens = VEnsemble(obs_dim, cfg_all["hidden_dim"], K=3).to(device)
        p = os.path.join(odir, f"v_ensemble_pess_seed{seed}.pt")
        ck = torch.load(p, map_location=device, weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()

        scores = score_trajectories(trajs, ens, device)
        costs = np.array([float(np.sum(t["costs"])) for t in trajs])
        unsafe = (costs > limit)
        tau = float(np.quantile(scores, Q1))
        sel = scores >= tau
        f1, u1 = float(sel.mean()), float(unsafe[sel].mean())
        out[f"{task}/seed{seed}"] = {
            "task": task, "seed": seed, "q": Q1, "n_trajs": len(trajs),
            "cost_limit": limit, "kept_frac": f1, "kept_unsafe": u1,
            "checkpoint": os.path.basename(p),
            "checkpoint_mtime": os.path.getmtime(p)}
        print(f"  {task} seed{seed}: n={len(trajs)} kept_frac={f1:.6f} "
              f"kept_unsafe={u1:.6f}  (base unsafe {unsafe.mean():.3f})")
    dst = os.path.join(BASE, "data", "q1_recovered.json")
    json.dump(out, open(dst, "w"), indent=2)
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
