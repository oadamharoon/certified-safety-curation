"""Mechanistic interpretability analysis of the preference-trained V.

All stats-only (saved checkpoints + offline data). Per task:
  A. Constraint-variable discovery: |corr(obs_i, cost)| ranking vs gradient
     saliency mean|dV/ds_i| ranking; overlap of top-3 dims.
  B. 1-D landscape: binned mean V vs the top constraint dim, with empirical
     per-step cost rate in the same bins.
  C. Anticipation: mean V binned by steps-to-next-violation.
  D. Example traces: V(t) with cost events for one safe + one unsafe traj.
Saves everything to iclr2027/data/interpretability.json.
"""
import json, os, pickle, sys
import numpy as np
import torch

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data"
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

LIMITS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
          "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
          "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLE = 50000


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out = {}
    rng = np.random.default_rng(3)
    for task, limit in LIMITS.items():
        with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
            trajs = pickle.load(f)
        obs_dim = trajs[0]["observations"].shape[1]
        ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
        ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed0.pt",
                        map_location=DEVICE, weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()

        # flat sample of (state, cost, steps-to-next-violation)
        obs_l, cost_l, ttv_l = [], [], []
        for t in trajs:
            o, c = t["observations"], np.asarray(t["costs"], dtype=np.float32)
            viol_idx = np.where(c > 0)[0]
            ttv = np.full(len(c), np.inf)
            if len(viol_idx):
                nxt = np.searchsorted(viol_idx, np.arange(len(c)))
                has = nxt < len(viol_idx)
                ttv[has] = viol_idx[nxt[has]] - np.arange(len(c))[has]
            obs_l.append(o); cost_l.append(c); ttv_l.append(ttv)
        obs = np.concatenate(obs_l); cost = np.concatenate(cost_l)
        ttv = np.concatenate(ttv_l)
        idx = rng.choice(len(obs), size=min(N_SAMPLE, len(obs)), replace=False)
        obs_s, cost_s, ttv_s = obs[idx], cost[idx], ttv[idx]

        # A1: constraint dims by |corr(obs_i, cost)|
        oc = obs_s - obs_s.mean(0); cc = cost_s - cost_s.mean()
        denom = oc.std(0) * cost_s.std() + 1e-12
        corr = np.abs((oc * cc[:, None]).mean(0) / denom)
        cost_rank = np.argsort(corr)[::-1]

        # A2: saliency mean|dV/ds_i|
        xt = torch.as_tensor(obs_s[:8192], dtype=torch.float32,
                             device=DEVICE).requires_grad_(True)
        v = ens(xt).sum()
        v.backward()
        sal = xt.grad.abs().mean(0).detach().cpu().numpy()
        sal_rank = np.argsort(sal)[::-1]
        top3_overlap = len(set(cost_rank[:3].tolist()) &
                           set(sal_rank[:3].tolist()))

        # B: landscape along top constraint dim
        d = int(cost_rank[0])
        xs = obs_s[:, d]
        with torch.no_grad():
            vs = np.concatenate([
                ens(torch.as_tensor(obs_s[j:j+8192], dtype=torch.float32
                    ).to(DEVICE)).cpu().numpy()
                for j in range(0, len(obs_s), 8192)])
        qs = np.quantile(xs, np.linspace(0.01, 0.99, 25))
        bins = np.digitize(xs, qs)
        land = [{"x": float(xs[bins == b].mean()),
                 "V": float(vs[bins == b].mean()),
                 "cost_rate": float(cost_s[bins == b].mean())}
                for b in range(1, 25) if (bins == b).sum() > 50]

        # C: anticipation curve
        edges = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        antic = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ttv_s >= lo) & (ttv_s < hi)
            if m.sum() > 30:
                antic.append({"ttv_lo": lo, "ttv_hi": hi,
                              "V": float(vs[m].mean()), "n": int(m.sum())})
        far = np.isinf(ttv_s) | (ttv_s >= 89)
        antic.append({"ttv_lo": 89, "ttv_hi": None,
                      "V": float(vs[far].mean()), "n": int(far.sum())})

        # D: example traces (highest-cost traj + a zero-cost traj)
        tcosts = np.array([float(np.sum(t["costs"])) for t in trajs])
        unsafe_i = int(np.argmin(np.abs(tcosts - 2 * limit)))
        safe_cands = np.where(tcosts == 0)[0]
        safe_i = int(safe_cands[0]) if len(safe_cands) else int(np.argmin(tcosts))
        traces = {}
        for name, i in (("unsafe", unsafe_i), ("safe", safe_i)):
            o = torch.as_tensor(trajs[i]["observations"],
                                dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                vt = np.concatenate([ens(o[j:j+8192]).cpu().numpy()
                                     for j in range(0, len(o), 8192)])
            traces[name] = {"V": vt[::5].tolist(),
                            "cost_steps": np.where(
                                np.asarray(trajs[i]["costs"]) > 0)[0].tolist(),
                            "stride": 5, "traj_cost": float(tcosts[i])}

        out[task] = {
            "cost_corr_top5": [[int(i), float(corr[i])] for i in cost_rank[:5]],
            "saliency_top5": [[int(i), float(sal[i])] for i in sal_rank[:5]],
            "top3_overlap": int(top3_overlap),
            "landscape_dim": d, "landscape": land,
            "anticipation": antic, "traces": traces,
        }
        print(f"{task}: constraint dim {d} (corr {corr[d]:.2f}), "
              f"saliency top dim {int(sal_rank[0])} (|dV/ds| {sal[sal_rank[0]]:.3f}), "
              f"top3 overlap {top3_overlap}/3", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "interpretability.json"), "w") as f:
        json.dump(out, f)
    print(f"Saved -> {OUT}/interpretability.json", flush=True)


if __name__ == "__main__":
    main()
