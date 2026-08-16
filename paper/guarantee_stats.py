"""Guarantee-validation statistics for the ICLR paper.

For each task and V-ensemble seed: score all trajectories once (cached),
then re-run the LTT calibration across many fresh calibration draws and
CAL_N values. Records per draw: certified?, chosen threshold quantile,
kept fraction, kept-set TRUE unsafe rate. Outputs JSON to iclr2027/data/.

Coverage claim validated: among CERTIFIED draws, the fraction whose
kept-set unsafe rate exceeds alpha should be <= delta.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np
import torch

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data"
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

TASKS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
         "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
         "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
ALPHA, DELTA = 0.25, 0.1
CAL_NS = [50, 100, 200, 400]
N_DRAWS = 200
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyper_pvalue(k, m, n_sel, alpha):
    from scipy.stats import hypergeom
    k_star = int(alpha * n_sel) + 1
    if k_star > n_sel:
        return 1.0
    return float(hypergeom.cdf(k, n_sel, k_star, m))


def ltt(scores, unsafe, rng, cal_n):
    cal_idx = rng.choice(len(scores), size=min(cal_n, len(scores)), replace=False)
    cs, cu = scores[cal_idx], unsafe[cal_idx]
    taus = [float(np.quantile(scores, q)) for q in QS]
    chosen_q, chosen_tau, certified = None, None, False
    for q, tau in zip(QS, taus):
        sel = cs >= tau
        m, k = int(sel.sum()), int(cu[sel].sum())
        n_sel = int((scores >= tau).sum())
        p = hyper_pvalue(k, m, n_sel, ALPHA) if m > 0 else 1.0
        if m > 0 and p <= DELTA:
            chosen_q, chosen_tau, certified = q, tau, True
        else:
            break  # strict fixed-sequence: stop at first failure
    if not certified:
        chosen_q, chosen_tau = 0.80, float(np.quantile(scores, 0.80))
    return chosen_q, chosen_tau, certified


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    results = {}
    for task, limit in TASKS.items():
        tc = cfg["tasks"][task]
        with open(tc["data_pickle"], "rb") as f:
            trajs = pickle.load(f)
        costs = np.array([float(np.sum(t["costs"])) for t in trajs])
        unsafe = (costs > limit).astype(int)
        obs_dim = trajs[0]["observations"].shape[1]
        task_out = {"limit": limit, "n_trajs": len(trajs),
                    "base_unsafe_rate": float(unsafe.mean()), "seeds": {}}
        for seed in (0, 1, 2):
            ens_p = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
            if not os.path.exists(ens_p):
                continue
            ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
            ck = torch.load(ens_p, map_location=DEVICE, weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            ens.eval()
            scores = np.zeros(len(trajs))
            with torch.no_grad():
                for i, t in enumerate(trajs):
                    o = torch.as_tensor(t["observations"], dtype=torch.float32).to(DEVICE)
                    vs = [ens(o[j:j+8192]).cpu() for j in range(0, len(o), 8192)]
                    scores[i] = torch.cat(vs).mean().item()
            seed_out = {}
            for cal_n in CAL_NS:
                rng = np.random.default_rng(7)
                draws = []
                for _ in range(N_DRAWS):
                    q, tau, cert = ltt(scores, unsafe, rng, cal_n)
                    kept = scores >= tau
                    draws.append({"q": q, "certified": bool(cert),
                                  "kept_frac": float(kept.mean()),
                                  "kept_unsafe": float(unsafe[kept].mean())})
                cert_draws = [d for d in draws if d["certified"]]
                viol = [d for d in cert_draws if d["kept_unsafe"] > ALPHA]
                seed_out[str(cal_n)] = {
                    "n_draws": N_DRAWS,
                    "cert_rate": len(cert_draws) / N_DRAWS,
                    "coverage_violation_rate":
                        (len(viol) / len(cert_draws)) if cert_draws else None,
                    "mean_kept_frac_certified":
                        float(np.mean([d["kept_frac"] for d in cert_draws]))
                        if cert_draws else None,
                    "mean_kept_unsafe_certified":
                        float(np.mean([d["kept_unsafe"] for d in cert_draws]))
                        if cert_draws else None,
                    "draws": draws,
                }
            task_out["seeds"][str(seed)] = seed_out
            print(f"{task} seed{seed}: " + " | ".join(
                f"n={c}: cert={task_out['seeds'][str(seed)][str(c)]['cert_rate']:.2f}"
                f" violrate={task_out['seeds'][str(seed)][str(c)]['coverage_violation_rate']}"
                for c in CAL_NS), flush=True)
        results[task] = task_out
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "guarantee_stats.json"), "w") as f:
        json.dump(results, f)
    print(f"Saved -> {OUT}/guarantee_stats.json", flush=True)


if __name__ == "__main__":
    main()
