"""Empirical validation of Proposition 1 by calibration resampling.

For each task, value-ensemble seed, and calibration size n, redraw the
calibration sample many times and record: certification rate, the
UNCONDITIONAL false-certification rate the theorem bounds, and the
CONDITIONAL violation rate a certificate-holder faces, each with
Clopper-Pearson intervals. CPU-only.

Writes iclr2027/data/review_response/guarantee_stats_2000.json
"""
import json, os, pickle, sys
import numpy as np
import torch
import yaml
from scipy.stats import hypergeom, beta

D = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble

torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
ALPHA, DELTA, N_DRAWS = 0.25, 0.1, 2000
SIZES = [50, 100, 200, 400]
TASKS = "halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2 ballrun_b ballcircle_b carcircle_b carrun_b dronerun_b".split()


def cp95(k, n):
    if n == 0: return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


out = {}
for task in TASKS:
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (cost > lim).astype(float)
    obs_dim = trajs[0]["observations"].shape[1]
    entry = {"limit": lim, "n_trajs": len(trajs),
             "base_unsafe_rate": float(unsafe.mean()), "seeds": {}}
    for seed in (0, 1, 2):
        fp = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(fp): continue
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, t in enumerate(trajs):
                o = torch.as_tensor(t["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0, len(o), 8192)]).mean().item()
        taus = [float(np.quantile(g, q)) for q in QS]
        Ns = [int((g >= t).sum()) for t in taus]
        ku = [float(unsafe[g >= t].mean()) for t in taus]
        sd = {}
        for n in SIZES:
            rng = np.random.default_rng(9000 + seed)
            n_cert = n_false = 0
            for _ in range(N_DRAWS):
                cal = rng.choice(len(g), n, replace=False)
                cs, cu = g[cal], unsafe[cal]
                pick = None
                for j, t in enumerate(taus):
                    sel = cs >= t; m, k = int(sel.sum()), int(cu[sel].sum())
                    ks = int(ALPHA * Ns[j]) + 1
                    p = float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0
                    if m > 0 and p <= DELTA: pick = j
                    else: break
                if pick is not None:
                    n_cert += 1
                    if ku[pick] > ALPHA: n_false += 1
            sd[str(n)] = {
                "n_draws": N_DRAWS,
                "cert_rate": n_cert / N_DRAWS,
                "false_cert_rate_uncond": n_false / N_DRAWS,
                "false_cert_cp95": cp95(n_false, N_DRAWS),
                "cond_viol_rate": (n_false / n_cert) if n_cert else None,
                "cond_viol_cp95": cp95(n_false, n_cert) if n_cert else None,
            }
        entry["seeds"][str(seed)] = sd
        print(f"{task} s{seed}: " + " ".join(
            f"n{n}:cert{sd[str(n)]['cert_rate']:.2f}/unc{sd[str(n)]['false_cert_rate_uncond']:.3f}"
            for n in SIZES), flush=True)
    out[task] = entry
dst = "/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/guarantee_stats_2000.json"
json.dump(out, open(dst, "w"), indent=1)
print("GUARANTEE VALIDATION DONE ->", dst, flush=True)
