"""Bullet certification operating curve: LTT resampled on archived scores
at alpha in {0.25, 0.40, 0.50}; scores computed once per (task, seed)."""
import json, pickle, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
import os
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
from scipy.stats import hypergeom
import yaml

torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def hp(k, m, n_sel, a):
    ks = int(a * n_sel) + 1
    return 1.0 if ks > n_sel else float(hypergeom.cdf(k, n_sel, ks, m))


out = {}
for t in ("ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b"):
    trajs = pickle.load(open(cfg["tasks"][t]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(x["costs"])) for x in trajs])
    unsafe = (costs > 10).astype(int)
    obs_dim = trajs[0]["observations"].shape[1]
    per_alpha = {a: [] for a in (0.25, 0.40, 0.50)}
    for seed in (0, 1, 2):
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(f"outputs/{t}/v_ensemble_pess_seed{seed}.pt",
                        map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, x in enumerate(trajs):
                o = torch.as_tensor(x["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j + 8192])
                                  for j in range(0, len(o), 8192)]).mean().item()
        taus = [float(np.quantile(g, q)) for q in QS]
        nsel = [int((g >= tau).sum()) for tau in taus]
        for alpha in (0.25, 0.40, 0.50):
            rng = np.random.default_rng(7)
            nc = 0
            for _ in range(2000):
                cal = rng.choice(len(g), size=200, replace=False)
                cs, cu = g[cal], unsafe[cal]
                pick = None
                for qi, tau in enumerate(taus):
                    sel = cs >= tau
                    m, k = int(sel.sum()), int(cu[sel].sum())
                    p = hp(k, m, nsel[qi], alpha) if m > 0 else 1.0
                    if m > 0 and p <= 0.1:
                        pick = qi
                    else:
                        break
                if pick is not None:
                    nc += 1
            per_alpha[alpha].append(nc / 2000)
    out[t] = {str(a): float(np.mean(v)) for a, v in per_alpha.items()}
    print(t, out[t], flush=True)
json.dump(out, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/bullet_alpha_curve.json", "w"), indent=1)
print("BULLET ALPHA CURVE DONE", flush=True)
