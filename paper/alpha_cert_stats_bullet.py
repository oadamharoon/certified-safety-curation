"""Certification behavior across guarantee levels alpha (stats-only).

For alpha in {0.05, 0.10, 0.25, 0.40} at n=200: certification rate,
unconditional false-cert rate, mean kept fraction among certified draws.
Strict fixed-sequence + exact hypergeometric (mirrors 04q).
Output: data/alpha_cert_stats.json
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

LIMITS = {"ballrun_b": 10, "ballcircle_b": 10, "carcircle_b": 10,
          "carrun_b": 10, "dronerun_b": 10}
ALPHAS = [0.05, 0.10, 0.25, 0.40]
DELTA, CAL_N, N_DRAWS = 0.1, 200, 200
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyper_pvalue(k, m, n_sel, alpha):
    from scipy.stats import hypergeom
    k_star = int(alpha * n_sel) + 1
    if k_star > n_sel:
        return 1.0
    return float(hypergeom.cdf(k, n_sel, k_star, m))


def ltt(scores, unsafe, rng, alpha):
    cal_idx = rng.choice(len(scores), size=CAL_N, replace=False)
    cs, cu = scores[cal_idx], unsafe[cal_idx]
    tau, cert = float(np.quantile(scores, 0.80)), False
    for q in QS:
        t = float(np.quantile(scores, q))
        sel = cs >= t
        m, k = int(sel.sum()), int(cu[sel].sum())
        n_sel = int((scores >= t).sum())
        p = hyper_pvalue(k, m, n_sel, alpha) if m > 0 else 1.0
        if m > 0 and p <= DELTA:
            tau, cert = t, True
        else:
            break
    return tau, cert


cfg = yaml.safe_load(open("config.yaml"))
out = {}
for task, limit in LIMITS.items():
    with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > limit).astype(int)
    obs_dim = trajs[0]["observations"].shape[1]
    task_out = {}
    for seed in (0, 1, 2):
        p = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(p):
            continue
        ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        scores = np.zeros(len(trajs))
        with torch.no_grad():
            for i, t in enumerate(trajs):
                o = torch.as_tensor(t["observations"], dtype=torch.float32).to(DEVICE)
                vs = [ens(o[j:j+8192]).cpu() for j in range(0, len(o), 8192)]
                scores[i] = torch.cat(vs).mean().item()
        seed_out = {}
        for alpha in ALPHAS:
            rng = np.random.default_rng(7)
            certs, fcs, keptfracs = 0, 0, []
            for _ in range(N_DRAWS):
                tau, cert = ltt(scores, unsafe, rng, alpha)
                if cert:
                    certs += 1
                    kept = scores >= tau
                    keptfracs.append(float(kept.mean()))
                    if unsafe[kept].mean() > alpha:
                        fcs += 1
            seed_out[str(alpha)] = {
                "cert_rate": certs / N_DRAWS,
                "false_cert_rate": fcs / N_DRAWS,
                "mean_kept_frac": float(np.mean(keptfracs)) if keptfracs else None}
        task_out[str(seed)] = seed_out
        print(f"{task} s{seed}: " + " | ".join(
            f"a={a}: cert={seed_out[str(a)]['cert_rate']:.2f}" for a in ALPHAS),
            flush=True)
    out[task] = task_out
with open(os.path.join(OUT, "review_response", "bullet_alpha_curve.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved data/alpha_cert_stats.json")
