"""Noise + preference-budget robustness statistics (stats-only).

For each V-ensemble variant (clean n=1000; noise 5/10/20/30%; n=100/300):
  - filter precision at the matched fraction
  - certification rate + false-cert rate at (alpha=0.25, delta=0.1, n=200),
    strict FS + exact hypergeometric, 100 draws
Output: data/robustness_stats.json
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
FRACS = {"halfcheetah_velocity": 0.182, "walker2d_velocity": 0.284,
         "ant_velocity": 0.197, "hopper_velocity": 0.111,
         "swimmer_velocity": 0.139, "cargoal1_dsrl": 0.399,
         "cargoal2": 0.324, "pointgoal1_dsrl": 0.518, "pointgoal2": 0.257}
VARIANTS = {"clean": "v_ensemble_pess_seed{s}.pt",
            "noise05": "v_ensemble_noise05_seed{s}.pt",
            "noise10": "v_ensemble_noise10_seed{s}.pt",
            "noise20": "v_ensemble_noise20_seed{s}.pt",
            "noise30": "v_ensemble_noise30_seed{s}.pt",
            "n100": "v_ensemble_n100_seed{s}.pt",
            "n300": "v_ensemble_n300_seed{s}.pt",
            "boltz": "v_ensemble_boltz_seed{s}.pt"}
ALPHA, DELTA, CAL_N, N_DRAWS = 0.25, 0.1, 200, 100
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyper_pvalue(k, m, n_sel, alpha):
    from scipy.stats import hypergeom
    k_star = int(alpha * n_sel) + 1
    return 1.0 if k_star > n_sel else float(hypergeom.cdf(k, n_sel, k_star, m))


def ltt(scores, unsafe, rng):
    cal_idx = rng.choice(len(scores), size=CAL_N, replace=False)
    cs, cu = scores[cal_idx], unsafe[cal_idx]
    tau, cert = float(np.quantile(scores, 0.80)), False
    for q in QS:
        t = float(np.quantile(scores, q))
        sel = cs >= t
        m, k = int(sel.sum()), int(cu[sel].sum())
        p = hyper_pvalue(k, m, int((scores >= t).sum()), ALPHA) if m > 0 else 1.0
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
    safe = 1 - unsafe
    obs_dim = trajs[0]["observations"].shape[1]
    n_keep = max(1, int(len(trajs) * FRACS[task]))
    task_out = {}
    for vname, pat in VARIANTS.items():
        seeds_out = []
        for s in (0, 1, 2):
            p = f"outputs/{task}/{pat.format(s=s)}"
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
            kept = np.argsort(scores)[::-1][:n_keep]
            prec = float(safe[kept].mean())
            rng = np.random.default_rng(7)
            certs, fcs = 0, 0
            for _ in range(N_DRAWS):
                tau, cert = ltt(scores, unsafe, rng)
                if cert:
                    certs += 1
                    if unsafe[scores >= tau].mean() > ALPHA:
                        fcs += 1
            seeds_out.append({"seed": s, "precision": prec,
                              "cert_rate": certs / N_DRAWS,
                              "false_cert_rate": fcs / N_DRAWS})
        task_out[vname] = seeds_out
        if seeds_out:
            mp = np.mean([x["precision"] for x in seeds_out])
            mc = np.mean([x["cert_rate"] for x in seeds_out])
            print(f"{task} {vname}: prec={mp:.3f} cert={mc:.2f}", flush=True)
    out[task] = task_out
with open(os.path.join(OUT, "robustness_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved data/robustness_stats.json")
