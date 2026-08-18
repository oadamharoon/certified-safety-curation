"""A2: rebuild certified selections from the standardized V ensembles.

For each (task, alpha) target, resample calibration draws to find which
grid thresholds certify and how often, then emit kept-index files for the
three most probable DISTINCT certified selections, ordered by realized
contamination. CPU-only so it does not contend with GPU queues.
"""
import json, os, pickle, sys
import numpy as np
import torch
import yaml
from scipy.stats import hypergeom

D = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/runs/selections"
sys.path.insert(0, D); os.chdir(D)
os.makedirs(OUT, exist_ok=True)
from model.policy import VEnsemble

torch.set_num_threads(4)
cfg = yaml.safe_load(open("config.yaml"))
QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
DELTA, NCAL, REPS = 0.1, 200, 500
# (task, alpha, v_seed) -- v_seed is the ensemble whose selections we deploy
JOBS = [("pointgoal1_dsrl", 0.25, 0), ("carrun_b", 0.25, 0),
        ("cargoal2", 0.40, 0), ("pointgoal2", 0.40, 0), ("pointgoal1_dsrl", 0.40, 0)]

summary = {}
for task, alpha, vs in JOBS:
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (cost > lim).astype(float)
    obs_dim = trajs[0]["observations"].shape[1]
    ens = VEnsemble(obs_dim, 256, K=3)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt", map_location="cpu", weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32)
            g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0, len(o), 8192)]).mean().item()
    taus = [float(np.quantile(g, q)) for q in QS]
    Ns = [int((g >= t).sum()) for t in taus]
    rng = np.random.default_rng(500 + vs)
    picks = {}
    for _ in range(REPS):
        cal = rng.choice(len(g), NCAL, replace=False)
        cs, cu = g[cal], unsafe[cal]
        chosen = None
        for j, t in enumerate(taus):
            sel = cs >= t; m, k = int(sel.sum()), int(cu[sel].sum())
            ks = int(alpha * Ns[j]) + 1
            p = float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0
            if m > 0 and p <= DELTA: chosen = j
            else: break
        if chosen is not None:
            picks[chosen] = picks.get(chosen, 0) + 1
    if not picks:
        print(f"{task} a{alpha}: NEVER CERTIFIES ({REPS} draws)", flush=True)
        summary[f"{task}_a{alpha}"] = {"cert_rate": 0.0, "selections": []}
        continue
    top = sorted(picks.items(), key=lambda kv: -kv[1])[:3]
    sels = []
    for j, cnt in top:
        kept = np.where(g >= taus[j])[0]
        contam = float(unsafe[kept].mean())
        name = f"{task}_a{int(alpha*100)}_q{int(QS[j]*100)}"
        json.dump({"kept": kept.tolist()}, open(f"{OUT}/{name}_kept.json", "w"))
        sels.append({"file": f"{name}_kept.json", "quantile": QS[j], "prob": cnt / REPS,
                     "n_kept": int(len(kept)), "contamination": contam})
    sels.sort(key=lambda s: s["contamination"])
    summary[f"{task}_a{alpha}"] = {"cert_rate": sum(picks.values()) / REPS, "selections": sels}
    print(f"{task} a{alpha}: cert {sum(picks.values())/REPS:.2f} | " +
          " | ".join(f"q{int(s['quantile']*100)} p={s['prob']:.2f} n={s['n_kept']} ahat={s['contamination']:.3f}"
                     for s in sels), flush=True)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("A2 SELECTIONS DONE", flush=True)
