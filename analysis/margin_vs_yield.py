"""Is certification yield governed by the purity margin below alpha?

For each task: compute the best achievable contamination over the
threshold grid (the purest selection the score can produce), define
margin = alpha - that, and relate it to the measured certification rate.
CPU-only.
"""
import json, os, pickle, sys
import numpy as np, torch, yaml
D = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
torch.set_num_threads(4)
cfg = yaml.safe_load(open("config.yaml"))
QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
ALPHA = 0.25
G = json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/guarantee_stats_2000.json"))
rows = []
for task, td in G.items():
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(x["costs"])) for x in trajs]); unsafe = (cost > lim).astype(float)
    obs_dim = trajs[0]["observations"].shape[1]
    margins, rates = [], []
    for seed, sd in td["seeds"].items():
        fp = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(fp) or "200" not in sd: continue
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, x in enumerate(trajs):
                o = torch.as_tensor(x["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0,len(o),8192)]).mean().item()
        best = min(float(unsafe[g >= np.quantile(g, q)].mean()) for q in QS)
        margins.append(ALPHA - best); rates.append(sd["200"]["cert_rate"])
    if margins:
        rows.append((task, float(np.mean(margins)), float(np.mean(rates)), float(unsafe.mean())))
rows.sort(key=lambda r: r[1])
print(f"{'task':22s} {'margin(a-purity)':>17s} {'cert rate':>10s} {'pool unsafe':>12s}")
for t, m, r, u in rows:
    print(f"{t:22s} {m:17.3f} {r:10.2f} {u:12.3f}")
x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
from scipy.stats import spearmanr, pearsonr
print(f"\nSpearman(margin, cert rate) = {spearmanr(x, y).statistic:.3f}")
print(f"Pearson (margin, cert rate) = {pearsonr(x, y)[0]:.3f}   (n = {len(rows)} tasks)")
neg = [r[0] for r in rows if r[1] <= 0]
print(f"tasks with margin <= 0 (purity cannot reach alpha): {neg}")
print(f"  their cert rates: {[round(r[2],3) for r in rows if r[1] <= 0]}")
json.dump([{"task": t, "margin": m, "cert_rate": r, "pool_unsafe": u} for t, m, r, u in rows],
          open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/margin_vs_yield.json", "w"), indent=1)
print("MARGIN VS YIELD DONE")
