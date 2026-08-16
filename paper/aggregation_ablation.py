"""Trajectory-score aggregation ablation (stats-only).

Why mean-V per trajectory? Compare filter precision at the matched fraction
for aggregation functions over per-state values: mean, median, min,
10th percentile (CVaR-flavored worst-state emphasis), and sum (length-
weighted). Uses saved K=3 ensembles; no training.
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
AGGS = {"mean": np.mean, "median": np.median, "min": np.min,
        "q10": lambda v: np.quantile(v, 0.10), "sum": np.sum}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = yaml.safe_load(open("config.yaml"))
rows = []
for task, limit in LIMITS.items():
    with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    safe = costs <= limit
    obs_dim = trajs[0]["observations"].shape[1]
    n_keep = max(1, int(len(trajs) * FRACS[task]))
    for seed in (0, 1, 2):
        p = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(p):
            continue
        ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        per_state = []
        with torch.no_grad():
            for t in trajs:
                o = torch.as_tensor(t["observations"], dtype=torch.float32).to(DEVICE)
                vs = torch.cat([ens(o[j:j+8192]).cpu()
                                for j in range(0, len(o), 8192)]).numpy()
                per_state.append(vs)
        row = {"task": task, "seed": seed, "base_rate": float(safe.mean())}
        for name, fn in AGGS.items():
            scores = np.array([fn(v) for v in per_state])
            kept = np.argsort(scores)[::-1][:n_keep]
            row[f"prec_{name}"] = float(safe[kept].mean())
        rows.append(row)
        print(f"{task} s{seed}: " + " ".join(
            f"{k[5:]}={row[k]:.3f}" for k in row if k.startswith("prec_")),
            flush=True)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "aggregation_ablation.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("saved data/aggregation_ablation.json")
