"""Mean-cost certificate variant (stats-only).

Same LTT machinery, second risk functional: certify that the MEAN episodic
cost of the selection is at most the task budget b. p-value via Hoeffding's
inequality, which holds for sampling without replacement (Hoeffding 1963,
Sec. 6): with m calibration trajectories in the selection showing sample
mean xbar, and costs bounded in [0, R] (R = pool max, known),
    p = exp(-2 m (b - xbar)_+^2 / R^2)
is super-uniform under H0: selection mean > b. Strict fixed sequence over
the same quantile grid. 100 draws per cell at n=200, delta=0.1.
Output: data/meancost_cert_stats.json
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
DELTA, CAL_N, N_DRAWS = 0.1, 200, 100
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = yaml.safe_load(open("config.yaml"))
out = {}
for task, b in LIMITS.items():
    with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    R = float(costs.max())
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
        rng = np.random.default_rng(7)
        certs, fcs, kept_fracs = 0, 0, []
        for _ in range(N_DRAWS):
            cal_idx = rng.choice(len(scores), size=CAL_N, replace=False)
            cs, cc = scores[cal_idx], costs[cal_idx]
            tau, cert = None, False
            for q in QS:
                t = float(np.quantile(scores, q))
                sel = cs >= t
                m = int(sel.sum())
                if m == 0:
                    break
                xbar = float(cc[sel].mean())
                pv = float(np.exp(-2 * m * max(0.0, b - xbar) ** 2 / R ** 2))
                if pv <= DELTA:
                    tau, cert = t, True
                else:
                    break
            if cert:
                certs += 1
                kept = scores >= tau
                kept_fracs.append(float(kept.mean()))
                if costs[kept].mean() > b:
                    fcs += 1
        task_out[str(seed)] = {"cert_rate": certs / N_DRAWS,
                               "false_cert_rate": fcs / N_DRAWS,
                               "mean_kept_frac": float(np.mean(kept_fracs)) if kept_fracs else None}
        print(f"{task} s{seed}: cert={certs/N_DRAWS:.2f} "
              f"falsecert={fcs/N_DRAWS:.2f}", flush=True)
    out[task] = task_out
with open(os.path.join(OUT, "meancost_cert_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved data/meancost_cert_stats.json")
