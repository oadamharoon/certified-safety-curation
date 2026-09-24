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

# --- paths ------------------------------------------------------------------
# Datasets, checkpoints and run output live outside the repository. Set CSC_WORKSPACE,
# or the individual roots, to point at yours. See the README.
import os as _os


def _csc_root(_p):
    """The repository root, found by the .csc-root marker rather than by depth."""
    _d = _os.path.dirname(_os.path.abspath(_p))
    while True:
        if _os.path.exists(_os.path.join(_d, ".csc-root")):
            return _d
        _up = _os.path.dirname(_d)
        if _up == _d:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(_p)))
        _d = _up


# __file__ is undefined when a script's source is exec'd in a fresh namespace, which the
# audit does to reuse the table builder's tables; fall back to the working directory, which
# the .csc-root walk resolves from anywhere inside the repository.
_self = globals().get("__file__") or _os.path.join(_os.getcwd(), "_")
CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(_self))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "datasets"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs ship with the repository, so they resolve on their own
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import json, os, pickle, sys
import numpy as np
import torch

REPO = CSC_WORK
OUT = CSC_PAPER_DATA
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
