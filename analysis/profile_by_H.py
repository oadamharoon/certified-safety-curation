"""Contamination profile of the threshold grid at each segment length H.

Proposition 3 part (ii) turns on u_1, the contamination at the FIRST grid
threshold, because the fixed-sequence walk stops at its first failure. The
purity margin uses min_j u_j instead. This records both so the H = 10
certification collapse can be attributed to one or the other.
CPU-only; the GPU is running the A3 CDT queue.
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
import numpy as np, torch, yaml

D = CSC_WORK
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
torch.set_num_threads(8)

QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
ALPHA = 0.25
TASKS = ["halfcheetah_velocity","walker2d_velocity","ant_velocity","hopper_velocity",
         "swimmer_velocity","cargoal1_dsrl","cargoal2","pointgoal1_dsrl","pointgoal2"]
ARMS = [("H10","config_h10.yaml","_h10"),("H30","config.yaml",""),("H50","config_h50.yaml","_h50")]

out = {}
for arm, cfgfile, suf in ARMS:
    cfg = yaml.safe_load(open(cfgfile)); out[arm] = {}
    for task in TASKS:
        tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
        trajs = pickle.load(open(tc["data_pickle"], "rb"))
        unsafe = (np.array([float(np.sum(x["costs"])) for x in trajs]) > lim).astype(float)
        profs = []
        for seed in range(3):
            fp = f"outputs/{task}{suf}/v_ensemble_pess_seed{seed}.pt"
            if not os.path.exists(fp): continue
            ck = torch.load(fp, map_location="cpu", weights_only=False)
            ens = VEnsemble(ck["obs_dim"], ck["hidden_dim"], K=ck["K"])
            ens.load_state_dict(ck["state_dict"]); ens.eval()
            g = np.zeros(len(trajs))
            with torch.no_grad():
                for i, x in enumerate(trajs):
                    o = torch.as_tensor(x["observations"], dtype=torch.float32)
                    g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0,len(o),8192)]).mean().item()
            profs.append([float(unsafe[g >= np.quantile(g, q)].mean()) for q in QS])
        if not profs: continue
        P = np.array(profs)
        out[arm][task] = {"u_profile_mean": P.mean(0).tolist(),
                          "u_first_mean": float(P[:, 0].mean()),
                          "u_min_mean": float(P.min(1).mean()),
                          "margin_mean": float(ALPHA - P.min(1).mean()),
                          "argmin_q": [float(QS[i]) for i in P.argmin(1)]}
        print(f"  {arm} {task:22s} u_first {P[:,0].mean():.3f}  u_min {P.min(1).mean():.3f}"
              f"  margin {ALPHA-P.min(1).mean():+.3f}", flush=True)

dst = CSC_PAPER_DATA + "/review_response/profile_by_H.json"
json.dump({"quantiles": QS, "alpha": ALPHA, "arms": out}, open(dst, "w"), indent=1)

print("\n=== attribution")
CERT = {"H10": (0, 27), "H30": (6, 33), "H50": (3, 27)}
for arm in ("H10","H30","H50"):
    a = out[arm]
    nf = sum(1 for v in a.values() if v["u_first_mean"] <= ALPHA)
    nm = sum(1 for v in a.values() if v["margin_mean"] > 0)
    c, n = CERT[arm]
    print(f"  {arm}: first-threshold clean {nf}/9 | positive margin {nm}/9 | certified {c}/{n}")
    print(f"       mean u_first {np.mean([v['u_first_mean'] for v in a.values()]):.3f}"
          f"  mean margin {np.mean([v['margin_mean'] for v in a.values()]):+.3f}")
print(f"\n  saved -> {dst}")
