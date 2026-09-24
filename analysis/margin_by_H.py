"""Does the purity margin explain the certification collapse at H = 10?

For each analysis task and each segment length H, load that arm's value
ensembles, score every trajectory, and compute the best achievable
contamination over the threshold grid. margin = alpha - that. Compare the
per-H mean margin against the certification counts measured in the sweep.
CPU-only; the GPU is occupied by the A3 CDT queue.
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
ARMS = [("H10","config_h10.yaml","_h10",3),
        ("H30","config.yaml","",3),
        ("H50","config_h50.yaml","_h50",3)]

out = {}
for arm, cfgfile, suf, nseed in ARMS:
    cfg = yaml.safe_load(open(cfgfile))
    out[arm] = {}
    for task in TASKS:
        tc = cfg["tasks"][task]
        lim = tc.get("cost_limit", cfg["cost_limit"])
        trajs = pickle.load(open(tc["data_pickle"], "rb"))
        cost = np.array([float(np.sum(x["costs"])) for x in trajs])
        unsafe = (cost > lim).astype(float)
        obs_dim = trajs[0]["observations"].shape[1]
        margins = []
        for seed in range(nseed):
            fp = f"outputs/{task}{suf}/v_ensemble_pess_seed{seed}.pt"
            if not os.path.exists(fp):
                continue
            ck = torch.load(fp, map_location="cpu", weights_only=False)
            sd = ck["state_dict"] if "state_dict" in ck else ck
            ens = VEnsemble(ck.get("obs_dim", obs_dim), ck.get("hidden_dim", 256),
                            K=ck.get("K", 3))
            ens.load_state_dict(sd); ens.eval()
            g = np.zeros(len(trajs))
            with torch.no_grad():
                for i, x in enumerate(trajs):
                    o = torch.as_tensor(x["observations"], dtype=torch.float32)
                    g[i] = torch.cat([ens(o[j:j+8192])
                                      for j in range(0, len(o), 8192)]).mean().item()
            best = min(float(unsafe[g >= np.quantile(g, q)].mean()) for q in QS)
            margins.append(ALPHA - best)
        if margins:
            out[arm][task] = {"margin_mean": float(np.mean(margins)),
                              "margin_seeds": [float(m) for m in margins],
                              "pool_unsafe": float(unsafe.mean())}
        print(f"  {arm} {task:22s} margin {np.mean(margins):+.3f}", flush=True)

dst = CSC_PAPER_DATA + "/review_response/margin_by_H.json"
json.dump(out, open(dst, "w"), indent=1)

print("\n=== summary")
CERT = {"H10": (0, 27), "H30": (6, 33), "H50": (3, 27)}
for arm in ("H10", "H30", "H50"):
    ms = [v["margin_mean"] for v in out[arm].values()]
    pos = [t for t, v in out[arm].items() if v["margin_mean"] > 0]
    c, n = CERT[arm]
    print(f"  {arm}: mean margin {np.mean(ms):+.4f}  median {np.median(ms):+.4f}  "
          f"positive-margin tasks {len(pos)}/9  certified {c}/{n}")
    print(f"       positive: {[t.split('_')[0] for t in pos]}")
print(f"\n  saved -> {dst}")
