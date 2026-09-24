"""Extend the calibration-size sweep for positive-margin tasks, so the
1/margin^2 scaling can actually be fitted. CPU-only."""

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
from scipy.stats import hypergeom
D = CSC_WORK
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30]
ALPHA, DELTA, REPS = 0.25, 0.1, 1000
M = {r["task"]: r["margin"] for r in json.load(open(CSC_PAPER_DATA + "/review_response/margin_vs_yield.json"))}
TASKS = [t for t, m in M.items() if m > 0]
EXTRA = [800, 1600, 3200]
out = {}
for task in sorted(TASKS, key=lambda t: -M[t]):
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(x["costs"])) for x in trajs]); unsafe = (cost > lim).astype(float)
    N = len(trajs); obs_dim = trajs[0]["observations"].shape[1]
    sizes = [n for n in EXTRA if n <= int(0.8 * N)]
    if not sizes:
        print(f"{task}: pool too small ({N}) for extension", flush=True); continue
    per = {}
    for seed in (0, 1, 2):
        fp = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
        if not os.path.exists(fp): continue
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
        g = np.zeros(N)
        with torch.no_grad():
            for i, x in enumerate(trajs):
                o = torch.as_tensor(x["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0,len(o),8192)]).mean().item()
        taus = [float(np.quantile(g, q)) for q in QS]; Ns = [int((g >= t).sum()) for t in taus]
        for n in sizes:
            rng = np.random.default_rng(4000 + seed); nc = 0
            for _ in range(REPS):
                cal = rng.choice(N, n, replace=False); cs, cu = g[cal], unsafe[cal]
                ok = False
                for j, t in enumerate(taus):
                    sel = cs >= t; m_, k = int(sel.sum()), int(cu[sel].sum())
                    ks = int(ALPHA * Ns[j]) + 1
                    p = float(hypergeom.cdf(k, Ns[j], ks, m_)) if (m_ > 0 and ks <= Ns[j]) else 1.0
                    if m_ > 0 and p <= DELTA: ok = True
                    else: break
                nc += ok
            per.setdefault(str(n), []).append(nc / REPS)
    out[task] = {"margin": M[task], "n_pool": N,
                 "rates": {n: float(np.mean(v)) for n, v in per.items()}}
    print(f"{task:22s} margin {M[task]:.3f} " +
          " ".join(f"n{n}:{np.mean(v):.2f}" for n, v in sorted(per.items(), key=lambda kv: int(kv[0]))), flush=True)
json.dump(out, open(CSC_PAPER_DATA + "/review_response/label_complexity_ext.json", "w"), indent=1)
print("EXTENSION DONE", flush=True)
