"""Emit kept-index files for the nine alpha=0.25 certified selections.

The nine are three DISTINCT quantile thresholds of the standardized V ensemble
per certifying task (see the PROVENANCE block in iclr2027/scripts/make_tables.py).
A selection is fully determined by (task, V seed, quantile), so this rebuilds
them deterministically and verifies n_kept and realized contamination against
the values the paper reports before writing anything.

Output goes to runs/selections/, NOT a scratchpad.
"""

# --- paths ------------------------------------------------------------------
# The research tree addressed itself by absolute path; these roots replace it. Set
# CSC_WORKSPACE (or the individual roots) to point at your own trees. See the README.
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
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "vlm-with-cpl", "new_data"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs are carried by the repository, so they resolve without a working tree
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import json, os, pickle, sys
import numpy as np, torch, yaml

D = CSC_WORK
OUT = CSC_RUNS + "/selections"
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
cfg = yaml.safe_load(open("config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

# (task, V seed, cost limit, [(quantile, expected n_kept, expected contamination)])
JOBS = [
    ("halfcheetah_velocity", 1, 20, [(0.85, 375, 0.219), (0.80, 499, 0.287), (0.75, 624, 0.351)]),
    ("cargoal1_dsrl",        3, 25, [(0.85, 251, 0.171), (0.80, 335, 0.212), (0.75, 418, 0.266)]),
    ("pointgoal1_dsrl",      0, 25, [(0.85, 304, 0.122), (0.80, 405, 0.141), (0.55, 910, 0.238)]),
]
ok = 0
for task, vs, lim, sels in JOBS:
    tc = cfg["tasks"][task]
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (cost > lim).astype(int)
    ens = VEnsemble(trajs[0]["observations"].shape[1], 256, K=3).to(dev)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt", map_location=dev, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32).to(dev)
            g[i] = torch.cat([ens(o[j:j + 8192]).cpu() for j in range(0, len(o), 8192)]).mean().item()
    for q, exp_n, exp_c in sels:
        tau = float(np.quantile(g, q))
        kept = np.where(g >= tau)[0]
        c = float(unsafe[kept].mean())
        if len(kept) != exp_n or abs(c - exp_c) > 0.002:
            print(f"  MISMATCH {task} q{int(q*100)}: n={len(kept)} (exp {exp_n}) "
                  f"contam={c:.3f} (exp {exp_c})"); continue
        p = os.path.join(OUT, f"{task}_a25_q{int(q*100)}_kept.json")
        json.dump({"kept": [int(i) for i in kept], "quantile": q, "tau": tau,
                   "n_kept": int(len(kept)), "contamination": c,
                   "v_seed": vs, "task": task}, open(p, "w"))
        ok += 1
        print(f"  {task} q{int(q*100)}: n_kept={len(kept)} contam={c:.3f} VERIFIED -> {os.path.basename(p)}")
print(f"{ok}/9 selections verified and written")
