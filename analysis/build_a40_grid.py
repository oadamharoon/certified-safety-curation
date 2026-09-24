"""Identify and emit kept-index files for the 24 alpha=0.40 grid selections.

The grid in iclr2027/scripts/make_tables.py (_A40SEL) names three selections per
task by realized contamination. The h5 files behind draw2/draw3 were written to
a session scratchpad and lost, but a selection is fully determined by
(task, V seed, quantile), so each target contamination is matched back to a
quantile of the deployed V ensemble and rebuilt. Nothing is written unless the
contamination matches the published value.

Output goes to runs/selections/, NOT a scratchpad.
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
OUT = CSC_RUNS + "/selections"
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
cfg = yaml.safe_load(open("config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
QS = [0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10]

# task -> (deployed V seed, cost limit, [target contaminations from _A40SEL])
JOBS = {
 "halfcheetah_velocity": (1, 20, [0.219, 0.287, 0.351]),
 "walker2d_velocity":    (0, 20, [0.276, 0.325, 0.393]),
 "ant_velocity":         (0, 20, [0.343, 0.360, 0.406]),
 "swimmer_velocity":     (1, 20, [0.221, 0.382, 0.462]),
 "cargoal1_dsrl":        (0, 25, [0.187, 0.353, 0.375]),
 "cargoal2":             (0, 25, [0.238, 0.287, 0.334]),
 "pointgoal1_dsrl":      (0, 25, [0.299, 0.318, 0.345]),
 "pointgoal2":           (1, 25, [0.269, 0.335, 0.382]),
}
tot = ok = 0
for task, (vs, lim, targets) in JOBS.items():
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
            g[i] = torch.cat([ens(o[j:j+8192]).cpu() for j in range(0, len(o), 8192)]).mean().item()
    table = []
    for q in QS:
        tau = float(np.quantile(g, q)); k = g >= tau
        table.append((q, tau, int(k.sum()), float(unsafe[k].mean())))
    print(f"  {task} (V seed {vs})")
    for tgt in targets:
        tot += 1
        best = min(table, key=lambda r: abs(r[3]-tgt))
        q, tau, n, c = best
        if abs(c-tgt) > 0.003:
            print(f"      target {tgt:.3f}: NO QUANTILE MATCHES (closest q{int(q*100)} c={c:.3f})")
            continue
        kept = np.where(g >= tau)[0]
        p = os.path.join(OUT, f"{task}_a40grid_q{int(q*100)}_kept.json")
        json.dump({"kept": [int(i) for i in kept], "quantile": q, "tau": tau,
                   "n_kept": n, "contamination": c, "v_seed": vs, "task": task,
                   "target_contamination": tgt}, open(p, "w"))
        ok += 1
        print(f"      target {tgt:.3f} -> q{int(q*100)} n={n} c={c:.3f} VERIFIED")
print(f"\n{ok}/{tot} grid selections identified and written")
