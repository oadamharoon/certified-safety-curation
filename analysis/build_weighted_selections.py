"""Probe 10: does coverage recovered by WEIGHTING convert into reward at equal cost?

Probe 9: only the final composition matters for an averaging learner.
Gating check: admitting the periphery (loose \ tight) at partial weight reaches
high-return-safe coverage that NO score threshold reaches at the same contamination.

This builds matched pairs so that COVERAGE is the only thing that differs:
  MIX_f  = all of the tight certified core + a random fraction f of the periphery
  THR_f  = the score-threshold selection with the closest contamination,
           randomly subsampled to the SAME size

Same n, same contamination, very different high-return-safe coverage. If the
coverage buys reward at equal cost, certified importance weighting is real.
Writes to runs/selections_recon/.
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
import numpy as np, yaml

D = CSC_WORK
OUT = CSC_RUNS + "/selections_recon"
sys.path.insert(0, D); os.chdir(D)
cfg = yaml.safe_load(open("config.yaml")); T, LIM = "pointgoal1_dsrl", 25
tr = pickle.load(open(cfg["tasks"][T]["data_pickle"], "rb"))
cost = np.array([float(np.sum(t["costs"])) for t in tr])
ret = np.array([float(np.sum(t["rewards"])) for t in tr])
unsafe = cost > LIM
safe = np.where(~unsafe)[0]
hi = set(safe[ret[safe] >= np.percentile(ret[safe], 90)].tolist())
sc = np.load(CSC_PAPER_DATA + "/e_scores/pointgoal1_dsrl_seed0.npz")["scores"]
core = sorted(set(json.load(open(f"outputs/{T}/kept_calfilt_csf_seed0.json"))["kept"]))
peri = sorted(set(json.load(open(f"outputs/{T}/kept_calfilt_a40_seed0.json"))["kept"]) - set(core))

def stats(idx):
    idx = np.array(sorted(idx))
    return float(unsafe[idx].mean()), float(np.mean([i in hi for i in idx])), len(idx)

rng = np.random.default_rng(0)
os.makedirs(OUT, exist_ok=True)
rows = []
for f in (0.05, 0.10):
    mix = core + list(rng.choice(peri, size=int(round(f * len(peri))), replace=False))
    cm, vm, nm = stats(mix)
    n_uns = int(round(cm * nm))
    # threshold selection whose contamination is closest, then STRATIFIED-subsampled
    # to the same n AND the same unsafe count, so only coverage can differ
    best = None
    for q in np.arange(0.95, 0.25, -0.005):
        idx = np.where(sc >= np.quantile(sc, q))[0]
        u, s = idx[unsafe[idx]], idx[~unsafe[idx]]
        if len(u) < n_uns or len(s) < nm - n_uns: continue
        c = float(unsafe[idx].mean())
        if best is None or abs(c - cm) < abs(best[1] - cm): best = (idx, c)
    idx, _ = best
    u, s = idx[unsafe[idx]], idx[~unsafe[idx]]
    thr = list(rng.choice(u, n_uns, replace=False)) + list(rng.choice(s, nm - n_uns, replace=False))
    ct, vt, nt = stats(thr)
    for tag, sel, c, v, n in ((f"mix{int(f*100):03d}", mix, cm, vm, nm),
                              (f"thr{int(f*100):03d}", thr, ct, vt, nt)):
        json.dump({"kept": [int(i) for i in sorted(sel)], "n_kept": n,
                   "contamination": c, "hi_return_safe_frac": v, "rule": tag},
                  open(os.path.join(OUT, f"{T}_{tag}_kept.json"), "w"))
        rows.append((tag, n, c, v))
print(f"  {'arm':8s} {'n':>6s} {'contamination':>14s} {'hi-ret-safe mass':>17s}")
for tag, n, c, v in rows:
    print(f"  {tag:8s} {n:6d} {c:14.4f} {v:17.4f}")
print("\n  pairs are matched on n and contamination; only coverage differs")
