"""Build data/h_seed_failures.json: the deployed calibrated filter's selection statistics
and 2000-episode cost for every (task, seed) cell of the main suite.

App. extended's seed-failure paragraph quotes this file (75 cells, the task-demeaned
Spearman of 2000-episode cost against selection size, kept fraction and contamination),
but the file had NO builder in the repo, so it could not be re-derived when F0 retrained
the six tasks' value ensembles. This script is that builder. It joins two per-run
artifacts and computes nothing new:
  outputs/<task>/calfilt_meta_calfilt_csf_seed<s>.json      (tau, n_kept, kept_frac,
                                                             kept_unsafe_rate, certified)
  outputs/<task>/eval_results_certn2k_calfilt_csf_seed<s>.json  (avg_cost over 2000 episodes)
Because both come from the runs themselves, the cohort of the output is the cohort of the
runs by construction.

--check compares against the installed file instead of overwriting it: the nine tasks F0
did not touch must reproduce exactly.
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

import json, os, sys


def _write_if_changed(path, text):
    """Write only when the content differs, so an identical regeneration keeps its mtime.

    The completeness gate's R11 re-runs these builders on every invocation to prove they
    reproduce. An unconditional write bumps the artifact's mtime, which R9 and R14 then read as
    "the table/figure is older than its input" -- the freshness rules would eat each other.
    """
    import os as _os
    if _os.path.exists(path) and open(path).read() == text:
        return False
    with open(path, "w") as _fh:
        _fh.write(text)
    return True


W = CSC_WORKSPACE
OUT = f"{W}/vlm-with-cpl/new_data/outputs"
DST = f"{W}/iclr2027/data/h_seed_failures.json"
TASKS = ("halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity "
         "swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2 "
         "pointbutton1 pointbutton2 carbutton1_t3 carbutton2 pointcircle1 "
         "pointcircle2").split()
SEEDS = range(5)

rows, missing = [], []
for task in TASKS:
    for s in SEEDS:
        mp = f"{OUT}/{task}/calfilt_meta_calfilt_csf_seed{s}.json"
        ep = f"{OUT}/{task}/eval_results_certn2k_calfilt_csf_seed{s}.json"
        if not (os.path.exists(mp) and os.path.exists(ep)):
            missing.append(f"{task}/s{s}")
            continue
        m, e = json.load(open(mp)), json.load(open(ep))
        rows.append({"task": task, "seed": s,
                     "cost2k": float(e["avg_cost"]),
                     "budget": float(e["cost_limit"]),
                     "kept_frac": float(m["kept_frac"]),
                     "kept_unsafe": float(m["kept_unsafe_rate"]),
                     "n_kept": int(m["n_kept"]),
                     "certified": bool(m["certified"]),
                     "tau": float(m["tau"])})
if missing:
    print(f"MISSING {len(missing)} cell(s): {missing}", file=sys.stderr)

if "--check" in sys.argv:
    _o = json.load(open(DST))
    old = {(r["task"], r["seed"]): r for r in (_o["cells"] if isinstance(_o, dict) else _o)}
    new = {(r["task"], r["seed"]): r for r in rows}
    same = [k for k in old if k in new and old[k] == new[k]]
    diff = [k for k in old if k in new and old[k] != new[k]]
    print(f"cells: old {len(old)}, new {len(new)}, identical {len(same)}, changed {len(diff)}")
    for k in sorted(diff):
        print(f"  CHANGED {k[0]} s{k[1]}: cost2k {old[k]['cost2k']} -> {new[k]['cost2k']}, "
              f"kept_frac {old[k]['kept_frac']:.4f} -> {new[k]['kept_frac']:.4f}, "
              f"certified {old[k]['certified']} -> {new[k]['certified']}")
    for k in sorted(set(old) ^ set(new)):
        print(f"  ONLY IN {'old' if k in old else 'new'}: {k}")
    sys.exit(0)

# The task-demeaned rank correlations App. extended quotes. They were computed by hand
# before, which is how they could go stale silently; computing them here binds each to an
# expression and to the cells above.
import numpy as np
from scipy.stats import spearmanr


def demean(field):
    v = np.array([r[field] for r in rows], dtype=float)
    out = v.copy()
    for t in {r["task"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["task"] == t]
        out[idx] -= v[idx].mean()
    return out


y = demean("cost2k")
stats = {}
for name, field in (("n_kept", "n_kept"), ("kept_frac", "kept_frac"),
                    ("contamination", "kept_unsafe")):
    rho, pv = spearmanr(demean(field), y)
    stats[name] = {"spearman_demeaned": float(rho), "p": float(pv)}

payload = {"n_cells": len(rows),
           "n_certified": int(sum(r["certified"] for r in rows)),
           "demeaned_spearman_vs_cost2k": stats,
           "cells": rows}
_write_if_changed(DST, json.dumps(payload, indent=1))
print(f"h_seed_failures: {len(rows)} cells, {payload['n_certified']} certified -> {DST}")
for k, v in stats.items():
    print(f"  demeaned Spearman(cost2k, {k:13s}) = {v['spearman_demeaned']:+.3f}  (p={v['p']:.3g})")
