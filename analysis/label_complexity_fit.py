"""Fit n50 vs margin across the combined sweep (n in 50..3200)."""

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

import json
import numpy as np
G = json.load(open(CSC_PAPER_DATA + "/review_response/guarantee_stats_2000.json"))
E = json.load(open(CSC_PAPER_DATA + "/review_response/label_complexity_ext.json"))
M = {r["task"]: r["margin"] for r in json.load(open(CSC_PAPER_DATA + "/review_response/margin_vs_yield.json"))}
TARGET = 0.5
rows = []
for task, m in M.items():
    curve = {}
    td = G.get(task, {})
    for s, sd in td.get("seeds", {}).items():
        for n, c in sd.items():
            curve.setdefault(int(n), []).append(c["cert_rate"])
    curve = {n: float(np.mean(v)) for n, v in curve.items()}
    curve.update({int(n): r for n, r in E.get(task, {}).get("rates", {}).items()})
    xs = sorted(curve)
    ys = [curve[n] for n in xs]
    n50 = None
    for i, (n, y) in enumerate(zip(xs, ys)):
        if y >= TARGET:
            if i == 0: n50 = float(n); break
            n0, y0 = xs[i-1], ys[i-1]
            f = (TARGET - y0) / (y - y0) if y != y0 else 0.0
            n50 = float(np.exp(np.log(n0) + f * (np.log(n) - np.log(n0)))); break
    rows.append((task, m, n50, curve))
rows.sort(key=lambda r: -r[1])
print(f"{'task':22s} {'margin':>8s} {'n50':>9s}")
for t, m, n, _ in rows:
    print(f"{t:22s} {m:8.3f} {('%.0f' % n) if n else '   never':>9s}")
fit = [(m, n) for _, m, n, _ in rows if n and m > 0]
x = np.log([m for m, _ in fit]); y = np.log([n for _, n in fit])
slope, b = np.polyfit(x, y, 1)
r2 = 1 - ((y - (slope*x+b))**2).sum() / ((y - y.mean())**2).sum()
print(f"\nfit on {len(fit)} tasks:  log n50 = {slope:.2f} log(margin) + {b:.2f}")
print(f"  slope {slope:.2f}  (theory -2; prereg band [-2.6, -1.4])   R^2 = {r2:.3f}")
print(f"  n50 ~ {np.exp(b):.2f} / margin^{-slope:.2f}")
json.dump({"slope": float(slope), "intercept": float(b), "r2": float(r2),
           "n50": {t: n for t, m, n, _ in rows}}, 
          open(CSC_PAPER_DATA + "/review_response/label_complexity.json","w"), indent=1)
