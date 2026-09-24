"""Probe 22 analysis: which selection statistic predicts clone cost beyond contamination?

Task-demeaned OLS with leave-one-task-out (LOTO) R^2, same protocol as the paper's
fig:contamcost analysis, so numbers are comparable to its 0.38 / 0.50.
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

import json, sys
import numpy as np
from scipy.stats import spearmanr

W = CSC_WORKSPACE
rows = json.load(open(f"{W}/runs/probe/distributional.json"))
FEATS = ["contam", "mean_cost_norm", "unsafe_excess", "safe_margin", "ret_z", "kept_frac",
         "cov_ratio", "act_multimodal", "act_within_safe", "sep_auc_lr", "sep_auc_knn",
         "act_conflict", "unsafe_isolation", "mean_margin", "p10_margin"]
tasks = np.array([r["task"] for r in rows])
y_raw = np.array([r["cost_norm"] for r in rows])
LOG = "--log" in sys.argv
y_raw = np.log1p(y_raw) if LOG else y_raw


def col(f):
    v = np.array([r.get(f, np.nan) for r in rows], dtype=float)
    return v


def demean(v):
    out = v.copy()
    for t in np.unique(tasks):
        m = tasks == t
        out[m] -= np.nanmean(v[m])
    return out


def fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def r2_in(X, y):
    b = fit(X, y)
    X1 = np.column_stack([np.ones(len(y)), X])
    return 1 - ((y - X1 @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_loto(X, y):
    sse = 0.0
    for t in np.unique(tasks):
        tr, te = tasks != t, tasks == t
        b = fit(X[tr], y[tr])
        sse += ((y[te] - np.column_stack([np.ones(te.sum()), X[te]]) @ b) ** 2).sum()
    return 1 - sse / ((y - y.mean()) ** 2).sum()


y = demean(y_raw)
F = {f: demean(col(f)) for f in FEATS}
# rows with every feature present (drops all-safe selections, which have no unsafe states)
full = np.all([~np.isnan(F[f]) for f in FEATS], axis=0)
print(f"rows: {len(rows)}   with all features: {full.sum()}   target: "
      f"{'log1p(cost/limit)' if LOG else 'cost/limit'}, task-demeaned\n")

print("=== single feature (rows with that feature present) ===")
print(f"  {'feature':18s} {'n':>4s} {'rho':>7s} {'R2 in':>7s} {'R2 LOTO':>8s}")
for f in FEATS:
    m = ~np.isnan(F[f])
    X = F[f][m].reshape(-1, 1)
    rho = spearmanr(F[f][m], y[m]).statistic
    # explicit LOTO on the subset
    sse = 0.0; ys = y[m]; ts = tasks[m]
    for t in np.unique(ts):
        tr, te = ts != t, ts == t
        b = fit(X[tr], ys[tr])
        sse += ((ys[te] - np.column_stack([np.ones(te.sum()), X[te]]) @ b) ** 2).sum()
    print(f"  {f:18s} {m.sum():4d} {rho:+7.2f} {r2_in(X, ys):7.3f} "
          f"{1 - sse / ((ys - ys.mean()) ** 2).sum():8.3f}")

print("\n=== incremental over contamination (rows with all features) ===")
yc = y[full]; tc = tasks[full]


def loto_sub(X, yy, tt):
    sse = 0.0
    for t in np.unique(tt):
        tr, te = tt != t, tt == t
        b = fit(X[tr], yy[tr])
        sse += ((yy[te] - np.column_stack([np.ones(te.sum()), X[te]]) @ b) ** 2).sum()
    return 1 - sse / ((yy - yy.mean()) ** 2).sum()


Xc = F["contam"][full].reshape(-1, 1)
base = loto_sub(Xc, yc, tc)
print(f"  contam alone: R2 in {r2_in(Xc, yc):.3f}   LOTO {base:.3f}")
print(f"  {'+ feature':18s} {'R2 in':>7s} {'R2 LOTO':>8s} {'dLOTO':>7s}")
res = []
for f in FEATS:
    if f == "contam":
        continue
    X = np.column_stack([Xc, F[f][full]])
    l = loto_sub(X, yc, tc)
    res.append((f, r2_in(X, yc), l, l - base))
for f, a, l, d in sorted(res, key=lambda t: -t[3]):
    print(f"  {f:18s} {a:7.3f} {l:8.3f} {d:+7.3f}")

Xm = np.column_stack([Xc, F["mean_margin"][full], F["p10_margin"][full]])
print(f"\n  paper's contam+margins: R2 in {r2_in(Xm, yc):.3f}   LOTO {loto_sub(Xm, yc, tc):.3f}")
best = sorted(res, key=lambda t: -t[3])[0][0]
Xb = np.column_stack([Xm, F[best][full]])
print(f"  contam+margins+{best}: R2 in {r2_in(Xb, yc):.3f}   LOTO {loto_sub(Xb, yc, tc):.3f}")

print("\n=== per-task within-task Spearman(feature, cost), 14 selections each ===")
show = ["contam", "mean_cost_norm", "ret_z", "cov_ratio", "act_conflict", "sep_auc_knn",
        "unsafe_isolation", "act_multimodal", "mean_margin"]
print(f"  {'task':22s}" + "".join(f"{s[:10]:>11s}" for s in show))
for t in np.unique(tasks):
    m = tasks == t
    out = []
    for f in show:
        v = col(f)[m]; ok = ~np.isnan(v)
        out.append(spearmanr(v[ok], y_raw[m][ok]).statistic if ok.sum() >= 5 else np.nan)
    print(f"  {t:22s}" + "".join(f"{v:+11.2f}" for v in out))
