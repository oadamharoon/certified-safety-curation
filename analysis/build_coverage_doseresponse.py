"""Probe 11: does recovered COVERAGE cause policy utility, or is it just more data?

Probe 10 left this unsettled: one clean positive, one null, an inverted
dose-response, and a design whose "only coverage differs" claim was too strong
(mix and thr were different trajectory sets differing in more than coverage).

This fixes all three. Core and periphery are defined from ONE score vector by
quantile, so nesting is exact. At each coverage level three arms are built with
IDENTICAL size and IDENTICAL unsafe count:

  mix   core + a random f-fraction of the periphery        (carries high-return safe)
  swap  core + an f-fraction drawn ONLY from periphery     (THE CONTROL: same size,
        items that are NOT high-return-safe                 same contamination, same
                                                            population, zero coverage)
  thr   the plain score-threshold selection, stratified-    (reference: what alpha
        subsampled to the same size and unsafe count         alone can reach)

mix vs swap is the causal test. If mix > swap, coverage specifically matters. If
mix == swap, the effect is "admitting periphery at all" and the coverage thread dies.
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
ICLR = CSC_PAPER
OUT = CSC_RUNS + "/selections_recon"
sys.path.insert(0, D); os.chdir(D)
cfg = yaml.safe_load(open("config.yaml"))
LEVELS = (0.05, 0.15, 0.30, 0.50)
Q_CORE, Q_POOL = 0.85, 0.30
TASKS = {"pointgoal1_dsrl": 25, "cargoal1_dsrl": 25}

def build(task, lim):
    tr = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in tr])
    ret = np.array([float(np.sum(t["rewards"])) for t in tr])
    unsafe = cost > lim
    sc = np.load(os.path.join(ICLR, "data", "e_scores", f"{task}_seed0.npz"))["scores"]
    safe = np.where(~unsafe)[0]
    hi = set(safe[ret[safe] >= np.percentile(ret[safe], 90)].tolist())
    core = set(np.where(sc >= np.quantile(sc, Q_CORE))[0].tolist())
    pool = set(np.where(sc >= np.quantile(sc, Q_POOL))[0].tolist())
    peri = np.array(sorted(pool - core))
    rng = np.random.default_rng(0)
    rows = []
    for f in LEVELS:
        k = int(round(f * len(peri)))
        mix_p = rng.choice(peri, k, replace=False)
        n = len(core) + k
        n_uns = int(unsafe[np.r_[sorted(core), mix_p]].sum())
        # SWAP: same size, same unsafe count, drawn only from non-high-return-safe periphery
        cand = np.array([i for i in peri if i not in hi])
        cu, cs = cand[unsafe[cand]], cand[~unsafe[cand]]
        n_u_p = int(unsafe[mix_p].sum()); n_s_p = k - n_u_p
        if len(cu) < n_u_p or len(cs) < n_s_p:
            print(f"    {task} f={f}: infeasible swap"); continue
        swap_p = np.r_[rng.choice(cu, n_u_p, replace=False), rng.choice(cs, n_s_p, replace=False)]
        # THR: plain threshold, stratified-subsampled to same n and unsafe count
        best = None
        for q in np.arange(0.95, 0.25, -0.005):
            idx = np.where(sc >= np.quantile(sc, q))[0]
            u, s = idx[unsafe[idx]], idx[~unsafe[idx]]
            if len(u) < n_uns or len(s) < n - n_uns: continue
            c = abs(float(unsafe[idx].mean()) - n_uns / n)
            if best is None or c < best[1]: best = (idx, c)
        idx, _ = best
        u, s = idx[unsafe[idx]], idx[~unsafe[idx]]
        thr = np.r_[rng.choice(u, n_uns, replace=False), rng.choice(s, n - n_uns, replace=False)]
        for tag, sel in (("mix", np.r_[sorted(core), mix_p]),
                         ("swap", np.r_[sorted(core), swap_p]), ("thr", thr)):
            sel = np.array(sorted(set(int(x) for x in sel)))
            cov = float(np.mean([i in hi for i in sel]))
            name = f"{task}_cov{int(f*100):02d}_{tag}"
            json.dump({"kept": [int(i) for i in sel], "n_kept": len(sel),
                       "contamination": float(unsafe[sel].mean()),
                       "hi_return_safe_frac": cov, "level": f, "arm": tag},
                      open(os.path.join(OUT, name + "_kept.json"), "w"))
            rows.append((name, len(sel), float(unsafe[sel].mean()), cov))
    return rows

os.makedirs(OUT, exist_ok=True)
for task, lim in TASKS.items():
    print(f"  === {task} ===")
    print(f"  {'selection':34s} {'n':>5s} {'contam':>8s} {'coverage':>9s}")
    for name, n, c, v in build(task, lim):
        print(f"  {name:34s} {n:5d} {c:8.4f} {v:9.4f}")
