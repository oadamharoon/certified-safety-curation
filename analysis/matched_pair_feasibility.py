"""Probe 5: is return-matched preference sampling IDENTIFIABLE?

Mechanism established by probes 3-4: pairs are sampled on cost contrast
(cost_contrast_min = 1.0), and where segment cost and return are correlated
such pairs also contrast return, so a state-only Bradley-Terry model can fit
the preference with return-predictive features. The proposed fix is to match
pairs on return.

But matching trades confounding for variance: conditioning on return removes
the confound AND removes identifying variation. If cost were a deterministic
function of return, no return-matched pair could contrast cost at all, and the
cost direction would be unidentified. So the question that gates the whole
follow-up is measurable BEFORE training anything:

  within a narrow return band, do cost-contrasting pairs still exist, and how
  many, relative to the 1000-pair budget?

This computes exactly that: bin segments by return decile, and inside each bin
count pairs meeting the SAME cost_contrast_min the deployed sampler uses.

Exploratory only; writes to runs/probe/.
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
from scipy.stats import spearmanr

D = CSC_WORK
OUT = CSC_RUNS + "/probe"
os.chdir(D)
cfg = yaml.safe_load(open("config.yaml"))
CMIN = float(cfg.get("cost_contrast_min", 1.0))
BUDGET = int(cfg.get("num_pairs", 1000))
NBIN, RNG, NSAMP = 10, np.random.default_rng(0), 400_000

TASKS = ["halfcheetah_velocity", "swimmer_velocity", "walker2d_velocity",
         "ant_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2", "pointcircle1"]
res = {}
for task in TASKS:
    f = f"outputs/{task}/active_segments.pkl"
    if not os.path.exists(f):
        continue
    seg = pickle.load(open(f, "rb"))
    cost = np.array([s["total_cost"] for s in seg], float)
    ret  = np.array([s["total_reward"] for s in seg], float)
    rho = spearmanr(cost, ret).statistic
    # unrestricted sampler: random pairs meeting the cost contrast
    i, j = RNG.integers(0, len(seg), NSAMP), RNG.integers(0, len(seg), NSAMP)
    ok = i != j
    i, j = i[ok], j[ok]
    dc, dr = np.abs(cost[i] - cost[j]), np.abs(ret[i] - ret[j])
    contrast = dc >= CMIN
    # return-matched: |delta return| within the 10th percentile of all |delta return|
    band = np.percentile(dr, 10)
    matched = dr <= band
    both = contrast & matched
    # how much cost variation survives inside a return decile
    edges = np.quantile(ret, np.linspace(0, 1, NBIN + 1)); edges[-1] += 1e-9
    frac_bins_ok, spans = 0, []
    for b in range(NBIN):
        m = (ret >= edges[b]) & (ret < edges[b + 1])
        if m.sum() < 2: continue
        span = cost[m].max() - cost[m].min()
        spans.append(span)
        if span >= CMIN: frac_bins_ok += 1
    res[task] = {
        "rho_segcost_segreturn": float(rho),
        "frac_pairs_cost_contrasting": float(contrast.mean()),
        "frac_pairs_matched_and_contrasting": float(both.mean()),
        "yield_ratio": float(both.mean() / max(contrast.mean(), 1e-12)),
        "return_deciles_with_cost_span": frac_bins_ok,
        "median_within_bin_cost_span": float(np.median(spans)) if spans else 0.0,
        "est_matched_pairs_available": int(both.mean() * len(seg) * (len(seg) - 1) / 2),
    }
    r = res[task]
    print(f"  {task:22s} rho(cost,ret)={rho:+.3f}  deciles with cost span>={CMIN:.0f}: "
          f"{frac_bins_ok}/{NBIN}  matched&contrasting yield {r['yield_ratio']*100:5.1f}% "
          f"(~{r['est_matched_pairs_available']:,} pairs vs budget {BUDGET})")
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "matched_pair_feasibility.json"), "w"), indent=2)
print(f"\n  wrote {OUT}/matched_pair_feasibility.json")
