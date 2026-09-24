"""Realized contamination of every selection the operator grid runs on.

make_tables' operator table took its $\\hat\\alpha$ column from profile_by_H's H30
u_profile_mean, which is the contamination at that quantile AVERAGED OVER THE THREE VALUE
ENSEMBLE SEEDS -- not the realized contamination of the selection the operator was actually
trained on. The composability grids report the latter, so the two tables carried different
numbers for the same selections (Walker2d 0.24/0.24/0.27 against 0.139/0.177/0.248) while
the prose called them the same selections. This computes the realized contamination of each
operator selection from its own membership file, which is the definition the rest of the
paper uses.

Writes iclr2027/data/operator_alpha_hat.json  {"<task>": {"<sel>": {"n", "contamination"}}}
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
import numpy as np
import yaml

D = CSC_WORK
W = CSC_WORKSPACE
SEL = f"{W}/runs/selections"
sys.path.insert(0, D)
os.chdir(D)
cfg = yaml.safe_load(open("config.yaml"))

# (task, selection tag as make_tables' _OPSEL names it, membership file basename)
SELECTIONS = [
    ("halfcheetah_velocity", "q85", "halfcheetah_velocity_a25new_q85"),
    ("halfcheetah_velocity", "q80", "halfcheetah_velocity_a25new_q80"),
    ("halfcheetah_velocity", "q75", "halfcheetah_velocity_a25new_q75"),
    ("walker2d_velocity", "q75", "walker2d_velocity_a25new_q75"),
    ("walker2d_velocity", "q70", "walker2d_velocity_a25new_q70"),
    ("walker2d_velocity", "q65", "walker2d_velocity_a25new_q65"),
    ("cargoal1_dsrl", "q85", "cargoal1_dsrl_a25new_q85"),
    ("cargoal1_dsrl", "q80", "cargoal1_dsrl_a25new_q80"),
    ("cargoal1_dsrl", "q75", "cargoal1_dsrl_a25new_q75"),
    # PointGoal1 is outside the F2 cohort; its operator sweep predates the composability
    # grid's triple and was run on q85/q70/q65, of which q85 is shared. No operator run
    # exists at q55, the composability grid's third selection.
    ("pointgoal1_dsrl", "q85", "pointgoal1_dsrl_a25_q85"),
    ("pointgoal1_dsrl", "q70", "pointgoal1_dsrl_a25_q70"),
    ("pointgoal1_dsrl", "q65", "pointgoal1_dsrl_a25_q65"),
    ("carrun_b", "echonew", "carrun_b_echonew_q85"),
]

out, cache = {}, {}
for task, tag, base in SELECTIONS:
    p = f"{SEL}/{base}_kept.json"
    if not os.path.exists(p):
        print(f"  MISSING {p}", file=sys.stderr)
        continue
    if task not in cache:
        tc = cfg["tasks"][task]
        trajs = pickle.load(open(tc["data_pickle"], "rb"))
        cache[task] = (np.array([float(np.sum(t["costs"])) for t in trajs]),
                       tc.get("cost_limit", cfg["cost_limit"]))
    cost, lim = cache[task]
    kept = json.load(open(p))["kept"]
    c = float((cost[kept] > lim).mean())
    out.setdefault(task, {})[tag] = {"selection": base, "n": len(kept), "contamination": c}
    print(f"  {task:22s} {tag:8s} n={len(kept):5d}  ahat={c:.4f}  ({base})")

dst = f"{W}/iclr2027/data/operator_alpha_hat.json"
json.dump(out, open(dst, "w"), indent=1)
print(f"\n  wrote {dst}")
