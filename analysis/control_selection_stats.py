"""Composition of the matched-size CONTROL selections, per task.

App. extended's selection-signal paragraph quotes the bottom-return control's kept-set mean
trajectory cost against the pool's ("on HalfCheetah, kept-set mean trajectory cost 39.9 against
the pool's 115.4") to make the point that genuinely low-cost training data still clones into an
unsafe policy. Nothing in the repo produced those two numbers, so they could not be rechecked
when the cohort moved. This does.

The selection rule is margin_expand.py's, unchanged: rank by episodic return, take the
ground-truth-matched fraction from the bottom (return selection takes it from the top), and
report the mean episodic COST of the trajectories kept. These controls read no value ensemble,
so they are cohort-independent by construction; the builder exists so the numbers are traceable.

Writes iclr2027/data/control_selection_stats.json
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
sys.path.insert(0, D)
os.chdir(D)
cfg = yaml.safe_load(open("config.yaml"))
TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
         "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2"]

out = {}
for task in TASKS:
    tc = cfg["tasks"][task]
    lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs])
    R = np.array([float(np.sum(t["rewards"])) for t in trajs])
    nsel = max(1, int(round(float((cost <= lim).mean()) * len(trajs))))
    order = np.argsort(R)[::-1]                      # descending return
    top, bot = order[:nsel], order[::-1][:nsel]
    out[task] = {
        "limit": lim, "n_pool": len(trajs), "n_selected": nsel,
        "pool_mean_traj_cost": float(cost.mean()),
        "retbot_kept_mean_traj_cost": float(cost[bot].mean()),
        "retbot_kept_contamination": float((cost[bot] > lim).mean()),
        "rettop_kept_mean_traj_cost": float(cost[top].mean()),
    }
    print(f"  {task:22s} pool {cost.mean():7.1f}  bottom-return kept {cost[bot].mean():7.1f}  "
          f"top-return kept {cost[top].mean():7.1f}  (n={nsel})", flush=True)

dst = f"{W}/iclr2027/data/control_selection_stats.json"
json.dump(out, open(dst, "w"), indent=1)
print(f"\n  wrote {dst}")
