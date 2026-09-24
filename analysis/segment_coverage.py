"""How much of each pool the 1000 training pairs actually constrain.

The Bradley-Terry loss sees only the states inside the segments it compares, so the
identification deficit of Section 5 is quantifiable: count the distinct (trajectory,
timestep) slots the 2000 drawn segments cover, against every state in the pool.

Supersedes the version that lived inside free_tier.py, which counted over ALL active
segments rather than the sampled pairs and looked up the trajectory under the keys
"traj_idx"/"traj_index". Segments carry "traj_id", so that lookup returned the -1
default for every segment, collapsing all trajectories onto one and reporting 1000
covered states for every task regardless of pool size.

Writes iclr2027/data/review_response/segment_coverage.json
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

import json
import os
import pickle

import yaml

W = os.environ.get("CSC_WORKSPACE", CSC_WORKSPACE)
D = os.path.join(W, "datasets")
OUT = os.path.join(W, "iclr2027", "data", "review_response", "segment_coverage.json")
TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
         "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2"]

cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
out = {}
for task in TASKS:
    segs = pickle.load(open(os.path.join(D, "outputs", task, "active_segments.pkl"), "rb"))
    pairs = json.load(open(os.path.join(D, "outputs", task, "gt_labels.json")))
    trajs = pickle.load(open(os.path.join(D, cfg["tasks"][task]["data_pickle"]), "rb"))
    total_states = sum(len(t["observations"]) for t in trajs)

    slots, n_slots = set(), 0
    for p in pairs:
        for key in ("seg_A_idx", "seg_B_idx"):
            s = segs[p[key]]
            start, end = int(s["start"]), int(s["end"])
            n_slots += end - start
            slots.update((int(s["traj_id"]), u) for u in range(start, end))

    covered = len(slots)
    out[task] = {
        "n_pairs": len(pairs),
        "n_segments_drawn": 2 * len(pairs),
        "slots_drawn": n_slots,                       # with multiplicity
        "states_covered": covered,                    # distinct
        "total_pool_states": total_states,
        "coverage_frac": covered / total_states,
        "overlap_frac": 1 - covered / n_slots,        # share of drawn slots that repeat
        "mean_multiplicity": n_slots / covered,       # segment sums per covered state
    }
    print(f"{task:24s} covered={covered:7d}/{total_states:9d} "
          f"({100 * covered / total_states:5.2f}%)  overlap={100 * out[task]['overlap_frac']:5.2f}%  "
          f"mult={out[task]['mean_multiplicity']:.2f}", flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("wrote", os.path.relpath(OUT, W))
