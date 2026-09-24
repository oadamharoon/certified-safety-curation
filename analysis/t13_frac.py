"""Print the calsafe fraction for (task, seed): CP-lower(0.1) on safe mass from a fresh 200-draw."""

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

import os, pickle, sys
import numpy as np
from scipy.stats import beta
import yaml
task, seed = sys.argv[1], int(sys.argv[2])
os.chdir(CSC_WORK)
cfg = yaml.safe_load(open("config.yaml"))
lim = 20 if "velocity" in task else (10 if task.endswith("_b") else 25)
trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
costs = np.array([float(np.sum(t["costs"])) for t in trajs])
rng = np.random.default_rng(2000 + seed)
cal = rng.choice(len(costs), 200, replace=False)
k = int((costs[cal] <= lim).sum())
lo = float(beta.ppf(0.1, k, 200 - k + 1)) if k > 0 else 0.0
frac = max(lo, 50 / len(costs))
print(f"{frac:.4f}")
