"""Find published results that predate the artifacts they were computed from.

General form of the alpha=0.40 grid bug: a V ensemble / label set / segment file
was regenerated AFTER the results depending on it were produced, so those
results cannot be reproduced from what is now on disk. Compares each
eval_results_*.json against the mtimes of its task's inputs.
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

import glob, json, os, datetime as dt
D = CSC_WORK
DEPS = ["v_ensemble_pess_seed0.pt", "gt_labels.json", "active_segments.pkl"]
def ts(p): return os.path.getmtime(p) if os.path.exists(p) else None
rows = []
for tdir in sorted(glob.glob(os.path.join(D, "outputs", "*"))):
    task = os.path.basename(tdir)
    dep = {d: ts(os.path.join(tdir, d)) for d in DEPS}
    newest = max([v for v in dep.values() if v] or [0])
    if not newest: continue
    for ev in sorted(glob.glob(os.path.join(tdir, "eval_results_*.json"))):
        arm = os.path.basename(ev)[len("eval_results_"):-len(".json")]
        e = ts(ev)
        if e and e < newest - 3600:            # result older than its inputs
            stale = [d for d, v in dep.items() if v and v > e + 3600]
            rows.append((task, arm, e, newest, stale))
print(f"  {len(rows)} result files predate their inputs\n")
bytask = {}
for task, arm, e, n, stale in rows:
    bytask.setdefault(task, []).append((arm, e, n, stale))
for task in sorted(bytask):
    items = bytask[task]
    newest = max(n for _, _, n, _ in items)
    deps = sorted({d for _, _, _, s in items for d in s})
    print(f"  {task}   inputs rebuilt {dt.datetime.fromtimestamp(newest):%Y-%m-%d}  ({', '.join(deps)})")
    arms = sorted({a for a, _, _, _ in items})
    oldest = min(e for _, e, _, _ in items)
    print(f"      {len(arms)} stale arms, oldest result {dt.datetime.fromtimestamp(oldest):%Y-%m-%d}")
    print(f"      {', '.join(arms[:14])}{' ...' if len(arms) > 14 else ''}")
