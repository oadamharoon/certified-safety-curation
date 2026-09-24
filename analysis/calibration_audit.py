"""The accepted hypergeometric test behind each deployed certificate.

App. hyper quotes, for the tasks that certify at alpha = 0.25, the number m of calibration
trajectories landing above the chosen threshold and the p-value the accepted test returned.
04q_calibrated_vfilter PRINTS that audit trail per grid quantile but stores none of it in
calfilt_meta_*.json, so the numbers existed only in run logs and had no producing expression.
This parses the trail out of the logs and records the accepted step -- the LAST quantile whose
exact test rejected, which is where the fixed-sequence walk stops.

Writes iclr2027/data/calibration_audit.json
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

import glob, json, os, re, sys


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
OUT = f"{W}/datasets/outputs"
LOGS = f"{W}/runs/logs"
DELTA = 0.1

certified = []
for p in sorted(glob.glob(f"{OUT}/*/calfilt_meta_calfilt_csf_seed*.json")):
    task = p.split("/")[-2]
    seed = int(re.search(r"seed(\d)", p).group(1))
    if json.load(open(p)).get("certified"):
        certified.append((task, seed))

out, missing = {}, []
for task, seed in certified:
    cands = [f for f in glob.glob(f"{LOGS}/**/*.log", recursive=True)
             if re.search(rf"{re.escape(task)}.*calfilt_csf.*seed{seed}\.log$", f)]
    cands = [f for f in cands if "[ltt]" in open(f, errors="ignore").read()]
    if not cands:
        missing.append(f"{task}/s{seed}")
        continue
    log = max(cands, key=os.path.getmtime)          # the most recent run of that cell
    txt = open(log, errors="ignore").read()
    blk = txt[txt.index("[ltt]"):]
    steps = [{"quantile": float(q), "m": int(m), "k": int(k), "p": float(pv)}
             for q, m, k, pv in re.findall(r"q=([\d.]+) m=\s*(\d+) k=\s*(\d+) p=([\d.]+)", blk)]
    acc = [s for s in steps if s["p"] <= DELTA]
    if not acc:
        missing.append(f"{task}/s{seed}: no rejecting step")
        continue
    # the fixed sequence stops at the first failure, so the accepted step is the last
    # rejecting one before that failure, not the smallest p anywhere on the grid
    accepted = None
    for s in steps:
        if s["p"] <= DELTA:
            accepted = s
        else:
            break
    out.setdefault(task, {})[str(seed)] = {
        "log": os.path.relpath(log, W), "accepted": accepted, "grid": steps}
    print(f"  {task:22s} s{seed}  accepted q={accepted['quantile']:.2f} "
          f"m={accepted['m']:3d} k={accepted['k']:2d} p={accepted['p']:.4f}")

if missing:
    print(f"  MISSING: {missing}", file=sys.stderr)
dst = f"{W}/iclr2027/data/calibration_audit.json"
_write_if_changed(dst, json.dumps(out, indent=1))
print(f"\n  wrote {dst} ({sum(len(v) for v in out.values())} certified cells)")
