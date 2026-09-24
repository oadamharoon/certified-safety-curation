"""Provenance check for the paper-consumed arms that record no selection.

vfilt/vawr/xlab/cf/qfilt/bcsafeseg write a policy and an eval_results file but
no selection record, so their published numbers cannot be content-verified
without retraining. What IS checkable is provenance: a result must not predate
the artifacts it was computed from. labels_only does record a meta, so its
internal consistency is checked too.
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

import glob, json, os, re, sys, collections, datetime as dt
D = CSC_WORK + "/outputs"
ICLR = CSC_PAPER
pats = dict(re.findall(r'"([\w]+)":\s*r"(\^[^"]+)"', open(os.path.join(ICLR, "scripts", "collect_results.py")).read()))
used = set(json.load(open(sys.argv[1]))) if len(sys.argv) > 1 else None

def arm_of(tag):
    for k, p in pats.items():
        if re.match(p, tag):
            return k
    return None

# Per-arm dependency map. 04p loads the V ensemble ONLY when SCORE_MODE=vscore,
# so the random/return/bottom-return controls do not read it; 04b (bc_all,
# bcsafeseg) and 04c (cpl_gt) never reference VEnsemble at all. Flagging those
# against the ensemble's mtime produces false positives.
ENS = "v_ensemble_pess_seed0.pt"; LAB = "gt_labels.json"; SEG = "active_segments.pkl"
DEPS = {
    "vfilt_random": (), "vfilt_return": (), "vfilt_retbot": (),
    "bc_all": (), "bcsafeseg": (SEG,), "cpl_gt": (SEG, LAB),
    "vfilt_matchgt": (ENS,), "vfilt_q25": (ENS,), "vfilt_calsafe": (ENS,),
}
def deps_for(arm):
    if arm in DEPS:
        return DEPS[arm]
    if arm.startswith(("vfilt", "vawr", "xlab", "cf_", "qfilt", "labels_only",
                       "labels_split", "bc_a", "wbc", "mwbc", "toph")):
        return (ENS,)
    return None
FAMILIES = ("vfilt", "vawr", "xlab", "cf_", "qfilt", "bcsafeseg", "labels_only",
            "labels_split", "cpl_gt", "bc_", "wbc", "mwbc", "toph")
stale = collections.defaultdict(list)
ok = 0
for tdir in sorted(glob.glob(os.path.join(D, "*"))):
    task = os.path.basename(tdir)
    deps = {}
    for dep in ("v_ensemble_pess_seed0.pt", "gt_labels.json", "active_segments.pkl"):
        p = os.path.join(tdir, dep)
        if os.path.exists(p):
            deps[dep] = os.path.getmtime(p)
    if not deps:
        continue
    for ev in glob.glob(os.path.join(tdir, "eval_results_*.json")):
        tag = os.path.basename(ev)[len("eval_results_"):-len(".json")]
        arm = arm_of(tag)
        if arm is None or (used is not None and arm not in used):
            continue
        if not any(arm.startswith(f) for f in FAMILIES):
            continue
        et = os.path.getmtime(ev)
        want = deps_for(arm)
        if want is None:
            continue
        older = {d: v for d, v in deps.items() if d in want and v > et + 3600}
        if older:
            stale[arm].append((task, tag, sorted(older)))
        else:
            ok += 1
import json as _j
_rows=[{"arm":a,"task":t,"tag":tag} for a,rows in stale.items() for t,tag,_ in rows]
_j.dump(_rows, open(os.environ.get("STALE_OUT","/tmp/stale_cells.json"),"w"), indent=1)
print(f"  paper-consumed non-calfilt result files whose inputs are NOT newer : {ok}")
print(f"  arms with at least one result predating its inputs                 : {len(stale)}")
for arm, rows in sorted(stale.items(), key=lambda kv: -len(kv[1])):
    tasks = sorted({r[0] for r in rows})
    print(f"      {arm:<24}{len(rows):>4} cells   tasks: {', '.join(tasks[:5])}{' ...' if len(tasks)>5 else ''}")
# labels_only meta internal consistency
print("\n  labels_only meta internal consistency:")
bad = 0; tot = 0
for mp in glob.glob(os.path.join(D, "*", "labelsonly_meta_*.json")):
    try:
        m = json.load(open(mp))
    except Exception:
        continue
    tot += 1
    kf = m.get("kept_frac"); fr = m.get("frac")
    if kf is None or fr is None:
        continue
    if abs(kf - fr) > 0.02:
        bad += 1
        if bad <= 5:
            print(f"      {os.path.basename(mp)}: kept_frac={kf:.3f} vs requested frac={fr:.3f}")
print(f"      {tot} metas, {bad} where realized kept_frac departs from the requested fraction by >2 points")
