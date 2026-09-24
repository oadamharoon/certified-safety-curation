"""Diff the pess-refresh rerun against the values it replaced.

The 38 learned-source xlab cells on CarGoal2/PointGoal1/PointGoal2 were produced in
July against a V-ensemble that was overwritten on 2026-08-16. Rerunning them on the
current ensemble makes Section 4's analysis and Section 5's method read the same
value function, at the cost of changing the numbers. This reports exactly what moved.

The line that matters is the safety verdict: a cell crossing the cost budget in
either direction changes a claim, while a cost shifting within the same verdict
usually only changes a cited figure.

Usage:  python runs/scripts/compare_pess_refresh.py
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

import json
import os

ROOT = CSC_WORKSPACE
BK = os.path.join(ROOT, "runs/archive/pess_refresh_pre_rerun_2026-08-27")
JOBS = os.path.join(ROOT, "runs/logs/reverify/jobs_pess_refresh.txt")
OUT = os.path.join(ROOT, "vlm-with-cpl/new_data/outputs")


def main():
    rows, flips, missing = [], [], []
    for line in open(JOBS):
        parts = line.split()
        if len(parts) != 4:
            continue
        task, _arm, tag, _seed = parts
        old_p = os.path.join(BK, f"{task}__{tag}.json")
        new_p = os.path.join(OUT, task, f"eval_results_{tag}.json")
        if not (os.path.exists(old_p) and os.path.exists(new_p)):
            missing.append((task, tag))
            continue
        o, n = json.load(open(old_p)), json.load(open(new_p))
        lim = n.get("cost_limit", o.get("cost_limit"))
        o_safe, n_safe = o["avg_cost"] <= lim, n["avg_cost"] <= lim
        rows.append((task, tag, o["avg_reward"], n["avg_reward"],
                     o["avg_cost"], n["avg_cost"], lim, o_safe, n_safe))
        if o_safe != n_safe:
            flips.append(rows[-1])

    print(f"compared {len(rows)} of 38 cells"
          + (f"; {len(missing)} not yet written" if missing else ""))
    print(f"\n{'task':<17}{'tag':<32}{'R old->new':>22}{'C old->new':>22}  verdict")
    for t, tag, oR, nR, oC, nC, lim, os_, ns_ in rows:
        v = ("safe" if os_ else "UNSAFE") + " -> " + ("safe" if ns_ else "UNSAFE")
        mark = "  <<< FLIP" if os_ != ns_ else ""
        print(f"{t:<17}{tag:<32}{oR:>10.3f}->{nR:<10.3f}{oC:>10.2f}->{nC:<10.2f}  "
              f"{v}{mark}")

    print(f"\nsafety-verdict flips: {len(flips)}")
    for t, tag, _oR, _nR, oC, nC, lim, os_, ns_ in flips:
        print(f"  {t} {tag}: cost {oC:.2f} -> {nC:.2f} against budget {lim}")

    if rows:
        dC = [abs(r[5] - r[4]) for r in rows]
        print(f"\ncost |delta|: median {sorted(dC)[len(dC)//2]:.2f}, max {max(dC):.2f}")
    if missing:
        print(f"\nnot yet written ({len(missing)}): "
              + ", ".join(f"{t}/{g}" for t, g in missing[:6])
              + (" ..." if len(missing) > 6 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
