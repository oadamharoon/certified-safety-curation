"""F2 dedupe (V2_REMEDIATION): where a task's deployed certified selection is the same kept set as
one of its three distinct grid selections (runs/selections/v2_summary.json), the deployed CDT/CPL
cells are not trained twice; the result keys of the deployed arm alias the identical grid cell.
Returns {task: {deployed_key: grid_key}} for the CDT (harvest_osrl) and CPL (collect_results) keys.
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

import json, os
SEL = CSC_RUNS + "/selections"


def aliases():
    S = json.load(open(f"{SEL}/v2_summary.json")); out = {}
    for task, rec in S.items():
        for a, dep_cdt, dep_cpl, grid_cdt, grid_cpl in (("a25", "cdt_cert", "cpl_gt_cert", "cdt_a25new_q{q}", "cpl_gt_a25q{q}"),
                                                        ("a40", "cdt_a40", "cpl_gt_a40", "cdt_a40new_q{q}", "cpl_gt_a40new_q{q}")):
            e = rec.get(a)
            if not e: continue
            dep = set(json.load(open(f"{SEL}/{e['deployed']['name']}_kept.json"))["kept"])
            for d in e["distinct"]:
                if set(json.load(open(f"{SEL}/{d['name']}_kept.json"))["kept"]) == dep:
                    q = int(round(d["q"] * 100)); out.setdefault(task, {})[dep_cdt] = grid_cdt.format(q=q); out[task][dep_cpl] = grid_cpl.format(q=q)
    return out


def duplicate_jobs():
    """(task, deployed selection name, result tag) triples whose F2 cells are skipped."""
    S = json.load(open(f"{SEL}/v2_summary.json")); A = aliases(); out = []
    for task, al in A.items():
        if "cdt_cert" in al: out.append((task, S[task]["a25"]["deployed"]["name"], "cplgt_cert"))
        if "cdt_a40" in al: out.append((task, S[task]["a40"]["deployed"]["name"], "cplgt_a40"))
    return out


if __name__ == "__main__":
    for t, al in aliases().items(): print(t, al)
    print(duplicate_jobs())
