"""Guard against the failure modes that have silently corrupted this snapshot.

Every bug found so far was silent: the harvest printed a healthy summary while
results were being dropped or duplicated. Each check below corresponds to one
that actually happened.

  A  dead pattern      a PATTERNS regex matching zero files on disk
                       (the vawr patterns were written r"(\\d+)$", so they
                       hunted for a literal backslash-d and matched nothing)
  B  orphan result     an eval_results file on disk matching no pattern, so it
                       never reaches the snapshot
  C  degenerate seeds  every seed of an arm identical, meaning the training
                       script ignored SEED_OVERRIDE and the "seeds" are copies
  D  short arm         fewer seeds than the protocol requires (3 for analysis
                       sweeps, 5 for headline arms)

Exit code is nonzero if anything fails, so this can gate a harvest.
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

import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = CSC_WORK
HEADLINE = {"calfilt_csf", "bc_all", "bc_safe", "vfilt"}


def load_patterns():
    src = open(os.path.join(BASE, "scripts", "collect_results.py")).read()
    body = src.split("PATTERNS", 1)[1]
    body = body[body.index("{"): body.index("\n}") + 2]
    return eval(body)


def main():
    cfgs = load_patterns()
    files = [os.path.basename(f)[len("eval_results_"):-len(".json")]
             for f in glob.glob(os.path.join(REPO, "outputs/*/eval_results_*.json"))]
    fails = []

    # A: patterns that match nothing on disk
    dead = [k for k, pat in cfgs.items() if not any(re.match(pat, n) for n in files)]
    print(f"A  dead patterns          : {len(dead)}")
    for k in sorted(dead):
        print(f"     {k}  ->  {cfgs[k]}")
    if dead:
        fails.append(f"{len(dead)} regex patterns match no file")

    # B: results on disk that no pattern claims
    orphans = sorted({n for n in files
                      if not any(re.match(p, n) for p in cfgs.values())})
    print(f"B  orphan result files    : {len(orphans)}")
    for n in orphans[:12]:
        print(f"     {n}")
    if len(orphans) > 12:
        print(f"     ... and {len(orphans)-12} more")

    snap = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))

    # C: arms whose seeds are all identical
    # Require BOTH cost and reward to tie. A cost tie alone is common and
    # genuine: a policy that never violates scores exactly 0.000 on every seed
    # while its reward still varies. Only a joint tie means copied runs.
    degen = []
    for task, arms in snap.items():
        for arm, seeds in arms.items():
            cs = [v.get("C") for v in seeds.values() if v.get("C") is not None]
            rs = [v.get("R") for v in seeds.values() if v.get("R") is not None]
            if (len(cs) > 1 and max(cs) - min(cs) < 1e-9
                    and len(rs) > 1 and max(rs) - min(rs) < 1e-9):
                degen.append((task, arm, len(cs)))
    print(f"C  degenerate-seed arms   : {len(degen)}")
    for t, a, n in degen[:12]:
        print(f"     {t}/{a} ({n} identical seeds)")
    if len(degen) > 12:
        print(f"     ... and {len(degen)-12} more")
    if degen:
        fails.append(f"{len(degen)} arms have identical results across seeds")

    # D: arms below the protocol's seed count
    short = []
    for task, arms in snap.items():
        for arm, seeds in arms.items():
            if arm == "bc_all":
                continue          # single training by design
            need = 5 if arm in HEADLINE else 3
            if 0 < len(seeds) < need:
                short.append((task, arm, len(seeds), need))
    print(f"D  short arms             : {len(short)}")
    for t, a, n, need in short[:12]:
        print(f"     {t}/{a}  {n} seeds, protocol wants {need}")
    if len(short) > 12:
        print(f"     ... and {len(short)-12} more")

    # E/F: generated tables and \tabinput calls must correspond one to one.
    # Five superseded tables sat in data/tables/ unread, three built on
    # calfilt_ltt, so inputting one would have put a known-bad number in the paper.
    import glob as _glob
    _paper = open(os.path.join(BASE, "paper.tex")).read()
    _gen = {os.path.basename(f) for f in _glob.glob(os.path.join(BASE, "data", "tables", "*.tex"))}
    _used = set(re.findall(r"\\tabinput\{data/tables/([^}]+)\}", _paper))
    _orphan, _missing = sorted(_gen - _used), sorted(_used - _gen)
    print(f"E  generated tables never input : {len(_orphan)}")
    for _t in _orphan:
        print(f"     {_t}")
    print(f"F  tabinput with no such table  : {len(_missing)}")
    for _t in _missing:
        print(f"     {_t}")
    if _orphan:
        fails.append(f"{len(_orphan)} generated tables the paper never inputs")
    if _missing:
        fails.append(f"{len(_missing)} tabinput calls with no generated table")

    print()
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

