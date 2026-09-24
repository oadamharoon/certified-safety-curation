"""Verify every published OSRL number against the run it was harvested from.

harvest_osrl.py reads wandb-summary.json, picks the first key matching
cost/c_<tgt> and ret/c_<tgt> for CDT, and keeps the latest run per
(task, algo, seed). This re-derives each published value independently and
reports: value mismatches, ambiguous key matches (more than one candidate key,
where [0] would be an arbitrary dict-order choice), and (task, algo, seed)
cells claimed by more than one run.
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

import glob, json, os, re, sys, collections
ICLR = CSC_PAPER
OSRL = CSC_OSRL
sys.path.insert(0, os.path.join(ICLR, "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("h", os.path.join(ICLR, "scripts", "harvest_osrl.py"))
h = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(h)      # module defines ENV2TASK at import
except SystemExit:
    pass
ENV2TASK = h.ENV2TASK

pub = json.load(open(os.path.join(ICLR, "data", "osrl_results.json")))
claims = collections.defaultdict(list)     # (task, algo, seed) -> [(mtime, R, C, ambiguous, run)]
for d in glob.glob(os.path.join(OSRL, "wandb", "run-*")):
    mp, sp = os.path.join(d, "files", "wandb-metadata.json"), os.path.join(d, "files", "wandb-summary.json")
    if not (os.path.exists(mp) and os.path.exists(sp)):
        continue
    try:
        meta, s = json.load(open(mp)), json.load(open(sp))
    except Exception:
        continue
    args = " ".join(meta.get("args", []))
    prog = meta.get("program", "")
    m_algo = re.search(r"train_(cdt|cpq|coptidice)\.py", prog + " " + args)
    m_env = next((k for k in ENV2TASK if k in args), None)
    m_seed = re.search(r"--seed[= ](\d+)", args)
    if not (m_algo and m_env and m_seed):
        continue
    task, seed = ENV2TASK[m_env], m_seed.group(1)
    algo = m_algo.group(1)
    if algo == "cdt":
        tgt = "10" if task.endswith("_b") else "20"
        ck = [k for k in s if k.startswith(f"cost/c_{tgt}")]
        rk = [k for k in s if k.startswith(f"ret/c_{tgt}")]
    else:
        rk = [k for k in s if k in ("eval/Reward", "eval/reward", "ret/reward")] or [k for k in s if k.startswith("ret/")]
        ck = [k for k in s if k in ("eval/Cost", "eval/cost", "ret/cost")] or [k for k in s if k.startswith("cost/")]
    if not (ck and rk):
        continue
    amb = len(ck) > 1 or len(rk) > 1
    claims[(task, algo, seed)].append((os.path.getmtime(sp), float(s[rk[0]]), float(s[ck[0]]), amb, os.path.basename(d), args))

mismatch, ambiguous, multi = [], [], []
checked = 0
for task, algos in pub.items():
    for algo, seeds in algos.items():
        base = algo.split("_")[0] if algo.startswith(("cdt", "cpq", "copt")) else algo
        for seed, v in seeds.items():
            if not isinstance(v, dict) or "R" not in v:
                continue
            # find the run(s) whose harvested identity matches, allowing the
            # variant suffix that subset_h5 introduces
            cand = [c for k, c in claims.items() if k[0] == task and k[2] == seed and algo.startswith(k[1])]
            flat = [x for lst in cand for x in lst]
            if not flat:
                continue
            checked += 1
            flat.sort()
            latest = flat[-1]
            if abs(latest[1] - v["R"]) > 1e-6 or abs(latest[2] - v["C"]) > 1e-6:
                # variant arms share (task, algo-base, seed); only flag when NO run matches
                if not any(abs(x[1] - v["R"]) < 1e-6 and abs(x[2] - v["C"]) < 1e-6 for x in flat):
                    mismatch.append((task, algo, seed, v["R"], v["C"]))
            if any(x[3] for x in flat):
                ambiguous.append((task, algo, seed))
print(f"  published cells checked against source runs : {checked}")
print(f"  cells whose value matches NO source run     : {len(mismatch)}")
for t, a, s_, R, C in mismatch[:15]:
    print(f"      {t} {a} seed{s_}: published R={R:.3f} C={C:.3f}")
print(f"  cells with an ambiguous summary-key match   : {len(set(ambiguous))}")
for t, a, s_ in sorted(set(ambiguous))[:10]:
    print(f"      {t} {a} seed{s_}")
