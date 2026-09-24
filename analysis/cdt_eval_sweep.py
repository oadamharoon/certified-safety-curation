"""Eval a CDT run dir at multiple cost targets, reusing its trained return target."""

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


CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(__file__))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "datasets"))
CSC_RUNS = _os.environ.get("CSC_RUNS", _os.path.join(_WS, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# -----------------------------------------------------------------------------

import json, os, re, subprocess, sys

run_dir, targets_csv, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
cfgp = None
for root, _, files in os.walk(run_dir):
    for f in files:
        if f in ("config.yaml", "config.json"):
            cfgp = os.path.join(root, f)
if cfgp is None:
    sys.exit(f"no config in {run_dir}")
run_dir = os.path.dirname(cfgp)  # eval script wants the dir holding config.yaml
import yaml
_cfg = yaml.unsafe_load(open(cfgp))
_tr = _cfg.get("target_returns")
pairs = [(float(a), float(b)) for a, b in _tr] if _tr else []
if not pairs:
    sys.exit(f"could not parse target_returns from {cfgp}")
# target_returns is [(ret, cost), ...]; take the return paired with the LOWEST cost
ret = min(pairs, key=lambda p: p[1])[0]
targets = [float(x) for x in targets_csv.split(",")]
rets = ",".join(str(ret) for _ in targets)
cmd = [_os.environ.get("PYTHON", "python"),
       "examples/eval/eval_cdt.py", "--path", run_dir,
       "--returns", f"[{rets}]", "--costs", f"[{targets_csv}]",
       "--eval_episodes", "100", "--device", "cpu", "--threads", "3"]
r = subprocess.run(cmd, capture_output=True, text=True,
                   cwd=CSC_OSRL,
                   env={**os.environ, "PYTHONNOUSERSITE": "1",
                        "PYTHONPATH": CSC_OSRL})
res = []
for line in r.stdout.splitlines():
    mm = re.search(r"real reward ([-\d.]+),.*target cost ([-\d.]+), real cost ([-\d.]+)", line)
    if mm:
        res.append({"target_cost": float(mm.group(2)), "R": float(mm.group(1)),
                    "C": float(mm.group(3))})
if not res:
    print(r.stdout[-2000:]); print(r.stderr[-2000:], file=sys.stderr)
    sys.exit("no eval lines parsed")
json.dump({"run_dir": run_dir, "return_target": ret, "evals": res}, open(out_json, "w"))
print(f"OK {os.path.basename(run_dir)}: {res}", flush=True)
