"""Eval a CDT run dir at multiple cost targets, reusing its trained return target."""
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
cmd = ["conda", "run", "-n", "safevlmcpl", "--no-capture-output", "python",
       "examples/eval/eval_cdt.py", "--path", run_dir,
       "--returns", f"[{rets}]", "--costs", f"[{targets_csv}]",
       "--eval_episodes", "100", "--device", "cpu", "--threads", "3"]
r = subprocess.run(cmd, capture_output=True, text=True,
                   cwd="/home/omniverse/workspace/safevlmcpl/osrl",
                   env={**os.environ, "PYTHONNOUSERSITE": "1",
                        "PYTHONPATH": "/home/omniverse/workspace/safevlmcpl/osrl"})
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
