"""Realized label-error rate of the Boltzmann preference labeler, per task.

App. extended's robustness paragraph states the labeler's temperature and its realized error
rate. 04n_train_v_only.py PRINTS that rate when it synthesizes the preferences
("[boltz] T=3.0: flipped 84/1000 (8.4%)") and stores it nowhere, so the number lived only in
training logs -- the same shape as the calibration audit trail. R4 of the remediation found the
old text claimed T_lab = 3 with 5 to 6 percent everywhere while the runs had used T_lab = 5 on
the velocity tasks; F0b retrained all of them at T_lab = 3 uniformly, and this reads the realized
rates back out of those runs.

Writes iclr2027/data/boltz_flip_rates.json
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

import glob, json, os, re, sys

W = CSC_WORKSPACE
LOGS = f"{W}/runs/logs/v2variants"
TASKS = ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
         "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2")


def _write_if_changed(path, text):
    if os.path.exists(path) and open(path).read() == text:
        return False
    with open(path, "w") as fh:
        fh.write(text)
    return True


out = {}
for f in sorted(glob.glob(f"{LOGS}/**/*.log", recursive=True) + glob.glob(f"{LOGS}/*.log")):
    txt = open(f, errors="ignore").read()
    hits = re.findall(r"\[boltz\] T=([\d.]+): flipped (\d+)/(\d+) \(([\d.]+)%\)", txt)
    if not hits:
        continue
    task = next((t for t in TASKS if t in os.path.basename(f)), None)
    if task is None:
        task = next((t for t in TASKS if t in f), None)
    if task is None:
        continue
    for T, k, n, pct in hits:
        out.setdefault(task, []).append(
            {"T_lab": float(T), "flipped": int(k), "n_pairs": int(n), "rate": float(pct) / 100.0,
             "log": os.path.relpath(f, W)})

if not out:
    print("no [boltz] lines found", file=sys.stderr); sys.exit(1)
temps = {r["T_lab"] for v in out.values() for r in v}
rates = [r["rate"] for v in out.values() for r in v]
payload = {"T_lab_values": sorted(temps), "n_tasks": len(out),
           "rate_min": min(rates), "rate_max": max(rates),
           "per_task": {t: {"T_lab": sorted({r["T_lab"] for r in v}),
                            "rate_min": min(r["rate"] for r in v),
                            "rate_max": max(r["rate"] for r in v),
                            "runs": v} for t, v in sorted(out.items())}}
dst = f"{W}/iclr2027/data/boltz_flip_rates.json"
_write_if_changed(dst, json.dumps(payload, indent=1))
for t, v in sorted(payload["per_task"].items()):
    print(f"  {t:22s} T_lab={v['T_lab']}  realized error {100*v['rate_min']:.1f} to {100*v['rate_max']:.1f} percent  ({len(v['runs'])} runs)")
print(f"\n  temperatures used: {sorted(temps)};  overall {100*min(rates):.1f} to {100*max(rates):.1f} percent")
print(f"  wrote {dst}")
