"""Rebuild the deployed certified selections (draw 1) as OSRL subset hdf5.

The originals were written to a session scratchpad and are gone, so the
composability arm cdt_cert could not be reproduced. Membership is taken from
the recovered kept_calfilt_csf_seed*.json files (recover_kept.py, each verified
against the recorded n_kept) rather than by re-running the calibration draw,
so this reconstructs the selection that was actually deployed.

The h5 is written on the raw-DSRL span route used by build_draw2.py, the
surviving sibling of the lost builder, which carries real terminals; the
pickle route in build_cdt_h5.py zeroes them and would not match.

Output goes to runs/selections/, NOT a scratchpad.
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

import json, os, pickle, sys
import numpy as np
import h5py
import yaml

D = CSC_WORK
OUT = CSC_RUNS + "/selections"
sys.path.insert(0, D); os.chdir(D)
os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open("config.yaml"))

# (task, pipeline seed) = the LOWEST certifying seed, which is the rule the
# lost filenames encode: halfcheetah_..._seed1, cargoal1_..._seed3,
# pointgoal1_..._seed0. build_draw2.py's TASKS list confirms the same triple.
JOBS = [("halfcheetah_velocity", 1), ("cargoal1_dsrl", 3), ("pointgoal1_dsrl", 0)]

import gymnasium as gym
try:
    import dsrl
    if hasattr(dsrl, "register_envs"):
        dsrl.register_envs()
except Exception:
    pass

for task, vseed in JOBS:
    tc = cfg["tasks"][task]
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    keptp = f"outputs/{task}/kept_calfilt_csf_seed{vseed}.json"
    rec = json.load(open(keptp))
    assert rec["certified"], f"{task} seed{vseed} is not a certified selection"
    kept_idx = np.array(rec["kept"], dtype=int)
    keep = np.zeros(len(trajs), dtype=bool); keep[kept_idx] = True

    name = tc.get("offline_env_name") or tc["env_name"].replace("Safety", "Offline", 1)
    env = gym.make(name); d = env.get_dataset(); env.close()
    term = np.asarray(d["terminals"], dtype=bool)
    if "timeouts" in d:
        end = np.logical_or(term, np.asarray(d["timeouts"], dtype=bool))
        tout = np.asarray(d["timeouts"], dtype=bool)
    else:
        end = term.copy(); end[999::1000] = True
        tout = np.zeros_like(term); tout[999::1000] = True
    if not end[-1]:
        end[-1] = True
    ends = np.where(end)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    spans = [(s, e) for s, e in zip(starts, ends) if e + 1 - s >= 2]
    assert len(spans) == len(trajs), f"{task}: span/pickle mismatch {len(spans)} vs {len(trajs)}"

    idx = np.concatenate([np.arange(s, e + 1) for (s, e), k in zip(spans, keep) if k])
    outp = os.path.join(OUT, f"{task}_cert_seed{vseed}.hdf5")
    with h5py.File(outp, "w") as f:
        keys = ["observations", "actions", "rewards", "costs"]
        if "next_observations" in d:
            keys.append("next_observations")
        for key in keys:
            f.create_dataset(key, data=np.asarray(d[key])[idx])
        if "next_observations" not in d:
            obs_all = np.asarray(d["observations"])
            nxt = obs_all.copy(); nxt[:-1] = obs_all[1:]
            for s_, e_ in spans:
                nxt[e_] = obs_all[e_]
            f.create_dataset("next_observations", data=nxt[idx])
        f.create_dataset("terminals", data=term[idx])
        f.create_dataset("timeouts", data=tout[idx])
    json.dump({"kept": [int(i) for i in kept_idx]},
              open(os.path.join(OUT, f"{task}_cert_seed{vseed}_kept.json"), "w"))
    lim = tc.get("cost_limit", cfg["cost_limit"])
    cost = np.array([float(np.sum(t["costs"])) for t in trajs])
    contam = float((cost[kept_idx] > lim).mean())
    print(f"  {task} seed{vseed}: {len(kept_idx)} trajs, {len(idx)} transitions, "
          f"contamination {contam:.3f} -> {os.path.basename(outp)}", flush=True)
print("CERT SUBSETS REBUILT", flush=True)
