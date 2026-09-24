"""Build flat DSRL-format hdf5 subsets from the regenerated certified selections,
so CDT can be retrained on exactly the certified data (A3)."""

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

import json, os, pickle, sys
import numpy as np, h5py, yaml
D=CSC_WORK
SEL=CSC_RUNS + "/selections"
os.chdir(D); sys.path.insert(0, D)
cfg=yaml.safe_load(open("config.yaml"))
# usage: build_cdt_h5.py [--force] [substring ...]
FORCE="--force" in sys.argv
ONLY=[a for a in sys.argv[1:] if not a.startswith("--")]
made=0
for f in sorted(os.listdir(SEL)):
    if not f.endswith("_kept.json"): continue
    tag=f[:-len("_kept.json")]
    task=None
    for t in cfg["tasks"]:
        if tag.startswith(t) and (task is None or len(t)>len(task)): task=t
    if task is None: print("  skip (no task):", tag); continue
    # Only build what is asked for, and never clobber an existing subset. The
    # *_cert_*.hdf5 files come from build_cert_h5.py, which uses the raw-DSRL span
    # route and carries real terminals; this pickle route zeroes them, so a blanket
    # rebuild would silently corrupt the cdt_cert inputs.
    if ONLY and not any(k in tag for k in ONLY):
        continue
    out_probe=os.path.join(SEL,f"{tag}.hdf5")
    if os.path.exists(out_probe) and not FORCE:
        print("  skip (exists):", tag); continue
    kept=json.load(open(os.path.join(SEL,f)))["kept"]
    trajs=pickle.load(open(cfg["tasks"][task]["data_pickle"],"rb"))
    # next_observations by within-trajectory shift; the final step repeats its
    # own observation, matching the DSRL convention for timeout-terminated data.
    nxt=[]
    for i in kept:
        o=np.asarray(trajs[i]["observations"])
        nxt.append(np.concatenate([o[1:], o[-1:]], axis=0))
    nobs=np.concatenate(nxt)
    obs=np.concatenate([trajs[i]["observations"] for i in kept])
    act=np.concatenate([trajs[i]["actions"] for i in kept])
    rew=np.concatenate([trajs[i]["rewards"] for i in kept])
    cost=np.concatenate([trajs[i]["costs"] for i in kept])
    term=np.zeros(len(obs),dtype=bool); tout=np.zeros(len(obs),dtype=bool)
    p=0
    for i in kept:
        p+=len(trajs[i]["rewards"]); tout[p-1]=True
    out=os.path.join(SEL,f"{tag}.hdf5")
    with h5py.File(out,"w") as h:
        for k,v in (("observations",obs),("next_observations",nobs),
                    ("actions",act),("rewards",rew),
                    ("costs",cost),("terminals",term),("timeouts",tout)):
            h.create_dataset(k,data=v)
    made+=1
    print(f"  {tag}: {len(kept)} trajs, {len(obs)} transitions -> {os.path.basename(out)}", flush=True)
print(f"built {made} hdf5 subsets")
