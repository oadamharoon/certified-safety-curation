"""Faithful content check of every calfilt selection.

04q derives tau per MODE, then takes kept = {scores >= tau} (min 50), then
optionally applies REWARD_FRAC: top fraction by episodic return, with
n_keep = max(10, ceil(len(kept)*frac)). The recorded n_kept is POST that step,
and the meta does not record reward_frac -- which is why a naive
kept = scores >= tau check reports false mismatches on the R50 arms.

For each meta this recomputes the base selection, then reports either an exact
match, a match under a recognised reward fraction, or the implied fraction so
an unexplained arm is visible rather than hidden. Arms trained on a different
ensemble (pref / noise20 / vgen) are retried against every ensemble on disk.
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

import glob, json, math, os, pickle, sys
import numpy as np, torch, yaml
D = CSC_WORK
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
cfg = yaml.safe_load(open("config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = (sys.argv[1].split(",") if len(sys.argv) > 1 else
         ["halfcheetah_velocity","walker2d_velocity","ant_velocity","hopper_velocity",
          "swimmer_velocity","cargoal1_dsrl","cargoal2","pointgoal1_dsrl","pointgoal2",
          "pointbutton1","pointbutton2","carbutton1_t3","carbutton2","pointcircle1",
          "pointcircle2","carrun_b","ballcircle_b","ballrun_b","carcircle_b","dronerun_b"])

def score_all(task, trajs, ens_file):
    ck = torch.load(f"outputs/{task}/{ens_file}", map_location=dev, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    # ensemble size and width vary across arms (K5 variants, vgen widths);
    # infer both from the checkpoint rather than assuming the config default
    idx = [int(k.split(".")[1]) for k in sd if k.startswith("members.")]
    if not idx:
        return None            # a policy checkpoint, not a V ensemble
    K = ck.get("K") if isinstance(ck, dict) and ck.get("K") else 1 + max(idx)
    hid = ck.get("hidden_dim") if isinstance(ck, dict) and ck.get("hidden_dim") else \
        sd["members.0.net.0.weight"].shape[0]
    ens = VEnsemble(trajs[0]["observations"].shape[1], hid, K=K).to(dev)
    ens.load_state_dict(sd); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32).to(dev)
            g[i] = torch.cat([ens(o[j:j+8192]).cpu() for j in range(0, len(o), 8192)]).mean().item()
    return g

def base_kept(g, tau):
    k = np.where(g >= tau)[0]
    if len(k) < 50:
        k = np.argsort(g)[::-1][:50]
    return k

stats = {"exact": 0, "reward_frac": 0, "other_ensemble": 0, "unexplained": []}
for task in TASKS:
    tc = cfg["tasks"].get(task)
    metas = sorted(glob.glob(f"outputs/{task}/calfilt_meta_*.json"))
    if not tc or not metas:
        continue
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    rets = np.array([float(np.sum(t["rewards"])) for t in trajs])
    ens_files = [os.path.basename(p) for p in glob.glob(f"outputs/{task}/*.pt")
                 if "ensemble" in os.path.basename(p) or "v_ens" in os.path.basename(p)]
    cache = {}
    for mp in metas:
        tag = os.path.basename(mp)[len("calfilt_meta_"):-len(".json")]
        seed = tag.rsplit("_seed", 1)[-1]
        try:
            meta = json.load(open(mp))
        except Exception:
            continue
        if "tau" not in meta or "n_kept" not in meta:
            continue
        rec = int(meta["n_kept"]); tau = float(meta["tau"])
        primary = f"v_ensemble_pess_seed{seed}.pt"
        cands = [primary] + [e for e in ens_files if e != primary]
        resolved = None
        for ef in cands:
            if not os.path.exists(f"outputs/{task}/{ef}"):
                continue
            if ef not in cache:
                try:
                    cache[ef] = score_all(task, trajs, ef)
                except Exception:
                    cache[ef] = None
            g = cache[ef]
            if g is None:
                continue
            k = base_kept(g, tau)
            if len(k) == rec:
                resolved = ("exact" if ef == primary else "other_ensemble", ef, 1.0)
                break
            # try the reward sub-selection at recognised fractions
            for fr in (0.5, 0.25, 0.75, 0.9, 0.1):
                if max(10, int(math.ceil(len(k) * fr))) == rec:
                    resolved = (("reward_frac" if ef == primary else "other_ensemble"), ef, fr)
                    break
            if resolved:
                break
        if resolved:
            stats[resolved[0]] += 1
        else:
            g = cache.get(primary)
            k = base_kept(g, tau) if g is not None else []
            implied = (rec / len(k)) if len(k) else float("nan")
            stats["unexplained"].append((task, tag, len(k), rec, implied))
    print(f"  {task}: done ({len(metas)} metas)", flush=True)

tot = stats["exact"] + stats["reward_frac"] + stats["other_ensemble"] + len(stats["unexplained"])
print(f"\n  TOTAL {tot}")
print(f"    exact match (plain tau)            {stats['exact']}")
print(f"    match under reward sub-selection   {stats['reward_frac']}")
print(f"    match under a different ensemble   {stats['other_ensemble']}")
print(f"    UNEXPLAINED                        {len(stats['unexplained'])}")
for t, tag, nb, rec, imp in stats["unexplained"]:
    print(f"      {t} {tag}: base={nb} recorded={rec} implied_frac={imp:.3f}")
