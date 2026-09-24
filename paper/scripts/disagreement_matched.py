"""Matched-supervision disagreement control: cross-seed advantage agreement for the
preference (pess) ensembles against dense cost-to-go (octg) ensembles, on the SAME
sampled transitions.

App. extraction's matched-supervision paragraph quotes this comparison ("on seven of the
nine analysis tasks the supervised ensembles agree less by advantage rank correlation than
the preference ensembles on the same samples"), but data/review_response/
disagreement_matched.json had no builder in the repo, so it could not be re-derived when F0
retrained the six tasks' preference ensembles. This script is that builder.

The B/C protocol is obstacle2_stats.py's, unchanged and deliberately so: the same
active_segments, the same rng(42) draw of N_TRANS transitions, the same advantage
V(s') - V(s), the same top-one-percent set. Only the ensemble family varies, which is what
makes the two arms comparable. The octg ensembles are the July dense-supervision control
and F0 did not touch them; the pess side moves with the regenerated cohort.
"""
from __future__ import annotations

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

import itertools, json, os, pickle, sys
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CSC_SRC", CSC_WORK)
sys.path.insert(0, REPO)
from model.policy import VEnsemble  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
         "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2"]
FAMILIES = {"pess": "v_ensemble_pess_seed{}.pt", "octg": "v_ensemble_octg_seed{}.pt"}
SEEDS = (0, 1, 2)
N_TRANS = 5000
HIDDEN = 256


def load(outdir, pat, seed, obs_dim):
    p = os.path.join(outdir, pat.format(seed))
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    ens = VEnsemble(obs_dim, ck.get("hidden_dim", HIDDEN), K=ck.get("K", 3)).to(DEVICE)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()
    return ens


def agreement(enss, s, sp):
    advs = []
    for ens in enss:
        with torch.no_grad():
            advs.append((ens(sp) - ens(s)).cpu().numpy().ravel())
    k = max(1, N_TRANS // 100)
    tops = [set(np.argsort(a)[-k:]) for a in advs]
    rhos, jacs = [], []
    for i, j in itertools.combinations(range(len(advs)), 2):
        rhos.append(float(spearmanr(advs[i], advs[j])[0]))
        jacs.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return {"n_seeds": len(advs), "mean_rho": float(np.mean(rhos)), "rhos": rhos,
            "mean_top1_jaccard": float(np.mean(jacs)), "jaccards": jacs}


def main():
    out = {}
    for task in TASKS:
        outdir = os.path.join(REPO, "outputs", task)
        seg_p = os.path.join(outdir, "active_segments.pkl")
        if not os.path.exists(seg_p):
            print(f"  {task:<22} skipped (no active_segments)"); continue
        active = pickle.load(open(seg_p, "rb"))
        obs_dim = active[0]["observations"].shape[1]
        T = active[0]["observations"].shape[0]
        r2 = np.random.default_rng(42)          # obstacle2_stats' draw, so the samples match
        sid = r2.integers(0, len(active), N_TRANS)
        st = r2.integers(0, T - 1, N_TRANS)
        s = torch.as_tensor(np.stack([active[i]["observations"][t] for i, t in zip(sid, st)]),
                            dtype=torch.float32).to(DEVICE)
        sp = torch.as_tensor(np.stack([active[i]["observations"][t + 1] for i, t in zip(sid, st)]),
                             dtype=torch.float32).to(DEVICE)
        rec = {}
        for fam, pat in FAMILIES.items():
            enss = [e for sd in SEEDS if (e := load(outdir, pat, sd, obs_dim)) is not None]
            if len(enss) < 2:
                print(f"  {task:<22} {fam}: <2 seeds, skipped"); continue
            rec[fam] = agreement(enss, s, sp)
        if rec:
            out[task] = rec
            line = "  ".join(f"{f} rho={rec[f]['mean_rho']:+.3f} jac={rec[f]['mean_top1_jaccard']:.3f}"
                             for f in rec)
            print(f"  {task:<22} {line}", flush=True)
    both = [t for t in out if "pess" in out[t] and "octg" in out[t]]
    n_octg_worse = sum(out[t]["octg"]["mean_rho"] < out[t]["pess"]["mean_rho"] for t in both)
    out["_summary"] = {"n_tasks_both_families": len(both),
                       "n_tasks_octg_agrees_less_by_rho": n_octg_worse}
    dst = os.path.join(os.path.dirname(HERE), "data", "review_response",
                       "disagreement_matched.json")
    json.dump(out, open(os.path.abspath(dst), "w"), indent=1)
    print(f"\n  supervised ensembles agree LESS by rho on {n_octg_worse} of {len(both)} tasks")
    print(f"  wrote {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
