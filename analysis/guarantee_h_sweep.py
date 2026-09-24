"""Proposition 1 resampling validation at each segment length H, the H-arm counterpart
of analysis/guarantee_validation.py.

The three artifacts data/review_response/guarantee_H{10,30,50}_2000.json were quoted by
the segment-length paragraph of App. extended but had NO builder anywhere in the repo, so
they could not be re-derived when F0 retrained the six tasks' base ensembles. This script
is that builder. It is guarantee_validation's procedure with the H arm's config and the
H arm's ensemble directory substituted, n = 200 only, on the nine analysis tasks.

Faithfulness test: H10 and H50 read outputs/{task}_h{10,50}, whose ensembles are the
2026-08-16 stated-protocol files that F0 did not touch, so this script must reproduce the
archived H10 and H50 files exactly. H30 reads the base directory, which F0 did change, so
that one is expected to move.

Usage: python runs/scripts/guarantee_h_sweep.py [--arm H10|H30|H50] [--outdir DIR]
"""

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

import argparse, json, os, sys
import numpy as np, torch, yaml
from scipy.stats import hypergeom, beta

D = CSC_WORK
sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble  # noqa: E402

torch.set_num_threads(6)
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
ALPHA, DELTA, N_DRAWS = 0.25, 0.1, 2000
SIZES = [200]
ARMS = {"H10": ("config_h10.yaml", "_h10"), "H30": ("config.yaml", ""),
        "H50": ("config_h50.yaml", "_h50")}
TASKS = ("halfcheetah_velocity walker2d_velocity ant_velocity hopper_velocity "
         "swimmer_velocity cargoal1_dsrl cargoal2 pointgoal1_dsrl pointgoal2").split()
DEFAULT_OUT = CSC_PAPER_DATA + "/review_response"


def cp95(k, n):
    if n == 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def run_arm(arm, outdir):
    cfgfile, suf = ARMS[arm]
    cfg = yaml.safe_load(open(cfgfile))
    out = {}
    for task in TASKS:
        tc = cfg["tasks"][task]
        lim = tc.get("cost_limit", cfg["cost_limit"])
        import pickle
        trajs = pickle.load(open(tc["data_pickle"], "rb"))
        cost = np.array([float(np.sum(t["costs"])) for t in trajs])
        unsafe = (cost > lim).astype(float)
        obs_dim = trajs[0]["observations"].shape[1]
        entry = {"limit": lim, "n_trajs": len(trajs),
                 "base_unsafe_rate": float(unsafe.mean()), "seeds": {}}
        for seed in (0, 1, 2):
            fp = f"outputs/{task}{suf}/v_ensemble_pess_seed{seed}.pt"
            if not os.path.exists(fp):
                continue
            ens = VEnsemble(obs_dim, 256, K=3)
            ck = torch.load(fp, map_location="cpu", weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            ens.eval()
            g = np.zeros(len(trajs))
            with torch.no_grad():
                for i, t in enumerate(trajs):
                    o = torch.as_tensor(t["observations"], dtype=torch.float32)
                    g[i] = torch.cat([ens(o[j:j + 8192])
                                      for j in range(0, len(o), 8192)]).mean().item()
            taus = [float(np.quantile(g, q)) for q in QS]
            Ns = [int((g >= t).sum()) for t in taus]
            ku = [float(unsafe[g >= t].mean()) for t in taus]
            sd = {}
            for n in SIZES:
                rng = np.random.default_rng(9000 + seed)
                n_cert = n_false = 0
                for _ in range(N_DRAWS):
                    cal = rng.choice(len(g), n, replace=False)
                    cs, cu = g[cal], unsafe[cal]
                    pick = None
                    for j, t in enumerate(taus):
                        sel = cs >= t
                        m, k = int(sel.sum()), int(cu[sel].sum())
                        ks = int(ALPHA * Ns[j]) + 1
                        p = float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0
                        if m > 0 and p <= DELTA:
                            pick = j
                        else:
                            break
                    if pick is not None:
                        n_cert += 1
                        if ku[pick] > ALPHA:
                            n_false += 1
                sd[str(n)] = {
                    "n_draws": N_DRAWS,
                    "cert_rate": n_cert / N_DRAWS,
                    "false_cert_rate_uncond": n_false / N_DRAWS,
                    "false_cert_cp95": cp95(n_false, N_DRAWS),
                    "cond_viol_rate": (n_false / n_cert) if n_cert else None,
                    "cond_viol_cp95": cp95(n_false, n_cert) if n_cert else None,
                }
            entry["seeds"][str(seed)] = sd
            print(f"{arm} {task} s{seed}: " + " ".join(
                f"n{n}:cert{sd[str(n)]['cert_rate']:.3f}/unc{sd[str(n)]['false_cert_rate_uncond']:.3f}"
                for n in SIZES), flush=True)
        out[task] = entry
    dst = os.path.join(outdir, f"guarantee_{arm}_2000.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"{arm} DONE -> {dst}", flush=True)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), action="append")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for arm in (a.arm or sorted(ARMS)):
        run_arm(arm, a.outdir)
