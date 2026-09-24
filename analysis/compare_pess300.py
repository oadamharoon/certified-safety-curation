"""B2 (V2_REMEDIATION): old-cohort ensembles (June/July, 50 epochs batch 1024 or the V-AWR
script) against the retrained ones at the stated protocol (300 epochs, batch 512), on the six
affected tasks, without retraining any policy.

Per task and seed, for each ensemble: trajectory score g = mean state value; score AUC vs
ground-truth safety; top-quantile purity u1 at q = 0.85; certification rate at n = 200,
alpha = 0.25, delta = 0.1 over 200 draws (paper's strict fixed-sequence walk); filter
precision at the ground-truth safe fraction; Spearman(g, cost). Applies decision rule B3.
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

import json, os, pickle, sys
import numpy as np, torch, yaml
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import roc_auc_score
D = CSC_WORK; sys.path.insert(0, D); os.chdir(D)
from model.policy import VEnsemble
cfg = yaml.safe_load(open("config.yaml"))
TASKS = ["halfcheetah_velocity", "cargoal1_dsrl", "walker2d_velocity", "ant_velocity", "hopper_velocity", "swimmer_velocity"]
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
ALPHA, DELTA, CAL_N, DRAWS = 0.25, 0.1, 200, 200
dev = "cuda" if torch.cuda.is_available() else "cpu"


def hyp_p(k, m, n_sel):
    ks = int(ALPHA * n_sel) + 1
    return 1.0 if ks > n_sel else float(hypergeom.cdf(k, n_sel, ks, m))


def walk(g, unsafe, cal):
    for q in QS:
        mask = g >= np.quantile(g, q); sc = mask[cal]; m = int(sc.sum()); k = int(unsafe[cal][sc].sum())
        if m > 0 and hyp_p(k, m, int(mask.sum())) <= DELTA: return True
        break
    return False


def score(task, trajs, path):
    ens = VEnsemble(trajs[0]["observations"].shape[1], 256, K=3).to(dev)
    ck = torch.load(path, map_location=dev, weights_only=False); ens.load_state_dict(ck.get("state_dict", ck)); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32, device=dev)
            g[i] = torch.cat([ens(o[j:j + 8192]) for j in range(0, len(o), 8192)]).mean().item()
    return g


out = {}
print(f"{'task':22s} {'seed':>4s} {'ens':>7s} {'AUC':>6s} {'u1':>6s} {'cert':>6s} {'prec@gt':>8s} {'rho_cost':>9s}")
for task in TASKS:
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"])
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs]); unsafe = (cost > lim); gt_frac = float((~unsafe).mean())
    nsel = max(1, int(round(gt_frac * len(trajs))))
    out[task] = {}
    for s in range(5):
        for name, fn in (("old", f"outputs/{task}/v_ensemble_pess_seed{s}.pt"), ("new300", f"outputs/{task}/v_ensemble_pess300_seed{s}.pt")):
            if not os.path.exists(fn): continue
            g = score(task, trajs, fn)
            m1 = g >= np.quantile(g, 0.85); u1 = float(unsafe[m1].mean())
            rng = np.random.default_rng(1); rate = np.mean([walk(g, unsafe, rng.choice(len(g), CAL_N, replace=False)) for _ in range(DRAWS)])
            prec = float((~unsafe[np.argsort(-g)[:nsel]]).mean()); rho = spearmanr(g, cost).statistic
            out[task].setdefault(str(s), {})[name] = {"auc": float(roc_auc_score(~unsafe, g)), "u1": u1, "cert_rate": float(rate), "prec_gt": prec, "rho_cost": float(rho)}
            print(f"{task:22s} {s:4d} {name:>7s} {out[task][str(s)][name]['auc']:6.3f} {u1:6.3f} {rate:6.2f} {prec:8.3f} {rho:9.3f}", flush=True)
json.dump(out, open(CSC_RUNS + "/probe/pess300_compare.json", "w"), indent=1)

print("\n=== B3 decision rule per task (mean over seeds present in both) ===")
verdict = {}
for task in TASKS:
    rows = [(v["old"], v["new300"]) for v in out[task].values() if "old" in v and "new300" in v]
    if not rows: print(f"{task}: incomplete"); continue
    du1 = np.mean([n["u1"] - o["u1"] for o, n in rows]); dpr = np.mean([n["prec_gt"] - o["prec_gt"] for o, n in rows])
    ro, rn = np.mean([o["cert_rate"] for o, n in rows]), np.mean([n["cert_rate"] for o, n in rows])
    same_status = (ro >= 0.05) == (rn >= 0.05)
    ok = abs(du1) <= 0.02 and same_status and abs(dpr) < 0.02
    verdict[task] = ok
    print(f"{task:22s} d(u1)={du1:+.3f} d(prec)={dpr:+.3f} cert {ro:.2f}->{rn:.2f} status_same={same_status}  -> {'IMMATERIAL' if ok else 'MATERIAL'}")
print("\nOVERALL:", "IMMATERIAL on all six: state both cohorts with this control" if all(verdict.values()) and len(verdict) == 6 else "MATERIAL on at least one task: regenerate downstream arms (B3 second branch)")
