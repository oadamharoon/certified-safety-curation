"""D1 (V2_REMEDIATION): the certificate as a triple, and its validation.

The procedure returns, besides the certified threshold, two numbers a deployer can read
from the calibration sample already spent, both consequences of Proposition 4 (rate):

  (b) estimated certification rate r_hat: Pr[certify] from Equation (rate) with the unsafe
      count U_1 of the first grid selection S(lambda_1) integrated over its posterior given
      the (m_1, k_1) calibration counts that fell in S(lambda_1) (Jeffreys
      Beta(k+1/2, m-k+1/2) on u_1, U_1 = round(u_1 N_1); N_POST posterior draws). Reported
      as the posterior mean and the posterior 0.1 quantile r_lo.
  (c) a bound on the deployer-facing conditional violation rate. The guarantee is
      unconditional, Pr[certify and out of spec] <= delta, so Pr[out of spec | certify]
      <= delta / Pr[certify]; plugging the conservative r_lo gives min(1, delta / r_lo).
      A plug-in posterior Pr[u(S(lambda_hat)) > alpha] from the counts in the returned
      selection is NOT reported as (c): it is computed here only to show that it is
      optimistic under selection (a certificate fires exactly when the sample looks cleaner
      than the pool), which is why the bound rather than the plug-in is the deliverable.

Validation, per task and V seed at n = 200 over N_DRAWS fresh calibration draws: mean r_hat
against the realized certification rate; the mean bound (c) over certified draws against
the realized Pr[violate | certified] of Table condviol; and the optimistic plug-in beside
them. Scores come from the cached e_scores npz (analysis tasks) or from v_ensemble_pess.
Output: data/certificate_triple.json and data/tables/cert_triple.tex.
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
import numpy as np
from scipy.stats import beta, hypergeom

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from cert_rate_theory import pr_certify  # noqa: E402
REPO = CSC_WORK
ALPHA, DELTA, CAL_N, N_DRAWS, N_POST = 0.25, 0.1, 200, 500, 200
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
         "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2",
         "pointbutton1", "pointbutton2", "carbutton1_t3", "carbutton2", "pointcircle1",
         "pointcircle2", "ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b"]
NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d", "ant_velocity": "Ant",
         "hopper_velocity": "Hopper", "swimmer_velocity": "Swimmer", "cargoal1_dsrl": "CarGoal1",
         "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1", "pointgoal2": "PointGoal2",
         "pointbutton1": "PointButton1", "pointbutton2": "PointButton2", "carbutton1_t3": "CarButton1",
         "carbutton2": "CarButton2", "pointcircle1": "PointCircle1", "pointcircle2": "PointCircle2",
         "ballrun_b": "BallRun", "ballcircle_b": "BallCircle", "carcircle_b": "CarCircle",
         "carrun_b": "CarRun", "dronerun_b": "DroneRun"}


def limit_of(t):
    return 20 if "velocity" in t else 10 if t.endswith("_b") else 25


def load_scores(task, seed):
    p = os.path.join(BASE, "data", "e_scores", f"{task}_seed{seed}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["scores"], z["unsafe"].astype(int)
    import torch, yaml
    sys.path.insert(0, REPO)
    from model.policy import VEnsemble
    cfg = yaml.safe_load(open(os.path.join(REPO, "config.yaml")))
    trajs = pickle.load(open(os.path.join(REPO, cfg["tasks"][task]["data_pickle"]) if not
                             os.path.isabs(cfg["tasks"][task]["data_pickle"]) else cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    ens_p = os.path.join(REPO, "outputs", task, f"v_ensemble_pess_seed{seed}.pt")
    if not os.path.exists(ens_p):
        return None, None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ens = VEnsemble(trajs[0]["observations"].shape[1], 256, K=3).to(dev)
    ck = torch.load(ens_p, map_location=dev, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32, device=dev)
            g[i] = torch.cat([ens(o[j:j + 8192]).cpu() for j in range(0, len(o), 8192)]).mean().item()
    return g, (costs > limit_of(task)).astype(int)


def walk(scores, unsafe, cal, taus):
    """Strict fixed sequence. Returns (certified, chosen index, [(m_j, k_j, N_j)] per grid)."""
    cs, cu = scores[cal], unsafe[cal]
    chosen, cert, counts = None, False, []
    for j, tau in enumerate(taus):
        sel = cs >= tau; m, k = int(sel.sum()), int(cu[sel].sum()); n_sel = int((scores >= tau).sum())
        counts.append((m, k, n_sel))
        ks = int(ALPHA * n_sel) + 1
        p = 1.0 if (m == 0 or ks > n_sel) else float(hypergeom.cdf(k, n_sel, ks, m))
        if m > 0 and p <= DELTA:
            chosen, cert = j, True
        else:
            break
    return cert, chosen, counts


_RATE_CACHE = {}


def triple(counts, chosen, npool, rng):
    """(r_hat mean, r_lo, bound delta/r_lo or None, optimistic plug-in or None)."""
    m1, k1, n1 = counts[0]
    u = beta.rvs(k1 + 0.5, m1 - k1 + 0.5, size=N_POST, random_state=rng)
    rs = []
    for U in np.round(u * n1).astype(int):
        key = (npool, n1, int(U))
        if key not in _RATE_CACHE:
            _RATE_CACHE[key] = pr_certify(npool, n1, int(U), CAL_N)
        rs.append(_RATE_CACHE[key])
    rs = np.array(rs)
    r_hat, r_lo = float(rs.mean()), float(np.quantile(rs, 0.1))
    if chosen is None:
        return r_hat, r_lo, None, None
    m, k, _ = counts[chosen]
    plug = float(beta.sf(ALPHA, k + 0.5, m - k + 0.5))
    bound = min(1.0, DELTA / r_lo) if r_lo > 0 else 1.0
    return r_hat, r_lo, bound, plug


def main():
    out, rows = {}, []
    for task in TASKS:
        out[task] = {}
        for seed in (0, 1, 2):
            scores, unsafe = load_scores(task, seed)
            if scores is None:
                continue
            taus = [float(np.quantile(scores, q)) for q in QS]
            rng = np.random.default_rng(11 + seed)
            est_rates, r_los, bounds, plugs, certs, viols = [], [], [], [], [], []
            for _ in range(N_DRAWS):
                cal = rng.choice(len(scores), CAL_N, replace=False)
                cert, chosen, counts = walk(scores, unsafe, cal, taus)
                r_hat, r_lo, bound, plug = triple(counts, chosen, len(scores), rng)
                est_rates.append(r_hat); r_los.append(r_lo); certs.append(cert)
                if cert:
                    bounds.append(bound); plugs.append(plug)
                    viols.append(bool(unsafe[scores >= taus[chosen]].mean() > ALPHA))
            rate = float(np.mean(certs))
            rec = {"realized_cert_rate": rate, "est_cert_rate_mean": float(np.mean(est_rates)),
                   "est_cert_rate_sd": float(np.std(est_rates)), "r_lo_mean": float(np.mean(r_los)),
                   "realized_cond_viol": (float(np.mean(viols)) if viols else None),
                   "bound_mean": (float(np.mean(bounds)) if bounds else None),
                   "bound_covers": (float(np.mean([b >= np.mean(viols) for b in bounds])) if bounds else None),
                   "plugin_mean": (float(np.mean(plugs)) if plugs else None),
                   "n_certified": int(sum(certs)), "n_draws": N_DRAWS}
            out[task][str(seed)] = rec
            print(f"{task:22s} s{seed} rate {rate:.2f} est {rec['est_cert_rate_mean']:.2f}+-{rec['est_cert_rate_sd']:.2f}"
                  f" | cond viol {rec['realized_cond_viol']} bound {rec['bound_mean']} plugin {rec['plugin_mean']}", flush=True)
    os.makedirs(os.path.join(BASE, "data", "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(BASE, "data", "certificate_triple.json"), "w"), indent=1)
    for task in TASKS:
        cells = list(out[task].values())
        if not cells:
            continue
        f = lambda xs: "--" if not xs else f"{np.mean(xs):.2f}"
        r = [c["realized_cert_rate"] for c in cells]; e = [c["est_cert_rate_mean"] for c in cells]
        cv = [c["realized_cond_viol"] for c in cells if c["realized_cond_viol"] is not None]
        bd = [c["bound_mean"] for c in cells if c["bound_mean"] is not None]
        pl = [c["plugin_mean"] for c in cells if c["plugin_mean"] is not None]
        nc = sum(c["n_certified"] for c in cells)
        rows.append(f"{NAMES[task]} & {f(r)} & {f(e)} & {f(cv)} & {f(bd)} & {f(pl)} & {nc} \\\\")
    open(os.path.join(BASE, "data", "tables", "cert_triple.tex"), "w").write("\n".join(rows) + "\n")
    print("wrote data/certificate_triple.json and data/tables/cert_triple.tex")


if __name__ == "__main__":
    main()
