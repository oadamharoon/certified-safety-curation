"""Review-response scoring-only statistics (all CPU, no training).

1. Spearman(g, episodic cost) and Spearman(g, episodic return) per task,
   across all available pess V-ensemble seeds (15 tasks).
2. Guarantee validation at N_DRAWS = 2000 with Clopper-Pearson intervals
   (9 analysis tasks, same schema as guarantee_stats.json).
3. Per-transition disagreement, BT (pess) vs per-step-supervised (octg)
   ensembles: cross-seed advantage Spearman + top-1% Jaccard on 5000
   fixed transitions per task.

Outputs to iclr2027/data/review_response/.
"""
import json, os, pickle, sys
import numpy as np
import torch
from scipy.stats import spearmanr, hypergeom, beta as beta_dist

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response"
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

TASKS15 = {"ballrun_b": 10, "ballcircle_b": 10, "carcircle_b": 10,
           "carrun_b": 10, "dronerun_b": 10}
ANALYSIS9 = list(TASKS15)
ALPHA, DELTA = 0.25, 0.1
CAL_NS = [50, 100, 200, 400]
N_DRAWS = 2000
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]

cfg = yaml.safe_load(open("config.yaml"))


def load_trajs(task):
    with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
        return pickle.load(f)


def load_ens(task, name, seed, obs_dim):
    p = f"outputs/{task}/v_ensemble_{name}_seed{seed}.pt"
    if not os.path.exists(p):
        return None
    ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()
    return ens


def score_trajs(ens, trajs):
    scores = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32)
            vs = [ens(o[j:j + 8192]) for j in range(0, len(o), 8192)]
            scores[i] = torch.cat(vs).mean().item()
    return scores


def hyper_pvalue(k, m, n_sel, alpha):
    k_star = int(alpha * n_sel) + 1
    if k_star > n_sel:
        return 1.0
    return float(hypergeom.cdf(k, n_sel, k_star, m))


def cp_interval(x, n, conf=0.95):
    lo = 0.0 if x == 0 else float(beta_dist.ppf((1 - conf) / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta_dist.ppf(1 - (1 - conf) / 2, x + 1, n - x))
    return lo, hi


# ---------- 1. correlations ----------
corr, score_cache = {}, {}
for task, limit in TASKS15.items():
    trajs = load_trajs(task)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    rets = np.array([float(np.sum(t["rewards"])) for t in trajs])
    obs_dim = trajs[0]["observations"].shape[1]
    rows = []
    for seed in range(5):
        ens = load_ens(task, "pess", seed, obs_dim)
        if ens is None:
            continue
        g = score_trajs(ens, trajs)
        score_cache[(task, seed)] = g
        rows.append({"seed": seed,
                     "rho_cost": float(spearmanr(g, costs)[0]),
                     "rho_return": float(spearmanr(g, rets)[0])})
        print(f"[corr] {task} seed{seed} rho_cost={rows[-1]['rho_cost']:+.3f} "
              f"rho_return={rows[-1]['rho_return']:+.3f}", flush=True)
    corr[task] = {"limit": limit,
                  "rho_cost_return_of_costs": float(spearmanr(costs, rets)[0]),
                  "seeds": rows,
                  "mean_rho_cost": float(np.mean([r["rho_cost"] for r in rows])),
                  "mean_rho_return": float(np.mean([r["rho_return"] for r in rows]))}
    with open(f"{OUT}/bullet_score_correlations.json", "w") as f:
        json.dump(corr, f, indent=1)
print("correlations done", flush=True)

# ---------- 2. guarantee at 2000 draws ----------
gres = {}
for task in ANALYSIS9:
    limit = TASKS15[task]
    trajs = load_trajs(task)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > limit).astype(int)
    task_out = {"limit": limit, "n_trajs": len(trajs),
                "base_unsafe_rate": float(unsafe.mean()), "seeds": {}}
    for seed in (0, 1, 2):
        if (task, seed) not in score_cache:
            trj = trajs  # ensemble may be missing
            ens = load_ens(task, "pess", seed, trajs[0]["observations"].shape[1])
            if ens is None:
                continue
            score_cache[(task, seed)] = score_trajs(ens, trj)
        scores = score_cache[(task, seed)]
        taus = [float(np.quantile(scores, q)) for q in QS]
        nsel = [int((scores >= t).sum()) for t in taus]
        kept_unsafe_full = [float(unsafe[scores >= t].mean()) for t in taus]
        kept_frac_full = [float((scores >= t).mean()) for t in taus]
        tau_fb = float(np.quantile(scores, 0.80))
        fb_i = QS.index(0.80)
        seed_out = {}
        for cal_n in CAL_NS:
            rng = np.random.default_rng(7)
            n_cert, n_viol = 0, 0
            for _ in range(N_DRAWS):
                cal_idx = rng.choice(len(scores), size=min(cal_n, len(scores)),
                                     replace=False)
                cs, cu = scores[cal_idx], unsafe[cal_idx]
                chosen, cert = None, False
                for qi, tau in enumerate(taus):
                    sel = cs >= tau
                    m, k = int(sel.sum()), int(cu[sel].sum())
                    p = hyper_pvalue(k, m, nsel[qi], ALPHA) if m > 0 else 1.0
                    if m > 0 and p <= DELTA:
                        chosen, cert = qi, True
                    else:
                        break
                if cert:
                    n_cert += 1
                    if kept_unsafe_full[chosen] > ALPHA:
                        n_viol += 1
            fc_lo, fc_hi = cp_interval(n_viol, N_DRAWS)
            seed_out[str(cal_n)] = {
                "n_draws": N_DRAWS, "cert_rate": n_cert / N_DRAWS,
                "false_cert_rate_uncond": n_viol / N_DRAWS,
                "false_cert_cp95": [fc_lo, fc_hi],
                "cond_viol_rate": (n_viol / n_cert) if n_cert else None,
                "cond_viol_cp95": list(cp_interval(n_viol, n_cert)) if n_cert else None,
            }
        task_out["seeds"][str(seed)] = seed_out
        print(f"[guar2000] {task} seed{seed} done", flush=True)
    gres[task] = task_out
    with open(f"{OUT}/bullet_guarantee_2000.json", "w") as f:
        json.dump(gres, f, indent=1)
print("guarantee 2000 done", flush=True)

print("BULLET GUARANTEE DONE", flush=True)
