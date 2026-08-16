"""T2.1 do-not-filter, T2.3 MT procedures, T3.3 segment coverage — archived data only."""
import json, os, pickle, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
from scipy.stats import hypergeom
import yaml

torch.set_num_threads(8)
cfg = yaml.safe_load(open("config.yaml"))
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response"
ANALYSIS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
            "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
            "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
BULLET = {"ballrun_b": 10, "ballcircle_b": 10, "carcircle_b": 10,
          "carrun_b": 10, "dronerun_b": 10}
ALPHA, DELTA, NCAL, REPS = 0.25, 0.1, 200, 2000
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]

# ---------- T2.1: do-not-filter pre-test ----------
t21 = {}
for task, lim in {**ANALYSIS, **BULLET}.items():
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    N = len(costs)
    unsafe_frac = float((costs > lim).mean())
    T_max = max(len(t["costs"]) for t in trajs)
    rng = np.random.default_rng(5)
    fires_frac = fires_mean_apriori = fires_mean_plugin = 0
    serf = np.sqrt(np.log(1 / DELTA) * (1 - NCAL / N) / (2 * NCAL))
    for _ in range(REPS):
        cal = rng.choice(N, NCAL, replace=False)
        k, m = int((costs[cal] > lim).sum()), NCAL
        ks = int(ALPHA * N) + 1
        p = float(hypergeom.cdf(k, N, ks, m)) if ks <= N else 1.0
        fires_frac += p <= DELTA
        sm = costs[cal].mean()
        fires_mean_apriori += (sm + T_max * serf) <= lim
        fires_mean_plugin += (sm + (costs[cal].max() - costs[cal].min()) * serf) <= lim
    t21[task] = {"pool_unsafe_frac": unsafe_frac, "pool_mean_cost": float(costs.mean()),
                 "budget": lim, "T_max": T_max,
                 "frac_test_rate": fires_frac / REPS,
                 "mean_test_apriori_rate": fires_mean_apriori / REPS,
                 "mean_test_plugin_rate": fires_mean_plugin / REPS}
    print(f"T2.1 {task}: unsafe={unsafe_frac:.2f} mean={costs.mean():.1f}/{lim} "
          f"frac={fires_frac/REPS:.2f} meanA={fires_mean_apriori/REPS:.2f} "
          f"meanP={fires_mean_plugin/REPS:.2f}", flush=True)
json.dump(t21, open(f"{OUT}/donotfilter.json", "w"), indent=1)

# ---------- T3.3: segment coverage / null space ----------
t33 = {}
for task in ANALYSIS:
    segp = f"outputs/{task}/active_segments.pkl"
    if not os.path.exists(segp):
        print(f"T3.3 {task}: no segments file", flush=True)
        continue
    segs = pickle.load(open(segp, "rb"))
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    total_states = sum(len(t["observations"]) for t in trajs)
    slots = set()
    n_slots = 0
    for s in segs:
        ti, st = s.get("traj_idx", s.get("traj_index", -1)), s.get("start", s.get("start_idx", 0))
        L = len(s.get("costs", [])) or 30
        n_slots += L
        for u in range(st, st + L):
            slots.add((ti, u))
    covered = len(slots)
    overlap = 1 - covered / max(n_slots, 1)
    t33[task] = {"n_segments": len(segs), "states_covered": covered,
                 "total_pool_states": total_states,
                 "coverage_frac": covered / total_states,
                 "within_segment_overlap": overlap}
    print(f"T3.3 {task}: segs={len(segs)} covered={covered}/{total_states} "
          f"({covered/total_states:.3f}) overlap={overlap:.3f}", flush=True)
json.dump(t33, open(f"{OUT}/segment_coverage.json", "w"), indent=1)

# ---------- T2.3: MT procedures ----------
t23 = {}
for task, lim in ANALYSIS.items():
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > lim).astype(float)
    obs_dim = trajs[0]["observations"].shape[1]
    task_out = {}
    for seed in (0, 1, 2):
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{seed}.pt",
                        map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, t in enumerate(trajs):
                o = torch.as_tensor(t["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j+8192])
                                  for j in range(0, len(o), 8192)]).mean().item()
        taus = [float(np.quantile(g, q)) for q in QS]
        J = len(taus)
        Ns = [int((g >= t).sum()) for t in taus]
        ku = [float(unsafe[g >= t].mean()) for t in taus]
        rng = np.random.default_rng(11)
        stats = {p: {"cert": 0, "false": 0} for p in ("fixedseq", "bonferroni", "holm")}
        for _ in range(REPS):
            cal = rng.choice(len(g), NCAL, replace=False)
            cs, cu = g[cal], unsafe[cal]
            pvals = []
            for j, tau in enumerate(taus):
                sel = cs >= tau
                m, k = int(sel.sum()), int(cu[sel].sum())
                ks = int(ALPHA * Ns[j]) + 1
                pvals.append(float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0)
            # fixed sequence: deepest prefix all <= delta
            pick = -1
            for j in range(J):
                if pvals[j] <= DELTA:
                    pick = j
                else:
                    break
            if pick >= 0:
                stats["fixedseq"]["cert"] += 1
                stats["fixedseq"]["false"] += ku[pick] > ALPHA
            # bonferroni: deepest j with p <= delta/J
            bidx = [j for j in range(J) if pvals[j] <= DELTA / J]
            if bidx:
                stats["bonferroni"]["cert"] += 1
                stats["bonferroni"]["false"] += ku[max(bidx)] > ALPHA
            # holm step-down on sorted pvals
            order = np.argsort(pvals)
            rej = set()
            for r, j in enumerate(order):
                if pvals[j] <= DELTA / (J - r):
                    rej.add(j)
                else:
                    break
            if rej:
                stats["holm"]["cert"] += 1
                stats["holm"]["false"] += ku[max(rej)] > ALPHA
        task_out[str(seed)] = {p: {"cert_rate": v["cert"] / REPS,
                                   "false_rate": v["false"] / REPS,
                                   "cond": (v["false"] / v["cert"]) if v["cert"] else None}
                               for p, v in stats.items()}
        print(f"T2.3 {task} s{seed}: " + " ".join(
            f"{p}={task_out[str(seed)][p]['cert_rate']:.2f}/{task_out[str(seed)][p]['false_rate']:.3f}"
            for p in stats), flush=True)
    t23[task] = task_out
json.dump(t23, open(f"{OUT}/mt_procedures.json", "w"), indent=1)
print("FREE TIER DONE", flush=True)
