"""T2.4: direct LTT certificate on the operator's weighted unsafe mass."""
import json, os, pickle, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
from scipy.stats import hypergeom
import yaml

torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
TASKS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
         "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
         "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25,
         "ballrun_b": 10, "ballcircle_b": 10, "carcircle_b": 10, "carrun_b": 10, "dronerun_b": 10}
ALPHA, DELTA, NCAL, REPS, CLIP = 0.25, 0.1, 200, 2000, 2.0
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def hoeff_wor(n, N, delta):
    """One-sided Hoeffding bound for sampling without replacement, range [0,1]."""
    if n >= N:
        return 0.0
    return float(np.sqrt(np.log(1.0 / delta) * (1.0 - (n - 1.0) / N) / (2.0 * n)))


out = {}
for task, lim in TASKS.items():
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    R = np.array([float(np.sum(t["rewards"])) for t in trajs])
    unsafe = (costs > lim).astype(float)
    obs_dim = trajs[0]["observations"].shape[1]
    task_out = {}
    for seed in (0, 1, 2):
        fp = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
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
                g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0, len(o), 8192)]).mean().item()
        taus = [float(np.quantile(g, q)) for q in QS]
        Ns = [int((g >= t).sum()) for t in taus]
        # true weighted risk per threshold (weights standardized WITHIN selection)
        Wtrue = []
        for t in taus:
            sel = g >= t
            Rs = R[sel]
            z = (Rs - Rs.mean()) / (Rs.std() + 1e-8)
            w = np.exp(np.clip(z, -CLIP, CLIP))
            Wtrue.append(float((w * unsafe[sel]).sum() / w.sum()))
        rng = np.random.default_rng(31 + seed)
        n_cert = n_false = 0
        n_cert_comp = 0
        for _ in range(REPS):
            cal = rng.choice(len(g), NCAL, replace=False)
            cs, cu, cr = g[cal], unsafe[cal], R[cal]
            pick = pick_comp = None
            for j, t in enumerate(taus):
                sel = cs >= t
                m = int(sel.sum())
                if m == 0:
                    break
                # composition certificate (reference)
                k = int(cu[sel].sum())
                ks = int(ALPHA * Ns[j]) + 1
                p = float(hypergeom.cdf(k, Ns[j], ks, m)) if ks <= Ns[j] else 1.0
                if p <= DELTA and (pick_comp is None or j > pick_comp):
                    if pick_comp is None or j == pick_comp + 1 or pick_comp == j - 1:
                        pick_comp = j
                # weighted-risk certificate. Weights are a function of returns and
                # the selection, both known for the whole pool, so the denominator
                # sum_{i in S} w_i is EXACT; only the numerator needs estimating.
                selfull = g >= t
                Rfull = R[selfull]
                zf = (Rfull - Rfull.mean()) / (Rfull.std() + 1e-8)
                wfull = np.exp(np.clip(zf, -CLIP, CLIP))
                Wsum = float(wfull.sum())                       # exact
                # calibration points carry the same (pool-level) weights
                idx_in = cal[sel]
                pos = np.searchsorted(np.where(selfull)[0], idx_in)
                wcal = wfull[np.clip(pos, 0, len(wfull) - 1)]
                # v_i = w_i(u_i - alpha) in [-alpha*wmax, (1-alpha)*wmax]
                v = wcal * (cu[sel] - ALPHA)
                rng_v = np.exp(CLIP)                            # conservative range scale
                vbar = float(v.mean())
                eps = hoeff_wor(m, Ns[j], DELTA) * rng_v
                # certify if the population sum is provably <= 0
                if (vbar + eps) * Ns[j] <= 0:
                    pick = j
                else:
                    break
            if pick is not None:
                n_cert += 1
                if Wtrue[pick] > ALPHA:
                    n_false += 1
        task_out[str(seed)] = {"weighted_cert_rate": n_cert / REPS,
                               "weighted_false_rate": n_false / REPS,
                               "true_weighted_risk_top": Wtrue[0]}
        print(f"{task} s{seed}: wcert={n_cert/REPS:.3f} wfalse={n_false/REPS:.4f} "
              f"Wtrue(top)={Wtrue[0]:.3f}", flush=True)
    out[task] = task_out
json.dump(out, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/weighted_cert.json", "w"), indent=1)
print("WEIGHTED CERT DONE", flush=True)
