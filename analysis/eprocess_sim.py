"""E-process sequential certification: betting martingale, without-replacement."""
import json, os, pickle, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
import yaml

torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
TASKS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
         "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
         "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
ALPHA, DELTA, REPS, CAP = 0.25, 0.1, 2000, 400
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
THRESH = 1.0 / DELTA

out = {}
for task, lim in TASKS.items():
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    unsafe = np.array([float(np.sum(t["costs"])) > lim for t in trajs], dtype=float)
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
        Ns = np.array([int((g >= t).sum()) for t in taus])
        Kstar = (np.floor(ALPHA * Ns) + 1).astype(float)
        ku_true = np.array([unsafe[g >= t].mean() for t in taus])
        rng = np.random.default_rng(1234)
        perms = np.stack([rng.permutation(len(g))[:CAP] for _ in range(REPS)])
        gsel = g[perms]                       # (REPS, CAP)
        usel = unsafe[perms]                  # (REPS, CAP)
        W = np.ones((REPS, J))
        icount = np.zeros((REPS, J))
        scount = np.zeros((REPS, J))
        crossed = np.zeros((REPS, J), dtype=bool)
        t_cert = np.full(REPS, -1)
        depth = np.zeros(REPS, dtype=int)
        active = np.ones(REPS, dtype=bool)
        for t in range(CAP):
            gt, ut = gsel[:, t], usel[:, t]
            for j in range(J):
                inS = (gt >= taus[j]) & active
                if not inS.any():
                    continue
                rem = Ns[j] - icount[:, j]
                mu = np.clip((Kstar[j] - scount[:, j]) / np.maximum(rem, 1), 1e-6, 1 - 1e-6)
                phat = (scount[:, j] + 0.5) / (icount[:, j] + 1.0)
                lam = np.clip((mu - phat) / (mu * (1 - mu) + 1e-4),
                              0.0, 0.5 / (1 - mu))
                mult = 1 + lam * (mu - ut)
                W[:, j] = np.where(inS, W[:, j] * mult, W[:, j])
                icount[:, j] += inS
                scount[:, j] += inS * ut
                crossed[:, j] |= (W[:, j] >= THRESH)
            newly = active & crossed[:, 0]
            if newly.any():
                t_cert[newly] = t + 1
                d = np.ones(newly.sum(), dtype=int)
                cn = crossed[newly]
                for j in range(1, J):
                    stillok = cn[:, :j + 1].all(axis=1)
                    d = np.where(stillok, j + 1, d)
                depth[newly] = d
                active[newly] = False
            if not active.any():
                break
        certified = t_cert > 0
        false_cert = certified & (ku_true[np.maximum(depth - 1, 0)] > ALPHA)
        stops = t_cert[certified]
        task_out[str(seed)] = {
            "cert_rate_by_400": float(certified.mean()),
            "false_cert_rate": float(false_cert.mean()),
            "median_labels_at_cert": float(np.median(stops)) if certified.any() else None,
            "mean_labels_at_cert": float(np.mean(stops)) if certified.any() else None,
        }
        print(f"{task} s{seed}: cert@400={certified.mean():.2f} "
              f"false={false_cert.mean():.4f} "
              f"median_n={task_out[str(seed)]['median_labels_at_cert']}", flush=True)
    out[task] = task_out
json.dump(out, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/eprocess_sim.json", "w"), indent=1)
print("EPROCESS SIM DONE", flush=True)
