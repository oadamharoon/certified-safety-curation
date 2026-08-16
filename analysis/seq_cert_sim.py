"""Sequential certification feasibility sim on archived scores."""
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
         "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
ALPHA, DELTA, REPS = 0.25, 0.1, 2000
LOOKS = [25, 50, 75, 100, 150, 200, 300, 400]
SPEND = [DELTA * 2 ** -(k + 1) for k in range(len(LOOKS))]  # sums < delta
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def hp(k, m, n_sel, a):
    ks = int(a * n_sel) + 1
    return 1.0 if ks > n_sel else float(hypergeom.cdf(k, n_sel, ks, m))


out = {}
for task, lim in TASKS.items():
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > lim).astype(int)
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
    # use seed-0 style: run per seed
        taus = [float(np.quantile(g, q)) for q in QS]
        nsel = [int((g >= t).sum()) for t in taus]
        ku_full = [float(unsafe[g >= t].mean()) for t in taus]
        rng = np.random.default_rng(99)
        stops, false_certs, cert_count = [], 0, 0
        for _ in range(REPS):
            perm = rng.permutation(len(g))
            done = False
            for look, dk in zip(LOOKS, SPEND):
                cal = perm[:look]
                cs, cu = g[cal], unsafe[cal]
                pick = None
                for qi, tau in enumerate(taus):
                    sel = cs >= tau
                    m, kk = int(sel.sum()), int(cu[sel].sum())
                    p = hp(kk, m, nsel[qi], ALPHA) if m > 0 else 1.0
                    if m > 0 and p <= dk:
                        pick = qi
                    else:
                        break
                if pick is not None:
                    cert_count += 1
                    stops.append(look)
                    if ku_full[pick] > ALPHA:
                        false_certs += 1
                    done = True
                    break
            if not done:
                stops.append(-1)
        certified_stops = [s for s in stops if s > 0]
        task_out[str(seed)] = {
            "cert_rate_by_400": cert_count / REPS,
            "false_cert_rate": false_certs / REPS,
            "median_labels_at_cert": (float(np.median(certified_stops))
                                      if certified_stops else None),
            "mean_labels_at_cert": (float(np.mean(certified_stops))
                                    if certified_stops else None),
        }
        print(f"{task} s{seed}: cert@400={cert_count/REPS:.2f} "
              f"false={false_certs/REPS:.4f} "
              f"median_n={task_out[str(seed)]['median_labels_at_cert']}", flush=True)
    out[task] = task_out
json.dump(out, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/seq_cert_sim.json", "w"), indent=1)
print("SEQ CERT SIM DONE", flush=True)
