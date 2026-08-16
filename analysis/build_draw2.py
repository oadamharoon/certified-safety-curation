"""Build second certified selections (fresh calibration draw) for the
selection-draw variance experiment. Writes OSRL-format subset hdf5 files
named *_draw2_* so the harvester keys them as cdt_cert_draw2.
"""
import os, pickle, sys
import numpy as np
import torch
import h5py
from scipy.stats import hypergeom

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certified_h5")
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)
cfg = yaml.safe_load(open("config.yaml"))

TASKS = [
    ("halfcheetah_velocity", 1, 20),
    ("cargoal1_dsrl", 3, 25),
    ("pointgoal1_dsrl", 0, 25),
]
ALPHA, DELTA, CAL_N = 0.25, 0.1, 200
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def hyper_pvalue(k, m, n_sel, alpha):
    k_star = int(alpha * n_sel) + 1
    if k_star > n_sel:
        return 1.0
    return float(hypergeom.cdf(k, n_sel, k_star, m))


for task, vseed, limit in TASKS:
    tc = cfg["tasks"][task]
    with open(tc["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > limit).astype(int)
    obs_dim = trajs[0]["observations"].shape[1]
    ens = VEnsemble(obs_dim, 256, K=3)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vseed}.pt",
                    map_location="cpu", weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()
    scores = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32)
            scores[i] = torch.cat([ens(o[j:j+8192]) for j in range(0, len(o), 8192)]).mean().item()
    taus = [float(np.quantile(scores, q)) for q in QS]
    nsel = [int((scores >= t).sum()) for t in taus]

    tau_star, redraws = None, 0
    for k in range(200):
        rng = np.random.default_rng(9000 + 37 * k)
        cal = rng.choice(len(scores), size=CAL_N, replace=False)
        cs, cu = scores[cal], unsafe[cal]
        chosen = None
        for qi, tau in enumerate(taus):
            sel = cs >= tau
            m, kk = int(sel.sum()), int(cu[sel].sum())
            p = hyper_pvalue(kk, m, nsel[qi], ALPHA) if m > 0 else 1.0
            if m > 0 and p <= DELTA:
                chosen = qi
            else:
                break
        if chosen is not None:
            tau_star, redraws = taus[chosen], k
            break
    assert tau_star is not None, f"{task}: no certified draw in 200 tries"
    kept = scores >= tau_star
    print(f"{task}: certified on draw {redraws}, kept {kept.sum()}/{len(trajs)} "
          f"({kept.mean():.3f}), true kept-unsafe {unsafe[kept].mean():.3f}", flush=True)

    # rebuild spans from the DSRL dataset with the converter's exact logic
    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass
    name = tc.get("offline_env_name") or tc["env_name"].replace("Safety", "Offline", 1)
    env = gym.make(name)
    d = env.get_dataset()
    env.close()
    term = np.asarray(d["terminals"], dtype=bool)
    if "timeouts" in d:
        end = np.logical_or(term, np.asarray(d["timeouts"], dtype=bool))
        tout = np.asarray(d["timeouts"], dtype=bool)
    else:
        end = term.copy(); end[999::1000] = True
        tout = np.zeros_like(term); tout[999::1000] = True
    if not end[-1]:
        end[-1] = True
    ends = np.where(end)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    spans = [(s, e) for s, e in zip(starts, ends) if e + 1 - s >= 2]
    assert len(spans) == len(trajs), f"{task}: span/pickle mismatch {len(spans)} vs {len(trajs)}"

    idx = np.concatenate([np.arange(s, e + 1) for (s, e), keep
                          in zip(spans, kept) if keep])
    outp = os.path.join(OUT, f"{task}_draw2_seed{vseed}.hdf5")
    with h5py.File(outp, "w") as f:
        keys = ["observations", "actions", "rewards", "costs"]
        if "next_observations" in d:
            keys.append("next_observations")
        for key in keys:
            f.create_dataset(key, data=np.asarray(d[key])[idx])
        if "next_observations" not in d:
            obs_all = np.asarray(d["observations"])
            nxt = obs_all.copy()
            nxt[:-1] = obs_all[1:]
            for s_, e_ in spans:
                nxt[e_] = obs_all[e_]
            f.create_dataset("next_observations", data=nxt[idx])
        f.create_dataset("terminals", data=term[idx])
        f.create_dataset("timeouts", data=tout[idx])
    print(f"  wrote {outp} ({len(idx)} transitions)", flush=True)
print("ALL DRAW2 SUBSETS BUILT", flush=True)
