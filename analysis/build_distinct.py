"""Build the missing distinct certified selections (top-3-probability
thresholds per task/level, minus those already run). Tags: {task}_{lvl}selq{QQ}.
Writes subset h5 (for CDT) + kept json (for BC)."""
import json, os, pickle, sys
import numpy as np
import torch
import h5py

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certified_h5")
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

torch.set_num_threads(8)
cfg = yaml.safe_load(open("config.yaml"))

# (task, vseed, limit, level_tag, quantiles of the MISSING distinct selections)
# existing distinct selections established from prior draws are excluded.
JOBS = [
    ("halfcheetah_velocity", "1", 20, "a25", [0.75]),        # have q85(d1=d3), q80(d2)
    ("cargoal1_dsrl", "3", 25, "a25", [0.75]),               # have q85-ish(d1=d3), q80(d2)
    ("walker2d_velocity", "0", 20, "a40", [0.55]),           # have q60(x2 dup), q65-ish -> add next-most-probable
    ("ant_velocity", "0", 20, "a40", [0.85]),                # have q80/q85 region duplicates
    ("swimmer_velocity", "1", 20, "a40", [0.80, 0.75]),      # have q85 only (x3)
    ("pointgoal1_dsrl", "0", 25, "a40", [0.35]),             # have q30(x2 dup) + d1
    ("pointgoal2", "1", 25, "a40", [0.80, 0.75]),            # have q85 only (x3)
]
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]

for task, vs, lim, lvl, quants in JOBS:
    tc = cfg["tasks"][task]
    trajs = pickle.load(open(tc["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > lim).astype(int)
    obs_dim = trajs[0]["observations"].shape[1]
    ens = VEnsemble(obs_dim, 256, K=3)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt",
                    map_location="cpu", weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32)
            g[i] = torch.cat([ens(o[j:j+8192]) for j in range(0, len(o), 8192)]).mean().item()

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
    tout = np.asarray(d["timeouts"], dtype=bool)
    end = np.logical_or(term, tout)
    if not end[-1]:
        end[-1] = True
    ends = np.where(end)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    spans = [(s, e) for s, e in zip(starts, ends) if e + 1 - s >= 2]
    assert len(spans) == len(trajs), task

    for q in quants:
        tau = float(np.quantile(g, q))
        kept = g >= tau
        ku = float(unsafe[kept].mean())
        tag = f"{task}_{lvl}selq{int(q*100)}_seed{vs}"
        idx = np.concatenate([np.arange(s, e + 1) for (s, e), keep
                              in zip(spans, kept) if keep])
        with h5py.File(os.path.join(OUT, f"{tag}.hdf5"), "w") as f:
            keys = ["observations", "actions", "rewards", "costs"]
            if "next_observations" in d:
                keys.append("next_observations")
            for key in keys:
                f.create_dataset(key, data=np.asarray(d[key])[idx])
            f.create_dataset("terminals", data=term[idx])
            f.create_dataset("timeouts", data=tout[idx])
        with open(os.path.join(OUT, f"{tag}_kept.json"), "w") as jf:
            json.dump({"kept": np.where(kept)[0].tolist(), "quantile": q,
                       "true_kept_unsafe": ku}, jf)
        print(f"{tag}: kept {int(kept.sum())}/{len(trajs)} ({kept.mean():.3f}) "
              f"true_unsafe={ku:.3f}", flush=True)
print("ALL DISTINCT SELECTIONS BUILT", flush=True)
