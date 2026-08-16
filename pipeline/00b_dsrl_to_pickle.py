"""Convert a DSRL offline dataset (auto-downloaded hdf5) into the project's
list-of-trajectories pickle format (keys: observations, actions, rewards,
costs, seed). Usage: SAFETY_VLM_TASK=<task> python scripts/00b_dsrl_to_pickle.py
Prints trajectory-cost percentiles for config threshold entry.
"""
from __future__ import annotations
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    out_p = cfg["data_pickle"]
    if os.path.exists(out_p):
        print(f"exists: {out_p}")
    else:
        import gymnasium as gym
        try:
            import dsrl
            if hasattr(dsrl, "register_envs"):
                dsrl.register_envs()
        except Exception:
            pass
        try:
            import bullet_safety_gym  # noqa: F401
        except Exception:
            pass
        name = cfg.get("offline_env_name") or cfg["env_name"].replace("Safety", "Offline", 1)
        print(f"Loading {name} ...", flush=True)
        env = gym.make(name)
        d = env.get_dataset()
        env.close()
        obs = np.asarray(d["observations"], dtype=np.float32)
        act = np.asarray(d["actions"], dtype=np.float32)
        rew = np.asarray(d["rewards"], dtype=np.float32)
        cost = np.asarray(d["costs"], dtype=np.float32)
        term = np.asarray(d["terminals"], dtype=bool)
        if "timeouts" in d:
            end = np.logical_or(term, np.asarray(d["timeouts"], dtype=bool))
        else:
            # Mirror-hosted files omit timeouts; episodes are fixed-length
            # (1000 steps), so mark an end every 1000 steps as fallback.
            end = term.copy()
            end[999::1000] = True
        if not end[-1]:
            end[-1] = True
        ends = np.where(end)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        trajs = []
        for s, e in zip(starts, ends):
            sl = slice(s, e + 1)
            if e + 1 - s < 2:
                continue
            trajs.append({"observations": obs[sl], "actions": act[sl],
                          "rewards": rew[sl], "costs": cost[sl], "seed": None})
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump(trajs, f)
        print(f"Saved {len(trajs)} trajectories -> {out_p}", flush=True)

    with open(out_p, "rb") as f:
        trajs = pickle.load(f)
    tc = np.array([float(np.sum(t["costs"])) for t in trajs])
    print(f"traj cost percentiles: p25={np.percentile(tc,25):.1f} "
          f"p50={np.percentile(tc,50):.1f} p75={np.percentile(tc,75):.1f} "
          f"max={tc.max():.1f} | n={len(trajs)}", flush=True)


if __name__ == "__main__":
    main()
