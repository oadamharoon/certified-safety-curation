"""Step 04e: Offline Safe RL baseline — BCQ-Lagrangian on `active_segments`.

Apples-to-apples comparison to BC+VLM-CPL and BC+GT-CPL: same offline data,
different method. BCQ-Lag is an offline-safe-RL algorithm (BCQ + Lagrangian
constraint) that can ingest a fixed transition dataset and learn a safe
policy without further env interaction — unlike PPO-Lag which is on-policy.

Pipeline:
  1. Load outputs/{task}/active_segments.pkl
  2. Flatten 30-step segments → per-transition arrays
  3. Save as a DSRL-format .npz with keys (obs, action, reward, cost, next_obs, done)
  4. Train omnisafe BCQLag with `dataset=<path-to-npz>`
  5. Save actor checkpoint and run the same 4-metric evaluation as 05_evaluate.py

Usage:
    MUJOCO_GL=egl PYTHONNOUSERSITE=1 python scripts/04e_train_bcq_lag.py --task pointgoal1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402


def segments_to_npz(segments, out_path: str) -> int:
    """Flatten 30-step segments into per-transition arrays in omnisafe's
    expected schema, save as .npz, return the number of transitions written.
    """
    obs_l, act_l, rew_l, cost_l, next_obs_l, done_l = [], [], [], [], [], []
    for seg in segments:
        obs  = seg["observations"]
        act  = seg["actions"]
        rew  = seg["rewards"]
        cost = seg["costs"]
        n = len(obs)
        for t in range(n):
            obs_l.append(obs[t])
            act_l.append(act[t])
            rew_l.append(rew[t])
            cost_l.append(cost[t])
            # Segments are length-30 slices of longer trajectories. The "next"
            # for steps 0..n-2 is the next step inside the segment. For the
            # last step we use the same obs as a soft terminal — fine because
            # `done=True` masks the Q-target bootstrap on that transition.
            if t < n - 1:
                next_obs_l.append(obs[t + 1])
                done_l.append(False)
            else:
                next_obs_l.append(obs[t])
                done_l.append(True)

    data = {
        "obs":      np.asarray(obs_l,      dtype=np.float32),
        "action":   np.asarray(act_l,      dtype=np.float32),
        "reward":   np.asarray(rew_l,      dtype=np.float32),
        "cost":     np.asarray(cost_l,     dtype=np.float32),
        "next_obs": np.asarray(next_obs_l, dtype=np.float32),
        "done":     np.asarray(done_l,     dtype=bool),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **data)
    print(f"[04e] flattened {len(segments)} segments → {data['obs'].shape[0]} transitions → {out_path}")
    return data["obs"].shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    ap.add_argument("--total_steps", type=int, default=1_000_000)
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── 1. Flatten active_segments → DSRL-style .npz ─────────────────────────
    seg_path = os.path.join(cfg["output_dir"], "active_segments.pkl")
    npz_path = os.path.join(cfg["output_dir"], "active_segments_offline.npz")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"{seg_path} missing — run 01_segment_and_filter.py first.")
    with open(seg_path, "rb") as f:
        segments = pickle.load(f)
    segments_to_npz(segments, npz_path)

    # ── 2. Train BCQ-Lag via omnisafe ────────────────────────────────────────
    import omnisafe

    # omnisafe's offline registry uses the base safety-gymnasium env names
    # (e.g. "SafetyPointGoal1-v0", not the DSRL "Gymnasium" suffix). Strip it.
    env_name = cfg["env_name"].replace("Gymnasium-v0", "-v0")

    custom_cfgs = {
        "train_cfgs": {
            "device":         "cuda" if torch.cuda.is_available() else "cpu",
            "total_steps":    args.total_steps,
            "dataset":        npz_path,
            "parallel":       1,
            "vector_env_nums": 1,
        },
        "lagrange_cfgs": {
            "cost_limit": float(cfg["cost_limit"]),
        },
        "logger_cfgs": {
            "log_dir":   os.path.join(cfg["output_dir"], "bcq_lag_logs"),
            "use_wandb": False,
        },
        "seed": int(cfg["seed"]),
    }

    print(f"[04e] env={env_name}  total_steps={args.total_steps:,}  cost_limit={cfg['cost_limit']}")
    agent = omnisafe.Agent("BCQLag", env_name, custom_cfgs=custom_cfgs)
    agent.learn()

    # ── 3. Export actor ──────────────────────────────────────────────────────
    out_path = os.path.join(cfg["output_dir"], "bcq_lag_policy.pt")
    torch.save({
        "actor_state": agent.agent.actor.state_dict(),
        "actor_type":  "omnisafe_bcqlag",
        "env_name":    env_name,
        "hidden_dim":  cfg["hidden_dim"],
    }, out_path)
    print(f"[04e] saved actor → {out_path}")

    # ── 4. Evaluate with the same 4 metrics as 05_evaluate.py ────────────────
    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass

    env = gym.make(cfg["env_name"])
    act_low  = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    actor = agent.agent.actor
    actor.eval()
    rewards, costs = [], []
    for ep in range(cfg["eval_episodes"]):
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        ep_rew = ep_cost = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor.predict(obs_t, deterministic=True).squeeze(0).cpu().numpy()
            action = np.clip(action, act_low, act_high)
            obs, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            ep_rew  += float(r)
            ep_cost += float(info.get("cost", 0.0))
        rewards.append(ep_rew)
        costs.append(ep_cost)
        print(f"Ep {ep:02d}: R={ep_rew:6.2f}  C={ep_cost:6.2f}")
    env.close()

    cost_limit = float(cfg["cost_limit"])
    avg_r = float(np.mean(rewards))
    avg_c = float(np.mean(costs))
    results = {
        "avg_reward":                avg_r,
        "avg_cost":                  avg_c,
        "normalized_cost":           avg_c / cost_limit,
        "constraint_violation_rate": float(np.mean(np.array(costs) > cost_limit)),
        "cost_limit":                cost_limit,
        "eval_episodes":             int(cfg["eval_episodes"]),
        "baseline":                  "BCQ-Lag (offline safe RL, active_segments)",
    }
    print(
        f"\nAvg Reward: {avg_r:.2f} | Avg Cost: {avg_c:.2f} | "
        f"Normalized Cost: {results['normalized_cost']:.3f} | "
        f"Violation > {cost_limit:g}: {results['constraint_violation_rate']:.2%}"
    )

    out_eval = os.path.join(cfg["output_dir"], "eval_results_bcq_lag.json")
    with open(out_eval, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_eval}")


if __name__ == "__main__":
    main()
