"""Step 04d: Online Safe RL baseline — PPO-Lagrangian (omnisafe).

Trains from scratch with full online interaction and the ground-truth
cost signal. This is the conventional safe-RL reference point against
our offline annotation-free approach.

Saves the trained actor's omnisafe state_dict + metadata to
``ppo_lag_policy.pt`` and writes its evaluation under
``eval_results_ppo_lag.json`` in the same 4-metric format as the
main eval script.

Usage:
    MUJOCO_GL=egl PYTHONNOUSERSITE=1 python scripts/04d_train_ppo_lag.py --task pointgoal1
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    ap.add_argument("--total_steps", type=int, default=1_000_000)
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    import omnisafe

    custom_cfgs = {
        "train_cfgs": {
            "total_steps": args.total_steps,
            "parallel":    1,
        },
        "algo_cfgs": {
            "steps_per_epoch": 4096,
            "update_iters":    10,
        },
        "lagrange_cfgs": {
            "cost_limit": float(cfg["cost_limit"]),
        },
        "logger_cfgs": {
            "log_dir":   os.path.join(cfg["output_dir"], "ppo_lag_logs"),
            "use_wandb": False,
        },
        "seed": int(cfg["seed"]),
    }

    # omnisafe's registry uses the base safety-gymnasium names (e.g.
    # "SafetyPointGoal1-v0"); the DSRL-style "*Gymnasium-v0" suffix in
    # our config refers to the same underlying env but isn't in omnisafe's
    # `support_envs()` list.
    omnisafe_env_name = cfg["env_name"].replace("Gymnasium-v0", "-v0")
    print(f"[ppo_lag] env={omnisafe_env_name}  total_steps={args.total_steps:,}  "
          f"cost_limit={cfg['cost_limit']}")
    agent = omnisafe.Agent("PPOLag", omnisafe_env_name, custom_cfgs=custom_cfgs)
    agent.learn()

    # ── Export actor checkpoint ───────────────────────────────────────────────
    import gymnasium as gym
    env_probe = gym.make(cfg["env_name"])
    obs_dim  = env_probe.observation_space.shape[0]
    act_dim  = env_probe.action_space.shape[0]
    act_low  = env_probe.action_space.low.astype(np.float32)
    act_high = env_probe.action_space.high.astype(np.float32)
    env_probe.close()

    out_path = os.path.join(cfg["output_dir"], "ppo_lag_policy.pt")
    torch.save({
        "actor_state": agent.agent.actor.state_dict(),
        "actor_type":  "omnisafe_ppolag",
        "obs_dim":     obs_dim,
        "act_dim":     act_dim,
        "act_low":     act_low,
        "act_high":    act_high,
        "hidden_dim":  cfg["hidden_dim"],
    }, out_path)
    print(f"[ppo_lag] Saved actor -> {out_path}")

    # ── Evaluate with the same 4 metrics as 05_evaluate.py ────────────────────
    env = gym.make(cfg["env_name"])
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
        "baseline":                  "PPO-Lag (online safe RL)",
    }
    print(
        f"\nAvg Reward: {avg_r:.2f} | Avg Cost: {avg_c:.2f} | "
        f"Normalized Cost: {results['normalized_cost']:.3f} | "
        f"Violation > {cost_limit:g}: {results['constraint_violation_rate']:.2%}"
    )

    out_eval = os.path.join(cfg["output_dir"], "eval_results_ppo_lag.json")
    with open(out_eval, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_eval}")


if __name__ == "__main__":
    main()
