"""Step 05: roll out the trained policy in the *online* SafetyGym env.

Reports 4 evaluation metrics (universal across safety-gym, MetaDrive, etc.):
    1. avg_reward                — task performance
    2. avg_cost                  — raw cumulative episode cost
    3. normalized_cost           — avg_cost / cost_limit  (cross-env comparable)
    4. constraint_violation_rate — fraction of episodes with ep_cost > cost_limit
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    ap.add_argument("--policy_file", default="cpl_policy.pt",
                    help="Policy checkpoint filename in output_dir (default: cpl_policy.pt)")
    ap.add_argument("--random", action="store_true",
                    help="Evaluate a uniform-random policy (no checkpoint loaded)")
    ap.add_argument("--results_suffix", default="",
                    help="Suffix for eval_results_<suffix>.json (default: empty → eval_results.json)")
    args, _ = ap.parse_known_args()  # allow --task to pass through
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = init_wandb(cfg, job_type="eval")

    policy = None
    if not args.random:
        policy_path = os.path.join(cfg["output_dir"], args.policy_file)
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found at {policy_path}. Train first.")
        ckpt = torch.load(policy_path, map_location=device, weights_only=False)
        policy = GaussianPolicy(
            ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden_dim"],
            dropout=ckpt.get("dropout", 0.0),
        ).to(device)
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        print(f"Loaded policy: {policy_path}")
    else:
        print("Random-policy evaluation (no checkpoint loaded)")

    # Use online env for evaluation (the offline id may not support .step()).
    if cfg.get("data_source") == "metadrive":
        from utils.metadrive_utils import make_metadrive_env
        env = make_metadrive_env(
            cfg["metadrive_task"], for_render=False,
            nonterminal=bool(cfg.get("metadrive_nonterminal", False)),
        )
    else:
        import gymnasium as gym
        try:
            import dsrl
            if hasattr(dsrl, "register_envs"):
                dsrl.register_envs()
        except Exception:
            pass
        try:
            import bullet_safety_gym  # noqa: F401  (registers Bullet Safety envs)
        except Exception:
            pass
        env = gym.make(cfg["env_name"])

    rewards, costs, lengths = [], [], []
    for ep in range(cfg["eval_episodes"]):
        ep_seed = cfg["seed"] + ep
        obs, _ = env.reset(seed=ep_seed)
        if args.random:
            # action_space has its own RNG, separate from env.reset() — seed it
            # explicitly so the random baseline is reproducible.
            env.action_space.seed(ep_seed)
        ep_rew = ep_cost = 0.0
        steps = 0
        done = False
        while not done:
            if args.random:
                action = env.action_space.sample()
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    mean, std = policy(obs_t)
                    if cfg["deterministic"]:
                        action = mean.cpu().numpy()
                    else:
                        action = torch.normal(mean, std).cpu().numpy()
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            ep_rew += float(reward)
            ep_cost += float(info.get("cost", 0.0))
            steps += 1
        rewards.append(ep_rew)
        costs.append(ep_cost)
        lengths.append(steps)
        print(f"Ep {ep:02d}: R={ep_rew:6.2f}  C={ep_cost:6.2f}  len={steps}")
    env.close()

    avg_r = float(np.mean(rewards))
    avg_c = float(np.mean(costs))
    cost_limit = float(cfg["cost_limit"])
    norm_c = avg_c / cost_limit
    viol = float(np.mean(np.array(costs) > cost_limit))
    results = {
        "avg_reward": avg_r,
        "avg_cost": avg_c,
        "normalized_cost": norm_c,
        "constraint_violation_rate": viol,
        "cost_limit": cost_limit,
        "eval_episodes": cfg["eval_episodes"],
    }
    print(
        f"\nAvg Reward: {avg_r:.2f} | Avg Cost: {avg_c:.2f} | "
        f"Normalized Cost: {norm_c:.3f} | "
        f"Violation > {cost_limit:g}: {viol:.2%}"
    )

    suffix = f"_{args.results_suffix}" if args.results_suffix else ""
    out_path = os.path.join(cfg["output_dir"], f"eval_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if run is not None:
        import wandb
        wandb.log({f"eval/{k}": v for k, v in results.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
