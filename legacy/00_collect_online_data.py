"""Data collection is handled in data_collect.ipynb, which collects
rollouts online with stored reset seeds.

This script is kept as a thin CLI wrapper that saves the same pickle
(``cfg['data_pickle']``) for use by the pipeline scripts without
needing to re-run the notebook.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402


def _make_env(cfg):
    if cfg.get("data_source") == "metadrive":
        from utils.metadrive_utils import make_metadrive_env
        return make_metadrive_env(
            cfg["metadrive_task"], for_render=False,
            nonterminal=bool(cfg.get("metadrive_nonterminal", False)),
            dsrl=bool(cfg.get("metadrive_dsrl", False)),
        )
    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass
    return gym.make(cfg["env_name"])


def _rollout_episode(
    env, bc_policy, bc_device, ep_seed: int, bc_noise: float,
    max_steps: int, md: bool, momentum: float, fwd_bias: float,
) -> dict:
    """One reproducible episode rollout. Returns a trajectory dict or None
    if the episode was too short to keep."""
    obs, _ = env.reset(seed=ep_seed)
    rng_action = np.random.default_rng(ep_seed + 1)
    obs_buf, act_buf, rew_buf, cost_buf = [], [], [], []
    prev_act = None
    act_low  = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)
    for _ in range(max_steps):
        if bc_policy is not None:
            import torch
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=bc_device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = bc_policy(obs_t)
            act = mean.squeeze(0).cpu().numpy().astype(np.float32)
            act = act + rng_action.normal(0.0, bc_noise, size=act.shape).astype(np.float32)
            act = np.clip(act, act_low, act_high)
        else:
            act = _sample_action(env, rng_action, prev_act, momentum, fwd_bias)
        next_obs, reward, term, trunc, info = env.step(act)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(act)
        rew_buf.append(float(reward))
        if md:
            # Two MetaDrive cost conventions:
            #   * DSRL/FSRL wrapper (our dsrl=True path): info["cost"] already
            #     includes per-step velocity_cost + proximity_cost on top of
            #     any crash/out_of_road single-step cost. Use it directly.
            #     Detected by the presence of "velocity_cost" in info.
            #   * Default MetaDriveEnv (legacy terminal / nonterminal paths):
            #     info["cost"] is just 0/1 from a single if/elif, but if the
            #     vehicle has simultaneous violations we want them all to
            #     count → sum the binary flags.
            if "velocity_cost" in info:
                cost_buf.append(float(info.get("cost", 0.0)))
            else:
                c = (float(info.get("out_of_road", 0.0))
                     + float(info.get("crash_vehicle", 0.0))
                     + float(info.get("crash_object", 0.0)))
                cost_buf.append(c)
        else:
            cost_buf.append(float(info.get("cost", 0.0)))
        obs = next_obs
        prev_act = act
        if term or trunc:
            break
    if len(obs_buf) < 2:
        return None
    return {
        "seed": ep_seed,
        "observations": np.stack(obs_buf),
        "actions": np.stack(act_buf),
        "rewards": np.array(rew_buf, dtype=np.float32),
        "costs": np.array(cost_buf, dtype=np.float32),
    }


def _calibrate_noise(
    env, bc_policy, bc_device, *, n_pilot: int, target_frac: tuple,
    target_cost: tuple, max_steps: int, md: bool, momentum: float, fwd_bias: float,
    sigma_lo: float = 0.05, sigma_hi: float = 0.60, max_iter: int = 5,
) -> float:
    """Bisect on bc_action_noise to hit a target cost distribution.

    target_frac : (lo, hi) — desired fraction of pilot episodes with cost > 0
    target_cost : (lo, hi) — desired MEDIAN cost AMONG violating episodes.
        Median (not mean) is used so the metric is robust to the
        catastrophic-tail problem: a single "stuck off-road" episode with
        cost=900 would shift the mean by hundreds but the median by very
        little — so we calibrate to the typical violation severity.
    """
    print(f"\n[calibrate] bisecting bc_action_noise on {n_pilot} pilot episodes per iteration", flush=True)
    print(f"  target: frac_violating in [{target_frac[0]:.2f}, {target_frac[1]:.2f}]  "
          f"and median_cost_when_violating in [{target_cost[0]:.0f}, {target_cost[1]:.0f}]", flush=True)
    lo, hi = sigma_lo, sigma_hi
    last_sigma = (lo + hi) / 2
    # Pilot seeds must stay inside MetaDrive's [start_seed, start_seed + num_scenarios)
    # window. Place pilot seeds in a band that doesn't overlap with main-collection
    # seeds (0..n_episodes-1) — and clamp so we never exceed 999 (num_scenarios=1000).
    # For metadrive: main uses 0..299, pilot uses 800..(800+max_iter*n_pilot-1) ≤ 874.
    PILOT_SEED_BASE = 800
    for it in range(max_iter):
        sigma = (lo + hi) / 2
        base = PILOT_SEED_BASE + it * n_pilot
        costs = []
        for k in range(n_pilot):
            traj = _rollout_episode(
                env, bc_policy, bc_device,
                ep_seed=base + k, bc_noise=sigma, max_steps=max_steps,
                md=md, momentum=momentum, fwd_bias=fwd_bias,
            )
            if traj is not None:
                costs.append(float(traj["costs"].sum()))
        costs = np.array(costs)
        frac = float((costs > 0).mean()) if len(costs) else 0.0
        viol = costs[costs > 0]
        median_viol = float(np.median(viol)) if len(viol) else 0.0
        p75_viol = float(np.percentile(viol, 75)) if len(viol) else 0.0
        max_cost = float(costs.max()) if len(costs) else 0.0
        print(f"  iter {it}: sigma={sigma:.3f}  frac_viol={frac:.2f}  "
              f"median_cost_viol={median_viol:.1f}  p75_cost_viol={p75_viol:.1f}  "
              f"max_cost={max_cost:.0f}", flush=True)
        last_sigma = sigma
        # Decide adjustment (median-based, robust to tail)
        if frac < target_frac[0]:
            lo = sigma                         # need more noise
        elif frac > target_frac[1]:
            hi = sigma                         # need less noise
        else:
            # fraction is in band; check median cost
            if median_viol > target_cost[1]:
                hi = sigma                     # typical violations too severe -> reduce
            elif median_viol > 0 and median_viol < target_cost[0]:
                lo = sigma                     # typical violations too small -> increase
            else:
                print(f"[calibrate] converged: sigma={sigma:.3f}", flush=True)
                return sigma
    print(f"[calibrate] did not converge after {max_iter} iters; using sigma={last_sigma:.3f}", flush=True)
    return last_sigma


def _sample_action(env, rng: np.random.Generator, prev_action: np.ndarray | None,
                   momentum: float, forward_bias: float) -> np.ndarray:
    """Mixed exploration policy:
       - random action sampled from action space via the passed rng
         (rng is seeded per-episode from seed_offset + ep_idx → reproducible)
       - blended with previous action (momentum) for smoother trajectories
       - small forward bias on the first action dim to encourage motion
    """
    low  = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    rand = rng.uniform(low, high).astype(np.float32)
    if prev_action is not None:
        rand = momentum * prev_action + (1.0 - momentum) * rand
    rand[0] += forward_bias  # bias toward "forward"
    return np.clip(rand, low, high).astype(np.float32)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()  # allow --task to pass through to load_cfg
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])

    coll = cfg.get("collect", {}) or {}
    n_episodes   = int(coll.get("n_episodes", cfg.get("online_episodes", 200)))
    max_steps    = int(coll.get("max_steps_per_episode", cfg.get("max_steps_per_episode", 1000)))
    momentum     = float(coll.get("action_momentum", 0.7))
    fwd_bias     = float(coll.get("forward_bias", 0.3))
    seed_offset  = int(coll.get("seed_offset", 100000))
    use_bc       = bool(coll.get("use_bc_policy", False))
    bc_noise     = float(coll.get("bc_action_noise", 0.1))

    env_name = cfg["env_name"]
    print(f"Collecting {n_episodes} episodes from {env_name} "
          f"(max {max_steps} steps each)")

    env = _make_env(cfg)

    # Optionally load the DSRL-distilled BC policy as the action source.
    bc_policy = None
    bc_device = None
    if use_bc:
        import torch
        from model.policy import GaussianPolicy
        bc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bc_path = os.path.join(cfg["output_dir"], "dsrl_bc_policy.pt")
        if not os.path.exists(bc_path):
            raise FileNotFoundError(
                f"BC policy not found at {bc_path}. "
                f"Run scripts/00a_train_bc_from_dsrl.py --task ... first."
            )
        ckpt = torch.load(bc_path, map_location=bc_device, weights_only=False)
        bc_policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden_dim"]).to(bc_device)
        bc_policy.load_state_dict(ckpt["state_dict"])
        bc_policy.eval()
        print(f"  action source: DSRL-BC policy ({bc_path})  noise_std={bc_noise}")
    else:
        print(f"  action source: random + forward_bias={fwd_bias}, momentum={momentum}")

    # MetaDrive enforces seed ∈ [start_seed, start_seed + num_scenarios);
    # use ep_idx directly so each episode is a distinct procedural scenario.
    # Safety-gym has no such bound, so we keep the seed_offset convention there.
    md = cfg.get("data_source") == "metadrive"

    # Optional auto-calibration of bc_action_noise: replaces a hand-tuned
    # noise std with a value that produces a target cost distribution. The
    # target is configured under cfg["collect"]["auto_calibrate"].
    auto = coll.get("auto_calibrate")
    if auto and bc_policy is not None:
        target_frac = tuple(auto.get("target_frac", [0.30, 0.50]))
        target_cost = tuple(auto.get("target_cost_when_violating", [5, 30]))
        bc_noise = _calibrate_noise(
            env, bc_policy, bc_device,
            n_pilot=int(auto.get("n_pilot", 15)),
            target_frac=target_frac, target_cost=target_cost,
            max_steps=max_steps, md=md, momentum=momentum, fwd_bias=fwd_bias,
            sigma_lo=float(auto.get("sigma_lo", 0.05)),
            sigma_hi=float(auto.get("sigma_hi", 0.60)),
            max_iter=int(auto.get("max_iter", 5)),
        )
        print(f"  ↳ using calibrated bc_action_noise={bc_noise:.3f} for the full collection",
              flush=True)

    trajectories = []
    for ep_idx in tqdm(range(n_episodes), desc="rollout"):
        ep_seed = int(ep_idx if md else seed_offset + ep_idx)  # reproducible
        traj = _rollout_episode(
            env, bc_policy, bc_device,
            ep_seed=ep_seed, bc_noise=bc_noise, max_steps=max_steps,
            md=md, momentum=momentum, fwd_bias=fwd_bias,
        )
        if traj is None:
            continue
        trajectories.append({
            "seed": traj["seed"],
            "observations": traj["observations"],
            "actions": traj["actions"],
            "rewards": traj["rewards"],
            "costs": traj["costs"],
        })
    env.close()

    # Quick diversity report
    total_costs = np.array([t["costs"].sum() for t in trajectories])
    total_rews  = np.array([t["rewards"].sum() for t in trajectories])
    print(f"Collected {len(trajectories)} episodes")
    print(f"  episode cost:  mean={total_costs.mean():.2f} | "
          f"min={total_costs.min():.2f} | max={total_costs.max():.2f}")
    print(f"  episode reward: mean={total_rews.mean():.2f} | "
          f"min={total_rews.min():.2f} | max={total_rews.max():.2f}")
    print(f"  fraction with cost > 0: "
          f"{(total_costs > 0).mean():.2%}")

    out_path = cfg["data_pickle"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
