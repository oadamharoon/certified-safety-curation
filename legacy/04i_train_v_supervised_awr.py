"""Step 04i: V-supervised + AWR. Phase 10c baseline.

Replaces V-from-preferences with V-from-direct-cost-regression. Same AWR
extraction. This is the apples-to-apples test of whether our preference
pipeline is the bottleneck:

  if V_pref AWR < V_sup AWR  →  preference pipeline is the bottleneck
  if V_pref AWR ≈ V_sup AWR  →  preferences are doing their job
  if V_pref AWR > V_sup AWR  →  preference signal carries info that
                                direct cost regression misses (unlikely
                                with oracle-cost preferences but possible
                                with VLM noise reduction)

The V_supervised target is per-segment summed cost (sign-flipped so
high V = safer). NOT per-state, to match exactly what BT loss on
segment sums constrains. This is the most direct comparison.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy, VNetwork  # noqa: E402
from model.train import train_v_supervised, train_awr_policy  # noqa: E402


def build_transitions(trajectories, k: int = 1):
    obs_l, act_l, next_obs_l = [], [], []
    for traj in trajectories:
        obs = traj["observations"]
        act = traj["actions"]
        if len(obs) < k + 1:
            continue
        obs_l.append(obs[:-k])
        act_l.append(act[:-k])
        next_obs_l.append(obs[k:])
    return (np.concatenate(obs_l, axis=0),
            np.concatenate(act_l, axis=0),
            np.concatenate(next_obs_l, axis=0))


def main() -> None:
    cfg = load_cfg()
    # Same env-var overrides as 04f for sweep parity
    if "AWR_BETA" in os.environ:
        cfg["awr_beta"] = float(os.environ["AWR_BETA"])
    if "AWR_NORMALIZE" in os.environ:
        cfg["awr_normalize_adv"] = int(os.environ["AWR_NORMALIZE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_v_sup_awr")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)

    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]

    # Build segment-level training data for V_supervised. All segments
    # contribute (not just those that appeared in the preference pool) so V
    # sees the full state distribution.
    seg_obs = torch.stack([
        torch.as_tensor(s["observations"], dtype=torch.float32) for s in active
    ])
    # Sign-flipped per-segment cumulative cost: high target = safer segment.
    seg_target = torch.tensor(
        [-float(s["total_cost"]) for s in active], dtype=torch.float32
    )
    print(f"V_supervised training data: {len(seg_target)} segments  "
          f"(target stats: min={seg_target.min().item():.2f} "
          f"mean={seg_target.mean().item():.2f} max={seg_target.max().item():.2f})",
          flush=True)

    # ---- Stage 1: train V_supervised ----
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"--- Stage 1: V-supervised training ({v_epochs} epochs) ---",
          flush=True)
    v_net = VNetwork(obs_dim, cfg["hidden_dim"]).to(device)
    train_v_supervised(
        v_net, seg_obs, seg_target,
        epochs=v_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )
    for p in v_net.parameters():
        p.requires_grad = False
    v_net.eval()

    # ---- Stage 2: AWR fine-tune from BC ----
    bc_path = os.path.join(cfg["output_dir"], "bc_policy.pt")
    print(f"--- Stage 2: AWR fine-tune from {bc_path} ---", flush=True)
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(
        bc_ckpt["obs_dim"], bc_ckpt["act_dim"], bc_ckpt["hidden_dim"]
    ).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np = build_transitions(trajectories, k=1)
    print(f"AWR transitions: {len(obs_np)} from {len(trajectories)} trajectories",
          flush=True)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)

    awr_beta = float(cfg.get("awr_beta", 0.5))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    awr_normalize = bool(int(cfg.get("awr_normalize_adv", 0)))
    print(f"--- AWR ({awr_epochs} epochs, beta={awr_beta}, "
          f"normalize_adv={awr_normalize}) ---", flush=True)
    train_awr_policy(
        policy, v_net,
        obs_t, act_t, next_obs_t,
        epochs=awr_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        beta=awr_beta,
        device=device,
        weight_clip=awr_weight_clip,
        normalize_adv=awr_normalize,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_p = os.path.join(cfg["output_dir"], "bc_v_sup_awr_policy.pt")
    out_v = os.path.join(cfg["output_dir"], "v_supervised.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"],
                "state_dict": v_net.state_dict()}, out_v)
    print(f"Saved policy -> {out_p}")
    print(f"Saved V_sup  -> {out_v}")
    if run is not None:
        import wandb
        wandb.save(out_p)
        wandb.finish()


if __name__ == "__main__":
    main()
