"""Step 04f: V-preference learning + Advantage-Weighted Regression (AWR).

Diagnostic experiment for the "safety = state occupancy" hypothesis. CPL
attributes preferences to ACTIONS given states. In MetaDrive, safe and
unsafe segments often share action distributions (forward driving in
both, just at different positions on the road), so CPL can't extract a
useful signal from action comparisons. This script instead:

  Stage 1: train V_theta(s) on segment-level state preferences via
           Bradley-Terry over summed V values. V never sees actions —
           it can only learn "states in preferred segments are higher
           value than states in dispreferred segments."

  Stage 2: AWR fine-tune the BC policy using one-step advantages
           A(s, a) = V(s') - V(s). Actions that lead to higher-V
           successor states get weighted more heavily in the BC loss.
           The V-preference signal flows into the policy through the
           successor state, not through V's evaluation of the action.

Inputs:
  - <output_dir>/active_segments.pkl   trajectory-filtered segments
  - <output_dir>/gt_labels.json         cost-derived preference pairs
  - <output_dir>/bc_policy.pt           BC starting point
  - <data_pickle>                        cached trajectories for AWR transitions

Output:
  - <output_dir>/bc_v_awr_policy.pt
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
from model.train import train_v_preference, train_awr_policy  # noqa: E402


def build_transitions(trajectories, k: int = 1):
    """Concat (s_t, a_t, s_{t+k}) over all trajectories.

    k controls the advantage horizon used by AWR:
       A(s_t, a_t) = V(s_{t+k}) - V(s_t)
    Larger k gives each action credit for a longer state change, which
    helps when one-step advantages are too subtle (driving envs).

    Only emits transitions where t+k is within the same episode; the last
    k steps of each episode are dropped. Episodes shorter than k+1 are
    skipped entirely."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    obs_l, act_l, next_obs_l = [], [], []
    for traj in trajectories:
        obs = traj["observations"]
        act = traj["actions"]
        if len(obs) < k + 1:
            continue
        # For each t in [0, T-k-1]: (obs[t], act[t], obs[t+k])
        obs_l.append(obs[:-k])
        act_l.append(act[:-k])
        next_obs_l.append(obs[k:])
    return (np.concatenate(obs_l, axis=0),
            np.concatenate(act_l, axis=0),
            np.concatenate(next_obs_l, axis=0))


def main() -> None:
    cfg = load_cfg()
    # Allow sweep runner to override AWR knobs via env vars without touching
    # config.yaml (avoids race conditions when multiple cells share a config).
    if "AWR_BETA" in os.environ:
        cfg["awr_beta"] = float(os.environ["AWR_BETA"])
    if "AWR_WEIGHT_CLIP" in os.environ:
        cfg["awr_weight_clip"] = float(os.environ["AWR_WEIGHT_CLIP"])
    if "AWR_K_STEP" in os.environ:
        cfg["awr_k_step"] = int(os.environ["AWR_K_STEP"])
    if "AWR_NORMALIZE" in os.environ:
        cfg["awr_normalize_adv"] = int(os.environ["AWR_NORMALIZE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_v_awr")

    # --- Load preference data ---
    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "gt_labels.json"), "r") as f:
        pref_data = json.load(f)
    if not pref_data:
        raise RuntimeError("No GT labels found - generate gt_labels.json first.")

    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]

    obs_A, obs_B, labels = [], [], []
    for entry in pref_data:
        sA = active[entry["seg_A_idx"]]
        sB = active[entry["seg_B_idx"]]
        obs_A.append(torch.as_tensor(sA["observations"], dtype=torch.float32))
        obs_B.append(torch.as_tensor(sB["observations"], dtype=torch.float32))
        labels.append(entry["preference"])
    pref_obs_A = torch.stack(obs_A)
    pref_obs_B = torch.stack(obs_B)
    pref_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Preference pairs: {pref_labels.size(0)}  "
          f"(state-only; actions ignored by V network)")

    # === Stage 1: V-preference training ===
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"--- Stage 1: V-preference training ({v_epochs} epochs) ---")
    v_net = VNetwork(obs_dim, cfg["hidden_dim"]).to(device)
    train_v_preference(
        v_net, pref_obs_A, pref_obs_B, pref_labels,
        epochs=v_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    # === Stage 2: AWR policy fine-tune ===
    bc_path = os.path.join(cfg["output_dir"], "bc_policy.pt")
    print(f"--- Stage 2: loading BC starting point from {bc_path} ---")
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(
        bc_ckpt["obs_dim"], bc_ckpt["act_dim"], bc_ckpt["hidden_dim"]
    ).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])

    awr_k = int(cfg.get("awr_k_step", 1))
    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np = build_transitions(trajectories, k=awr_k)
    print(f"AWR transitions: {len(obs_np)} from {len(trajectories)} trajectories "
          f"(k_step={awr_k})")
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)

    awr_beta = float(cfg.get("awr_beta", 1.0))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    awr_normalize = bool(int(cfg.get("awr_normalize_adv", 0)))
    print(f"--- AWR fine-tune ({awr_epochs} epochs, beta={awr_beta}, "
          f"k_step={awr_k}, weight_clip={awr_weight_clip}, "
          f"normalize_adv={awr_normalize}) ---")
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

    _tag = os.environ.get("OUT_TAG")
    out_path = os.path.join(cfg["output_dir"],
                            f"bc_v_awr_{_tag}_policy.pt" if _tag else "bc_v_awr_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_path)
    print(f"Saved BC+V-AWR policy -> {out_path}")
    # Also save V network for diagnostics / future reuse
    v_path = os.path.join(cfg["output_dir"], "v_network.pt")
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"],
                "state_dict": v_net.state_dict()}, v_path)
    print(f"Saved V network -> {v_path}")

    if run is not None:
        import wandb
        wandb.save(out_path)
        wandb.finish()


if __name__ == "__main__":
    main()
