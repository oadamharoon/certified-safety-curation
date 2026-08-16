"""Step 04g: V-IQL — V-preference + offline-TD Q + AWR.

Phase 7d. Extends V-AWR (04f) by inserting an action-conditioned critic
between the state-preference V and the AWR policy update. The motivation
is that one-step V advantage A = V(s') - V(s) is myopic, while a Q
trained to bootstrap V_θ as a per-state reward captures the discounted
cumulative state-preference value of acting (s, a) — long-horizon credit
assignment that V-AWR misses.

Three stages, fully offline:

  1. V_θ(s) — Bradley-Terry on segment state-value sums. Same as 04f.
  2. Q_φ(s, a) — TD targets y = V_θ(s) + γ Q_target(s', a'_data).
     a'_data is the behavior-policy next-action from the cached
     trajectories (offline TD, no policy involved). Soft-updates a
     target network.
  3. Policy π — initialize from BC, then AWR with
       A(s, a) = Q_φ(s, a) - V_θ(s)

Inputs (same as 04f):
  - <output_dir>/active_segments.pkl
  - <output_dir>/gt_labels.json
  - <output_dir>/bc_policy.pt
  - <data_pickle>            # for (s, a, s', a') quadruples

Output:
  - <output_dir>/bc_v_iql_policy.pt
  - <output_dir>/v_network_iql.pt
  - <output_dir>/q_network_iql.pt
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
from model.policy import GaussianPolicy, VNetwork, QNetwork  # noqa: E402
from model.train import (  # noqa: E402
    train_v_preference, train_q_td, train_awr_with_q,
)


def build_quadruples(trajectories):
    """Concat (s_t, a_t, s_{t+1}, a_{t+1}) over all trajectories.

    Needs both this-action and next-action for the offline TD target. The
    last two steps of each episode are dropped (no s_{t+1}, a_{t+1} pair).
    Episodes shorter than 3 steps are skipped.
    """
    obs_l, act_l, next_obs_l, next_act_l = [], [], [], []
    for traj in trajectories:
        obs = traj["observations"]
        act = traj["actions"]
        T = len(obs)
        if T < 3:
            continue
        # t in [0, T-2): (obs[t], act[t], obs[t+1], act[t+1])
        obs_l.append(obs[:-2])
        act_l.append(act[:-2])
        next_obs_l.append(obs[1:-1])
        next_act_l.append(act[1:-1])
    return (np.concatenate(obs_l, axis=0),
            np.concatenate(act_l, axis=0),
            np.concatenate(next_obs_l, axis=0),
            np.concatenate(next_act_l, axis=0))


def main() -> None:
    cfg = load_cfg()
    # Env-var overrides so a sweep runner can change AWR knobs without
    # editing config.yaml. Same convention as 04f.
    if "AWR_BETA" in os.environ:
        cfg["awr_beta"] = float(os.environ["AWR_BETA"])
    if "IQL_GAMMA" in os.environ:
        cfg["iql_gamma"] = float(os.environ["IQL_GAMMA"])
    if "IQL_TAU" in os.environ:
        cfg["iql_tau"] = float(os.environ["IQL_TAU"])
    # Per-stage epoch overrides. Useful when GPU is contended and we
    # want to ride out a plateau in fewer wall-clock minutes.
    if "V_EPOCHS" in os.environ:
        cfg["v_epochs"] = int(os.environ["V_EPOCHS"])
    if "Q_EPOCHS" in os.environ:
        cfg["q_epochs"] = int(os.environ["Q_EPOCHS"])
    if "AWR_EPOCHS" in os.environ:
        cfg["awr_epochs"] = int(os.environ["AWR_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_v_iql")

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
    print(f"Preference pairs: {pref_labels.size(0)}")

    # === Stage 1: V_θ from state-preferences ===
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
    for p in v_net.parameters():
        p.requires_grad = False
    v_net.eval()

    # === Stage 2: Q_φ via offline TD with V_θ reward ===
    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np, next_act_np = build_quadruples(trajectories)
    print(f"TD quadruples: {len(obs_np)} from {len(trajectories)} trajectories")
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)
    next_act_t = torch.as_tensor(next_act_np, dtype=torch.float32)

    q_net = QNetwork(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    q_target = QNetwork(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    q_target.load_state_dict(q_net.state_dict())

    q_epochs = int(cfg.get("q_epochs", cfg["cpl_epochs"]))
    gamma = float(cfg.get("iql_gamma", 0.99))
    tau = float(cfg.get("iql_tau", 0.005))
    print(f"--- Stage 2: Q-TD ({q_epochs} epochs, γ={gamma}, τ={tau}) ---")
    train_q_td(
        q_net, q_target, v_net,
        obs_t, act_t, next_obs_t, next_act_t,
        epochs=q_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        gamma=gamma, tau=tau,
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    # === Stage 3: AWR policy fine-tune ===
    bc_path = os.path.join(cfg["output_dir"], "bc_policy.pt")
    print(f"--- Stage 3: AWR fine-tune from {bc_path} ---")
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(
        bc_ckpt["obs_dim"], bc_ckpt["act_dim"], bc_ckpt["hidden_dim"]
    ).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])

    awr_beta = float(cfg.get("awr_beta", 0.5))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    print(f"--- AWR-Q ({awr_epochs} epochs, β={awr_beta}) ---")
    train_awr_with_q(
        policy, q_net, v_net,
        obs_t, act_t,
        epochs=awr_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        beta=awr_beta,
        device=device,
        weight_clip=awr_weight_clip,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_policy = os.path.join(cfg["output_dir"], "bc_v_iql_policy.pt")
    out_v = os.path.join(cfg["output_dir"], "v_network_iql.pt")
    out_q = os.path.join(cfg["output_dir"], "q_network_iql.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_policy)
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"],
                "state_dict": v_net.state_dict()}, out_v)
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": q_net.state_dict()}, out_q)
    print(f"Saved policy -> {out_policy}")
    print(f"Saved V     -> {out_v}")
    print(f"Saved Q     -> {out_q}")

    if run is not None:
        import wandb
        wandb.save(out_policy)
        wandb.finish()


if __name__ == "__main__":
    main()
