"""Step 04j: Proper IQL with expectile V_psi + V-ensemble V_theta. Phase 10b.

Fixes the wrong-baseline bug in 04g (Phase 7d). The earlier V-IQL used
A = Q(s,a) - V_theta(s) where Q was discounted-cumulative and V_theta was
per-state — scale mismatch papered over by ad-hoc advantage normalization.

This script implements the canonical IQL adaptation:

  Stage 1: train V-ensemble V_theta (Phase 10a, K=3) on state preferences.
  Stage 2: alternate:
             V_psi(s)  <- expectile_tau( Q_target(s, a) over data actions )
             Q(s,a)    -> V_theta(s') + gamma * V_psi(s')
           Q_target tracks Q via Polyak averaging.
  Stage 3: AWR fine-tune from BC with A = Q(s,a) - V_psi(s).

V_theta is an ensemble so the per-state BT-loss underspec noise is
laundered before IQL bootstraps it through Q (the other Claude's
laundering observation made concrete).
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
from model.policy import GaussianPolicy, VEnsemble, VNetwork, QNetwork  # noqa: E402
from model.train import (  # noqa: E402
    train_v_ensemble, train_iql_q_vpsi, train_awr_with_q,
)


def build_quadruples(trajectories):
    obs_l, act_l, next_obs_l, next_act_l = [], [], [], []
    for traj in trajectories:
        obs = traj["observations"]
        act = traj["actions"]
        T = len(obs)
        if T < 3:
            continue
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
    # Env-var overrides for sweep parity
    if "AWR_BETA" in os.environ:
        cfg["awr_beta"] = float(os.environ["AWR_BETA"])
    if "AWR_NORMALIZE" in os.environ:
        cfg["awr_normalize_adv"] = int(os.environ["AWR_NORMALIZE"])
    if "ENS_K" in os.environ:
        cfg["v_ensemble_k"] = int(os.environ["ENS_K"])
    if "IQL_GAMMA" in os.environ:
        cfg["iql_gamma"] = float(os.environ["IQL_GAMMA"])
    if "IQL_TAU_POLYAK" in os.environ:
        cfg["iql_tau_polyak"] = float(os.environ["IQL_TAU_POLYAK"])
    if "IQL_TAU_EXPECTILE" in os.environ:
        cfg["iql_tau_expectile"] = float(os.environ["IQL_TAU_EXPECTILE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_v_iql_proper")

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
    print(f"Preference pairs: {pref_labels.size(0)}", flush=True)

    # === Stage 1: V-ensemble V_theta ===
    ens_K = int(cfg.get("v_ensemble_k", 3))
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"--- Stage 1: V-ensemble (K={ens_K}) preference training "
          f"({v_epochs} epochs) ---", flush=True)
    v_theta = VEnsemble(obs_dim, cfg["hidden_dim"], K=ens_K).to(device)
    train_v_ensemble(
        v_theta, pref_obs_A, pref_obs_B, pref_labels,
        epochs=v_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )
    for p in v_theta.parameters():
        p.requires_grad = False
    v_theta.eval()

    # === Stage 2: IQL Q and V_psi ===
    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np, next_act_np = build_quadruples(trajectories)
    print(f"IQL quadruples: {len(obs_np)} from {len(trajectories)} trajectories",
          flush=True)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)
    next_act_t = torch.as_tensor(next_act_np, dtype=torch.float32)

    q_net = QNetwork(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    q_target = QNetwork(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    q_target.load_state_dict(q_net.state_dict())
    v_psi = VNetwork(obs_dim, cfg["hidden_dim"]).to(device)

    iql_epochs = int(cfg.get("q_epochs", cfg["cpl_epochs"]))
    gamma = float(cfg.get("iql_gamma", 0.99))
    tau_polyak = float(cfg.get("iql_tau_polyak", 0.005))
    tau_expectile = float(cfg.get("iql_tau_expectile", 0.7))
    print(f"--- Stage 2: IQL Q+V_psi ({iql_epochs} epochs, "
          f"gamma={gamma}, tau_polyak={tau_polyak}, "
          f"tau_expectile={tau_expectile}) ---", flush=True)
    train_iql_q_vpsi(
        q_net, q_target, v_psi, v_theta,
        obs_t, act_t, next_obs_t, next_act_t,
        epochs=iql_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        gamma=gamma,
        tau_polyak=tau_polyak,
        tau_expectile=tau_expectile,
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    # === Stage 3: AWR with A = Q - V_psi ===
    bc_path = os.path.join(cfg["output_dir"], "bc_policy.pt")
    print(f"--- Stage 3: AWR fine-tune from {bc_path} ---", flush=True)
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(
        bc_ckpt["obs_dim"], bc_ckpt["act_dim"], bc_ckpt["hidden_dim"]
    ).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])

    awr_beta = float(cfg.get("awr_beta", 0.5))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    awr_normalize = bool(int(cfg.get("awr_normalize_adv", 0)))
    print(f"--- AWR-Q ({awr_epochs} epochs, beta={awr_beta}, "
          f"normalize_adv={awr_normalize}) ---", flush=True)
    # train_awr_with_q computes A = q_net(s,a) - v_baseline(s). Here we pass
    # v_psi as the baseline (the PROPER IQL advantage), not v_theta.
    train_awr_with_q(
        policy, q_net, v_psi,
        obs_t, act_t,
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

    out_p = os.path.join(cfg["output_dir"], "bc_v_iql_proper_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    print(f"Saved policy -> {out_p}")
    # also save the critics for analysis
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"], "K": ens_K,
                "state_dict": v_theta.state_dict()},
               os.path.join(cfg["output_dir"], "v_theta_ens_iql.pt"))
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"],
                "state_dict": v_psi.state_dict()},
               os.path.join(cfg["output_dir"], "v_psi_iql.pt"))
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": q_net.state_dict()},
               os.path.join(cfg["output_dir"], "q_iql.pt"))
    if run is not None:
        import wandb
        wandb.save(out_p)
        wandb.finish()


if __name__ == "__main__":
    main()
