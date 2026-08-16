"""Step 04k: K-coupled IQL with Q-ensemble disagreement pessimism (Phase 13).

The diagnostic experiment we never ran on DSRL tasks:

  Stage 1: V-ensemble V_theta (K=3) on BT preferences. (same as 04j)
  Stage 2: K-COUPLED IQL training:
             V_psi_k(s) <- expectile_tau(Q_target_k(s, a))   for each k
             Q_k(s, a)  -> V_theta_k(s') + γ V_psi_k(s')      for each k
             Q_target_k tracked via Polyak.
  Stage 3: AWR with disagreement-pessimistic Q-advantage:
             A_pess(s, a) = mean_k(Q_k - V_psi_k) − λ * std_k(Q_k - V_psi_k)
             w(s,a) ∝ exp(A_pess / β)

Decision rule on diagnostic tasks (HC, CG1, PG1, Hopper):
- PG1 moves to safe → state-only V worked; one-step AWR was the bottleneck
- PG1 still fails → unsafe consequence isn't one-step state-decodable
  (sharper structural claim than the prior 'state-action ambiguity')

Env vars (parity with 04h/04j):
  AWR_BETA, AWR_NORMALIZE, ENS_K, V_EPOCHS, Q_EPOCHS, AWR_EPOCHS,
  BATCH_SIZE, ADAPTIVE_BETA (ignored here; A is already centered by V_psi),
  IQL_GAMMA, IQL_TAU_POLYAK, IQL_TAU_EXPECTILE,
  DISAGREEMENT_LAMBDA (default 1.0 — the working value from Phase 12e),
  SEED_OVERRIDE (lets parallel runners skip config.yaml race).
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
from model.policy import GaussianPolicy, VEnsemble, QEnsemble  # noqa: E402
from model.train import (  # noqa: E402
    train_v_ensemble, train_iql_q_vpsi_ensemble, train_awr_q_ensemble_pess,
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
    if "AWR_BETA" in os.environ:
        cfg["awr_beta"] = float(os.environ["AWR_BETA"])
    if "AWR_NORMALIZE" in os.environ:
        cfg["awr_normalize_adv"] = int(os.environ["AWR_NORMALIZE"])
    if "ENS_K" in os.environ:
        cfg["v_ensemble_k"] = int(os.environ["ENS_K"])
    if "V_EPOCHS" in os.environ:
        cfg["v_epochs"] = int(os.environ["V_EPOCHS"])
    if "Q_EPOCHS" in os.environ:
        cfg["q_epochs"] = int(os.environ["Q_EPOCHS"])
    if "AWR_EPOCHS" in os.environ:
        cfg["awr_epochs"] = int(os.environ["AWR_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    if "IQL_GAMMA" in os.environ:
        cfg["iql_gamma"] = float(os.environ["IQL_GAMMA"])
    if "IQL_TAU_POLYAK" in os.environ:
        cfg["iql_tau_polyak"] = float(os.environ["IQL_TAU_POLYAK"])
    if "IQL_TAU_EXPECTILE" in os.environ:
        cfg["iql_tau_expectile"] = float(os.environ["IQL_TAU_EXPECTILE"])
    if "DISAGREEMENT_LAMBDA" in os.environ:
        cfg["awr_disagreement_lambda"] = float(os.environ["DISAGREEMENT_LAMBDA"])
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_iql_q_ensemble")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "gt_labels.json"), "r") as f:
        pref_data = json.load(f)
    if not pref_data:
        raise RuntimeError("No GT labels — generate gt_labels.json first.")

    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]

    obs_A, obs_B, labels = [], [], []
    for entry in pref_data:
        obs_A.append(torch.as_tensor(active[entry["seg_A_idx"]]["observations"],
                                     dtype=torch.float32))
        obs_B.append(torch.as_tensor(active[entry["seg_B_idx"]]["observations"],
                                     dtype=torch.float32))
        labels.append(entry["preference"])
    pref_obs_A = torch.stack(obs_A)
    pref_obs_B = torch.stack(obs_B)
    pref_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Preference pairs: {pref_labels.size(0)}", flush=True)

    # === Stage 1: V-ensemble V_theta on BT preferences ===
    K = int(cfg.get("v_ensemble_k", 3))
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"--- Stage 1: V-ensemble V_theta (K={K}, {v_epochs} epochs) ---",
          flush=True)
    v_theta = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)
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

    # === Stage 2: K-coupled IQL Q + V_psi ===
    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np, next_act_np = build_quadruples(trajectories)
    print(f"IQL quadruples: {len(obs_np)} from {len(trajectories)} trajectories",
          flush=True)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)
    next_act_t = torch.as_tensor(next_act_np, dtype=torch.float32)

    q_ens = QEnsemble(obs_dim, act_dim, cfg["hidden_dim"], K=K).to(device)
    q_target_ens = QEnsemble(obs_dim, act_dim, cfg["hidden_dim"], K=K).to(device)
    q_target_ens.load_state_dict(q_ens.state_dict())
    v_psi_ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)

    iql_epochs = int(cfg.get("q_epochs", cfg["cpl_epochs"]))
    gamma = float(cfg.get("iql_gamma", 0.99))
    tau_polyak = float(cfg.get("iql_tau_polyak", 0.005))
    tau_expectile = float(cfg.get("iql_tau_expectile", 0.7))
    print(f"--- Stage 2: K-coupled IQL ({iql_epochs} epochs, K={K}, "
          f"γ={gamma}, τ_polyak={tau_polyak}, τ_exp={tau_expectile}) ---",
          flush=True)
    train_iql_q_vpsi_ensemble(
        q_ens, q_target_ens, v_psi_ens, v_theta,
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

    # === Stage 3: AWR with disagreement-pessimistic Q-advantage ===
    bc_filename = os.environ.get("BC_POLICY", "bc_policy.pt")
    bc_path = os.path.join(cfg["output_dir"], bc_filename)
    print(f"--- Stage 3: AWR-Qens fine-tune from {bc_path} ---", flush=True)
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(
        bc_ckpt["obs_dim"], bc_ckpt["act_dim"], bc_ckpt["hidden_dim"]
    ).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])

    awr_beta = float(cfg.get("awr_beta", 0.5))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    awr_normalize = bool(int(cfg.get("awr_normalize_adv", 1)))   # default ON for Q-AWR
    awr_dis_lambda = float(cfg.get("awr_disagreement_lambda", 1.0))
    print(f"--- AWR-Qens ({awr_epochs} epochs, β={awr_beta}, "
          f"normalize={awr_normalize}, λ_dis={awr_dis_lambda}) ---", flush=True)
    train_awr_q_ensemble_pess(
        policy, q_ens, v_psi_ens,
        obs_t, act_t,
        epochs=awr_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        beta=awr_beta,
        device=device,
        weight_clip=awr_weight_clip,
        normalize_adv=awr_normalize,
        disagreement_lambda=awr_dis_lambda,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_p = os.path.join(cfg["output_dir"], "bc_iql_q_ens_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    print(f"Saved policy -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.save(out_p)
        wandb.finish()


if __name__ == "__main__":
    main()
