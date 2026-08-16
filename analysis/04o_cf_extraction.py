"""Step 04o: Counterfactual extraction via one-step dynamics (Phase B'.3).

The load-bearing extraction experiment. B'.0 established that even
ground-truth V cannot push through exp-AWR on PG1/CG2: AWR can only
reweight actions that exist in the dataset at each state. This script
lifts that restriction.

Pipeline:
  Stage 1: V-ensemble (load saved checkpoint or train fresh).
  Stage 2: one-step dynamics ensemble f(s,a) -> s' on offline transitions
           (load dyn_ensemble_seed{seed}.pt if present, else train + save).
  Stage 3: counterfactual AWR: at each data state, score
           {data action} + M BC-sampled candidates through V(f(s,a)) with
           dual disagreement pessimism (V-members + dynamics-members),
           softmax-imitate. OOD candidates are auto-suppressed by dynamics
           disagreement; if all counterfactuals are bad, weight collapses
           to the data action (graceful BC fallback).

Env vars (beyond the 04h/04m standards):
  M_CANDIDATES       number of BC-sampled candidate actions   (default 8)
  CAND_SIGMA_SCALE   candidate sampling std multiplier        (default 1.0)
  CF_BETA            softmax temperature over candidates      (default 0.1)
  LAMBDA_V           V-ensemble disagreement penalty          (default 1.0)
  LAMBDA_DYN         dynamics-ensemble disagreement penalty   (default 1.0)
  DYN_K              dynamics ensemble size                   (default 5)
  DYN_EPOCHS         dynamics training epochs                 (default 25)
  V_ENSEMBLE_FILE    saved V ensemble to load (else trains fresh)
  DYN_FILE           saved dynamics ensemble to load/save
                     (default dyn_ensemble_seed{seed}.pt)
  OUT_TAG            output tag (default cf_M<m>_seed<seed>)
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
from model.policy import GaussianPolicy, VEnsemble, DynamicsEnsemble  # noqa: E402
from model.train import (  # noqa: E402
    train_v_ensemble, train_dynamics_ensemble, train_cf_awr,
)


def build_transitions(trajectories):
    obs_l, act_l, next_obs_l = [], [], []
    for traj in trajectories:
        obs, act = traj["observations"], traj["actions"]
        if len(obs) < 2:
            continue
        obs_l.append(obs[:-1]); act_l.append(act[:-1]); next_obs_l.append(obs[1:])
    return (np.concatenate(obs_l, 0), np.concatenate(act_l, 0),
            np.concatenate(next_obs_l, 0))


def main() -> None:
    cfg = load_cfg()
    for var, key, cast in (
        ("AWR_EPOCHS", "awr_epochs", int), ("V_EPOCHS", "v_epochs", int),
        ("BATCH_SIZE", "batch_size", int), ("ENS_K", "v_ensemble_k", int),
        ("SEED_OVERRIDE", "seed", int),
    ):
        if var in os.environ:
            cfg[key] = cast(os.environ[var])
    seed = int(cfg["seed"])
    M = int(os.environ.get("M_CANDIDATES", 8))
    cand_scale = float(os.environ.get("CAND_SIGMA_SCALE", 1.0))
    cf_beta = float(os.environ.get("CF_BETA", 0.1))
    lambda_v = float(os.environ.get("LAMBDA_V", 1.0))
    lambda_dyn = float(os.environ.get("LAMBDA_DYN", 1.0))
    perstate_norm = bool(int(os.environ.get("CF_PERSTATE_NORM", 0)))
    dyn_k = int(os.environ.get("DYN_K", 5))
    dyn_epochs = int(os.environ.get("DYN_EPOCHS", 25))
    out_tag = os.environ.get("OUT_TAG", f"cf_M{M}_seed{seed}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | M={M} scale={cand_scale} cf_beta={cf_beta} "
          f"lam_v={lambda_v} lam_dyn={lambda_dyn} psnorm={perstate_norm} "
          f"dyn_K={dyn_k} seed={seed} tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="cf_extraction")

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_np, act_np, next_obs_np = build_transitions(trajectories)
    obs_dim, act_dim = obs_np.shape[1], act_np.shape[1]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)
    print(f"Transitions: {len(obs_np)}", flush=True)

    # --- Stage 1: V ensemble ---
    K = int(cfg.get("v_ensemble_k", 3))
    v_ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)
    ens_file = os.environ.get("V_ENSEMBLE_FILE", "")
    if ens_file:
        p = os.path.join(cfg["output_dir"], ens_file)
        ck = torch.load(p, map_location=device, weights_only=False)
        v_ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        print(f"Loaded V-ensemble: {p}", flush=True)
    else:
        with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
            active = pickle.load(f)
        with open(os.path.join(cfg["output_dir"], "gt_labels.json"), "r") as f:
            pref_data = json.load(f)
        oA = torch.stack([torch.as_tensor(active[e["seg_A_idx"]]["observations"],
                                          dtype=torch.float32) for e in pref_data])
        oB = torch.stack([torch.as_tensor(active[e["seg_B_idx"]]["observations"],
                                          dtype=torch.float32) for e in pref_data])
        lab = torch.tensor([e["preference"] for e in pref_data], dtype=torch.long)
        train_v_ensemble(v_ens, oA, oB, lab,
                         epochs=int(cfg.get("v_epochs", cfg["cpl_epochs"])),
                         batch_size=cfg["batch_size"], lr=cfg["learning_rate"],
                         device=device, log_every=cfg["log_every"], wandb_run=run)

    # --- Stage 2: dynamics ensemble ---
    dyn_file = os.environ.get("DYN_FILE", f"dyn_ensemble_seed{seed}.pt")
    dyn_path = os.path.join(cfg["output_dir"], dyn_file)
    dyn = DynamicsEnsemble(obs_dim, act_dim, cfg["hidden_dim"], K=dyn_k).to(device)
    if os.path.exists(dyn_path):
        ck = torch.load(dyn_path, map_location=device, weights_only=False)
        dyn.load_state_dict(ck["state_dict"])
        print(f"Loaded dynamics ensemble: {dyn_path}", flush=True)
    else:
        print(f"--- Stage 2: dynamics ensemble (K={dyn_k}, {dyn_epochs} epochs) ---",
              flush=True)
        train_dynamics_ensemble(dyn, obs_t, act_t, next_obs_t,
                                epochs=dyn_epochs, batch_size=cfg["batch_size"],
                                lr=cfg["learning_rate"], device=device,
                                log_every=cfg["log_every"], wandb_run=run)
        torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                    "hidden_dim": cfg["hidden_dim"], "K": dyn_k,
                    "state_dict": dyn.state_dict()}, dyn_path)
        print(f"Saved dynamics -> {dyn_path}", flush=True)

    # --- Stage 3: counterfactual AWR from BC ---
    bc_filename = os.environ.get("BC_POLICY", "bc_policy.pt")
    bc_path = os.path.join(cfg["output_dir"], bc_filename)
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(bc_ckpt["obs_dim"], bc_ckpt["act_dim"],
                            bc_ckpt["hidden_dim"]).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])
    bc_ref = GaussianPolicy(bc_ckpt["obs_dim"], bc_ckpt["act_dim"],
                            bc_ckpt["hidden_dim"]).to(device)
    bc_ref.load_state_dict(bc_ckpt["state_dict"])
    print(f"BC start + reference: {bc_path}", flush=True)

    train_cf_awr(
        policy, bc_ref, v_ens, dyn, obs_t, act_t,
        epochs=int(cfg.get("awr_epochs", cfg["cpl_epochs"])),
        batch_size=cfg["batch_size"], lr=cfg["learning_rate"], device=device,
        n_candidates=M, cand_sigma_scale=cand_scale, cf_beta=cf_beta,
        lambda_v=lambda_v, lambda_dyn=lambda_dyn, perstate_norm=perstate_norm,
        log_every=cfg["log_every"], wandb_run=run,
    )

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    print(f"Saved policy -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
