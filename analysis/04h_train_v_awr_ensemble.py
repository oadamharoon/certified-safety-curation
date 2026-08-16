"""Step 04h: V-ensemble + AWR. Phase 10a.

Direct attack on BT-loss underspecification. Stage 1 trains K independent
V_θ networks on the same preference data; each member fills the
zero-sum-per-segment null space differently due to its random init.
Stage 2 runs AWR with A = V_bar(s') - V_bar(s) where V_bar is the
ensemble mean. If seed 1's catastrophic policy is driven by idiosyncratic
per-state V errors, the mean across members should cancel them out and
the seed-1 outcome should normalize toward seeds 0 and 2.

Hyperparameters mirror 04f (V-AWR baseline) so the only changed axis is
the ensemble width K.

Outputs:
  - <output_dir>/bc_v_ens_awr_policy.pt
  - <output_dir>/v_ensemble.pt
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
from model.policy import GaussianPolicy, VEnsemble  # noqa: E402
from model.train import train_v_ensemble, train_awr_policy  # noqa: E402


def build_transitions(trajectories, k: int = 1):
    """Concat (s_t, a_t, s_{t+k}) over all trajectories. Same as 04f."""
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
    if "AWR_K_STEP" in os.environ:
        cfg["awr_k_step"] = int(os.environ["AWR_K_STEP"])
    if "AWR_NORMALIZE" in os.environ:
        cfg["awr_normalize_adv"] = int(os.environ["AWR_NORMALIZE"])
    if "ENS_K" in os.environ:
        cfg["v_ensemble_k"] = int(os.environ["ENS_K"])
    # Per-stage epoch + batch-size overrides for large-data tasks
    # (4M-transition CarGoal2 makes default 300-epoch AWR a multi-hour job).
    if "V_EPOCHS" in os.environ:
        cfg["v_epochs"] = int(os.environ["V_EPOCHS"])
    if "AWR_EPOCHS" in os.environ:
        cfg["awr_epochs"] = int(os.environ["AWR_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    if "ADAPTIVE_BETA" in os.environ:
        cfg["awr_adaptive_beta"] = int(os.environ["ADAPTIVE_BETA"])
    if "V_WEIGHT_DECAY" in os.environ:
        cfg["v_weight_decay"] = float(os.environ["V_WEIGHT_DECAY"])
    if "DISAGREEMENT_LAMBDA" in os.environ:
        cfg["awr_disagreement_lambda"] = float(os.environ["DISAGREEMENT_LAMBDA"])
    # SEED_OVERRIDE env var: lets parallel runners control seed without
    # patching config.yaml (which races when multiple tasks run concurrently).
    # Backwards compatible: if unset, cfg["seed"] from config.yaml is used.
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_v_awr_ensemble")

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
        sA = active[entry["seg_A_idx"]]; sB = active[entry["seg_B_idx"]]
        obs_A.append(torch.as_tensor(sA["observations"], dtype=torch.float32))
        obs_B.append(torch.as_tensor(sB["observations"], dtype=torch.float32))
        labels.append(entry["preference"])
    pref_obs_A = torch.stack(obs_A)
    pref_obs_B = torch.stack(obs_B)
    pref_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Preference pairs: {pref_labels.size(0)}", flush=True)

    K = int(cfg.get("v_ensemble_k", 3))
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    v_wd = float(cfg.get("v_weight_decay", 0.0))
    print(f"--- Stage 1: V-ensemble (K={K}) preference training "
          f"({v_epochs} epochs, weight_decay={v_wd}) ---", flush=True)
    ensemble = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)
    train_v_ensemble(
        ensemble, pref_obs_A, pref_obs_B, pref_labels,
        epochs=v_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        device=device,
        weight_decay=v_wd,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    # Optional override of BC initialization (e.g. bc_safe_policy.pt).
    bc_filename = os.environ.get("BC_POLICY", "bc_policy.pt")
    bc_path = os.path.join(cfg["output_dir"], bc_filename)
    print(f"--- Stage 2: loading BC starting point from {bc_path} ---",
          flush=True)
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
          f"(k_step={awr_k})", flush=True)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)

    awr_beta = float(cfg.get("awr_beta", 1.0))
    awr_weight_clip = float(cfg.get("awr_weight_clip", 20.0))
    awr_epochs = int(cfg.get("awr_epochs", cfg["cpl_epochs"]))
    awr_normalize = bool(int(cfg.get("awr_normalize_adv", 0)))
    awr_adaptive = bool(int(cfg.get("awr_adaptive_beta", 0)))
    awr_dis_lambda = float(cfg.get("awr_disagreement_lambda", 0.0))
    print(f"--- AWR fine-tune ({awr_epochs} epochs, beta={awr_beta}, "
          f"k_step={awr_k}, normalize_adv={awr_normalize}, "
          f"adaptive_beta={awr_adaptive}, disagreement_lambda={awr_dis_lambda}, "
          f"ensemble-V K={K}) ---", flush=True)
    # AWR consumes V_bar(s) via ensemble.forward (mean over members) —
    # train_awr_policy doesn't know it's an ensemble, just calls v_net(obs).
    train_awr_policy(
        policy, ensemble,
        obs_t, act_t, next_obs_t,
        epochs=awr_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        beta=awr_beta,
        device=device,
        weight_clip=awr_weight_clip,
        normalize_adv=awr_normalize,
        adaptive_beta=awr_adaptive,
        disagreement_lambda=awr_dis_lambda,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_policy = os.path.join(cfg["output_dir"], "bc_v_ens_awr_policy.pt")
    out_ens = os.path.join(cfg["output_dir"], "v_ensemble.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_policy)
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"],
                "K": K,
                "state_dict": ensemble.state_dict()}, out_ens)
    print(f"Saved policy -> {out_policy}")
    print(f"Saved V-ens  -> {out_ens}")

    if run is not None:
        import wandb
        wandb.save(out_policy)
        wandb.finish()


if __name__ == "__main__":
    main()
