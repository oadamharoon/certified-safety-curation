"""Step 04c: BC pretrain + CPL fine-tune using GROUND-TRUTH cost-based
preference labels (from ``03b_label_by_cost.py``).

Identical to ``04_train_cpl.py`` except it consumes ``gt_labels.json``
instead of ``vlm_labels.json`` and saves ``bc_gt_cpl_policy.pt``.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy  # noqa: E402
from model.train import bc_pretrain, train_cpl  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_cpl_gt")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "gt_labels.json"), "r") as f:
        pref_data = json.load(f)
    if not pref_data:
        raise RuntimeError("No GT labels found - run 03b_label_by_cost.py first.")

    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]

    obs_A, act_A, obs_B, act_B, labels = [], [], [], [], []
    for entry in pref_data:
        sA, sB = active[entry["seg_A_idx"]], active[entry["seg_B_idx"]]
        obs_A.append(torch.as_tensor(sA["observations"], dtype=torch.float32))
        act_A.append(torch.as_tensor(sA["actions"], dtype=torch.float32))
        obs_B.append(torch.as_tensor(sB["observations"], dtype=torch.float32))
        act_B.append(torch.as_tensor(sB["actions"], dtype=torch.float32))
        labels.append(entry["preference"])
    pref_obs_A = torch.stack(obs_A)
    pref_act_A = torch.stack(act_A)
    pref_obs_B = torch.stack(obs_B)
    pref_act_B = torch.stack(act_B)
    pref_labels = torch.tensor(labels, dtype=torch.long)

    bc_obs = torch.cat(
        [torch.as_tensor(s["observations"], dtype=torch.float32) for s in active], dim=0
    )
    bc_act = torch.cat(
        [torch.as_tensor(s["actions"], dtype=torch.float32) for s in active], dim=0
    )
    print(f"Preference pairs (GT): {pref_labels.size(0)} | BC transitions: {bc_obs.size(0)}")

    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)

    print("--- BC pretraining ---")
    bc_pretrain(
        policy, bc_obs, bc_act,
        batch_size=cfg["batch_size"],
        epochs=cfg["bc_pretrain_epochs"],
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    print("--- CPL training (GT labels) ---")
    train_cpl(
        policy,
        pref_obs_A, pref_act_A,
        pref_obs_B, pref_act_B,
        pref_labels,
        bc_obs, bc_act,
        epochs=cfg["cpl_epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        lambda_bc=cfg["lambda_bc"],
        temp=cfg["temperature_cpl"],
        conservative_bias=float(os.environ.get("CPL_LAMBDA", 0.5)),
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_path = os.path.join(
        cfg["output_dir"], os.environ.get("POLICY_OUT", "bc_gt_cpl_policy.pt"))
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_path)
    print(f"Saved BC+GT-CPL policy -> {out_path}")
    if run is not None:
        import wandb
        wandb.save(out_path)
        wandb.finish()


if __name__ == "__main__":
    main()
