"""Step 04b: BC-only baseline (no CPL fine-tuning).

Trains the Gaussian policy with the BC objective alone for `bc_only_epochs`
(defaults to 350 — matches the paper's BC baseline spec).

Saves ``bc_policy.pt`` to the task output dir; downstream evaluation
selects it via ``05_evaluate.py --policy_file bc_policy.pt``.
"""
from __future__ import annotations

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy  # noqa: E402
from model.train import bc_pretrain  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    if "BC_EPOCHS" in os.environ:
        cfg["bc_only_epochs"] = int(os.environ["BC_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_bc")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)

    # BC-Safe mode: filter to segments from trajectories whose total cost is
    # below a threshold. Matches PReSa's BC-Safe baseline.
    if "BC_SAFE_THRESHOLD" in os.environ:
        thr = float(os.environ["BC_SAFE_THRESHOLD"])
        before = len(active)
        active = [s for s in active if s.get("traj_total_cost", 0) <= thr]
        print(f"[BC-Safe] threshold = {thr}, kept {len(active)}/{before} segments "
              f"({100*len(active)/before:.1f}%)")
        if len(active) == 0:
            raise RuntimeError(f"BC-Safe filter rejected all segments at thr={thr}")

    # BC-Safe-Seg mode: keep only zero-cost SEGMENTS regardless of their
    # parent trajectory. Matches PReSa's BC-Safe-Seg baseline.
    if "BC_SAFE_SEG" in os.environ and int(os.environ["BC_SAFE_SEG"]):
        before = len(active)
        active = [s for s in active if s.get("total_cost", 0) <= 0]
        print(f"[BC-Safe-Seg] kept {len(active)}/{before} zero-cost segments "
              f"({100*len(active)/max(before,1):.1f}%)")
        if len(active) == 0:
            raise RuntimeError("BC-Safe-Seg filter rejected all segments")

    obs_dim = active[0]["observations"].shape[1]
    act_dim = active[0]["actions"].shape[1]

    bc_obs = torch.cat(
        [torch.as_tensor(s["observations"], dtype=torch.float32) for s in active], dim=0
    )
    bc_act = torch.cat(
        [torch.as_tensor(s["actions"], dtype=torch.float32) for s in active], dim=0
    )
    print(f"BC transitions: {bc_obs.size(0)}  obs_dim={obs_dim} act_dim={act_dim}")

    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)

    # BC-only baseline runs for `bc_only_epochs` (paper spec: 350).
    epochs = int(cfg.get("bc_only_epochs", 350))
    print(f"--- BC-only training for {epochs} epochs ---")
    bc_pretrain(
        policy, bc_obs, bc_act,
        batch_size=cfg["batch_size"],
        epochs=epochs,
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_path = os.path.join(cfg["output_dir"],
                            os.environ.get("BC_OUT", "bc_policy.pt"))
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_path)
    print(f"Saved BC policy -> {out_path}")
    if run is not None:
        import wandb
        wandb.save(out_path)
        wandb.finish()


if __name__ == "__main__":
    main()
