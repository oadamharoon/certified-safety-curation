"""Step 00a: Train a BC policy on DSRL offline data.

DSRL provides offline trajectories from trained safe-RL policies (TRPO-Lag /
CPO from FSRL) — high-quality, diverse, but without reset seeds, so the
trajectories can't be re-rendered for VLM labeling.

This script distills DSRL into a GaussianPolicy via behaviour cloning. The
trained BC policy is then used by ``00_collect_online_data.py`` (when
``collect.use_bc_policy`` is enabled) as the action source for *fresh*
online collection with stored seeds — giving us DSRL-quality behaviour
*and* seed-matched rendering downstream.

Usage:
    PYTHONNOUSERSITE=1 python scripts/00a_train_bc_from_dsrl.py --task pointgoal1
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy  # noqa: E402
from model.train import bc_pretrain  # noqa: E402


def _load_dsrl(offline_env_name: str):
    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"):
            dsrl.register_envs()
    except Exception:
        pass
    env = gym.make(offline_env_name)
    dataset = env.get_dataset()
    env.close()
    return dataset


def _load_metadrive_hdf5(cfg) -> dict:
    """Load a DSRL MetaDrive offline dataset straight from its hdf5 file.

    dsrl's MetaDrive gym envs are incompatible with metadrive 0.4.3, so we read
    the cached hdf5 directly. Path comes from cfg['dsrl_hdf5'] or is globbed
    from the standard DSRL cache by cfg['dsrl_hdf5_glob'].
    """
    import glob
    import h5py
    path = cfg.get("dsrl_hdf5")
    if not path:
        pattern = cfg.get("dsrl_hdf5_glob")
        if not pattern:
            raise KeyError("metadrive task needs 'dsrl_hdf5' or 'dsrl_hdf5_glob' in config")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No DSRL hdf5 matched {pattern!r}")
        path = matches[0]
    print(f"Loading MetaDrive DSRL hdf5: {path}")
    with h5py.File(path, "r") as f:
        return {"observations": f["observations"][:], "actions": f["actions"][:]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="BC training epochs (default: cfg.dsrl_bc_epochs or 50)")
    args, _ = ap.parse_known_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run = init_wandb(cfg, job_type="train_dsrl_bc")

    if cfg.get("data_source") == "metadrive":
        dataset = _load_metadrive_hdf5(cfg)
    else:
        # The DSRL offline env carries the dataset. If the config doesn't have
        # a proper "Offline..." name, derive it from env_name.
        offline_env_name = cfg.get("offline_env_name")
        if not offline_env_name or not offline_env_name.startswith("Offline"):
            offline_env_name = cfg["env_name"].replace("Safety", "Offline", 1)
        print(f"Loading DSRL dataset from {offline_env_name} ...")
        dataset = _load_dsrl(offline_env_name)

    obs = np.asarray(dataset["observations"], dtype=np.float32)
    act = np.asarray(dataset["actions"],      dtype=np.float32)
    n_transitions = obs.shape[0]
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]
    print(f"Loaded {n_transitions} transitions  obs_dim={obs_dim} act_dim={act_dim}")

    bc_obs = torch.from_numpy(obs)
    bc_act = torch.from_numpy(act)

    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)

    epochs = int(args.epochs if args.epochs is not None else cfg.get("dsrl_bc_epochs", 50))
    print(f"Training BC for {epochs} epochs ...")
    bc_pretrain(
        policy, bc_obs, bc_act,
        batch_size=cfg["batch_size"],
        epochs=epochs,
        lr=cfg["learning_rate"],
        device=device,
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_path = os.path.join(cfg["output_dir"], "dsrl_bc_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_path)
    print(f"Saved DSRL-BC policy -> {out_path}")
    if run is not None:
        import wandb
        wandb.save(out_path)
        wandb.finish()


if __name__ == "__main__":
    main()
