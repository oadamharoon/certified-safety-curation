"""Step 04p: Learned-V trajectory filtering + BC (Phase A synthesis).

The V->action mapping realized at the granularity where the preference-
trained V actually has signal. B'.0-B'.3 established that per-transition
V-differences are noise-dominated while segment/trajectory-level V sums
rank safety at 0.83-1.0 held-out accuracy. Tier 1.1 established that
trajectory-filtered BC (with a ground-truth cost filter) is safe on 7/8
tasks. This script closes the loop: replace the ground-truth filter with
the LEARNED V-ensemble score, keeping the pipeline cost-oracle-free.

  1. Score every offline trajectory: score(tau) = mean_t V_bar(s_t).
  2. Keep the top FILTER_FRAC fraction by score.
  3. BC on the kept trajectories' transitions.

Also reports filter quality vs the ground-truth cost filter (precision of
the kept set against traj_cost <= limit) since costs are available for
diagnostics.

Env vars:
  FILTER_FRAC     fraction of trajectories kept          (required)
  COST_LIMIT      task cost limit for the diagnostic     (required)
  V_ENSEMBLE_FILE V ensemble checkpoint in output_dir    (required)
  BC_EPOCHS, BATCH_SIZE, SEED_OVERRIDE, OUT_TAG          as usual
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy, VEnsemble  # noqa: E402
from model.train import bc_pretrain  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    if "BC_EPOCHS" in os.environ:
        cfg["bc_only_epochs"] = int(os.environ["BC_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    seed = int(cfg["seed"])
    frac = float(os.environ["FILTER_FRAC"])
    cost_limit = float(os.environ["COST_LIMIT"])
    ens_file = os.environ["V_ENSEMBLE_FILE"]
    out_tag = os.environ.get("OUT_TAG", f"vfilt{int(100*frac)}_seed{seed}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | frac={frac} seed={seed} tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="vfilter_bc")

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]

    score_mode_early = os.environ.get("SCORE_MODE", "vscore")
    ens = None
    if score_mode_early == "vscore":
        ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=3).to(device)
        ck = torch.load(os.path.join(cfg["output_dir"], ens_file),
                        map_location=device, weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()

    # Trajectory scoring. SCORE_MODE controls the ranking signal:
    #   vscore (default) - mean ensemble V over states (the method)
    #   return           - episodic return (%BC-style control: does the
    #                      safety-V ranking add anything beyond return?)
    #   random           - uniform random permutation (selection-size control)
    score_mode = os.environ.get("SCORE_MODE", "vscore")
    if score_mode == "return":
        scores = np.array([float(np.sum(t["rewards"])) for t in trajectories])
    elif score_mode == "return_bottom":
        scores = -np.array([float(np.sum(t["rewards"])) for t in trajectories])
    elif score_mode == "random":
        scores = np.random.default_rng(seed).permutation(len(trajectories)).astype(float)
    else:
        scores = np.zeros(len(trajectories))
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                o = torch.as_tensor(traj["observations"], dtype=torch.float32).to(device)
                vs = []
                for j in range(0, len(o), 8192):
                    vs.append(ens(o[j:j+8192]).cpu())
                _v = torch.cat(vs)
                _agg = os.environ.get("AGG_MODE", "mean")
                if _agg == "min":
                    scores[i] = _v.min().item()
                elif _agg == "p10":
                    scores[i] = _v.quantile(0.1).item()
                else:
                    scores[i] = _v.mean().item()

    n_keep = max(1, int(len(trajectories) * frac))
    kept_idx = set(np.argsort(scores)[::-1][:n_keep].tolist())

    # Diagnostic: precision of the learned filter against the GT cost filter.
    traj_costs = np.array([float(np.sum(t["costs"])) for t in trajectories])
    gt_safe = set(np.where(traj_costs <= cost_limit)[0].tolist())
    inter = len(kept_idx & gt_safe)
    kept_costs = traj_costs[sorted(kept_idx)]
    print(f"[filter] kept {n_keep}/{len(trajectories)} "
          f"({100*frac:.0f}%) | precision vs GT-safe: {inter/n_keep:.3f} "
          f"(GT-safe base rate {len(gt_safe)/len(trajectories):.3f}) | "
          f"kept-set mean traj cost {kept_costs.mean():.1f} "
          f"vs all {traj_costs.mean():.1f}", flush=True)

    obs = torch.cat([torch.as_tensor(trajectories[i]["observations"],
                                     dtype=torch.float32) for i in sorted(kept_idx)])
    act = torch.cat([torch.as_tensor(trajectories[i]["actions"],
                                     dtype=torch.float32) for i in sorted(kept_idx)])
    print(f"BC transitions: {len(obs)}", flush=True)

    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    bc_pretrain(policy, obs, act,
                batch_size=cfg["batch_size"],
                epochs=int(cfg.get("bc_only_epochs", 100)),
                lr=cfg["learning_rate"], device=device,
                log_every=cfg["log_every"], wandb_run=run)

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    print(f"Saved -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
