"""Step 04m: Extraction lab (Phase B' of the roadmap).

One script for the three extraction experiments that isolate WHERE the
V-to-policy bridge loses information:

  B'.0 oracle-V ablation   — ADV_SOURCE=oracle_step | oracle_ctg
       Swap learned V_theta for ground-truth cost-derived values in the
       IDENTICAL AWR pipeline. oracle ~= learned -> extraction is the
       bottleneck, not V quality.
         oracle_step: V(s_t) = -c_t          (the exact BT generator:
                      segment sums of -c reproduce segment-cost preferences)
         oracle_ctg:  V(s_t) = -G(t), G(t) = c_t + gamma*G(t+1)
                      (anticipatory smoothed variant)
  B'.1 weight-mode         — WEIGHT_MODE=exp | rank | binary
       rank consumes only advantage ORDERING (cross-seed stable, rho .63-.84);
       binary is the CRR indicator 1[A>0]. Both discard the cardinal tail
       that the BT loss does not identify (top-1% Jaccard 0.15).
  B'.2 horizon ablation    — AWR_K_STEP=1|3|5|10
       A_k = V(s_{t+k}) - V(s_t): maps the curve between k=1 (protective
       myopia, Phase 2) and k=inf (stationary collapse, Phase 3).

Env vars:
  ADV_SOURCE   learned | oracle_step | oracle_ctg   (default learned)
  WEIGHT_MODE  exp | rank | binary                  (default exp)
  AWR_K_STEP   int                                  (default 1)
  ORACLE_GAMMA float                                (default 0.99)
  V_ENSEMBLE_FILE  checkpoint in output_dir to LOAD for learned mode
                   (skips V training; e.g. v_ensemble_pess_seed0.pt).
                   If unset, trains a fresh ensemble (V_EPOCHS).
  OUT_TAG      output name tag (default: xlab_<source>_<mode>_k<k>_seed<seed>)
  BC_POLICY, SEED_OVERRIDE, AWR_BETA, AWR_EPOCHS, V_EPOCHS, BATCH_SIZE,
  AWR_NORMALIZE, DISAGREEMENT_LAMBDA  — as in 04h.

Saves <output_dir>/bc_<OUT_TAG>_policy.pt; eval via
  05_evaluate.py --task <task> --policy_file bc_<OUT_TAG>_policy.pt \
                 --results_suffix <OUT_TAG>
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
from model.train import train_v_ensemble, train_awr_flex  # noqa: E402


def build_transitions_with_costs(trajectories, k: int, gamma: float):
    """(s_t, a_t, s_{t+k}) plus aligned oracle values at t and t+k."""
    obs_l, act_l, next_obs_l = [], [], []
    c_t_l, c_tk_l, g_t_l, g_tk_l = [], [], [], []
    for traj in trajectories:
        obs = traj["observations"]
        act = traj["actions"]
        costs = np.asarray(traj["costs"], dtype=np.float32)
        if len(obs) < k + 1:
            continue
        # discounted cost-to-go, backward pass (includes current step)
        G = np.zeros_like(costs)
        G[-1] = costs[-1]
        for t in range(len(costs) - 2, -1, -1):
            G[t] = costs[t] + gamma * G[t + 1]
        obs_l.append(obs[:-k]); act_l.append(act[:-k]); next_obs_l.append(obs[k:])
        c_t_l.append(costs[:-k]); c_tk_l.append(costs[k:])
        g_t_l.append(G[:-k]);     g_tk_l.append(G[k:])
    cat = lambda xs: np.concatenate(xs, axis=0)
    return (cat(obs_l), cat(act_l), cat(next_obs_l),
            cat(c_t_l), cat(c_tk_l), cat(g_t_l), cat(g_tk_l))


@torch.no_grad()
def precompute_learned_adv(ensemble, obs_np, next_obs_np, device,
                           chunk: int = 8192) -> torch.Tensor:
    """Per-member advantages [K, N] from a frozen V ensemble."""
    ensemble.eval()
    outs = []
    for i in range(0, len(obs_np), chunk):
        bo = torch.as_tensor(obs_np[i:i+chunk], dtype=torch.float32).to(device)
        bn = torch.as_tensor(next_obs_np[i:i+chunk], dtype=torch.float32).to(device)
        outs.append((ensemble.forward_all(bn) - ensemble.forward_all(bo)).cpu())
    return torch.cat(outs, dim=1)  # [K, N]


def main() -> None:
    cfg = load_cfg()
    for var, key, cast in (
        ("AWR_BETA", "awr_beta", float),
        ("AWR_EPOCHS", "awr_epochs", int),
        ("V_EPOCHS", "v_epochs", int),
        ("BATCH_SIZE", "batch_size", int),
        ("AWR_NORMALIZE", "awr_normalize_adv", int),
        ("DISAGREEMENT_LAMBDA", "awr_disagreement_lambda", float),
        ("ENS_K", "v_ensemble_k", int),
        ("SEED_OVERRIDE", "seed", int),
    ):
        if var in os.environ:
            cfg[key] = cast(os.environ[var])

    adv_source = os.environ.get("ADV_SOURCE", "learned")
    weight_mode = os.environ.get("WEIGHT_MODE", "exp")
    k_step = int(os.environ.get("AWR_K_STEP", 1))
    oracle_gamma = float(os.environ.get("ORACLE_GAMMA", 0.99))
    assert adv_source in ("learned", "oracle_step", "oracle_ctg",
                          "oracle_rtg"), adv_source

    seed = int(cfg["seed"])
    out_tag = os.environ.get(
        "OUT_TAG", f"xlab_{adv_source}_{weight_mode}_k{k_step}_seed{seed}")

    set_seed(seed)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | source={adv_source} mode={weight_mode} "
          f"k={k_step} seed={seed} tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="extraction_lab")

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    (obs_np, act_np, next_obs_np,
     c_t, c_tk, g_t, g_tk) = build_transitions_with_costs(
        trajectories, k=k_step, gamma=oracle_gamma)
    print(f"Transitions: {len(obs_np)} from {len(trajectories)} trajs "
          f"(k={k_step})", flush=True)

    # --- advantage members [K, N] ---
    if adv_source == "oracle_rtg":
        # SANITY CHECK (reward task): V = +discounted reward-to-go.
        # If the AWR machinery works, this must raise reward above BC.
        import numpy as _np
        g_r_t, g_r_tk = [], []
        gamma = float(os.environ.get('ORACLE_GAMMA', 0.99))
        for traj in trajectories:
            r = _np.asarray(traj["rewards"], dtype=_np.float32)
            if len(traj["observations"]) < k_step + 1:
                continue
            G = _np.zeros_like(r); G[-1] = r[-1]
            for t in range(len(r) - 2, -1, -1):
                G[t] = r[t] + gamma * G[t + 1]
            g_r_t.append(G[:-k_step]); g_r_tk.append(G[k_step:])
        adv = torch.as_tensor(_np.concatenate(g_r_tk) - _np.concatenate(g_r_t),
                              dtype=torch.float32).unsqueeze(0)
    elif adv_source == "oracle_step":
        # V = -c  ->  A = c_t - c_{t+k}
        adv = torch.as_tensor(c_t - c_tk, dtype=torch.float32).unsqueeze(0)
    elif adv_source == "oracle_ctg":
        # V = -G  ->  A = G(t) - G(t+k)
        adv = torch.as_tensor(g_t - g_tk, dtype=torch.float32).unsqueeze(0)
    else:  # learned
        obs_dim = obs_np.shape[1]
        K = int(cfg.get("v_ensemble_k", 3))
        ens_file = os.environ.get("V_ENSEMBLE_FILE", "")
        ensemble = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)
        if ens_file:
            p = os.path.join(cfg["output_dir"], ens_file)
            ck = torch.load(p, map_location=device, weights_only=False)
            sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            ensemble.load_state_dict(sd)
            print(f"Loaded V-ensemble from {p}", flush=True)
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
            v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
            print(f"Training fresh V-ensemble (K={K}, {v_epochs} epochs, "
                  f"{lab.size(0)} pairs)", flush=True)
            train_v_ensemble(ensemble, oA, oB, lab,
                             epochs=v_epochs, batch_size=cfg["batch_size"],
                             lr=cfg["learning_rate"], device=device,
                             log_every=cfg["log_every"], wandb_run=run)
        adv = precompute_learned_adv(ensemble, obs_np, next_obs_np, device)

    # --- policy from BC ---
    bc_filename = os.environ.get("BC_POLICY", "bc_policy.pt")
    bc_path = os.path.join(cfg["output_dir"], bc_filename)
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    policy = GaussianPolicy(bc_ckpt["obs_dim"], bc_ckpt["act_dim"],
                            bc_ckpt["hidden_dim"]).to(device)
    policy.load_state_dict(bc_ckpt["state_dict"])
    print(f"BC start: {bc_path}", flush=True)

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    act_t = torch.as_tensor(act_np, dtype=torch.float32)

    train_awr_flex(
        policy, obs_t, act_t, adv,
        epochs=int(cfg.get("awr_epochs", cfg["cpl_epochs"])),
        batch_size=cfg["batch_size"],
        lr=cfg["learning_rate"],
        beta=float(cfg.get("awr_beta", 0.5)),
        device=device,
        weight_mode=weight_mode,
        weight_clip=float(cfg.get("awr_weight_clip", 20.0)),
        normalize_adv=bool(int(cfg.get("awr_normalize_adv", 0))),
        disagreement_lambda=float(cfg.get("awr_disagreement_lambda", 1.0)),
        log_every=cfg["log_every"],
        wandb_run=run,
    )

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": bc_ckpt["obs_dim"], "act_dim": bc_ckpt["act_dim"],
                "hidden_dim": bc_ckpt["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    print(f"Saved policy -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
