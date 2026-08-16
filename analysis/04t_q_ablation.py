"""Step 04t: Q(s,a) ablation for the state-only design choice.

Trains an action-conditioned ensemble Q(s,a) with the IDENTICAL
Bradley-Terry segment-sum loss used for V(s) (only the input changes:
concat(obs, act)), scores each trajectory by the mean of Q over its
(state, action) pairs, keeps the top FILTER_FRAC fraction, and
behavior-clones the selection. Everything downstream of supervision is
unchanged, so any difference is attributable to action-conditioning.

Env: FILTER_FRAC (required), SEED_OVERRIDE, ENS_K (3), V_EPOCHS,
     BATCH_SIZE, BC_EPOCHS, OUT_TAG (default qfilt_seed{seed}).
Saves <output_dir>/bc_<OUT_TAG>_policy.pt; eval via 05_evaluate.py.
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
from model.train import train_v_ensemble, bc_pretrain  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    for var, key, cast in (("ENS_K", "v_ensemble_k", int),
                           ("V_EPOCHS", "v_epochs", int),
                           ("BATCH_SIZE", "batch_size", int),
                           ("SEED_OVERRIDE", "seed", int)):
        if var in os.environ:
            cfg[key] = cast(os.environ[var])
    seed = int(cfg["seed"])
    set_seed(seed)
    frac = float(os.environ["FILTER_FRAC"])
    out_tag = os.environ.get("OUT_TAG", f"qfilt_seed{seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | frac={frac} seed={seed} tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="q_ablation")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    with open(os.path.join(cfg["output_dir"], "gt_labels.json")) as f:
        pref_data = json.load(f)

    def seg_input(s):
        return np.concatenate([s["observations"], s["actions"]], axis=1)

    xA = torch.stack([torch.as_tensor(seg_input(active[e["seg_A_idx"]]),
                                      dtype=torch.float32) for e in pref_data])
    xB = torch.stack([torch.as_tensor(seg_input(active[e["seg_B_idx"]]),
                                      dtype=torch.float32) for e in pref_data])
    lab = torch.tensor([e["preference"] for e in pref_data], dtype=torch.long)
    in_dim = xA.shape[-1]
    K = int(cfg.get("v_ensemble_k", 3))
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"Q(s,a): in_dim={in_dim}, K={K}, {v_epochs} epochs, "
          f"{lab.size(0)} pairs", flush=True)
    ens = VEnsemble(in_dim, cfg["hidden_dim"], K=K).to(device)
    train_v_ensemble(ens, xA, xB, lab, epochs=v_epochs,
                     batch_size=cfg["batch_size"], lr=cfg["learning_rate"],
                     device=device, log_every=cfg["log_every"], wandb_run=run)

    with open(cfg["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    ens.eval()
    scores = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            x = np.concatenate([t["observations"], t["actions"]], axis=1)
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            vs = [ens.forward_all(x[j:j + 8192]).mean(0).cpu()
                  for j in range(0, len(x), 8192)]
            scores[i] = torch.cat(vs).mean().item()

    n_keep = max(1, int(len(trajs) * frac))
    keep = np.argsort(scores)[-n_keep:]
    limit = float(cfg["cost_limit"])
    unsafe = np.array([float(np.sum(t["costs"])) > limit for t in trajs])
    print(f"kept {n_keep}/{len(trajs)} | kept unsafe rate "
          f"{unsafe[keep].mean():.3f} (dataset {unsafe.mean():.3f})", flush=True)

    obs = torch.cat([torch.as_tensor(trajs[i]["observations"],
                                     dtype=torch.float32) for i in keep])
    act = torch.cat([torch.as_tensor(trajs[i]["actions"],
                                     dtype=torch.float32) for i in keep])
    policy = GaussianPolicy(obs.shape[1], act.shape[1],
                            cfg["hidden_dim"]).to(device)
    bc_pretrain(policy, obs, act, batch_size=int(cfg["batch_size"]),
                epochs=int(os.environ.get("BC_EPOCHS",
                                          cfg.get("bc_epochs", 100))),
                lr=cfg["learning_rate"], device=device,
                log_every=cfg["log_every"], wandb_run=run)

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": obs.shape[1], "act_dim": act.shape[1],
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    with open(os.path.join(cfg["output_dir"], f"qfilt_meta_{out_tag}.json"),
              "w") as f:
        json.dump({"kept_unsafe_rate": float(unsafe[keep].mean()),
                   "frac": frac}, f)
    print(f"Saved -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
