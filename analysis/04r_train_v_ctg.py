"""Step 04r: oracle-supervised value (distilled cost-to-go).

Trains a VEnsemble by per-state MSE regression to V*(s_t) = -G(t), the
discounted ground-truth cost-to-go. This value has NO Bradley-Terry
identifiability ambiguity and generalizes off the data manifold, so it
composes with the one-step dynamics ensemble to form an ACTION-CONDITIONED
ORACLE advantage A(s, a) = V*(f_hat(s, a)) - V*(s): the strongest test of
whether per-transition extraction fails even with exact, action-aware
signal (reviewer weakness 2ii).

Env: SEED_OVERRIDE, CTG_GAMMA (0.99), V_EPOCHS (25), BATCH_SIZE, V_OUT.
"""
from __future__ import annotations
import os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import VEnsemble  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    seed = int(cfg["seed"])
    gamma = float(os.environ.get("CTG_GAMMA", 0.99))
    epochs = int(os.environ.get("V_EPOCHS", 25))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = init_wandb(cfg, job_type="train_v_ctg")

    with open(cfg["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    obs_l, tgt_l = [], []
    for t in trajs:
        c = np.asarray(t["costs"], dtype=np.float32)
        G = np.zeros_like(c)
        G[-1] = c[-1]
        for i in range(len(c) - 2, -1, -1):
            G[i] = c[i] + gamma * G[i + 1]
        obs_l.append(t["observations"]); tgt_l.append(-G)
    obs = torch.as_tensor(np.concatenate(obs_l), dtype=torch.float32)
    tgt = torch.as_tensor(np.concatenate(tgt_l), dtype=torch.float32)
    # standardize targets for stable regression; scale is irrelevant to
    # ranking and the CF softmax is per-state normalized downstream
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-8)
    obs_dim = obs.shape[1]
    print(f"CTG distillation: {len(obs)} states, seed={seed}", flush=True)

    ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=3).to(device)
    opt = torch.optim.Adam(ens.parameters(), lr=cfg["learning_rate"])
    bs = int(cfg["batch_size"])
    n = len(obs)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            bo, bt = obs[idx].to(device), tgt[idx].to(device)
            pred = ens.forward_all(bo)                # [K, B]
            loss = F.mse_loss(pred, bt.unsqueeze(0).expand_as(pred))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{epochs} mse={tot/max(1,nb):.4f}", flush=True)

    out = os.path.join(cfg["output_dir"],
                       os.environ.get("V_OUT", f"v_ensemble_octg_seed{seed}.pt"))
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"], "K": 3,
                "state_dict": ens.state_dict()}, out)
    print(f"Saved -> {out}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
