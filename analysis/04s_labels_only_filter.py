"""Step 04s: labels-only filter control.

Answers "why preferences at all -- would the 200 calibration labels alone
suffice?" Trains a supervised state-level safety classifier on ONLY the
CAL_N uniformly sampled trajectories (identical draw protocol to 04q:
rng = default_rng(1000 + seed)), scores every trajectory by mean predicted
safety, keeps the top FILTER_FRAC fraction, and behavior-clones the
selection. Same architecture and scoring rule as the V-ensemble route;
the only change is the supervision source (200 episodic-cost labels vs
1000 segment preferences). Matched-fraction control: no certificate is
claimed (training and calibrating on the same labels would double-dip).

Env: CAL_N (200), FILTER_FRAC (required unless SPLIT_CAL), SEED_OVERRIDE,
     CLS_EPOCHS (50), BATCH_SIZE, OUT_TAG (default labelsonly_seed{seed}).
     SPLIT_CAL (optional): train/calibrate split control. The CAL_N draw is
     split; the first CAL_N - SPLIT_CAL trajectories train the classifier
     and the last SPLIT_CAL run strict fixed-sequence LTT (exact
     hypergeometric tests, quantile grid 0.85..0.30) on classifier scores,
     mirroring 04q. Uncertified fallback: top 20 percent.
Saves <output_dir>/bc_<OUT_TAG>_policy.pt; eval via 05_evaluate.py.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import GaussianPolicy, VEnsemble  # noqa: E402
from model.train import bc_pretrain  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    seed = int(cfg["seed"])
    set_seed(seed)
    cal_n = int(os.environ.get("CAL_N", 200))
    split_cal = int(os.environ.get("SPLIT_CAL", 0))
    frac = float(os.environ["FILTER_FRAC"]) if not split_cal else None
    epochs = int(os.environ.get("CLS_EPOCHS", 50))
    batch_size = int(os.environ.get("BATCH_SIZE", cfg["batch_size"]))
    out_tag = os.environ.get("OUT_TAG", f"labelsonly_seed{seed}")
    limit = float(cfg["cost_limit"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | cal_n={cal_n} frac={frac} seed={seed} "
          f"tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="labels_only_filter")

    with open(cfg["data_pickle"], "rb") as f:
        trajs = pickle.load(f)
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    unsafe = (costs > limit).astype(np.float32)

    # identical calibration draw to 04q
    label_draw_seed = int(os.environ.get("LABEL_DRAW_SEED", seed))
    rng = np.random.default_rng(1000 + label_draw_seed)
    cal_idx = rng.choice(len(trajs), size=cal_n, replace=False)
    print(f"calibration sample: {cal_n} trajs, unsafe rate "
          f"{unsafe[cal_idx].mean():.3f}", flush=True)
    ltt_idx = None
    if split_cal:
        ltt_idx = cal_idx[cal_n - split_cal:]
        cal_idx = cal_idx[:cal_n - split_cal]
        print(f"split: {len(cal_idx)} train / {len(ltt_idx)} calibrate", flush=True)

    # state-level classifier: states labeled by parent trajectory's safety.
    # STATE_FRAC (mechanism test): subsample this fraction of states per
    # labeled trajectory, shrinking the effective weak-label count while
    # holding the trajectory-label budget fixed.
    state_frac = float(os.environ.get("STATE_FRAC", 1.0))
    sub_rng = np.random.default_rng(4200 + seed)
    Xs, ys = [], []
    for i in cal_idx:
        o = trajs[i]["observations"]
        if state_frac < 1.0:
            m_states = max(1, int(len(o) * state_frac))
            sel_states = sub_rng.choice(len(o), size=m_states, replace=False)
            o = o[sel_states]
        Xs.append(torch.as_tensor(o, dtype=torch.float32))
        ys.append(torch.full((len(o),), float(1 - unsafe[i])))
    X = torch.cat(Xs).to(device)
    y = torch.cat(ys).to(device)
    obs_dim = X.shape[1]

    # same architecture family as one V-ensemble member
    net = VEnsemble(obs_dim, cfg["hidden_dim"], K=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    n = len(X)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logit = net.forward_all(X[idx]).squeeze(0)
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == 1:
            print(f"  cls epoch {ep}/{epochs} loss={tot/n:.4f}", flush=True)

    # score every trajectory: mean predicted safety logit
    net.eval()
    scores = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32).to(device)
            vs = [net.forward_all(o[j:j + 8192]).squeeze(0).cpu()
                  for j in range(0, len(o), 8192)]
            scores[i] = torch.cat(vs).mean().item()

    certified, tau = None, None
    if split_cal:
        from scipy.stats import hypergeom
        alpha, delta = 0.25, 0.1
        qs = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
              0.35, 0.30]
        # Validity: conditioned on the classifier, the calibration half is a
        # uniform draw from D' = D minus the training trajectories, so the
        # certificate is computed entirely over D' (selection sizes, grid
        # quantiles, and the certified selection). Known-safe training
        # trajectories are unioned into the clone set afterward; their labels
        # are certain, so the certified bound on the unsafe fraction of the
        # final set still holds.
        train_set = set(int(i) for i in cal_idx)
        pool = np.array([i for i in range(len(trajs)) if i not in train_set])
        pool_scores = scores[pool]
        cal_scores = scores[ltt_idx]
        cal_unsafe = unsafe[ltt_idx].astype(int)
        chosen, certified = None, False
        for q in qs:
            t_q = float(np.quantile(pool_scores, q))
            sel = cal_scores >= t_q
            m = int(sel.sum()); k = int(cal_unsafe[sel.astype(bool)].sum())
            n_sel = int((pool_scores >= t_q).sum())
            k_star = int(alpha * n_sel) + 1
            p = float(hypergeom.cdf(k, n_sel, k_star, m)) if (m > 0 and k_star <= n_sel) else 1.0
            if m > 0 and p <= delta:
                chosen, certified = t_q, True
            else:
                break
        known_safe_train = np.array([i for i in cal_idx if unsafe[i] == 0],
                                    dtype=int)
        if certified:
            tau = chosen
            keep = pool[pool_scores >= tau]
            if len(keep) < 50:
                keep = pool[np.argsort(pool_scores)[-50:]]
        else:
            keep = pool[np.argsort(pool_scores)[-max(50, int(0.2 * len(pool))):]]
        keep = np.concatenate([keep, known_safe_train])
        print(f"LTT: certified={certified} tau={tau} "
              f"(+{len(known_safe_train)} known-safe train trajs)", flush=True)
    else:
        n_keep = max(1, int(len(trajs) * frac))
        keep = np.argsort(scores)[-n_keep:]
    kept_unsafe = unsafe[keep].mean()
    print(f"kept {len(keep)}/{len(trajs)} | kept unsafe rate {kept_unsafe:.3f} "
          f"(dataset {unsafe.mean():.3f})", flush=True)

    obs = torch.cat([torch.as_tensor(trajs[i]["observations"],
                                     dtype=torch.float32) for i in keep])
    act = torch.cat([torch.as_tensor(trajs[i]["actions"],
                                     dtype=torch.float32) for i in keep])
    act_dim = act.shape[1]
    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    bc_pretrain(policy, obs, act, batch_size=batch_size,
                epochs=int(os.environ.get("BC_EPOCHS",
                                          cfg.get("bc_epochs", 100))),
                lr=cfg["learning_rate"], device=device,
                log_every=cfg["log_every"], wandb_run=run)

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    import json
    with open(os.path.join(cfg["output_dir"], f"labelsonly_meta_{out_tag}.json"),
              "w") as f:
        json.dump({"cal_n": cal_n, "frac": frac, "split_cal": split_cal,
                   "certified": certified, "tau": tau,
                   "kept_frac": float(len(keep) / len(trajs)),
                   "kept_unsafe_rate": float(kept_unsafe),
                   "dataset_unsafe_rate": float(unsafe.mean())}, f)
    print(f"Saved -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
