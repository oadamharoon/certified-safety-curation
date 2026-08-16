"""Step 04n: train and save a V-ensemble ONLY (no AWR stage).

Purpose: fill the held-out V-accuracy gap for tasks whose 5-seed final runs
did not retain the V-ensemble checkpoint (walker2d/ant/hopper/swimmer).
Saves with the v_ensemble_pess_seed{seed}.pt naming so the existing
diagnostic script's checkpoint-pattern lookup finds them.

Env vars: SEED_OVERRIDE, ENS_K, V_EPOCHS, BATCH_SIZE (as in 04h).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed, init_wandb  # noqa: E402
from model.policy import VEnsemble  # noqa: E402
from model.train import train_v_ensemble  # noqa: E402


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = init_wandb(cfg, job_type="train_v_only")

    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    labels_file = os.environ.get("PREF_LABELS_FILE", "gt_labels.json")
    with open(os.path.join(cfg["output_dir"], labels_file), "r") as f:
        pref_data = json.load(f)

    # Preference-budget ablation: train on a random subset of N pairs.
    if "PREF_SUBSET" in os.environ:
        n_sub = int(os.environ["PREF_SUBSET"])
        if n_sub < len(pref_data):
            import numpy as np
            rng = np.random.default_rng(4000 + int(cfg["seed"]))
            keep = rng.choice(len(pref_data), size=n_sub, replace=False)
            pref_data = [pref_data[i] for i in keep]
            print(f"[subset] using {len(pref_data)} preference pairs", flush=True)
        elif n_sub > len(pref_data):
            print(f"[subset] requested {n_sub} > available "
                  f"{len(pref_data)}; using all", flush=True)

    obs_dim = active[0]["observations"].shape[1]
    oA = torch.stack([torch.as_tensor(active[e["seg_A_idx"]]["observations"],
                                      dtype=torch.float32) for e in pref_data])
    oB = torch.stack([torch.as_tensor(active[e["seg_B_idx"]]["observations"],
                                      dtype=torch.float32) for e in pref_data])
    lab = torch.tensor([e["preference"] for e in pref_data], dtype=torch.long)

    # Boltzmann labeler (PREF_BOLTZ = temperature T > 0): resample each
    # label with P(prefer the truly-safer segment) = sigmoid(cost_gap / T),
    # where cost_gap is the pair's segment-cost difference. Errors thus
    # concentrate on borderline pairs, a more realistic noise model than
    # uniform flips. T is specified in cost units.
    if "PREF_BOLTZ" in os.environ:
        import numpy as np
        T = float(os.environ["PREF_BOLTZ"])
        rng_b = np.random.default_rng(7000 + seed)
        with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f2:
            act2 = pickle.load(f2)
        flips = 0
        for i, e in enumerate(pref_data):
            cA = act2[e["seg_A_idx"]]["total_cost"]
            cB = act2[e["seg_B_idx"]]["total_cost"]
            gap = abs(cA - cB)
            p_correct = 1.0 / (1.0 + np.exp(-gap / T))
            if rng_b.random() > p_correct:
                lab[i] = 1 - lab[i]
                flips += 1
        print(f"[boltz] T={T}: flipped {flips}/{len(pref_data)} "
              f"({100*flips/len(pref_data):.1f}%)", flush=True)

    # Label-noise robustness (PREF_NOISE in [0,1]): flip each preference
    # label independently with the given probability. Seeded by the run seed
    # so noise realizations are reproducible and differ across seeds.
    noise = float(os.environ.get("PREF_NOISE", 0))
    if noise > 0:
        import numpy as np
        rng = np.random.default_rng(9000 + seed)
        flip = torch.as_tensor(rng.random(lab.shape[0]) < noise)
        lab = torch.where(flip, 1 - lab, lab)
        print(f"[noise] flipped {int(flip.sum())}/{lab.shape[0]} labels "
              f"(p={noise})", flush=True)

    K = int(cfg.get("v_ensemble_k", 3))
    v_epochs = int(cfg.get("v_epochs", cfg["cpl_epochs"]))
    print(f"V-only: K={K}, {v_epochs} epochs, {lab.size(0)} pairs, seed={seed}",
          flush=True)
    ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=K).to(device)
    train_v_ensemble(ens, oA, oB, lab, epochs=v_epochs,
                     batch_size=cfg["batch_size"], lr=cfg["learning_rate"],
                     device=device, log_every=cfg["log_every"], wandb_run=run)

    out = os.path.join(cfg["output_dir"],
                       os.environ.get("V_OUT", f"v_ensemble_pess_seed{seed}.pt"))
    torch.save({"obs_dim": obs_dim, "hidden_dim": cfg["hidden_dim"], "K": K,
                "state_dict": ens.state_dict()}, out)
    print(f"Saved -> {out}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
