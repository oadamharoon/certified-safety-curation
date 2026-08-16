"""Two stats-only analyses for the paper (no new training).

1. Composition -> policy transfer: scatter of kept-set true unsafe rate vs
   resulting policy cost (normalized by task budget) across all calibrated
   runs. Addresses the 'certificate covers data, not policy' limitation
   with an empirical transfer curve. Saves data + figure.

2. Ensemble-K ablation for trajectory scoring: filter precision at the
   matched fraction using single ensemble members (K=1) vs the K=3 mean
   (and K=5 on HalfCheetah where those checkpoints exist). No policy
   training; precision of the selection is the quantity the certificate
   consumes.
"""
import json, glob, os, re, pickle, sys
import numpy as np
import torch

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data"
sys.path.insert(0, REPO)
os.chdir(REPO)
from model.policy import VEnsemble
import yaml

LIMITS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
          "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
          "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
GT_FRACS = {"halfcheetah_velocity": 0.182, "walker2d_velocity": 0.284,
            "ant_velocity": 0.197, "hopper_velocity": 0.111,
            "swimmer_velocity": 0.139, "cargoal1_dsrl": 0.399,
            "cargoal2": 0.324, "pointgoal1_dsrl": 0.518, "pointgoal2": 0.257}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfer_points():
    pts = []
    for f in glob.glob("outputs/*/calfilt_meta_calfilt_*_seed*.json"):
        task = f.split("/")[1]
        if task not in LIMITS:   # analysis set only; other tasks live in the main tables
            continue
        m = re.search(r"calfilt_(lttv2R50|lttv2|pref)_seed(\d+)", f)
        if not m:
            continue
        variant, seed = m.groups()
        ev = f"outputs/{task}/eval_results_calfilt_{variant}_seed{seed}.json"
        if not os.path.exists(ev):
            continue
        meta = json.load(open(f))
        r = json.load(open(ev))
        pts.append({"task": task, "variant": variant, "seed": int(seed),
                    "kept_unsafe": meta["kept_unsafe_rate"],
                    "kept_frac": meta["kept_frac"],
                    "policy_cost_norm": r["avg_cost"] / LIMITS[task],
                    "policy_R": r["avg_reward"]})
    return pts


def k_ablation():
    cfg = yaml.safe_load(open("config.yaml"))
    rows = []
    for task, frac in GT_FRACS.items():
        with open(cfg["tasks"][task]["data_pickle"], "rb") as f:
            trajs = pickle.load(f)
        costs = np.array([float(np.sum(t["costs"])) for t in trajs])
        safe = costs <= LIMITS[task]
        obs_dim = trajs[0]["observations"].shape[1]
        n_keep = max(1, int(len(trajs) * frac))

        def precision_from_scores(scores):
            kept = np.argsort(scores)[::-1][:n_keep]
            return float(safe[kept].mean())

        for seed in (0, 1, 2):
            p = f"outputs/{task}/v_ensemble_pess_seed{seed}.pt"
            if not os.path.exists(p):
                continue
            ens = VEnsemble(obs_dim, 256, K=3).to(DEVICE)
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            ens.eval()
            member_scores = np.zeros((3, len(trajs)))
            with torch.no_grad():
                for i, t in enumerate(trajs):
                    o = torch.as_tensor(t["observations"],
                                        dtype=torch.float32).to(DEVICE)
                    vs = [ens.forward_all(o[j:j+8192]).cpu()
                          for j in range(0, len(o), 8192)]
                    member_scores[:, i] = torch.cat(vs, dim=1).mean(dim=1).numpy()
            p_members = [precision_from_scores(member_scores[k])
                         for k in range(3)]
            p_mean = precision_from_scores(member_scores.mean(axis=0))
            rows.append({"task": task, "seed": seed,
                         "precision_K1_members": p_members,
                         "precision_K1_mean": float(np.mean(p_members)),
                         "precision_K3": p_mean,
                         "base_rate": float(safe.mean())})
            print(f"{task} seed{seed}: K1 {np.mean(p_members):.3f} "
                  f"(members {[round(x,3) for x in p_members]}) "
                  f"K3 {p_mean:.3f} base {safe.mean():.3f}", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pts = transfer_points()
    with open(os.path.join(OUT, "transfer_points.json"), "w") as f:
        json.dump(pts, f, indent=1)
    print(f"transfer points: {len(pts)} -> data/transfer_points.json")
    rows = k_ablation()
    with open(os.path.join(OUT, "k_ablation.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"k-ablation rows: {len(rows)} -> data/k_ablation.json")


if __name__ == "__main__":
    main()
