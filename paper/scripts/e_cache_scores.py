"""Item E, stage 1: cache per-trajectory scores and costs for the analysis tasks.

E tests Proposition 4's prediction directly by constructing pools at controlled unsafe
fractions and re-running the calibration walk on them. The score of a trajectory does
not depend on which other trajectories are in the pool, so every constructed pool can
reuse ONE scoring pass per (task, pipeline seed). This stage does that pass and stores
scores alongside episodic costs; stage 2 (e_contamination_sweep.py) needs no GPU.

Scoring replays 04q_calibrated_vfilter.score_trajectories exactly (same VEnsemble(K=3),
same checkpoint, mean over observations), the rule already validated in
recover_q1_stats.py by reproducing recorded cells to 1e-6.

Writes data/e_scores/<task>_seed<s>.npz with arrays: scores, costs, unsafe, limit.
"""
import os, pickle, sys
import numpy as np, torch, yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(os.path.dirname(BASE), "datasets")
sys.path.insert(0, REPO)
from model.policy import VEnsemble  # noqa: E402

TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
         "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2"]
OUTD = os.path.join(BASE, "data", "e_scores")


def score(ens, trajs, dev):
    out = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32).to(dev)
            vs = [ens(o[j:j + 8192]).cpu() for j in range(0, len(o), 8192)]
            out[i] = torch.cat(vs).mean().item()
    return out


def main():
    os.makedirs(OUTD, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(REPO, "config.yaml")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for t in TASKS:
        tc = cfg["tasks"][t]
        limit = float(tc.get("cost_limit", cfg["cost_limit"]))
        trajs = pickle.load(open(os.path.join(REPO, tc["data_pickle"]), "rb"))
        costs = np.array([float(np.sum(x["costs"])) for x in trajs])
        obs_dim = trajs[0]["observations"].shape[1]
        for s in (0, 1, 2):
            p = os.path.join(REPO, tc["output_dir"], f"v_ensemble_pess_seed{s}.pt")
            ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=3).to(dev)
            ck = torch.load(p, map_location=dev, weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval()
            sc = score(ens, trajs, dev)
            np.savez(os.path.join(OUTD, f"{t}_seed{s}.npz"), scores=sc, costs=costs,
                     unsafe=(costs > limit), limit=limit, checkpoint_mtime=os.path.getmtime(p))
            print(f"  {t:<22} seed{s}: N={len(sc)} base_unsafe={np.mean(costs>limit):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
