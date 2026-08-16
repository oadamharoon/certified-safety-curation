"""Held-out V-ensemble diagnostics (manuscript Sec 4.4).

A) Held-out preference accuracy per task: 1000 fresh preference pairs
   sampled with a held-out RNG seed, predicted by each trained V ensemble
   (v_ensemble_pess_seed{0,1,2}.pt); reports mean/std across ensembles.
B) Cross-seed advantage Spearman on HC (all seed pairs, 5000 transitions).
"""
from __future__ import annotations
import itertools, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.policy import VEnsemble
from utils.segment_utils import sample_pair_indices

TASKS = [
    ("halfcheetah_velocity", None), ("walker2d_velocity", None),
    ("cargoal1_dsrl", None), ("cargoal2", None), ("pointgoal1_dsrl", None),
    ("ant_velocity", None), ("hopper_velocity", None), ("swimmer_velocity", None),
]
CKPT_CANDIDATES = ["v_ensemble_pess_seed{seed}.pt",
                   "v_ensemble_principled_seed{seed}.pt",
                   "v_ensemble_K3_nonorm_seed{seed}.pt"]
HIDDEN_DIM, K, N_PAIRS, HELDOUT_SEED, N_TRANS = 256, 3, 1000, 100, 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_ensemble(outdir, seed, obs_dim):
    for pat in CKPT_CANDIDATES:
        p = os.path.join(outdir, pat.format(seed=seed))
        if os.path.exists(p):
            ens = VEnsemble(obs_dim, HIDDEN_DIM, K=K).to(DEVICE)
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            ens.eval()
            return ens
    return None


def heldout_pref_accuracy(task):
    outdir = os.path.join(REPO, "outputs", task)
    segp = os.path.join(outdir, "active_segments.pkl")
    if not os.path.exists(segp):
        return None, "no segments"
    with open(segp, "rb") as f:
        active = pickle.load(f)
    obs_dim = active[0]["observations"].shape[1]
    enss = [e for s in (0, 1, 2) if (e := load_ensemble(outdir, s, obs_dim))]
    if not enss:
        return None, "no checkpoints"
    seen_t, costs = set(), []
    for s in active:
        if s["traj_id"] not in seen_t:
            seen_t.add(s["traj_id"]); costs.append(s["traj_total_cost"])
    costs = np.array(costs)
    safe_max, unsafe_min = np.percentile(costs, 25), np.percentile(costs, 75)
    rng = np.random.default_rng(HELDOUT_SEED)
    pairs, labels, seen = [], [], set()
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
        attempts += 1
        try:
            i, j = sample_pair_indices(active, cost_safe_threshold=0.0,
                                       cost_contrast_min=1.0, rng=rng,
                                       traj_safe_max_cost=float(safe_max),
                                       traj_safe_min_reward=-1.0,
                                       traj_unsafe_min_cost=float(unsafe_min))
        except Exception:
            continue
        key = (min(i, j), max(i, j))
        if key in seen: continue
        seen.add(key)
        a, b = active[i], active[j]
        if a["total_cost"] == b["total_cost"]: continue
        pairs.append((i, j)); labels.append(0 if a["total_cost"] < b["total_cost"] else 1)
    if len(pairs) < 200:
        return None, f"only {len(pairs)} pairs"
    accs = []
    for ens in enss:
        correct = 0
        for (i, j), lab in zip(pairs, labels):
            with torch.no_grad():
                va = ens(torch.as_tensor(active[i]["observations"], dtype=torch.float32).to(DEVICE)).sum().item()
                vb = ens(torch.as_tensor(active[j]["observations"], dtype=torch.float32).to(DEVICE)).sum().item()
            correct += int((0 if va > vb else 1) == lab)
        accs.append(correct / len(pairs))
    return (float(np.mean(accs)), float(np.std(accs)), len(pairs), len(enss)), "ok"


def cross_seed_spearman_hc():
    from scipy.stats import spearmanr
    outdir = os.path.join(REPO, "outputs", "halfcheetah_velocity")
    with open(os.path.join(outdir, "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    obs_dim = active[0]["observations"].shape[1]
    enss = [e for s in (0, 1, 2) if (e := load_ensemble(outdir, s, obs_dim))]
    if len(enss) < 2: return None
    T = active[0]["observations"].shape[0]
    rng = np.random.default_rng(42)
    sid = rng.integers(0, len(active), N_TRANS); st = rng.integers(0, T - 1, N_TRANS)
    s = torch.as_tensor(np.stack([active[i]["observations"][t] for i, t in zip(sid, st)]), dtype=torch.float32).to(DEVICE)
    sp = torch.as_tensor(np.stack([active[i]["observations"][t + 1] for i, t in zip(sid, st)]), dtype=torch.float32).to(DEVICE)
    advs = []
    for ens in enss:
        with torch.no_grad():
            advs.append((ens(sp) - ens(s)).cpu().numpy())
    return [((i, j), float(spearmanr(advs[i], advs[j])[0]))
            for i, j in itertools.combinations(range(len(advs)), 2)]


if __name__ == "__main__":
    print("=== A: held-out preference accuracy ===")
    for task, _ in TASKS:
        res, status = heldout_pref_accuracy(task)
        if res is None:
            print(f"{task:<22} skipped ({status})")
        else:
            m, sd, np_, ns = res
            print(f"{task:<22} acc={m:.3f} std={sd:.3f} pairs={np_} ensembles={ns}")
    print("\n=== B: cross-seed Spearman (HC) ===")
    r = cross_seed_spearman_hc()
    if r:
        for (i, j), rho in r:
            print(f"  seed {i} vs {j}: rho={rho:+.4f}")
