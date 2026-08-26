"""Regenerate the four tab:obstacle2 quantities, reproducibly.

The printed table was hardcoded and its source analysis was lost. This
recomputes all four rows from artifacts on disk and writes JSON that
make_tables.py consumes, so the table can no longer drift.

Rows, and the scope each is measured over:
  A held-out segment-pair accuracy   per task, range over the nine analysis tasks
  B cross-seed advantage Spearman    per task, over all seed pairs of that task
  C top-1 percent advantage Jaccard  per task, over all seed pairs of that task
  D action-value separation          per task, spread of V(f(s,a)) over BC-sampled
                                     candidate actions at a fixed state, using the
                                     same one-step dynamics ensembles that
                                     04o_cf_extraction scores with

Every row is computed on all nine tasks with an identical protocol, so the
reported ranges are cross-task ranges, which is what the caption claims. The
previous table mixed scopes: its per-transition rows were HalfCheetah seed-pair
statistics presented as cross-task ranges.
"""
from __future__ import annotations
import itertools, json, os, pickle, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from model.policy import VEnsemble                      # noqa: E402
from utils.segment_utils import sample_pair_indices     # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
         "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2"]
SEEDS = (0, 1, 2)          # the analysis-sweep convention the paper uses
N_TRANS = 5000
N_PAIRS = 1000
HIDDEN = 256


def load_ensemble(outdir, seed, obs_dim):
    p = os.path.join(outdir, f"v_ensemble_pess_seed{seed}.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    ens = VEnsemble(obs_dim, ck.get("hidden_dim", HIDDEN), K=ck.get("K", 3)).to(DEVICE)
    ens.load_state_dict(ck["state_dict"])
    ens.eval()
    return ens


def task_stats(task):
    outdir = os.path.join(REPO, "outputs", task)
    seg_p = os.path.join(outdir, "active_segments.pkl")
    if not os.path.exists(seg_p):
        return None
    active = pickle.load(open(seg_p, "rb"))
    obs_dim = active[0]["observations"].shape[1]
    enss = [e for s in SEEDS if (e := load_ensemble(outdir, s, obs_dim)) is not None]
    if len(enss) < 2:
        return None
    T = active[0]["observations"].shape[0]

    # --- A: held-out preference accuracy, fresh pairs from a held-out rng
    # identical protocol to the pipeline: parents drawn from the bottom and top
    # quartiles of episodic cost, which is what the paper documents
    seen_t, tcosts = set(), []
    for sg in active:
        if sg["traj_id"] not in seen_t:
            seen_t.add(sg["traj_id"]); tcosts.append(sg["traj_total_cost"])
    tcosts = np.asarray(tcosts)
    safe_max, unsafe_min = np.percentile(tcosts, 25), np.percentile(tcosts, 75)
    rng = np.random.default_rng(20260825)
    accs = []
    pairs, tries = [], 0
    while len(pairs) < N_PAIRS and tries < N_PAIRS * 50:
        tries += 1
        try:
            pairs.append(sample_pair_indices(
                active, cost_safe_threshold=0.0, cost_contrast_min=1.0, rng=rng,
                traj_safe_max_cost=float(safe_max), traj_safe_min_reward=-1.0,
                traj_unsafe_min_cost=float(unsafe_min)))
        except Exception:
            continue
    for ens in enss:
        ok = 0
        for i, j in pairs:
            with torch.no_grad():
                oi = torch.as_tensor(active[i]["observations"], dtype=torch.float32).to(DEVICE)
                oj = torch.as_tensor(active[j]["observations"], dtype=torch.float32).to(DEVICE)
                si, sj = float(ens(oi).mean()), float(ens(oj).mean())
            ci = float(np.sum(active[i]["costs"])); cj = float(np.sum(active[j]["costs"]))
            if ci == cj:
                continue
            ok += int((si > sj) == (ci < cj))
        accs.append(ok / len(pairs))

    # --- B and C: cross-seed advantage agreement on matched transitions
    r2 = np.random.default_rng(42)
    sid = r2.integers(0, len(active), N_TRANS); st = r2.integers(0, T - 1, N_TRANS)
    s = torch.as_tensor(np.stack([active[i]["observations"][t] for i, t in zip(sid, st)]),
                        dtype=torch.float32).to(DEVICE)
    sp = torch.as_tensor(np.stack([active[i]["observations"][t + 1] for i, t in zip(sid, st)]),
                         dtype=torch.float32).to(DEVICE)
    advs = []
    for ens in enss:
        with torch.no_grad():
            advs.append((ens(sp) - ens(s)).cpu().numpy().ravel())
    from scipy.stats import spearmanr
    rhos, jacs = [], []
    k = max(1, N_TRANS // 100)
    tops = [set(np.argsort(a)[-k:]) for a in advs]
    for i, j in itertools.combinations(range(len(advs)), 2):
        rhos.append(float(spearmanr(advs[i], advs[j])[0]))
        jacs.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))

    # --- D: spread of V over candidate actions at a fixed state, via dynamics
    sep = None; noise_floor = None
    dp = os.path.join(outdir, "dyn_ensemble_seed0.pt")
    if os.path.exists(dp):
        ck = torch.load(dp, map_location=DEVICE, weights_only=False)
        act_dim = int(ck.get("act_dim", active[0]["actions"].shape[1]))
        from model.policy import DynamicsEnsemble
        dyn = DynamicsEnsemble(obs_dim, act_dim, ck.get("hidden_dim", HIDDEN),
                          K=ck.get("K", 5)).to(DEVICE)
        dyn.load_state_dict(ck["state_dict"]); dyn.eval()
        bcp = os.path.join(outdir, "bc_policy.pt")
        if os.path.exists(bcp):
            from model.policy import GaussianPolicy
            bck = torch.load(bcp, map_location=DEVICE, weights_only=False)
            bc = GaussianPolicy(obs_dim, act_dim, bck.get("hidden_dim", HIDDEN)).to(DEVICE)
            bc.load_state_dict(bck["state_dict"]); bc.eval()
            M, nS = 8, 512
            idx = r2.integers(0, N_TRANS, nS)
            s_sub = s[idx]
            spreads = []
            with torch.no_grad():
                mean, log_std = bc(s_sub)
                std = log_std.exp() if torch.is_tensor(log_std) else torch.ones_like(mean)
                for _ in range(M):                    # BC-sampled candidates
                    a = mean + torch.randn_like(mean) * std
                    spreads.append(enss[0](dyn(s_sub, a)).cpu().numpy().ravel())
            arr = np.stack(spreads)                   # M x nS
            # like-for-like with the noise floor below: both are standard
            # deviations at the same states, one across candidate actions and
            # one across ensemble members. Comparing a range to a std would
            # inflate the ratio by roughly 3x for free.
            sep = float(np.mean(arr.std(0)))
            # The paper's claim is that candidates separate by LESS than the
            # value's own noise floor. Measure that floor on the same states:
            # disagreement across ensemble members at the same next-states.
            with torch.no_grad():
                a0 = mean
                nxt0 = dyn(s_sub, a0)
                per_member = np.stack([e(nxt0).cpu().numpy().ravel() for e in enss])
            noise_floor = float(np.mean(per_member.std(0)))
    return dict(accuracy=accs, rho=rhos, jaccard=jacs, separation=sep,
                noise_floor=noise_floor, n_seeds=len(enss))


def main():
    out = {}
    for t in TASKS:
        r = task_stats(t)
        if r is None:
            print(f"  {t:<22} skipped (missing artifacts)"); continue
        out[t] = r
        print(f"  {t:<22} acc={np.mean(r['accuracy']):.3f}  "
              f"rho[{min(r['rho']):.2f},{max(r['rho']):.2f}]  "
              f"jac[{min(r['jaccard']):.2f},{max(r['jaccard']):.2f}]  "
              f"sep={r['separation'] if r['separation'] is None else round(r['separation'],4)}")
    dst = os.path.join(REPO, "..", "..", "iclr2027", "data", "obstacle2_stats.json")
    json.dump(out, open(os.path.abspath(dst), "w"), indent=1)
    print(f"\n  wrote {os.path.abspath(dst)}  ({len(out)} tasks)")


if __name__ == "__main__":
    main()
