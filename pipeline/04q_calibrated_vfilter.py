"""Step 04q: Calibrated V-filter threshold selection (Phase D).

Replaces the ad-hoc filter fraction of 04p with a principled threshold.
Two modes:

MODE=ltt (headline): split-conformal risk control via Learn-then-Test
  (Angelopoulos et al., 2021, fixed-sequence variant).
  - Reveal ground-truth costs on a small uniform calibration sample of
    CAL_N trajectories (the only cost supervision in the whole pipeline;
    baselines like PReSa/PREFINE consume cost labels on ALL trajectories).
  - Grid of score thresholds tau from conservative (high) to permissive.
  - For each tau: among calibration trajs with score >= tau (m of them,
    k unsafe, unsafe = traj cost > limit), the exact binomial p-value for
    H0: P(unsafe | selected) > ALPHA is p = BinomCDF(k; m, ALPHA).
  - Fixed-sequence testing: walk the grid from most to least conservative,
    stop at the first tau whose p > DELTA; return the last passing tau.
  - Guarantee: with prob >= 1-DELTA over the calibration draw, the selected
    population's unsafe rate <= ALPHA. (Monotone risk assumed: lowering tau
    weakly increases risk.)
  - If NO tau passes, falls back to the most conservative grid point and
    reports uncertified=True.

MODE=pref (ablation): pure-preference calibration, zero cost labels.
  - Hold out fresh preference pairs (held-out RNG, as in diag_v_quality).
  - Score the PARENT trajectories of preferred vs dispreferred segments.
  - tau = the PREF_Q quantile of the dispreferred-parent score distribution
    (keep only trajectories scoring above almost all known-dispreferred ones).
  - No formal guarantee (labels are relative), but fully oracle-free.

Then: filter trajectories by tau, BC on the kept set, save + (external) eval.

Env: MODE, CAL_N (200), ALPHA (0.25), DELTA (0.1), PREF_Q (0.9),
     COST_LIMIT, V_ENSEMBLE_FILE, SEED_OVERRIDE, BC_EPOCHS, BATCH_SIZE, OUT_TAG.
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
from model.train import bc_pretrain  # noqa: E402


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Bin(n, p). Retained for reference."""
    from math import comb
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1)))


def hyper_pvalue(k: int, m: int, n_sel: int, alpha: float) -> float:
    """Exact p-value for H0: unsafe count in the selection exceeds
    alpha * n_sel, given k unsafe among m calibration points sampled
    without replacement from the selection. Under H0 the least-favorable
    unsafe count is K* = floor(alpha * n_sel) + 1; the hypergeometric CDF
    at fixed k is decreasing in K (stochastic monotonicity), so
    p = P(Hyp(n_sel, K*, m) <= k) is super-uniform under H0."""
    from scipy.stats import hypergeom
    k_star = int(alpha * n_sel) + 1
    if k_star > n_sel:
        return 1.0
    return float(hypergeom.cdf(k, n_sel, k_star, m))


def score_trajectories(trajectories, ens, device) -> np.ndarray:
    scores = np.zeros(len(trajectories))
    with torch.no_grad():
        for i, traj in enumerate(trajectories):
            o = torch.as_tensor(traj["observations"], dtype=torch.float32).to(device)
            vs = [ens(o[j:j+8192]).cpu() for j in range(0, len(o), 8192)]
            scores[i] = torch.cat(vs).mean().item()
    return scores


def calibrate_ltt(scores, traj_costs, limit, rng, cal_n, alpha, delta):
    """Fixed-sequence LTT over score-quantile thresholds. Returns
    (tau, certified, cal_idx, audit)."""
    n = len(scores)
    cal_idx = rng.choice(n, size=min(cal_n, n), replace=False)
    cal_scores = scores[cal_idx]
    cal_unsafe = (traj_costs[cal_idx] > limit).astype(int)
    # Threshold grid: quantiles of the FULL score distribution, conservative
    # first. The grid deliberately STARTS at 0.85, not deeper into the tail:
    # (a) above q~0.9 the calibration selection m is too small for the
    # binomial test to ever reject at delta=0.1 (e.g. m=4 zero-violation
    # gives p=(1-alpha)^4 = 0.32), so extreme thresholds are uncertifiable
    # at practical CAL_N; (b) the extreme tail of the V-score ranking is
    # exactly the BT-underspecified region (manuscript Sec 4.10) and is
    # empirically unreliable (Walker2d: top-2% of trajectories by score had
    # 59% TRUE unsafe rate in the Phase-D v1 run).
    qs = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
          0.35, 0.30]
    taus = [float(np.quantile(scores, q)) for q in qs]
    # STRICT fixed-sequence testing (Angelopoulos et al., 2021): test the
    # ordered hypotheses conservative-first, each with an exact binomial
    # test at level delta, and STOP AT THE FIRST FAILURE TO REJECT. This is
    # what makes the family-wise error exactly delta without multiplicity
    # correction. (An earlier variant continued past initial failures; that
    # weakens the worst-case guarantee toward a union bound and is not used.)
    # p-values are EXACT hypergeometric tests against the finite pool
    # (the selection size is known in the transductive setting), which is
    # both exactly valid and more powerful than the binomial approximation.
    chosen, certified, audit = taus[0], False, []
    for q, tau in zip(qs, taus):
        sel = cal_scores >= tau
        m = int(sel.sum())
        k = int(cal_unsafe[sel].sum())
        n_sel = int((scores >= tau).sum())
        p = hyper_pvalue(k, m, n_sel, alpha) if m > 0 else 1.0
        audit.append({"quantile": q, "tau": tau, "m": m, "k": k, "p": round(p, 4)})
        if m > 0 and p <= delta:
            chosen, certified = tau, True     # rejected: continue the sequence
        else:
            break                              # first failure: stop (strict FS)
    return chosen, certified, cal_idx, audit


def calibrate_pref(trajectories, scores, cfg, rng, pref_q, device):
    """Threshold from held-out preference pairs (no cost labels)."""
    from utils.segment_utils import sample_pair_indices
    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    seen_t, tcosts = set(), []
    for s in active:
        if s["traj_id"] not in seen_t:
            seen_t.add(s["traj_id"]); tcosts.append(s["traj_total_cost"])
    tcosts = np.array(tcosts)
    safe_max = float(np.percentile(tcosts, 25))
    unsafe_min = float(np.percentile(tcosts, 75))
    dis_scores, n_pairs, attempts = [], 0, 0
    while n_pairs < 300 and attempts < 300 * 50:
        attempts += 1
        try:
            i, j = sample_pair_indices(active, cost_safe_threshold=0.0,
                                       cost_contrast_min=1.0, rng=rng,
                                       traj_safe_max_cost=safe_max,
                                       traj_safe_min_reward=-1.0,
                                       traj_unsafe_min_cost=unsafe_min)
        except Exception:
            continue
        a, b = active[i], active[j]
        if a["total_cost"] == b["total_cost"]:
            continue
        dis = b if a["total_cost"] < b["total_cost"] else a
        dis_scores.append(scores[dis["traj_id"]])
        n_pairs += 1
    tau = float(np.quantile(np.array(dis_scores), pref_q))
    return tau, n_pairs


def main() -> None:
    cfg = load_cfg()
    if "SEED_OVERRIDE" in os.environ:
        cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
    if "BC_EPOCHS" in os.environ:
        cfg["bc_only_epochs"] = int(os.environ["BC_EPOCHS"])
    if "BATCH_SIZE" in os.environ:
        cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
    seed = int(cfg["seed"])
    mode = os.environ.get("MODE", "ltt")
    cal_n = int(os.environ.get("CAL_N", 200))
    alpha = float(os.environ.get("ALPHA", 0.25))
    delta = float(os.environ.get("DELTA", 0.1))
    pref_q = float(os.environ.get("PREF_Q", 0.9))
    limit = float(os.environ["COST_LIMIT"])
    ens_file = os.environ["V_ENSEMBLE_FILE"]
    out_tag = os.environ.get("OUT_TAG", f"calfilt_{mode}_seed{seed}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | mode={mode} cal_n={cal_n} alpha={alpha} "
          f"delta={delta} seed={seed} tag={out_tag}", flush=True)
    run = init_wandb(cfg, job_type="calibrated_vfilter")

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    obs_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]

    ens = VEnsemble(obs_dim, cfg["hidden_dim"], K=3).to(device)
    ck = torch.load(os.path.join(cfg["output_dir"], ens_file),
                    map_location=device, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()

    scores = score_trajectories(trajectories, ens, device)
    traj_costs = np.array([float(np.sum(t["costs"])) for t in trajectories])
    rng = np.random.default_rng(1000 + seed)

    if mode == "ltt":
        tau, certified, cal_idx, audit = calibrate_ltt(
            scores, traj_costs, limit, rng, cal_n, alpha, delta)
        print(f"[ltt] tau={tau:.4f} certified={certified}", flush=True)
        for a in audit:
            print(f"      q={a['quantile']:.2f} m={a['m']:>4} k={a['k']:>3} "
                  f"p={a['p']}", flush=True)
    else:
        tau, n_pairs = calibrate_pref(trajectories, scores, cfg, rng,
                                      pref_q, device)
        certified = False
        print(f"[pref] tau={tau:.4f} from {n_pairs} held-out pairs "
              f"(q={pref_q})", flush=True)

    if mode == "ltt" and not certified:
        t2d = os.environ.get("TIER2_DELTA")
        if t2d:
            # Two-tier rule: rerun the identical fixed-sequence walk at a
            # weaker evidence standard (delta_2, e.g. 0.5) and take the
            # deepest passing threshold, flagged uncertified. One statistic,
            # one grid, two evidence levels; if nothing passes, minimum
            # selection at the top of the grid.
            t2 = float(t2d)
            cal_s2 = scores[cal_idx]
            cal_u2 = (traj_costs[cal_idx] > limit).astype(int)
            chosen2 = None
            for q2 in (0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                       0.45, 0.40, 0.35, 0.30):
                tq = float(np.quantile(scores, q2))
                sel2 = cal_s2 >= tq
                m2 = int(sel2.sum())
                k2 = int(cal_u2[sel2].sum())
                p2 = hyper_pvalue(k2, m2, int((scores >= tq).sum()),
                                  alpha) if m2 > 0 else 1.0
                if m2 > 0 and p2 <= t2:
                    chosen2 = tq
                else:
                    break
            if chosen2 is not None:
                tau = chosen2
                print(f"[ltt] TIER-2 fallback (delta2={t2}): "
                      f"tau={tau:.4f}", flush=True)
            else:
                tau = float(np.quantile(scores, 0.85))
                print("[ltt] TIER-2 empty; minimum selection q0.85",
                      flush=True)
        elif os.environ.get("FALLBACK_MODE") == "calsafe":
            # Calibration-informed fallback: keep the fraction given by the
            # one-sided Clopper-Pearson lower bound (level delta) on the pool
            # safe mass, estimated from the calibration labels already spent.
            from scipy.stats import beta as _beta
            n_c = len(cal_idx)
            k_safe = int((traj_costs[cal_idx] <= limit).sum())
            lo = float(_beta.ppf(delta, k_safe, n_c - k_safe + 1)) if k_safe > 0 else 0.0
            tau = float(np.quantile(scores, 1 - lo)) if lo > 0 else float("inf")
            print(f"[ltt] UNCERTIFIED calsafe fallback: safe {k_safe}/{n_c} "
                  f"CP-lower={lo:.3f} tau={tau:.4f}", flush=True)
        else:
            # No threshold certified: fall back to keeping the top 20% by
            # score (uncertified, reported as such).
            tau = float(np.quantile(scores, 0.80))
            print(f"[ltt] UNCERTIFIED fallback: tau=q0.80={tau:.4f}", flush=True)
    kept = np.where(scores >= tau)[0]
    if len(kept) < 50:
        kept = np.argsort(scores)[::-1][:50]
        print("[warn] <50 trajs above tau; keeping top-50", flush=True)
    unsafe_rate = float((traj_costs[kept] > limit).mean())
    print(f"[filter] kept {len(kept)}/{len(trajectories)} "
          f"({100*len(kept)/len(trajectories):.1f}%) | "
          f"kept-set TRUE unsafe rate {unsafe_rate:.3f} (target alpha={alpha}) | "
          f"kept-set mean cost {traj_costs[kept].mean():.1f} vs all "
          f"{traj_costs.mean():.1f}", flush=True)

    # Reward-aware sub-selection: within the score-certified set, keep the
    # top REWARD_FRAC by trajectory return. Rewards are dataset-native
    # (every baseline consumes them); the scarce signal is cost. Note the
    # LTT guarantee formally covers the score-selected set; we report the
    # post-sub-selection true unsafe rate so any composition shift is visible.
    reward_frac = float(os.environ.get("REWARD_FRAC", 0))
    if reward_frac > 0:
        rets = np.array([float(np.sum(trajectories[i]["rewards"])) for i in kept])
        n_keep = max(10, int(np.ceil(len(kept) * reward_frac)))
        kept = kept[np.argsort(rets)[::-1][:n_keep]]
        unsafe_rate = float((traj_costs[kept] > limit).mean())
        print(f"[reward-select] kept {len(kept)} "
              f"({100*len(kept)/len(trajectories):.1f}% of data) | "
              f"post-selection TRUE unsafe rate {unsafe_rate:.3f} | "
              f"mean cost {traj_costs[kept].mean():.1f} | "
              f"mean return {rets[np.argsort(rets)[::-1][:n_keep]].mean():.1f} "
              f"vs certified-set {rets.mean():.1f}", flush=True)
    kept_set = set(kept.tolist())

    obs = torch.cat([torch.as_tensor(trajectories[i]["observations"],
                                     dtype=torch.float32) for i in sorted(kept_set)])
    act = torch.cat([torch.as_tensor(trajectories[i]["actions"],
                                     dtype=torch.float32) for i in sorted(kept_set)])
    policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
    bc_pretrain(policy, obs, act, batch_size=cfg["batch_size"],
                epochs=int(cfg.get("bc_only_epochs", 100)),
                lr=cfg["learning_rate"], device=device,
                log_every=cfg["log_every"], wandb_run=run)

    out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
    torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
                "hidden_dim": cfg["hidden_dim"],
                "state_dict": policy.state_dict()}, out_p)
    meta = {"mode": mode, "tau": float(tau), "certified": bool(certified),
            "n_kept": int(len(kept)), "kept_frac": float(len(kept)/len(trajectories)),
            "kept_unsafe_rate": unsafe_rate, "alpha": alpha, "delta": delta,
            "cal_n": cal_n}
    with open(os.path.join(cfg["output_dir"], f"calfilt_meta_{out_tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved -> {out_p}", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
