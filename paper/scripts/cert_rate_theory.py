"""Exact finite-sample certification probability, and its validation.

Proposition 2 says certification succeeds when the purest attainable selection is
below alpha and fails when it is above, but its positive half is asymptotic and the
paper's link between the purity margin and the observed certification rate is a
Spearman correlation. This computes Pr[certify] exactly, in closed form, from
quantities a practitioner can estimate before spending any labels.

The procedure is strict fixed-sequence (04q line 120 breaks on the first failure to
reject), so certification happens if and only if the FIRST and most selective grid
threshold is rejected. That collapses the whole sequence to one hypergeometric event:

    m ~ Hypergeom(Npool, N1, n)                  calibration points inside S1
    k | m ~ Hypergeom(N1, K1, m)                 unsafe ones among them
    reject iff  hypergeom.cdf(k, N1, k*, m) <= delta,   k* = floor(alpha*N1) + 1

so  Pr[certify] = sum_m P(m) * sum_{k: reject} P(k | m).

Corollary (interpretable form). With eps = alpha - u1 the purity margin and m the
calibration points landing in S1, three Hoeffding steps for sampling without
replacement give: if m >= 2 ln(1/delta) / eps^2 then Pr[certify | m] >= 1 - exp(-m
eps^2 / 2). This recovers both halves of Proposition 2 and shows the rate is governed
by m eps^2, i.e. by the margin AND by how much calibration mass lands in the
selection, which is why a large margin on a very thin selection still certifies rarely.

Usage:  python iclr2027/scripts/cert_rate_theory.py [--write] [--plan]
"""
import argparse
import json
import math
import os


def _write_if_changed(path, text):
    """Write only when the content differs, so an identical regeneration keeps its mtime.

    The completeness gate's R11 re-runs these builders on every invocation to prove they
    reproduce. An unconditional write bumps the artifact's mtime, which R9 and R14 then read as
    "the table/figure is older than its input" -- the freshness rules would eat each other.
    """
    import os as _os
    if _os.path.exists(path) and open(path).read() == text:
        return False
    with open(path, "w") as _fh:
        _fh.write(text)
    return True


import numpy as np
from scipy.stats import hypergeom

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, DELTA, Q1 = 0.25, 0.1, 0.85

_recp = os.path.join(BASE, "data", "q1_recovered.json")
_REC = json.load(open(_recp)) if os.path.exists(_recp) else {}


def pr_certify(npool, n1, k1, n, alpha=ALPHA, delta=DELTA):
    """Exact Pr[certify] for strict fixed-sequence LTT with hypergeometric tests."""
    ks = int(alpha * n1) + 1
    if ks > n1:
        return 1.0
    total = 0.0
    for m in range(0, min(n, n1) + 1):
        pm = hypergeom.pmf(m, npool, n1, n)
        if pm < 1e-12:
            continue
        kk = np.arange(0, min(m, k1) + 1)
        rej = hypergeom.cdf(kk, n1, ks, m) <= delta
        if not rej.any():
            continue
        total += pm * float(hypergeom.pmf(kk, n1, k1, m)[rej].sum())
    return float(total)


def hoeffding_floor(m, eps, delta=DELTA):
    """Corollary: valid lower bound on Pr[certify | m], or None if m is too small."""
    if eps <= 0 or m < 2 * math.log(1 / delta) / eps ** 2:
        return None
    return 1.0 - math.exp(-m * eps ** 2 / 2.0)


def min_n_for(npool, n1, k1, target, alpha=ALPHA, delta=DELTA, cap=4000):
    """Smallest calibration budget reaching `target` certification probability."""
    lo, hi = 1, cap
    if pr_certify(npool, n1, k1, cap, alpha, delta) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if pr_certify(npool, n1, k1, mid, alpha, delta) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def cells():
    """One record per (task, seed, calibration budget).

    The grid's first threshold is Q1, but guarantee_stats records kept_frac and
    kept_unsafe only for the quantile a draw actually SELECTED. Reading Q1 from
    the draws of the same cell therefore drops every cell in which Q1 was never
    chosen -- an outcome-dependent filter that biases the validation, and it cost
    26 of 108 cells before this was caught. (f1, u1) at Q1 is a property of the
    task and the pipeline seed, not of the calibration budget, so it is pooled
    across that seed's budgets instead; the values agree exactly wherever more
    than one budget records them. Eight cells (hopper seeds 1 and 2)
    still have no Q1 record at any budget and are reported as excluded rather
    than silently dropped.
    """
    d = json.load(open(os.path.join(BASE, "data", "guarantee_stats.json")))
    out, excluded = [], []
    for task, td in d.items():
        npool = td["n_trajs"]
        for seed, sd in td["seeds"].items():
            q1rec = None
            for _nk, _cell in sd.items():
                for x in _cell["draws"]:
                    if abs(x["q"] - Q1) < 1e-9:
                        q1rec = (x["kept_frac"], x["kept_unsafe"])
            if q1rec is None:
                # Seeds whose draws never selected Q1: recovered by replaying
                # 04q's scoring on the same checkpoint (recover_q1_stats.py),
                # which reproduces recorded cells exactly where the ensemble is
                # unchanged. Both of these have Jul-11 checkpoints, predating
                # guarantee_stats, so the replay is against the same weights.
                r = _REC.get(f"{task}/seed{seed}")
                if r:
                    q1rec = (r["kept_frac"], r["kept_unsafe"])
            for nkey, cell in sd.items():
                if q1rec is None:
                    excluded.append((task, seed, nkey))
                    continue
                f1, u1 = q1rec
                n1 = int(round(f1 * npool))
                out.append({"task": task, "seed": seed, "n": int(nkey),
                            "npool": npool, "n1": n1, "k1": int(round(u1 * n1)),
                            "u1": u1, "eps": ALPHA - u1, "f1": f1,
                            "obs": cell["cert_rate"], "n_draws": cell["n_draws"]})
    if excluded:
        print(f"  excluded {len(excluded)} of {len(out) + len(excluded)} cells "
              f"with no q={Q1} record: "
              + ", ".join(sorted({f"{t} seed{s}" for t, s, _ in excluded})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    rows = cells()
    zs, recs = [], []
    for c in rows:
        p = pr_certify(c["npool"], c["n1"], c["k1"], c["n"])
        se = math.sqrt(max(p * (1 - p), 1e-9) / c["n_draws"])
        z = (c["obs"] - p) / se
        zs.append(z)
        c.update(pred=p, z=z,
                 floor=hoeffding_floor(c["n"] * c["f1"], c["eps"]))
        recs.append(c)

    diffs = [abs(c["obs"] - c["pred"]) for c in recs]
    r = float(np.corrcoef([c["pred"] for c in recs], [c["obs"] for c in recs])[0, 1])
    print(f"validated on {len(recs)} cells "
          f"({len({c['task'] for c in recs})} tasks x seeds x budgets)")
    print(f"  mean |pred - obs| = {np.mean(diffs):.4f}   max {max(diffs):.4f}")
    print(f"  Pearson r         = {r:.4f}")
    print(f"  z-scores vs Monte Carlo SE of the {recs[0]['n_draws']}-draw estimate:")
    print(f"    mean {np.mean(zs):+.3f}  sd {np.std(zs):.3f}  "
          f"|z|>2 on {sum(abs(z) > 2 for z in zs)}/{len(zs)} cells "
          f"(expect ~{0.0455 * len(zs):.1f} if the formula is exact)")

    print("\n  Proposition 2 recovered:")
    neg = [c for c in recs if c["eps"] <= 0]
    pos = [c for c in recs if c["eps"] > 0]
    if neg:
        print(f"    u1 > alpha  ({len(neg)} cells): max predicted "
              f"{max(c['pred'] for c in neg):.4f}, max observed "
              f"{max(c['obs'] for c in neg):.4f}  (both <= delta = {DELTA})")
    if pos:
        big = [c for c in pos if c["n"] == max(x["n"] for x in pos)]
        print(f"    u1 < alpha, largest budget: predicted certification rises to "
              f"{max(c['pred'] for c in big):.3f}")
    nf = [c for c in recs if c["floor"] is not None]
    print(f"    Hoeffding corollary is non-vacuous on {len(nf)}/{len(recs)} cells "
          f"(it needs m >= 2ln(1/delta)/eps^2, which thin selections fail)")

    if args.plan:
        print("\n  labels needed for a 0.9 certification probability:")
        seen = set()
        for c in sorted(recs, key=lambda c: -c["eps"]):
            if c["task"] in seen or c["eps"] <= 0:
                continue
            seen.add(c["task"])
            nn = min_n_for(c["npool"], c["n1"], c["k1"], 0.9)
            print(f"    {c['task']:<22} margin {c['eps']:+.3f}  "
                  + (f"n >= {nn}" if nn else "not reachable by n = 4000"))

    if args.write:
        p = os.path.join(BASE, "data", "cert_rate_theory.json")
        _write_if_changed(p, json.dumps({"alpha": ALPHA, "delta": DELTA, "q1": Q1,
                   "mean_abs_err": float(np.mean(diffs)), "pearson_r": r,
                   "z_mean": float(np.mean(zs)), "z_sd": float(np.std(zs)),
                   "cells": recs}, indent=1))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
