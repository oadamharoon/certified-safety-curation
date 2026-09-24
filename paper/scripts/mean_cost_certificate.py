"""Distribution-free upper confidence bound on a deployed clone's EXPECTED cost.

policy_certificate.py bounds Pr[cost > budget] from one bit per episode. This bounds
E[cost], which is the criterion the paper and DSRL actually use, and therefore turns
the paper's headline safety statement into a certified one rather than an estimated
one.

Estimator, fixed in advance (Maurer-Pontil empirical Bernstein):

    E[C] <= Cbar + sigmahat*sqrt(2 ln(2/delta)/n) + 7*R*ln(2/delta)/(3(n-1))

valid for i.i.d. C in [0, R]. R = 1000 is an a priori bound (episode length, with
per-step cost an indicator), NOT the observed maximum: using the observed max would
make the bound data-dependent and void it.

The 7R/(3n) term is why the published 100-episode evaluations cannot support this and
2000 fresh episodes per cell were rolled out (run_policycert.sh, re-rolled for the six
regenerated tasks by run_policycert_v2.sh). No
retraining: the policies are the published ones.

Per-episode costs are parsed from the evaluation logs, since eval_results stores only
aggregates.

Usage:  python iclr2027/scripts/mean_cost_certificate.py [--delta 0.1] [--write]
"""
import argparse
import json
import math
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BASE)
LOGD = os.path.join(REPO, "runs/logs/policycert")
OUT = os.path.join(REPO, "datasets/outputs")
RANGE = 1000.0  # a priori: episode length, per-step cost an indicator

ORDER = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
         "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
         "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]


def mp_upper(costs, delta, rng=RANGE):
    """Maurer-Pontil empirical-Bernstein upper confidence bound on the mean."""
    n = len(costs)
    if n < 2:
        return None
    L = math.log(2.0 / delta)
    sd = float(np.std(costs, ddof=1))
    return (float(np.mean(costs)) + sd * math.sqrt(2 * L / n)
            + 7.0 * rng * L / (3.0 * (n - 1)))


_SNAP = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))


def certified(task, seed):
    """True where the paper's Calibrated column reports the gated R50 policy."""
    e = _SNAP.get(task, {}).get("calfilt_csf", {}).get(str(seed), {})
    return bool(e.get("meta", {}).get("certified", False))


# F3e (2026-09-23) re-rolled the six regenerated tasks because the 09-01 evaluations described
# policies F0 had retrained. Its logs live in runs/logs/policycert_v2 under a different naming,
# so the v2 log is preferred wherever it exists and the 09-01 log is the fallback for the cells
# F3e did not need to redo (the nine untouched tasks, and PointGoal1's gated cells whose policies
# are unchanged since 08-19). Preferring the newer file is what keeps the two cohorts from mixing.
LOGD_V2 = os.path.join(REPO, "runs/logs/policycert_v2")


def _parse(path):
    if not os.path.exists(path):
        return None
    v = [float(x) for x in re.findall(r"^Ep \d+:.*?C=\s*([0-9.]+)",
                                      open(path, errors="ignore").read(), re.M)]
    return np.array(v) if v else None


def episode_costs(task, seed):
    # Gated cells were rolled out separately, against the R50 policy the paper
    # actually reports; certifying calfilt_csf there would bound a policy the
    # paper never shows.
    if certified(task, seed):
        for cand in (os.path.join(LOGD_V2, f"{task}_certn2k_gated_seed{seed}.log"),
                     os.path.join(LOGD, f"gated_{task}_seed{seed}.log")):
            v = _parse(cand)
            if v is not None:
                return v
    for cand in (os.path.join(LOGD_V2, f"{task}_certn2k_calfilt_csf_seed{seed}.log"),
                 os.path.join(LOGD, f"{task}_seed{seed}.log")):
        v = _parse(cand)
        if v is not None:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rec = {"delta": args.delta, "range": RANGE, "estimator": "maurer-pontil",
           "per_task": {}}
    print(f"delta={args.delta}, a priori cost range [0, {RANGE:.0f}], "
          f"Maurer-Pontil empirical Bernstein\n")
    print(f"{'task':<22}{'seeds':>6}{'n/seed':>8}{'meanC':>8}{'sd':>7}"
          f"{'slack':>8}{'E[C]<=':>9}{'budget':>8}  certified")
    ncert = ntot = 0
    for t in ORDER:
        # A seed counts only once its run FINISHED: a still-running cell has a
        # partially flushed log, and while a prefix of independent episodes is
        # still independent, mixing truncated and complete cells would make the
        # per-task n silently uneven. The result json is the completion marker.
        cells, lim = [], None
        for s in range(5):
            pf = os.path.join(OUT, t, (
                f"eval_results_certn2k_gated_seed{s}.json" if certified(t, s)
                else f"eval_results_certn2k_calfilt_csf_seed{s}.json"))
            if not os.path.exists(pf):
                continue
            c = episode_costs(t, s)
            if c is None:
                continue
            n_expected = int(json.load(open(pf))["eval_episodes"])
            if len(c) != n_expected:
                print(f"  [warn] {t} seed{s}: log has {len(c)} episodes, "
                      f"json says {n_expected}; skipped")
                continue
            lim = float(json.load(open(pf))["cost_limit"])
            cells.append((s, c))
        if not cells or lim is None:
            continue
        rec["per_task"][t] = {"cost_limit": lim, "seeds": {}}
        ub, mc, sds, ns = [], [], [], []
        for s, c in cells:
            u = mp_upper(c, args.delta)
            rec["per_task"][t]["seeds"][str(s)] = {
                "n": int(len(c)), "mean": float(np.mean(c)),
                "sd": float(np.std(c, ddof=1)), "ub": u,
                "certified": bool(u <= lim)}
            ub.append(u)
            mc.append(float(np.mean(c)))
            sds.append(float(np.std(c, ddof=1)))
            ns.append(len(c))
        worst = max(ub)
        allc = all(x <= lim for x in ub)
        ncert += int(allc)
        ntot += 1
        print(f"{t:<22}{len(cells):>6}{int(np.mean(ns)):>8}{np.mean(mc):>8.2f}"
              f"{np.mean(sds):>7.1f}{worst - np.mean(mc):>8.2f}{worst:>9.2f}"
              f"{lim:>8.0f}  {'yes' if allc else 'no'}")
        rec["per_task"][t]["all_seeds_certified"] = bool(allc)
    print(f"\n  E[cost] certified within budget on {ncert}/{ntot} tasks "
          f"(every seed, delta={args.delta})")
    if args.write:
        p = os.path.join(BASE, "data", "mean_cost_certificate.json")
        json.dump(rec, open(p, "w"), indent=2)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
