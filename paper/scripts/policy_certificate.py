"""Distribution-free upper bound on a deployed clone's budget-violation probability.

The paper certifies the composition of the training SELECTION. A practitioner deploys
a POLICY, and the gap between the two is the first item in our Limitations. This closes
it from the other side, with no concentrability argument: after training the clone is a
fixed function, so its evaluation episodes are independent draws from the task's
initial-state distribution, the number exceeding the budget is Binomial(n, p), and an
exact Clopper-Pearson upper limit on p is a genuine (1 - delta) certificate on the
POLICY itself.

It consumes exactly the supervisor the paper already assumes: one bit per episode,
"did this run exceed budget". No per-transition cost, so the supervision claim is
untouched.

Two honest caveats, both reported rather than hidden:

1. It bounds Pr[cost > budget], not E[cost], which is the paper's (and DSRL's) safety
   criterion. Bits carry no magnitudes. mean_bound_feasibility() shows why 100 episodes
   cannot support a mean bound: with episodic cost ranging over [0, T] the Maurer-Pontil
   range term swamps the deviation term at this n.
2. The criterion gap is large and is NOT specific to our method. Cloning the
   ground-truth safe subset also leaves roughly a quarter of episodes over budget. That
   is a property of mean-cost safety and of cloning, and is worth reporting as such.

Two objects are certified, and they are different statements:
  per-seed  -- one trained policy, n episodes. The deployable statement.
  pooled    -- the randomized policy that draws a training seed uniformly and runs it.
               Valid, but about the procedure, not about any one policy you hold.

Usage:  python iclr2027/scripts/policy_certificate.py [--delta 0.1] [--write]
"""
import argparse
import json
import math
import os

from scipy.stats import beta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(BASE), "datasets/outputs")

ORDER = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
         "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
         "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]

# These are the arms the paper's main table actually reports (make_tables.main_gated):
# BC-All is stored seedless because it is a single training; the V-filter column is
# vfilt_calsafe, NOT vfilt_matchgt; and the Calibrated column is GATED -- it reports
# calfilt_lttR50 on any (task, seed) whose calibration run returned a certificate and
# calfilt_csf elsewhere. Certifying calfilt_csf everywhere would bound a policy the
# paper does not report, on the 6 cells that certify.
ARMS = [("bc", "BC-All"), ("bcsafe", "BC-Safe"), ("vfilt_calsafe", "V-filter"),
        ("__gated__", "Calibrated")]

_SNAP = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))


def _certified(task, seed):
    e = _SNAP.get(task, {}).get("calfilt_csf", {}).get(str(seed), {})
    return bool(e.get("meta", {}).get("certified", False))


def cp_upper(k, n, delta):
    """Exact one-sided Clopper-Pearson upper limit on p given k of n."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def cells(task, arm):
    """(k, n, avg_cost, cost_limit) per evaluation of this arm on this task."""
    d = os.path.join(OUT, task)
    if not os.path.isdir(d):
        return []
    if arm == "__gated__":
        # per-seed: the R50 policy where that seed certified, csf otherwise
        fns = []
        for s_ in range(8):
            base = f"eval_results_calfilt_csf_seed{s_}.json"
            r50 = f"eval_results_calfilt_lttR50_seed{s_}.json"
            if not os.path.exists(os.path.join(d, base)):
                continue
            fns.append(r50 if (_certified(task, s_)
                               and os.path.exists(os.path.join(d, r50))) else base)
    elif arm == "bc":
        fns = ["eval_results_bc.json"]
    else:
        fns = [f for f in sorted(os.listdir(d))
               if f.startswith(f"eval_results_{arm}_seed") and f.endswith(".json")]
    rows = []
    for fn in fns:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        n = int(j.get("eval_episodes", 0))
        vr = j.get("constraint_violation_rate")
        if not n or vr is None:
            continue
        # viol is stored as a rate over n episodes; recover the integer count.
        rows.append((int(round(vr * n)), n, float(j["avg_cost"]),
                     float(j["cost_limit"])))
    return rows


def mean_bound_feasibility(sigma, rng, delta, slack):
    """Smallest n for which Maurer-Pontil empirical Bernstein fits inside slack.

    mean + sigma*sqrt(2 ln(2/delta)/n) + 7*rng*ln(2/delta)/(3(n-1)). The second term
    is O(rng/n) and dominates at small n, which is why more evaluation episodes, not a
    better policy, is what a mean bound needs.
    """
    L = math.log(2.0 / delta)
    for n in range(50, 500001, 50):
        s = sigma * math.sqrt(2 * L / n) + 7 * rng * L / (3 * (n - 1))
        if s <= slack:
            return n, s
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--write", action="store_true",
                    help="write data/policy_certificate.json and the LaTeX table")
    args = ap.parse_args()
    d_bonf = args.delta / len(ORDER)

    rec = {"delta": args.delta, "delta_bonferroni": d_bonf,
           "n_tasks": len(ORDER), "arms": {}, "per_task": {}}

    print(f"delta={args.delta}; joint claim over {len(ORDER)} tasks uses "
          f"Bonferroni delta={d_bonf:.5f}\n")
    hdr = f"{'task':<22}" + "".join(f"{nm:>20}" for _, nm in ARMS)
    print(hdr + "\n" + f"{'':<22}" + "".join(f"{'meanC   p<=':>20}" for _ in ARMS))

    agg = {a: [0, 0, 0, 0] for a, _ in ARMS}
    for t in ORDER:
        row, rec["per_task"][t] = f"{t:<22}", {}
        for arm, nm in ARMS:
            c = cells(t, arm)
            if not c:
                row += f"{'--':>20}"
                continue
            K, N = sum(x[0] for x in c), sum(x[1] for x in c)
            mc, lim = sum(x[2] for x in c) / len(c), c[0][3]
            u, uj = cp_upper(K, N, args.delta), cp_upper(K, N, d_bonf)
            agg[arm][0] += K
            agg[arm][1] += N
            agg[arm][2] += int(mc <= lim)
            agg[arm][3] += 1
            rec["per_task"][t][arm] = {
                "k": K, "n": N, "seeds": len(c), "mean_cost": mc,
                "cost_limit": lim, "mean_cost_safe": bool(mc <= lim),
                "cp_upper": u, "cp_upper_joint": uj}
            row += f"{mc:9.1f} {u:9.3f}"
        print(row)

    print("\n=== pooled: the procedure, i.e. a uniformly drawn training seed ===")
    for arm, nm in ARMS:
        K, N, S, T = agg[arm]
        if not N:
            continue
        rec["arms"][arm] = {"name": nm, "k": K, "n": N, "raw_rate": K / N,
                            "cp_upper": cp_upper(K, N, args.delta),
                            "mean_cost_safe_tasks": S, "tasks": T}
        print(f"  {nm:<12} {K:>5}/{N:<5} violate | raw {K/N:.3f} | "
              f"p<= {cp_upper(K, N, args.delta):.3f} | mean-cost-safe {S}/{T}")

    print("\n=== why Pr[violation] and not E[cost] ===")
    for sig, rng_ in ((15.0, 1000.0), (15.0, 200.0)):
        n, s = mean_bound_feasibility(sig, rng_, args.delta, 19.0)
        print(f"  sigma={sig:.0f}, cost range [0,{rng_:.0f}], slack 19 (budget 25 "
              f"vs mean ~6): " + (f"n >= {n} episodes" if n else "infeasible"))

    if args.write:
        p = os.path.join(BASE, "data", "policy_certificate.json")
        with open(p, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nwrote {p}")
        rows = []
        for t in ORDER:
            cellsr = []
            for arm, _ in ARMS:
                e = rec["per_task"][t].get(arm)
                cellsr.append("--" if not e else f"{e['cp_upper']:.3f}")
            rows.append(t.replace("_", r"\_") + " & " + " & ".join(cellsr) + r" \\")
        tp = os.path.join(BASE, "data", "tables", "policy_cert.tex")
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        with open(tp, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"wrote {tp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
