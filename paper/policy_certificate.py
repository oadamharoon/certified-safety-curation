"""Deployment-time certificate, and the empirical data-to-policy amplification.

Two questions, both answered from rollouts already on disk:
  1. Does a Clopper-Pearson bound on the deployed policy's episode violation
     rate certify at (alpha, delta)? This needs environment access, which the
     dataset certificate does not, so the two are not substitutes.
  2. How tightly does selected-set contamination predict policy violation?
"""
import json, os, statistics as st
from scipy.stats import beta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, DELTA, NEP = 0.25, 0.1, 100


def load():
    S = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))
    out = {}
    for task, algs in S.items():
        cells = [(e.get("meta") or {}, e) for e in algs.get("calfilt_csf", {}).values()]
        cells = [(m, e) for m, e in cells
                 if "kept_unsafe_rate" in m and e.get("viol") is not None]
        if cells:
            out[task] = cells
    return out


def cp_upper(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - DELTA, k + 1, n - k))


def main():
    data = load()
    print(f"alpha={ALPHA}  delta={DELTA}  {NEP} eval episodes per seed\n")
    hdr = (f"{'task':<22}{'data u':>8}{'dcert':>7} | {'viol':>7}{'pooled UB':>10}"
           f"{'  pooled':>9}{'  all-seed':>10}")
    print(hdr); print("-" * len(hdr))
    pooled = strict = dcert = 0
    rows = []
    for task, cells in sorted(data.items(), key=lambda kv: st.mean(e["viol"] for _, e in kv[1])):
        k = sum(round(e["viol"] * NEP) for _, e in cells); n = NEP * len(cells)
        ub = cp_upper(k, n)
        per = [cp_upper(round(e["viol"] * NEP), NEP) <= ALPHA for _, e in cells]
        dc = any(m.get("certified") for m, _ in cells)
        u = st.mean(m["kept_unsafe_rate"] for m, _ in cells)
        pooled += ub <= ALPHA; strict += all(per); dcert += dc
        rows.append((task, u, k / n))
        print(f"{task:<22}{u:8.3f}{('yes' if dc else 'no'):>7} | {k/n:7.3f}{ub:10.3f}"
              f"{('  CERT' if ub <= ALPHA else '  no'):>9}"
              f"{('  CERT' if all(per) else '  no'):>10}")
    N = len(rows)
    print(f"\n  dataset certified            : {dcert}/{N}")
    print(f"  policy certified, pooled     : {pooled}/{N}   (bounds the pick-a-seed mixture)")
    print(f"  policy certified, every seed : {strict}/{N}   (bounds each trained policy)")

    us = [r[1] for r in rows]; vs = [r[2] for r in rows]
    mu, mv = st.mean(us), st.mean(vs)
    num = sum((a - mu) * (b - mv) for a, b in zip(us, vs))
    den = (sum((a - mu) ** 2 for a in us) * sum((b - mv) ** 2 for b in vs)) ** 0.5
    print(f"\n  Pearson r(selected-set contamination, policy violation) = {num/den:+.3f}"
          f"  over {N} tasks")
    print("  amplification v/u ranges "
          f"{min(v/u for _, u, v in rows if u > 0):.2f} to {max(v/u for _, u, v in rows if u > 0):.2f}")


if __name__ == "__main__":
    main()
