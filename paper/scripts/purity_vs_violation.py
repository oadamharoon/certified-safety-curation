"""Does a purer selection give a safer policy?

The cross-task correlation between selected-set contamination and policy
violation is near zero, but that comparison is confounded: tasks differ in
dynamics, cost limit and difficulty, so between-task variation swamps the
effect. The question the pipeline actually rests on is within-task, where the
same environment is held fixed and only the selection changes. Measured that
way the relationship is strong and positive.
"""
import json, os, statistics as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else None


def arms(algs):
    """Mean (contamination, violation) per arm, over seeds."""
    out = []
    for arm, seeds in algs.items():
        pair = [((e.get("meta") or {}).get("kept_unsafe_rate"), e.get("viol"))
                for e in seeds.values()]
        pair = [p for p in pair if p[0] is not None and p[1] is not None]
        if pair:
            out.append((arm, st.mean(p[0] for p in pair), st.mean(p[1] for p in pair)))
    return sorted(out, key=lambda t: t[1])


def main():
    S = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))
    within, cross = [], []
    print(f"{'task':<24}{'arms':>5}{'within-task r':>15}")
    for task, algs in sorted(S.items()):
        a = arms(algs)
        if len(a) >= 3:
            r = pearson([x[1] for x in a], [x[2] for x in a])
            if r is not None:
                within.append((task, r, len(a)))
                print(f"{task:<24}{len(a):>5}{r:>+15.2f}")
        dep = [x for x in a if x[0] == "calfilt_csf"]
        if dep:
            cross.append((dep[0][1], dep[0][2]))

    rs = [r for _, r, _ in within]
    print(f"\n  within-task, median r  : {st.median(rs):+.2f}   "
          f"positive in {sum(1 for r in rs if r > 0)}/{len(rs)} tasks")
    print(f"  cross-task r           : "
          f"{pearson([c[0] for c in cross], [c[1] for c in cross]):+.2f}   "
          f"(confounded by task heterogeneity, do not read this as the effect)")
    neg = [(t, r) for t, r, _ in within if r < 0]
    if neg:
        print("  negative exceptions    : " + ", ".join(f"{t} ({r:+.2f})" for t, r in neg))


if __name__ == "__main__":
    main()
