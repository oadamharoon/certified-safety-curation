"""Does the calibration budget needed to certify scale as 1/margin^2?"""
import json
import numpy as np
G = json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/guarantee_stats_2000.json"))
M = {r["task"]: r["margin"] for r in json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/margin_vs_yield.json"))}
SIZES = [50, 100, 200, 400]
TARGET = 0.5

def n_at(rates):
    """smallest n reaching TARGET, by log-linear interpolation; None if never."""
    xs = [n for n in SIZES if str(n) in rates]
    ys = [rates[str(n)]["cert_rate"] for n in xs]
    if not ys or max(ys) < TARGET:
        return None
    for i, (n, y) in enumerate(zip(xs, ys)):
        if y >= TARGET:
            if i == 0: return float(n)
            n0, y0 = xs[i-1], ys[i-1]
            if y == y0: return float(n)
            f = (TARGET - y0) / (y - y0)
            return float(np.exp(np.log(n0) + f * (np.log(n) - np.log(n0))))
    return None

rows = []
for task, td in G.items():
    m = M.get(task)
    if m is None: continue
    per_seed = []
    for s, sd in td["seeds"].items():
        v = n_at(sd)
        if v: per_seed.append(v)
    rows.append((task, m, float(np.mean(per_seed)) if per_seed else None, len(per_seed)))
rows.sort(key=lambda r: -r[1])
print(f"{'task':22s} {'margin':>8s} {'n for 50% cert':>15s}")
for t, m, n, k in rows:
    print(f"{t:22s} {m:8.3f} {('%.0f' % n) if n else '   never':>15s}")
fit = [(m, n) for _, m, n, _ in rows if n and m > 0]
if len(fit) >= 3:
    x = np.log(np.array([m for m, _ in fit])); y = np.log(np.array([n for _, n in fit]))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nfit on {len(fit)} certifying tasks:  log n50 = {slope:.2f} * log(margin) + {intercept:.2f}")
    print(f"  slope {slope:.2f} (theory: -2)   R^2 = {r2:.3f}")
    print(f"  implied constant c (n = c/m^2): {np.exp(intercept):.3f}")
    never = [t for t, m, n, _ in rows if not n]
    print(f"  never reaching 50%: {never}")
    json.dump({"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
               "points": [{"task": t, "margin": m, "n50": n} for t, m, n, _ in rows]},
              open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/label_complexity.json", "w"), indent=1)
else:
    print("\ninsufficient certifying tasks to fit")
print("LABEL COMPLEXITY DONE")
