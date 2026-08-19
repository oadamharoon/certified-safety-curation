"""Bootstrap and leave-one-out intervals for the n50-vs-margin power law.

Six tasks reach the 50 percent certification level, so the two-parameter fit
needs an honest uncertainty statement. Reports a paired bootstrap over tasks
and a leave-one-out sweep of the exponent.
"""
import json
import numpy as np

B = "/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/"
G = json.load(open(B + "guarantee_stats_2000.json"))
E = json.load(open(B + "label_complexity_ext.json"))
M = {r["task"]: r["margin"] for r in json.load(open(B + "margin_vs_yield.json"))}
TARGET = 0.5

rows = []
for task, m in M.items():
    curve = {}
    for s, sd in G.get(task, {}).get("seeds", {}).items():
        for n, c in sd.items():
            curve.setdefault(int(n), []).append(c["cert_rate"])
    curve = {n: float(np.mean(v)) for n, v in curve.items()}
    curve.update({int(n): r for n, r in E.get(task, {}).get("rates", {}).items()})
    xs = sorted(curve); ys = [curve[n] for n in xs]
    n50 = None
    for i, (n, y) in enumerate(zip(xs, ys)):
        if y >= TARGET:
            if i == 0: n50 = float(n); break
            n0, y0 = xs[i-1], ys[i-1]
            f = (TARGET - y0) / (y - y0) if y != y0 else 0.0
            n50 = float(np.exp(np.log(n0) + f * (np.log(n) - np.log(n0)))); break
    if n50 and m > 0:
        rows.append((task, m, n50))

x = np.log([m for _, m, _ in rows])
y = np.log([n for _, _, n in rows])
slope, b = np.polyfit(x, y, 1)
r2 = 1 - ((y - (slope * x + b)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"  point estimate over {len(rows)} tasks: exponent {-slope:.2f}, "
      f"prefactor {np.exp(b):.2f}, R^2 {r2:.3f}\n")

rng = np.random.default_rng(0)
bs = []
for _ in range(10000):
    idx = rng.integers(0, len(rows), len(rows))
    if len(set(idx.tolist())) < 2:
        continue
    xs_, ys_ = x[idx], y[idx]
    if np.ptp(xs_) == 0:
        continue
    s_, _ = np.polyfit(xs_, ys_, 1)
    bs.append(-s_)
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  paired bootstrap over tasks ({len(bs)} resamples)")
print(f"    exponent 95 percent CI [{lo:.2f}, {hi:.2f}]   median {np.median(bs):.2f}")
print(f"    Pr[exponent < 2 (shallower than a fixed-threshold test)] = {(bs < 2).mean():.3f}")

print("\n  leave-one-out")
loo = []
for k, (t, _, _) in enumerate(rows):
    keep = [i for i in range(len(rows)) if i != k]
    s_, b_ = np.polyfit(x[keep], y[keep], 1)
    loo.append(-s_)
    print(f"    drop {t:22s} exponent {-s_:.2f}")
print(f"    LOO range [{min(loo):.2f}, {max(loo):.2f}]")

json.dump({"n_tasks": len(rows), "exponent": float(-slope),
           "prefactor": float(np.exp(b)), "r2": float(r2),
           "boot_ci95": [float(lo), float(hi)],
           "boot_median": float(np.median(bs)),
           "p_below_2": float((bs < 2).mean()),
           "loo_range": [float(min(loo)), float(max(loo))]},
          open(B + "label_complexity_ci.json", "w"), indent=1)
print(f"\n  saved -> label_complexity_ci.json")
