"""Invert the exact certification rate: the calibration budget that reaches a target rate.

Equation (rate) gives Pr[certify] in closed form from (N, N1, K1, n) at a fixed (alpha, delta).
It inverts, so a practitioner can price the labels before spending them. This writes the inverted
budget per task, which the paper quotes and which is otherwise a derived quantity with no stored
value for the prose audit to check against.

Writes data/label_budget_plan.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cert_rate_theory import cells, min_n_for, pr_certify, ALPHA, DELTA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET, DEPLOYED_N = 0.9, 200

first = {}
for c in cells():
    if c["task"] not in first and c.get("eps", 0) > 0:
        first[c["task"]] = c
out = {"alpha": ALPHA, "delta": DELTA, "target_rate": TARGET, "deployed_n": DEPLOYED_N, "tasks": {}}
for task, c in first.items():
    n = min_n_for(c["npool"], c["n1"], c["k1"], TARGET)
    out["tasks"][task] = {
        "margin": c["eps"], "npool": c["npool"], "n1": c["n1"], "k1": c["k1"],
        "n_for_target": n,                                   # None when unreachable by n = 4000
        "rate_at_deployed_n": pr_certify(c["npool"], c["n1"], c["k1"], DEPLOYED_N),
    }
json.dump(out, open(os.path.join(BASE, "data", "label_budget_plan.json"), "w"), indent=1)
for t, r in sorted(out["tasks"].items(), key=lambda x: -x[1]["margin"]):
    n = r["n_for_target"]
    print(f"  {t:22s} margin {r['margin']:+.3f}  " + (f"n >= {n}" if n else "not reachable by n = 4000")
          + f"   rate at n={DEPLOYED_N}: {r['rate_at_deployed_n']:.3f}")
