"""Print the calsafe fraction for (task, seed): CP-lower(0.1) on safe mass from a fresh 200-draw."""
import os, pickle, sys
import numpy as np
from scipy.stats import beta
import yaml
task, seed = sys.argv[1], int(sys.argv[2])
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
cfg = yaml.safe_load(open("config.yaml"))
lim = 20 if "velocity" in task else (10 if task.endswith("_b") else 25)
trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
costs = np.array([float(np.sum(t["costs"])) for t in trajs])
rng = np.random.default_rng(2000 + seed)
cal = rng.choice(len(costs), 200, replace=False)
k = int((costs[cal] <= lim).sum())
lo = float(beta.ppf(0.1, k, 200 - k + 1)) if k > 0 else 0.0
frac = max(lo, 50 / len(costs))
print(f"{frac:.4f}")
