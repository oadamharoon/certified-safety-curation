"""Labeling-cost Pareto figure: fraction of cost-labeled trajectories (x)
vs mean DSRL-normalized reward on each method's safe tasks (y).

Marker text annotates how many of the 15 tasks the method keeps within
budget. Methods with zero label consumption sit at x=0 (BC-All, CPL);
ours consumes 200 labels per task (x = mean 200/N); full-label methods
sit at x=1. Wave-2 baselines (CPQ, COptiDICE) join automatically once
harvested.
"""
import json
import os
import pickle

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/omniverse/workspace/safevlmcpl/iclr2027"
REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
T15 = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
       "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
       "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
       "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]
LIM = {t: 20 if "velocity" in t else 25 for t in T15}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

snap = json.load(open(f"{BASE}/data/results_snapshot.json"))
osrl = json.load(open(f"{BASE}/data/osrl_results.json"))
cfg = yaml.safe_load(open(f"{REPO}/config.yaml"))

# dataset sizes and reward extrema for normalization
NTRAJ, EXTREMA = {}, {}
for t in T15:
    with open(os.path.join(REPO, cfg["tasks"][t]["data_pickle"]), "rb") as f:
        trajs = pickle.load(f)
    NTRAJ[t] = len(trajs)
    rets = np.array([float(np.sum(x["rewards"])) for x in trajs])
    EXTREMA[t] = (rets.min(), rets.max())


def stats(getter):
    """getter(task) -> (R, C) or None; returns (mean norm R on safe tasks,
    n_safe, mean norm R on all covered tasks)."""
    safe_R, all_R, n_safe, n_cov = [], [], 0, 0
    for t in T15:
        rc = getter(t)
        if rc is None:
            continue
        R, C = rc
        lo, hi = EXTREMA[t]
        Rn = (R - lo) / (hi - lo)
        n_cov += 1
        all_R.append(Rn)
        if C <= LIM[t]:
            n_safe += 1
            safe_R.append(Rn)
    return (np.mean(safe_R) if safe_R else np.nan, n_safe,
            np.mean(all_R) if all_R else np.nan, n_cov)


def from_snap(config):
    def g(t):
        e = snap.get(t, {}).get(config, {})
        if not e:
            return None
        return (np.mean([v["R"] for v in e.values()]),
                np.mean([v["C"] for v in e.values()]))
    return g


def from_osrl(algo):
    def g(t):
        e = osrl.get(t, {}).get(algo, {})
        if not e:
            return None
        return (np.mean([v["R"] for v in e.values()]),
                np.mean([v["C"] for v in e.values()]))
    return g


mean_frac = np.mean([200.0 / NTRAJ[t] for t in T15])

_CDT_SWEEP = json.load(open(os.path.join(BASE, "data", "review_response",
                                         "cdt_target_sweep.json")))


def deployed_getter(t):
    """The calibrated filter as actually deployed: the LTT threshold where the
    run certifies, the uncertified conservative fallback otherwise."""
    # calfilt_csf is the deployed procedure end to end: it runs the full
    # Learn-then-Test walk and falls back to the Clopper-Pearson selection only
    # when nothing certifies. No task-level ltt/csf switch is needed.
    use = snap.get(t, {}).get("calfilt_csf", {})
    if not use:
        return None
    return (np.mean([v["R"] for v in use.values()]),
            np.mean([v["C"] for v in use.values()]))


def cdt_sweep_getter(t):
    """CDT at the safest target of its cost-target sweep, per the caption."""
    d = _CDT_SWEEP.get(t)
    if not d:
        return None
    C, tg = min((v["C"], k) for k, v in d.items())
    return (d[tg]["R"], C)


METHODS = [
    # (label, supervision rung, getter, color, marker)
    #   0 none | 1 ordinal comparisons | 2 comparisons + sparse budget judgments
    #   3 budget label on every trajectory | 4 cost value on every transition
    ("BC-All", 0, from_snap("bc_all"), "#999999", "o"),
    ("CPL (prefs)", 1, from_snap("cpl_gt"), "#D55E00", "^"),
    ("Ours (calibrated)", 2, deployed_getter, "#0072B2", "*"),
    ("BC-Safe", 3, from_snap("bcsafe"), "#009E73", "s"),
    ("CDT", 4, cdt_sweep_getter, "#CC79A7", "D"),
    ("CPQ", 4, from_osrl("cpq"), "#E69F00", "v"),
    ("COptiDICE", 4, from_osrl("coptidice"), "#8c564b", "X"),
]


def gated_getter(t):
    ltt = snap.get(t, {}).get("calfilt_ltt", {})
    r50 = snap.get(t, {}).get("calfilt_lttR50", {})
    if not ltt:
        return None
    Rs, Cs = [], []
    for s, e in ltt.items():
        cert = e.get("meta", {}).get("certified", False)
        src = r50.get(s, e) if cert else e
        Rs.append(src["R"]); Cs.append(src["C"])
    return (np.mean(Rs), np.mean(Cs))


plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(4.6, 3.1))
for label, x, getter, color, marker in METHODS:
    if getter is None:
        getter = gated_getter
    sR, n_safe, aR, n_cov = stats(getter)
    if n_cov == 0:
        continue
    y = n_safe
    ax.scatter([x], [y], s=110 if "Ours" in label else 70, color=color,
               edgecolors=color, linewidths=1.4, marker=marker, zorder=3)
    dx, ha = (9, "left") if x < 2.5 else (-9, "right")
    ax.annotate(label, (x, y),
                xytext=(dx, -2), textcoords="offset points", fontsize=6.8,
                ha=ha, color=INK)
ax.set_xlabel("supervision the method requires")
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(["none", "clip\ncomparisons", "comparisons\n+ 200 budget\njudgments",
                    "budget label\nper trajectory", "cost value\nper transition"], fontsize=6)
ax.set_xlim(-0.4, 4.4)
ax.set_ylabel(r"tasks within budget, of 15 ($\uparrow$ better)")
ax.set_ylim(-0.8, 13.5)
ax.set_yticks(range(0, 13, 2))
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5)
fig.tight_layout()
fig.savefig(f"{BASE}/figures/pareto.pdf", bbox_inches="tight")
fig.savefig(f"{BASE}/figures/pareto.png", dpi=200, bbox_inches="tight")
print("saved figures/pareto.{pdf,png}")
for label, x, getter, color, marker in METHODS:
    if getter is None:
        getter = gated_getter
    sR, n_safe, aR, n_cov = stats(getter)
    print(f"{label:18s} x={x:.3f} safeR={sR:.2f} n_safe={n_safe} allR={aR:.2f} cov={n_cov}")
