"""Labeling-cost Pareto figure: fraction of cost-labeled trajectories (x)
vs mean DSRL-normalized reward on each method's safe tasks (y).

Marker text annotates how many of the 15 tasks the method keeps within
budget. Methods with zero label consumption sit at x=0 (BC-All, CPL);
ours consumes 200 labels per task (x = mean 200/N); full-label methods
sit at x=1. Wave-2 baselines (CPQ, COptiDICE) join automatically once
harvested.
"""

# --- paths ------------------------------------------------------------------
# The research tree addressed itself by absolute path; these roots replace it. Set
# CSC_WORKSPACE (or the individual roots) to point at your own trees. See the README.
import os as _os


def _csc_root(_p):
    """The repository root, found by the .csc-root marker rather than by depth."""
    _d = _os.path.dirname(_os.path.abspath(_p))
    while True:
        if _os.path.exists(_os.path.join(_d, ".csc-root")):
            return _d
        _up = _os.path.dirname(_d)
        if _up == _d:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(_p)))
        _d = _up


# __file__ is undefined when a script's source is exec'd in a fresh namespace, which the
# audit does to reuse the table builder's tables; fall back to the working directory, which
# the .csc-root walk resolves from anywhere inside the repository.
_self = globals().get("__file__") or _os.path.join(_os.getcwd(), "_")
CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(_self))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "vlm-with-cpl", "new_data"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs are carried by the repository, so they resolve without a working tree
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import json
import os
import pickle

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = CSC_PAPER
REPO = CSC_WORK
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


# The old axis ran none -> comparisons -> comparisons+judgments -> label per
# trajectory -> cost per transition, which silently converts a QUANTITY difference
# into a KIND difference: a budget label on every trajectory is the same judgment our
# calibration asks for, just 12x more of it, yet it sat a whole rung higher. That is
# also why labels-only had nowhere to go and was left off a figure about supervision.
#
# The tier is now the strongest judgment the supervisor must be able to make:
#   0 none
#   1 ordinal   -- which of two clips is safer (no budget, no scale)
#   2 binary    -- did this episode exceed its budget
#   3 cardinal  -- the numeric cost of a transition
# and position WITHIN a tier orders by how many judgments are needed, so amount is
# visible without being confused for kind. Ours and labels-only share tier 2: same
# supervisor, and the paper's claim is about what each does with it.
MEAN_TRAJ = float(np.mean([NTRAJ[t] for t in T15]))

# (label, tier, certifies, getter, color, marker). Judgment counts live in the
# labels, not in the position: ordering within a tier by log-count pushed BC-Safe so
# far right it read as sitting between binary and cardinal, which is exactly the
# kind/amount confusion this axis exists to remove. Methods now centre on their tier
# and only spread sideways to separate ties.
METHODS = [
    ("BC-All", 0, False, from_snap("bc_all"), "#999999", "o"),
    ("CPL", 1, False, from_snap("cpl_gt"), "#D55E00", "^"),
    ("BC-Safe-Seg", 2, False, from_snap("bcsafeseg"), "#7f7f7f", "P"),
    ("Labels-only 200", 2, False, from_snap("labels_only"), "#B44B0A", "o"),
    ("Labels-only 400", 2, False, from_snap("labels_only_n400"), "#B44B0A", "o"),
    ("Labels-split", 2, True, from_snap("labels_split_s100"), "#B44B0A", "P"),
    ("V-filter", 2, False, from_snap("vfilt_calsafe"), "#56B4E9", "d"),
    ("Calibrated", 2, True, deployed_getter, "#0072B2", "*"),
    ("BC-Safe", 2, False, from_snap("bcsafe"), "#009E73", "s"),
    ("CDT", 3, False, cdt_sweep_getter, "#CC79A7", "D"),
    ("CPQ", 3, False, from_osrl("cpq"), "#E69F00", "v"),
    ("COptiDICE", 3, False, from_osrl("coptidice"), "#8c564b", "X"),
]


# Resolve every point first so ties can be centred on their tier.
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


RES = []
for label, tier, cert, getter, color, marker in METHODS:
    if getter is None:
        getter = gated_getter
    sR, n_safe, aR, n_cov = stats(getter)
    if n_cov == 0:
        continue
    RES.append([label, tier, cert, n_safe, color, marker])

groups = {}
for r in RES:
    groups.setdefault((r[1], r[3]), []).append(r)

PLACED = []
for (tier, y), members in groups.items():
    k = len(members)
    # keep METHODS declaration order, not alphabetical: sorting by name silently
    # reshuffles which tie member sits left or right whenever a label is renamed,
    # which desynchronises the hand-tuned label offsets below.
    for i, (label, _t, cert, _y, color, marker) in enumerate(members):
        x = tier + (i - (k - 1) / 2.0) * 0.17
        filled = tier <= 1 or label == "Calibrated"
        ax.scatter([x], [y], s=150 if label == "Calibrated" else 60, marker=marker,
                   facecolors=color if filled else "none", edgecolors=color,
                   linewidths=1.4, zorder=3)
        if cert:
            ax.scatter([x], [y], s=320 if label == "Calibrated" else 210, marker="o",
                       facecolors="none", edgecolors=color, linewidths=0.8,
                       linestyle=":", zorder=2)
        PLACED.append((label, x, y))

OFF = {"BC-All": (9, -3, "left"), "CPL": (9, -3, "left"),
       # the binary tier is dense: fan the labels outward from the cluster centre
       "Labels-only 400": (0, 11, "center"),
       "V-filter": (-8, 8, "right"), "BC-Safe": (8, 8, "left"),
       "Labels-only 200": (-11, 0, "right"), "Calibrated": (14, 0, "left"),
       "Labels-split": (0, -12, "center"),
       # BC-Safe-Seg sits at y=2 directly left of COptiDICE, so its label goes left
       "BC-Safe-Seg": (-10, -2, "right"),
       "CDT": (-11, -3, "right"), "CPQ": (-11, -3, "right"),
       "COptiDICE": (-11, -3, "right")}
for label, x, y in PLACED:
    dx, dy, ha = OFF.get(label, (9, -3, "left"))
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.4, ha=ha, va="center", color=INK)

ax.set_xlabel("judgment the supervisor must be able to make "
              r"($\rightarrow$ stronger)")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["none", "ordinal\nwhich clip is safer",
                    "binary\nwas this episode\nover budget",
                    "cardinal\ncost of every\ntransition"], fontsize=6)
ax.set_xlim(-0.45, 3.45)
ax.set_ylabel(r"tasks within budget, of 15 ($\uparrow$ better)")
ax.set_ylim(-0.8, 13.5)
ax.set_yticks(range(0, 13, 2))
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5)
# No legend for the dotted rings: only two methods carry one, so the caption names
# them directly rather than spending a legend box inside the axes.
fig.tight_layout()
fig.savefig(f"{BASE}/figures/pareto.pdf", bbox_inches="tight")
fig.savefig(f"{BASE}/figures/pareto.png", dpi=200, bbox_inches="tight")
print("saved figures/pareto.{pdf,png}")
for label, x, y in sorted(PLACED, key=lambda r: (r[1], r[2])):
    print(f"{label:20s} x={x:6.3f}  safe={y}/15")

# Archive the placements. The figure needs the trajectory pickles of the working tree;
# the audit needs only these counts, so it reads this record where the tree is absent.
with open(f"{BASE}/data/pareto_counts.json", "w") as _fh:
    json.dump({label: {"x": round(x, 3), "safe": y} for label, x, y in PLACED}, _fh,
              indent=1, sort_keys=True)
print("wrote data/pareto_counts.json")
