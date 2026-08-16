"""Generate paper figures from archived data (no recollection needed).

Figure 1 (coverage.pdf): guarantee validation.
  Panel A: unconditional false-certification rate vs calibration size n,
           per task (mean over 3 V-ensemble seeds), dashed line at delta.
  Panel B: certification rate vs n, per task.

Data: data/guarantee_stats.json (from scripts/guarantee_stats.py).
Style: validated colorblind-safe categorical palette (fixed slot order),
per-series marker shapes as secondary encoding, recessive grid, text in
ink colors (never series colors).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/omniverse/workspace/safevlmcpl/iclr2027"
ALPHA, DELTA = 0.25, 0.1
CAL_NS = [50, 100, 200, 400]

# Validated categorical palette, fixed slot order (reference instance).
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
TASKS = [  # fixed order: entity->color binding never changes across panels
    ("halfcheetah_velocity", "HalfCheetah"),
    ("walker2d_velocity", "Walker2d"),
    ("ant_velocity", "Ant"),
    ("hopper_velocity", "Hopper"),
    ("swimmer_velocity", "Swimmer"),
    ("cargoal1_dsrl", "CarGoal1"),
    ("cargoal2", "CarGoal2"),
    ("pointgoal1_dsrl", "PointGoal1"),
    ("pointgoal2", "PointGoal2"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
assert len(PALETTE) >= len(TASKS) and len(MARKERS) >= len(TASKS), \
    "palette/markers shorter than TASKS: zip would silently drop tasks"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    # Prefer the 2000-draw rerun (review response); fall back to 200-draw file.
    p2000 = os.path.join(BASE, "data", "review_response",
                         "guarantee_stats_2000.json")
    if os.path.exists(p2000):
        with open(p2000) as f:
            stats = json.load(f)
    else:
        with open(os.path.join(BASE, "data", "guarantee_stats.json")) as f:
            stats = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for (task, label), color, marker in zip(TASKS, PALETTE, MARKERS):
        seeds = stats[task]["seeds"]
        fc_by_n, cert_by_n = [], []
        for n in CAL_NS:
            fcs, certs = [], []
            for s in seeds.values():
                cell = s[str(n)]
                if "draws" in cell:
                    draws = cell["draws"]
                    fcs.append(np.mean([d["certified"] and
                                        d["kept_unsafe"] > ALPHA
                                        for d in draws]))
                    certs.append(np.mean([d["certified"] for d in draws]))
                else:
                    fcs.append(cell["false_cert_rate_uncond"])
                    certs.append(cell["cert_rate"])
            fc_by_n.append(np.mean(fcs))
            cert_by_n.append(np.mean(certs))
        kw = dict(color=color, marker=marker, markersize=4.5,
                  linewidth=1.4, markeredgewidth=0, label=label)
        ax1.plot(CAL_NS, fc_by_n, **kw)
        ax2.plot(CAL_NS, cert_by_n, **kw)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(CAL_NS)
        ax.set_xticklabels([str(n) for n in CAL_NS])
        ax.minorticks_off()
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(r"calibration size $n$")
        ax.tick_params(length=2.5)

    ax1.axhline(DELTA, color=INK2, linestyle="--", linewidth=0.9)
    ax1.annotate(r"$\delta = 0.1$", xy=(CAL_NS[-1], DELTA),
                 xytext=(-2, 3), textcoords="offset points",
                 ha="right", fontsize=7.5, color=INK2)
    ax1.set_ylim(-0.02, 0.5)
    ax1.set_ylabel("false-certification rate")
    ax1.set_title("Guarantee holds in every cell", fontsize=8.5, color=INK)

    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel("certification rate")
    ax2.set_title("Certification tracks achievable purity", fontsize=8.5,
                  color=INK)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=9, frameon=False,
               bbox_to_anchor=(0.5, 1.13), columnspacing=0.9,
               handletextpad=0.4, handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(BASE, "figures", "coverage.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "coverage.png"), dpi=200,
                bbox_inches="tight")
    print("saved figures/coverage.{pdf,png}")


if __name__ == "__main__":
    main()


def transfer_figure():
    """Composition -> policy transfer scatter from archived points."""
    with open(os.path.join(BASE, "data", "transfer_points.json")) as f:
        pts = json.load(f)
    task_slot = {t: i for i, (t, _) in enumerate(TASKS)}
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for (task, label), color, marker in zip(TASKS, PALETTE, MARKERS):
        xs = [p["kept_unsafe"] for p in pts if p["task"] == task]
        ys = [p["policy_cost_norm"] for p in pts if p["task"] == task]
        ax.scatter(xs, ys, s=14, color=color, marker=marker,
                   linewidths=0, label=label, alpha=0.85)
    ax.axhline(1.0, color=INK2, linestyle="--", linewidth=0.9)
    ax.annotate("budget", xy=(0.98, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK2)
    ax.axvline(0.25, color=INK2, linestyle=":", linewidth=0.9)
    ax.annotate(r"$\alpha$", xy=(0.25, 0.98), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=7.5,
                color=INK2)
    ax.set_xlabel("selected-set unsafe fraction (true)")
    ax.set_ylabel("policy cost / budget")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, columnspacing=0.7,
              handletextpad=0.2, loc="upper left", borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "transfer.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "transfer.png"), dpi=200,
                bbox_inches="tight")
    print("saved figures/transfer.{pdf,png}")


if __name__ == "__main__" and os.environ.get("FIG") == "transfer":
    transfer_figure()


def interp_figure():
    """Interpretability: constraint landscape, anticipation, example trace."""
    import matplotlib.gridspec as gridspec
    with open(os.path.join(BASE, "data", "interpretability.json")) as f:
        d = json.load(f)

    fig = plt.figure(figsize=(5.4, 2.3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.38)

    # Panel B: anticipation curves, per-task z-scored V
    axb = fig.add_subplot(gs[0])
    for (task, label), color, marker in zip(TASKS, PALETTE, MARKERS):
        a = d[task]["anticipation"]
        vs = np.array([x["V"] for x in a])
        z = (vs - vs.mean()) / (vs.std() + 1e-9)
        xpos = np.arange(len(a))
        axb.plot(xpos, z, color=color, marker=marker, markersize=3.4,
                 linewidth=1.2, markeredgewidth=0, label=label)
    a0 = d["halfcheetah_velocity"]["anticipation"]
    labels_all = [str(x["ttv_lo"]) if x["ttv_hi"] is not None else
                  f"{x['ttv_lo']}+" for x in a0]
    shown = list(range(0, len(a0), 2))
    if (len(a0) - 1) not in shown:
        shown.append(len(a0) - 1)
    axb.set_xticks(shown)
    axb.set_xticklabels([labels_all[i] for i in shown], fontsize=7)
    axb.set_xticks(range(len(a0)), minor=True)
    axb.set_xlabel("steps to violation")
    axb.set_ylabel(r"$\bar{V}$ (per-task z-score)")
    axb.set_title("Anticipation", fontsize=8, color=INK)

    # Panel C: example unsafe trace, HalfCheetah
    axc = fig.add_subplot(gs[1])
    tr = d["halfcheetah_velocity"]["traces"]["unsafe"]
    stride = tr["stride"]
    t = np.arange(len(tr["V"])) * stride
    v_raw = np.asarray(tr["V"])
    axc.plot(t, v_raw, color=PALETTE[0], linewidth=0.6, alpha=0.35,
             label=r"$\bar{V}(s_t)$ (per state)")
    w = 25
    roll = np.convolve(v_raw, np.ones(w) / w, mode="same")
    axc.plot(t, roll, color=PALETTE[0], linewidth=1.7,
             label="rolling mean (125 steps)")
    for i, cs in enumerate(tr["cost_steps"]):
        axc.axvline(cs, color=PALETTE[5], linewidth=0.35, alpha=0.35,
                    zorder=0, label="cost event" if i == 0 else None)
    axc.set_xlabel("timestep")
    axc.set_ylabel(r"$\bar{V}$")
    axc.set_title("Unsafe trajectory (HalfCheetah)", fontsize=8, color=INK)
    axc.legend(frameon=False, fontsize=6.5, loc="upper right")

    for ax in (axb, axc):
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, labelsize=6.8)

    handles, labels = axb.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=9, frameon=False,
               bbox_to_anchor=(0.5, 1.16), columnspacing=0.8,
               handletextpad=0.35, handlelength=1.2, fontsize=6.8)
    fig.savefig(os.path.join(BASE, "figures", "interpretability.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "interpretability.png"),
                dpi=200, bbox_inches="tight")
    print("saved figures/interpretability.{pdf,png}")


if __name__ == "__main__" and os.environ.get("FIG") == "interp":
    interp_figure()


def alpha_curve_figure():
    """Certification operating curve across both suites: cert rate vs alpha."""
    with open(os.path.join(BASE, "data", "alpha_cert_stats.json")) as f:
        dsrl = json.load(f)
    with open(os.path.join(BASE, "data", "review_response",
                           "bullet_alpha_curve.json")) as f:
        bullet = json.load(f)
    BTASKS = [("ballrun_b", "BallRun"), ("ballcircle_b", "BallCircle"),
              ("carcircle_b", "CarCircle"), ("carrun_b", "CarRun"),
              ("dronerun_b", "DroneRun")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for (task, label), color, marker in zip(TASKS, PALETTE, MARKERS):
        if task not in dsrl:
            continue
        alphas, rates = [], []
        for a in ("0.05", "0.1", "0.25", "0.4"):
            vals = [dsrl[task][s][a]["cert_rate"] for s in dsrl[task]
                    if a in dsrl[task][s]]
            if vals:
                alphas.append(float(a))
                rates.append(np.mean(vals))
        ax1.plot(alphas, rates, color=color, marker=marker, markersize=4.5,
                 linewidth=1.4, markeredgewidth=0, label=label)
    for (task, label), color, marker in zip(BTASKS, PALETTE, MARKERS):
        alphas = [0.25, 0.40, 0.50]
        rates = [bullet[task][str(a)] for a in ("0.25", "0.4", "0.5")]
        ax2.plot(alphas, rates, color=color, marker=marker, markersize=4.5,
                 linewidth=1.4, markeredgewidth=0, label=label)
    for ax, title in ((ax1, "DSRL (nine analysis tasks)"),
                      (ax2, "BulletSafetyGym (five tasks)")):
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"guarantee level $\alpha$")
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5)
        ax.set_title(title, fontsize=8.5, color=INK)
    ax1.set_ylabel("certification rate")
    ax1.legend(frameon=False, fontsize=6.0, ncol=2, columnspacing=0.6,
               handletextpad=0.3, loc="upper left")
    ax2.legend(frameon=False, fontsize=6.3, loc="upper left",
               handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "alpha_curve.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "alpha_curve.png"), dpi=200,
                bbox_inches="tight")
    print("saved figures/alpha_curve.{pdf,png}")


if __name__ == "__main__" and os.environ.get("FIG") == "alpha":
    alpha_curve_figure()


def tradeoff_figure():
    """Reward-cost tradeoff per analysis task across alpha levels and
    selection fractions, with BC-Safe as reference."""
    with open(os.path.join(BASE, "data", "results_snapshot.json")) as f:
        snap = json.load(f)
    LIMS = {t: (20 if "velocity" in t else 25) for t, _ in TASKS}

    def mean_rc(task, cfg):
        e = snap.get(task, {}).get(cfg, {})
        if not e:
            return None
        return (np.mean([v["C"] for v in e.values()]) / LIMS[task],
                np.mean([v["R"] for v in e.values()]))

    fig, axes = plt.subplots(3, 3, figsize=(6.6, 5.4))
    for ax, (task, label) in zip(axes.flat, TASKS):
        pts = [mean_rc(task, c) for c in ("calfilt_a10", "calfilt_ltt",
                                          "calfilt_a40")]
        alpha_pts = [p for p in pts if p]
        if alpha_pts:
            ax.plot([p[0] for p in alpha_pts], [p[1] for p in alpha_pts],
                    color=PALETTE[0], marker="o", markersize=4,
                    linewidth=1.2, markeredgewidth=0,
                    label=r"calibrated, $\alpha \in \{.1, .25, .4\}$")
        for cfg, mk, lab in (("vfilt_matchgt", "s", "V-filter (matched)"),
                             ("vfilt_q25", "^", "V-filter (top 25\%)"),
                             ("bcsafe", "X", "BC-Safe")):
            p = mean_rc(task, cfg)
            if p:
                ax.scatter([p[0]], [p[1]], s=22, marker=mk,
                           color=PALETTE[1] if cfg != "bcsafe" else INK2,
                           linewidths=0, label=lab)
        ax.axvline(1.0, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_title(label, fontsize=7.5, color=INK)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5)
    for ax in axes[-1]:
        ax.set_xlabel("cost / budget")
    for ax in axes[:, 0]:
        ax.set_ylabel("reward")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=6.8,
               columnspacing=0.9, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(BASE, "figures", "tradeoff.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "tradeoff.png"), dpi=200,
                bbox_inches="tight")
    print("saved figures/tradeoff.{pdf,png}")


if __name__ == "__main__" and os.environ.get("FIG") == "tradeoff":
    tradeoff_figure()


def contamination_cost_figure():
    """T2.5: clone cost vs realized contamination across archived selections."""
    with open(os.path.join(BASE, "data", "review_response", "margin_expand.json")) as f:
        d = json.load(f)
    rows = d["rows"]
    tasks = sorted({r["task"] for r in rows})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, t in enumerate(tasks):
        xs = [r["contam"] for r in rows if r["task"] == t]
        ys = [r["cost_norm"] for r in rows if r["task"] == t]
        ax.scatter(xs, ys, s=14, color=PALETTE[i % len(PALETTE)], alpha=0.75,
                   linewidths=0, label=None)
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    ax.set_xlabel(r"realized contamination $\hat\alpha$ of the selection")
    ax.set_ylabel("clone cost / budget")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_title(f"{len(rows)} selection-clone pairs, {len(tasks)} tasks", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "contam_cost.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "contam_cost.png"), dpi=200, bbox_inches="tight")
    print("saved figures/contam_cost.{pdf,png}")


contamination_cost_figure()


def corridor_figure():
    """Corridor-CMDP simulation validating Proposition prop:corridor.

    Behavior is a benchmark-like mixture: fraction rho of trajectories come
    from an all-safe expert, the rest take the safe action i.i.d. with prob p.
    Any per-state cloning operator keeps only the per-step marginal
    q = rho + (1-rho) p; selection keeps the zero-cost subset exactly.
    """
    rng = np.random.default_rng(3)
    p, rho, N, REPS = 0.7, 0.2, 10000, 50
    q = rho + (1 - rho) * p
    Ts = [10, 30, 100, 300, 1000]
    arms = [(1.0, "BC-All (unweighted)"), (20.0, "reweighted, clip 20"),
            (400.0, "reweighted, clip 400")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (W, lbl), color in zip(arms, PALETTE[:3]):
        sims, theo = [], []
        for T in Ts:
            pp = q * W / (q * W + 1 - q)
            theo.append(pp ** T)
            acc = []
            for _ in range(REPS):
                n_exp = rng.binomial(N, rho)
                qhat = (n_exp + rng.binomial(N - n_exp, p, size=T)) / N
                pprime = qhat * W / (qhat * W + 1 - qhat)
                acc.append(np.prod(pprime))
            sims.append(np.mean(acc))
        ax.plot(Ts, theo, color=color, linestyle="--", linewidth=1)
        ax.plot(Ts, sims, color=color, marker="o", markersize=3.5,
                linewidth=1.4, markeredgewidth=0, label=lbl)
    sel = []
    for T in Ts:
        ok = []
        for _ in range(REPS):
            n_zero = rng.binomial(N, rho + (1 - rho) * p ** T)
            ok.append(1.0 if n_zero > 0 else 0.0)
        sel.append(np.mean(ok))
    ax.plot(Ts, sel, color=PALETTE[3], marker="s", markersize=3.5,
            linewidth=1.4, markeredgewidth=0, label="trajectory selection")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-10, 3)
    ax.set_xlabel(r"horizon $T$")
    ax.set_ylabel("all-safe probability")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "corridor.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "corridor.png"), dpi=200, bbox_inches="tight")
    print("saved figures/corridor.{pdf,png}")


corridor_figure()


def pareto_targets_figure():
    """Reward vs cost/budget: CDT across cost targets against our alpha sweep."""
    with open(os.path.join(BASE, "data", "review_response", "cdt_target_sweep.json")) as f:
        cdt = json.load(f)
    with open(os.path.join(BASE, "data", "results_snapshot.json")) as f:
        snap = json.load(f)
    LIM = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
           "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
           "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25}
    NM = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
          "ant_velocity": "Ant", "hopper_velocity": "Hopper", "swimmer_velocity": "Swimmer",
          "cargoal1_dsrl": "CarGoal1", "cargoal2": "CarGoal2",
          "pointgoal1_dsrl": "PointGoal1", "pointgoal2": "PointGoal2"}
    tasks = list(NM)
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 5.8))
    for ax, task in zip(axes.ravel(), tasks):
        lim = LIM[task]
        d = cdt.get(task, {})
        tg = sorted((int(k) for k in d), key=int)
        xs = [d[str(t)]["C"] / lim for t in tg]
        ys = [d[str(t)]["R"] for t in tg]
        if xs:
            ax.plot(xs, ys, "-o", color=PALETTE[0], markersize=4, linewidth=1.2,
                    markeredgewidth=0, label="CDT (targets)")
        axs_, ays = [], []
        for cfg, lbl in (("calfilt_a10", r"$\alpha$=0.10"), ("calfilt_ltt", r"$\alpha$=0.25"),
                         ("calfilt_a40", r"$\alpha$=0.40")):
            e = snap.get(task, {}).get(cfg, {})
            if not e:
                continue
            axs_.append(np.mean([v["C"] for v in e.values()]) / lim)
            ays.append(np.mean([v["R"] for v in e.values()]))
        if axs_:
            ax.plot(axs_, ays, "-s", color=PALETTE[2], markersize=3.5, linewidth=1.2,
                    markeredgewidth=0, label=r"ours ($\alpha$ sweep)")
        ev = snap.get(task, {}).get("vfilt_calsafe", {})
        if ev:
            ax.plot([np.mean([v["C"] for v in ev.values()]) / lim],
                    [np.mean([v["R"] for v in ev.values()])], "D", color=PALETTE[1],
                    markersize=4.5, markeredgewidth=0, label="V-filter")
        e = snap.get(task, {}).get("bcsafe", {})
        if e:
            ax.plot([np.mean([v["C"] for v in e.values()]) / lim],
                    [np.mean([v["R"] for v in e.values()])], "^", color=PALETTE[3],
                    markersize=5, markeredgewidth=0, label="BC-Safe")
        ax.axvline(1.0, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_title(NM[task], fontsize=8.5, color=INK)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        allx = xs + axs_ + ([np.mean([v["C"] for v in e.values()]) / lim] if e else []) \
               + ([np.mean([v["C"] for v in ev.values()]) / lim] if ev else [])
        hi = min(max(allx + [1.2]) * 1.15, 4.0) if allx else 1.2
        ax.set_xlim(0, hi)
        ax.tick_params(length=2.5)
    for ax in axes[-1]:
        ax.set_xlabel("cost / budget")
    for ax in axes[:, 0]:
        ax.set_ylabel("reward")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.0,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(BASE, "figures", "pareto_targets.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "figures", "pareto_targets.png"), dpi=200, bbox_inches="tight")
    print("saved figures/pareto_targets.{pdf,png}")


pareto_targets_figure()
