"""Figure 1: what the procedure actually decides, on real PointGoal1 data.

Three panels, no schematic boxes. (a) trajectory scores separate safe from
unsafe pools, which is why thresholding is possible at all. (b) contamination
of the kept set against the threshold grid, with alpha: the gap is the purity
margin that governs whether certification is attainable. (c) the fixed-sequence
walk of hypergeometric p-values against delta, which is the decision itself.
CPU-only.
"""
import json, os, pickle, sys
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.abspath(os.path.join(BASE, "..", "vlm-with-cpl", "new_data"))
sys.path.insert(0, NEW_DATA)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import hypergeom
from utils.common import load_cfg
from model.policy import VEnsemble

QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
ALPHA, DELTA, CAL_N, SEED = 0.25, 0.1, 200, 0
CACHE = os.path.join(BASE, "data", "method_fig_scores.pkl")
INK = "#333333"


def scores_and_labels():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    os.chdir(NEW_DATA)
    cfg = load_cfg()
    lim = cfg["cost_limit"]
    trajs = pickle.load(open(cfg["data_pickle"], "rb"))
    unsafe = (np.array([float(np.sum(t["costs"])) for t in trajs]) > lim).astype(int)
    ck = torch.load(os.path.join(NEW_DATA, cfg["output_dir"],
                                 f"v_ensemble_pess_seed{SEED}.pt"),
                    map_location="cpu", weights_only=False)
    ens = VEnsemble(ck["obs_dim"], ck["hidden_dim"], K=ck["K"])
    ens.load_state_dict(ck["state_dict"]); ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32)
            g[i] = torch.cat([ens.forward_all(o[j:j+8192]).mean(0)
                              for j in range(0, len(o), 8192)]).mean().item()
    out = (g, unsafe, lim)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def main():
    g, unsafe, lim = scores_and_labels()
    N = len(g)
    rng = np.random.default_rng(1000 + SEED)
    cal = rng.choice(N, CAL_N, replace=False)
    cs, cu = g[cal], unsafe[cal]

    taus = [float(np.quantile(g, q)) for q in QS]
    Ns = [int((g >= t).sum()) for t in taus]
    contam = [float(unsafe[g >= t].mean()) for t in taus]
    # every grid point, so the figure can show that the walk stops at the first
    # failure even where deeper thresholds would have passed on their own
    pvals, stop, walking = [], None, True
    for j, t in enumerate(taus):
        sel = cs >= t
        m, k = int(sel.sum()), int(cu[sel].sum())
        ks = int(ALPHA * Ns[j]) + 1
        p = float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0
        pvals.append(p)
        if walking:
            if p <= DELTA: stop = j
            else: walking = False

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.65))
    plt.subplots_adjust(left=0.055, right=0.995, wspace=0.30, bottom=0.19, top=0.86)

    # (a) the score separates the two pools
    ax = axes[0]
    bins = np.linspace(g.min(), g.max(), 46)
    ax.hist(g[unsafe == 0], bins=bins, color="#0072B2", alpha=0.80,
            label=f"safe ({int((unsafe==0).sum())})", linewidth=0)
    ax.hist(g[unsafe == 1], bins=bins, color="#D55E00", alpha=0.75,
            label=f"unsafe ({int(unsafe.sum())})", linewidth=0)
    if stop is not None:
        ax.axvline(taus[stop], color=INK, lw=1.4)
        ax.annotate("certified\nthreshold", xy=(taus[stop], ax.get_ylim()[1]*0.72),
                    xytext=(6, 0), textcoords="offset points", fontsize=7,
                    color=INK, va="center")
    ax.set_xlabel(r"trajectory score $g(\tau)$", fontsize=8)
    ax.set_ylabel("trajectories", fontsize=8)
    ax.set_title("(a) the score ranks whole trajectories", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7)

    # (b) contamination against the grid, with alpha
    ax = axes[1]
    ax.plot(QS, contam, marker="o", markersize=4, color="#0072B2", lw=1.5,
            markeredgewidth=0)
    ax.axhline(ALPHA, color="#D55E00", ls="--", lw=1.2)
    ax.annotate(r"$\alpha$", xy=(QS[0], ALPHA), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color="#D55E00")
    jbest = int(np.argmin(contam))
    ax.annotate("", xy=(QS[jbest], contam[jbest]), xytext=(QS[jbest], ALPHA),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.annotate("purity margin", xy=(QS[jbest], (ALPHA + contam[jbest]) / 2),
                xytext=(7, 0), textcoords="offset points", fontsize=7, color=INK,
                va="center")
    ax.set_xlabel("threshold, as a score quantile", fontsize=8)
    ax.set_ylabel("unsafe fraction of kept set", fontsize=8)
    ax.set_title("(b) what the pool allows", fontsize=8.5)
    ax.invert_xaxis(); ax.tick_params(labelsize=7)

    # (c) the fixed-sequence walk
    ax = axes[2]
    reach = (stop + 1) if stop is not None else 0
    ax.plot(QS[:reach+1], pvals[:reach+1], marker="o", markersize=4.5,
            color="#0072B2", lw=1.5, markeredgewidth=0, label="walk reaches")
    ax.plot(QS[reach+1:], pvals[reach+1:], marker="o", markersize=3.5,
            color="0.65", lw=1.2, ls=":", markeredgewidth=0,
            label="never tested")
    ax.axhline(DELTA, color="#D55E00", ls="--", lw=1.2)
    ax.annotate(r"$\delta$", xy=(QS[-1], DELTA), xytext=(-10, 4),
                textcoords="offset points", fontsize=8, color="#D55E00", ha="right")
    if stop is not None and stop + 1 < len(pvals):
        ax.plot([QS[stop+1]], [pvals[stop+1]], marker="X", markersize=8,
                color="#D55E00", markeredgewidth=0, zorder=5)
        ax.annotate("first failure,\nwalk stops here", xy=(QS[stop+1], pvals[stop+1]),
                    xytext=(14, 16), textcoords="offset points", fontsize=7,
                    color=INK, ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.set_yscale("log")
    ax.set_xlabel("threshold, as a score quantile", fontsize=8)
    ax.set_ylabel("hypergeometric $p$-value", fontsize=8)
    ax.set_title("(c) certify, or refuse and fall back", fontsize=8.5)
    ax.invert_xaxis(); ax.tick_params(labelsize=7)

    dst = os.path.join(BASE, "figures", "method")
    fig.savefig(dst + ".pdf")
    fig.savefig(dst + ".png", dpi=200)
    json.dump({"task": "pointgoal1_dsrl", "seed": SEED, "N": int(N),
               "pool_unsafe": float(unsafe.mean()), "alpha": ALPHA, "delta": DELTA,
               "quantiles": QS, "contamination": contam, "pvalues": pvals,
               "stopped_at_index": stop, "certified_quantile": QS[stop] if stop is not None else None,
               "margin": float(ALPHA - min(contam))},
              open(os.path.join(BASE, "data", "method_fig.json"), "w"), indent=1)
    print(f"  N={N} pool unsafe {unsafe.mean():.3f} | margin {ALPHA-min(contam):+.3f} "
          f"| walk certified through q={QS[stop] if stop is not None else None}")
    print("  saved figures/method.{pdf,png}")


if __name__ == "__main__":
    main()
