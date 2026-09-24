"""Item E, stage 2: certification against controlled contamination.

The certification-rate proposition (prop:rate) predicts the certification
probability from a pool's purity structure
alone. Its validation so far used the nine real pools, which offer one purity margin
per task. This sweep manufactures pools at many contamination levels by subsampling
each real pool -- dropping unsafe trajectories to purify it, dropping safe ones to
dirty it -- and on every constructed pool runs the actual calibration walk
(guarantee_stats.py's ltt, same grid, same strict stopping, 200 draws at n = 200)
and evaluates the closed form from that pool's own (N, N_1, K_1). Scores are the
cached ones from e_cache_scores.py; nothing is retrained, and the grid quantiles are
recomputed on each constructed pool exactly as 04q would on a dataset of that shape.

Two things this can show that the real pools cannot: whether the dichotomy is a clean
threshold at margin zero when margin is varied continuously, and whether the closed
form holds off the nine points it was validated on.

Writes data/e_contamination.json and figures/contamination.{pdf,png}.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import hypergeom

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from cert_rate_theory import pr_certify  # noqa: E402

ALPHA, DELTA, N_CAL, N_DRAWS = 0.25, 0.1, 200, 200
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
TARGET_BASE = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]   # pool unsafe fractions


def ltt(scores, unsafe, rng):
    """guarantee_stats.py's walk, verbatim."""
    idx = rng.choice(len(scores), size=min(N_CAL, len(scores)), replace=False)
    cs, cu = scores[idx], unsafe[idx]
    taus = [float(np.quantile(scores, q)) for q in QS]
    cert = False
    for tau in taus:
        sel = cs >= tau; m = int(sel.sum()); k = int(cu[sel].sum())
        nsel = int((scores >= tau).sum()); ks = int(ALPHA * nsel) + 1
        if ks > nsel: break
        p = hypergeom.cdf(k, nsel, ks, m) if m > 0 else 1.0
        if m > 0 and p <= DELTA: cert = True
        else: break
    return cert


def subsample(scores, unsafe, target, rng, keep_min=400):
    """Drop safe or unsafe trajectories uniformly at random to hit `target` base rate."""
    N = len(scores); U = int(unsafe.sum()); S = N - U
    if target >= U / N:                      # need to remove safe items
        keep_u = U; keep_s = int(round(U * (1 - target) / target))
        keep_s = min(keep_s, S)
    else:                                    # need to remove unsafe items
        keep_s = S; keep_u = int(round(S * target / (1 - target)))
        keep_u = min(keep_u, U)
    if keep_u + keep_s < keep_min: return None
    ui = rng.choice(np.where(unsafe)[0], size=keep_u, replace=False)
    si = rng.choice(np.where(~unsafe)[0], size=keep_s, replace=False)
    keep = np.sort(np.concatenate([ui, si]))
    return scores[keep], unsafe[keep]


def main():
    rng = np.random.default_rng(7)
    rows = []
    for p in sorted(glob.glob(os.path.join(BASE, "data", "e_scores", "*.npz"))):
        d = np.load(p); task, seed = os.path.basename(p)[:-4].rsplit("_seed", 1)
        for tb in TARGET_BASE:
            sub = subsample(d["scores"], d["unsafe"].astype(bool), tb, rng)
            if sub is None: continue
            sc, un = sub
            N = len(sc); tau1 = float(np.quantile(sc, QS[0]))
            sel1 = sc >= tau1; N1 = int(sel1.sum()); K1 = int(un[sel1].sum())
            u1 = K1 / N1
            obs = np.mean([ltt(sc, un, rng) for _ in range(N_DRAWS)])
            pred = pr_certify(N, N1, K1, N_CAL)
            rows.append(dict(task=task, seed=int(seed), target_base=tb, N=N,
                             base_unsafe=float(un.mean()), N1=N1, K1=K1, u1=u1,
                             margin=ALPHA - u1, observed=float(obs), predicted=float(pred)))
            print(f"  {task:<20} s{seed} base={un.mean():.2f} u1={u1:.3f} "
                  f"margin={ALPHA-u1:+.3f}  obs={obs:.3f} pred={pred:.3f}", flush=True)
    o = np.array([r["observed"] for r in rows]); q = np.array([r["predicted"] for r in rows])
    mg = np.array([r["margin"] for r in rows])
    r = float(np.corrcoef(o, q)[0, 1]); mae = float(np.mean(np.abs(o - q)))
    neg = o[mg <= 0]; pos = o[mg > 0]
    summary = dict(n_cells=len(rows), pearson_r=r, mae=mae,
                   worst_rate_negative_margin=float(neg.max()) if len(neg) else None,
                   n_negative=int(len(neg)), n_positive=int(len(pos)),
                   mean_rate_positive=float(pos.mean()) if len(pos) else None)
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(BASE, "data", "e_contamination.json"), "w"), indent=1)
    print("\n  cells:", len(rows), "| r =", round(r, 4), "| MAE =", round(mae, 4))
    print("  margin<=0: n =", len(neg), "worst observed rate =", round(float(neg.max()), 4) if len(neg) else None,
          " (Prop 2(i) says <= delta =", DELTA, ")")
    print("  margin> 0: n =", len(pos), "mean rate =", round(float(pos.mean()), 3) if len(pos) else None)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.scatter(mg, o, s=14, color="#0072B2", alpha=0.75, linewidths=0, label="observed, constructed pools")
    ax.scatter(mg, q, s=10, marker="x", color="#B44B0A", alpha=0.8, linewidths=0.8, label="closed form")
    ax.axvline(0, color=INK2, lw=0.8, ls="--"); ax.axhline(DELTA, color=INK2, lw=0.8, ls=":")
    ax.annotate(r"$\delta$", xy=(mg.min(), DELTA), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel(r"purity margin $\alpha - u_1$ of the constructed pool")
    ax.set_ylabel(r"certification rate at $n = 200$")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(BASE, "figures", f"contamination.{ext}"), dpi=200, bbox_inches="tight")
    print("  saved figures/contamination.{pdf,png} and data/e_contamination.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
