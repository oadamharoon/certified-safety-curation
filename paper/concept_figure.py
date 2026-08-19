"""Figure 1: the idea, drawn rather than diagrammed.

Three stages read left to right. (1) The dataset is a bundle of whole
trajectories through a hazard field; some graze hazards, most of the pool is
mixed, and the method is never told which is which. (2) Each trajectory
collapses to a single number, so the readout is at trajectory scale rather than
per transition; the two hidden populations separate but overlap. (3) A
threshold cuts the score axis and a small labeled sample audits what survives,
which either certifies the kept set or refuses.

Schematic: shapes are illustrative, not measured.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a9a", "#2b2b2b"
rng = np.random.default_rng(4)


def kde(x, pts, bw):
    return np.exp(-0.5 * ((x[:, None] - pts[None]) / bw) ** 2).sum(1) / (len(pts) * bw)


def main():
    fig = plt.figure(figsize=(12.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.06, 1.58], wspace=0.13,
                          left=0.015, right=0.985, top=0.84, bottom=0.10)

    # ================= (1) trajectories through a hazard field =================
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 0.8), 9.4, 8.6, fill=False, ec="0.75", lw=1.0))
    haz = [(3.2, 6.6), (6.2, 4.6), (4.6, 2.6), (7.8, 7.2)]
    for hx, hy in haz:
        ax.add_patch(Circle((hx, hy), 0.80, facecolor=ORANGE, alpha=0.22, ec=ORANGE,
                            ls=(0, (3, 2)), lw=1.1, zorder=1))
    def path(y0, y1, bow, n=140):
        t = np.linspace(0, 1, n)
        x = 0.7 + 8.6 * t
        y = y0 + (y1 - y0) * t + bow * np.sin(np.pi * t)
        return x, y + 0.16 * np.sin(6.5 * t + y0)
    # colour is derived from the geometry, not asserted: a path is drawn orange
    # exactly when it enters a hazard disc. Asserting the labels independently
    # let blue paths run straight through hazards.
    unsafe = [(1.6, 4.3, 1.5), (5.1, 3.5, -1.7), (8.6, 5.6, 1.0),
              (6.5, 2.6, 0.7)]
    safe = [(1.6, 4.2, -1.4), (2.8, 3.4, -1.4), (3.9, 4.5, 1.8), (5.1, 4.1, -1.1),
            (6.3, 8.8, 1.0), (7.4, 3.4, -1.6), (8.6, 3.4, 1.8)]
    hz = np.array(haz)
    for y0, y1, b in safe + unsafe:
        x, y = path(y0, y1, b)
        d = np.min(np.linalg.norm(np.stack([x, y], 1)[:, None] - hz[None], axis=2))
        through = d < 0.80                       # hazard radius drawn below
        ax.plot(x, y, color=ORANGE if through else BLUE,
                lw=1.25 if through else 1.15,
                alpha=0.80 if through else 0.68, zorder=3 if through else 2)
    ax.set_title("a pool of whole trajectories", fontsize=9, color=INK, pad=14)
    ax.text(5.0, 10.0, "some pass through hazards, most do not;\nwhich is which is never given to the method",
            ha="center", va="bottom", fontsize=7, color=GREY, linespacing=1.4)
    ax.text(0.3, 0.15, "hazard field", fontsize=6.8, color=ORANGE, ha="left", va="bottom")

    # ================= (2) each trajectory collapses to one number =============
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(-0.5, 10.5); ax.set_ylim(0, 12); ax.axis("off")
    xs = np.linspace(0, 10, 400)
    s_pts = rng.normal(6.6, 1.35, 260); u_pts = rng.normal(3.5, 1.25, 240)
    ds, du = kde(xs, s_pts, 0.75), kde(xs, u_pts, 0.72)
    sc = 6.2 / max(ds.max(), du.max())
    ax.fill_between(xs, 1.6, 1.6 + sc * du, color=ORANGE, alpha=0.55, lw=0, zorder=2)
    ax.fill_between(xs, 1.6, 1.6 + sc * ds, color=BLUE, alpha=0.55, lw=0, zorder=3)
    ax.plot(xs, 1.6 + sc * du, color=ORANGE, lw=1.2, zorder=4)
    ax.plot(xs, 1.6 + sc * ds, color=BLUE, lw=1.2, zorder=4)
    ax.annotate("", xy=(10.2, 1.6), xytext=(-0.3, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(10.2, 0.95, r"score $g(\tau)=\frac{1}{|\tau|}\sum_{s_t\in\tau}\bar V(s_t)$", ha="right",
            va="top", fontsize=7.6, color=INK)
    ax.text(3.3, 1.6 + sc * du.max() + 0.5, "unsafe", ha="center", fontsize=7.4, color=ORANGE)
    ax.text(7.0, 1.6 + sc * ds.max() + 0.5, "safe", ha="center", fontsize=7.4, color=BLUE)
    # the supervision that trains the value, which the pipeline otherwise hides
    for k, (yb, col) in enumerate([(1.20, "#4d4d4d"), (0.72, "#b0b0b0")]):
        tt = np.linspace(0, 1, 40)
        ax.plot(-0.35 + 0.95 * tt, yb + 0.13 * np.sin(4.2 * tt + k), color=col, lw=1.1)
    ax.text(0.72, 1.20, r"$\sigma^{+}$", fontsize=6.8, color="#4d4d4d", va="center")
    ax.text(0.72, 0.72, r"$\sigma^{-}$", fontsize=6.8, color="#b0b0b0", va="center")
    ax.text(-0.35, 0.12, "which is safer, not by how much", fontsize=6.4,
            color=GREY, va="center", ha="left")
    ax.set_title("one number per trajectory", fontsize=9, color=INK, pad=14)
    ax.text(5.0, 10.0, "aggregating over a trajectory is where\npreferences identify the value",
            ha="center", va="bottom", fontsize=7, color=GREY, linespacing=1.4)

    # ================= (3) cut, audit, certify or refuse =======================
    ax = fig.add_subplot(gs[0, 2]); ax.set_xlim(-0.6, 15.2); ax.set_ylim(0, 12); ax.axis("off")
    ax.fill_between(xs, 1.6, 1.6 + sc * du, color=ORANGE, alpha=0.20, lw=0, zorder=2)
    ax.fill_between(xs, 1.6, 1.6 + sc * ds, color=BLUE, alpha=0.20, lw=0, zorder=3)
    TAU = 6.05
    keep = xs >= TAU
    ax.fill_between(xs[keep], 1.6, 1.6 + sc * du[keep], color=ORANGE, alpha=0.70, lw=0, zorder=4)
    ax.fill_between(xs[keep], 1.6, 1.6 + sc * ds[keep], color=BLUE, alpha=0.70, lw=0, zorder=5)
    ax.plot([TAU, TAU], [1.6, 8.3], color=INK, lw=1.5, zorder=6)
    ax.text(TAU - 0.15, 8.15, r"threshold $\lambda$", ha="right", va="bottom",
            fontsize=7.6, color=INK)
    ax.annotate("", xy=(11.1, 1.6), xytext=(-0.4, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(8.6, 2.85, "selected", ha="center", fontsize=7.6, color=INK)
    ax.text(2.6, 2.85, "discarded", ha="center", fontsize=7.6, color=GREY)
    # the labeled audit sample
    smp = rng.uniform(TAU + 0.3, 10.15, 11)
    ax.scatter(smp, np.full_like(smp, 8.85), s=17, marker="o",
               facecolor="white", edgecolor=INK, linewidth=0.9, zorder=7)
    bad = smp[:2]
    ax.scatter(bad, np.full_like(bad, 8.85), s=17, marker="o",
               facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.9, zorder=8)
    ax.text(8.3, 9.25, "200 budget-exceedance labels", ha="center", va="bottom",
            fontsize=7.2, color=INK)
    ax.add_patch(FancyBboxPatch((4.2, 0.05), 6.6, 2.0,
                                boxstyle="round,pad=0.14,rounding_size=0.25",
                                facecolor="white", edgecolor=GREEN, lw=1.2, zorder=9))
    ax.text(7.5, 1.55, "certified", ha="center", va="center", fontsize=8.4,
            color=GREEN, zorder=10)
    ax.text(7.5, 0.62, r"$\Pr[\,$certify $\wedge\ \widehat{\mathrm{unsafe}}>\alpha\,]\leq\delta$",
            ha="center", va="center", fontsize=7.2, color=INK, zorder=10, linespacing=1.4)
    ax.text(-0.5, 0.85, "or refuse, and keep a\nconservative selection", ha="left",
            va="center", fontsize=7.0, color=GREY, linespacing=1.4, style="italic")
    ax.annotate("", xy=(11.9, 1.05), xytext=(10.95, 1.05),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(12.1, 1.05, "clone the\nselection", ha="left", va="center", fontsize=7.0,
            color=INK, linespacing=1.4)
    ax.set_title("cut, then audit what survives", fontsize=9, color=INK, pad=14)
    ax.text(5.2, 10.0, "the guarantee is about the training set,\nnot the policy trained on it",
            ha="center", va="bottom", fontsize=7, color=GREY, linespacing=1.4)

    # stage arrows
    fig.canvas.draw()
    axl = fig.get_axes()
    for a, b in ((axl[0], axl[1]), (axl[1], axl[2])):
        x0, x1 = a.get_position().x1, b.get_position().x0
        mid = 0.5 * (x0 + x1)
        fig.add_artist(FancyArrowPatch((mid - 0.011, 0.42), (mid + 0.011, 0.42),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, lw=1.3, color=INK))
    dst = os.path.join(BASE, "figures", "concept")
    fig.savefig(dst + ".pdf"); fig.savefig(dst + ".png", dpi=200)
    print("  saved figures/concept.{pdf,png}")


if __name__ == "__main__":
    main()
