"""Figure 1: the pipeline, as a schematic.

Conceptual rather than empirical, so nothing here is task-specific and there is
nothing to cherry-pick. What it has to carry that prose does not: the
supervision entering at two distinct points and in two distinct forms, the
readout happening at whole-trajectory scale, and refusal being a first-class
outcome rather than a failure path.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8a8a8a", "#2b2b2b"
FS_T, FS_B, FS_S = 8.6, 7.8, 6.9          # title / body / small


def box(ax, x, y, w, h, label, sub=None, ec=INK, fc="white", lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h - 4.4, label, ha="center", va="top", fontsize=FS_T,
            color=INK, zorder=3, linespacing=1.35)
    if sub:
        ax.text(x + w / 2, y + 3.4, sub, ha="center", va="bottom", fontsize=FS_S,
                color=GREY, zorder=3, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw, color=color,
                                 linestyle=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(12.2, 3.25))
    ax.set_xlim(0, 215); ax.set_ylim(0, 66); ax.axis("off")
    plt.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)

    # ---- supervision enters at two places, in two different forms
    box(ax, 2, 34, 32, 24, "Segment comparisons",
        "\u201cwhich clip looks safer\u201d\nordinal, no magnitudes", ec=BLUE)
    box(ax, 2, 4, 32, 20, "Budget checks",
        "\u201cdid this episode go over?\u201d\n200 binary labels", ec=ORANGE)

    # ---- main pipeline
    box(ax, 44, 34, 32, 24, "State-only value  $\\bar V$",
        "Bradley-Terry on segment sums\nensemble of $K$; never sees actions")
    box(ax, 86, 34, 32, 24, "Score whole trajectories",
        "$g(\\tau)=\\mathrm{mean}_t\\,\\bar V(s_t)$\naggregated, not per transition")
    box(ax, 128, 34, 30, 24, "Learn-then-Test",
        "walk the threshold grid,\nstop at the first failure")
    arrow(ax, 34.6, 46, 43.4, 46, color=BLUE)
    arrow(ax, 76.6, 46, 85.4, 46)
    arrow(ax, 118.6, 46, 127.4, 46)

    # ---- the labels route underneath, not through, the pipeline
    arrow(ax, 34.6, 14, 141.5, 14, color=ORANGE, style="-")
    arrow(ax, 141.5, 14, 141.5, 33.4, color=ORANGE)
    ax.text(88, 24.5, "the only place cost information enters, and only as one bit "
                      "per sampled trajectory", ha="center", va="center",
            fontsize=FS_S, color=ORANGE)

    # ---- both outcomes are first-class
    box(ax, 172, 38, 30, 20, "Certified selection",
        "unsafe fraction $\\leq\\alpha$\nwith probability $1-\\delta$", ec=GREEN)
    box(ax, 172, 4, 30, 20, "Refusal",
        "conservative selection,\nreported as uncertified", ec=GREY, ls=(0, (4, 2)))
    arrow(ax, 158.6, 50, 171.4, 50, color=GREEN)
    ax.text(165.0, 51.8, "certifies", ha="center", va="bottom", fontsize=FS_S, color=GREEN)
    arrow(ax, 158.6, 40, 171.4, 18, color=GREY, rad=-0.26, ls=(0, (4, 2)))
    ax.text(166.5, 22.0, "refuses", ha="center", va="center",
            fontsize=FS_S, color=GREY)

    # ---- both are cloned
    ax.add_patch(FancyBboxPatch((206, 4), 7, 54,
                                boxstyle="round,pad=0.6,rounding_size=2",
                                linewidth=1.1, edgecolor=INK, facecolor="white", zorder=2))
    ax.text(209.5, 31, "behavior cloning", ha="center", va="center", fontsize=FS_B,
            color=INK, rotation=90, zorder=3)
    arrow(ax, 202.6, 48, 205.4, 48, color=GREEN)
    arrow(ax, 202.6, 14, 205.4, 14, color=GREY, ls=(0, (4, 2)))

    # ---- what the method never touches
    ax.text(2, 63, "never used:", fontsize=FS_S, color=GREY, va="center")
    ax.text(25.5, 63, "$c(s,a)$", fontsize=FS_B, color=GREY, va="center")
    ax.plot([22.4, 33.6], [63, 63], color=ORANGE, lw=1.1, zorder=5)
    ax.text(36, 63, "a cost value on every transition, which every full-label "
                    "baseline requires", fontsize=FS_S, color=GREY, va="center")

    dst = os.path.join(BASE, "figures", "concept")
    fig.savefig(dst + ".pdf")
    fig.savefig(dst + ".png", dpi=200)
    print("  saved figures/concept.{pdf,png}")


if __name__ == "__main__":
    main()
