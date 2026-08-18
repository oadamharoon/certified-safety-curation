"""Compose the three PointGoal1 layout panels into the appendix figure."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (7, 11, 23)
panels = [mpimg.imread(os.path.join(
    BASE, "figures", f"landscape_pointgoal1_dsrl_seed{s}.png")) for s in SEEDS]
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
for ax, im in zip(axes, panels):
    ax.imshow(im)
    ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.01)
out = os.path.join(BASE, "figures", "landscape_pointgoal1_layouts")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("saved figures/landscape_pointgoal1_layouts.{pdf,png}")
