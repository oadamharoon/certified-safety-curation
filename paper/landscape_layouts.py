"""Safety landscape on three PointGoal1 layouts, plus radial profiles.

Rolls the BC-All and BC-Safe behavior policies in a fixed layout with action
noise, bins visited states by (x, y),
and renders mean ensemble V per cell. Cells never visited are left gray, so the
map shows the value on the distribution the policy actually induces. The fourth
panel reduces the same cells to a radial profile against distance to the
nearest hazard, with the per-layout cell-level Spearman correlation.
CPU-only.
"""
import json, os, pickle, sys
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.abspath(os.path.join(BASE, "..", "vlm-with-cpl", "new_data"))
sys.path.insert(0, NEW_DATA)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from utils.common import load_cfg
from model.policy import VEnsemble, GaussianPolicy

SEEDS = (7, 11, 23)
EPISODES = 100          # 100 rollouts per layout, split across the two policies
NBIN, EXT = 30, 1.8


CACHE = os.path.join(BASE, "data", "landscape_panels.pkl")


def main():
    if "--replot" in sys.argv and os.path.exists(CACHE):
        panels = pickle.load(open(CACHE, "rb"))
        print(f"  replotting from cache ({len(panels)} layouts)")
        render(panels)
        return
    os.chdir(NEW_DATA)
    cfg = load_cfg()
    out = os.path.join(NEW_DATA, cfg["output_dir"])
    vck = torch.load(os.path.join(out, "v_ensemble_pess_seed0.pt"),
                     map_location="cpu", weights_only=False)
    ens = VEnsemble(vck["obs_dim"], vck["hidden_dim"], K=vck["K"])
    ens.load_state_dict(vck["state_dict"]); ens.eval()
    # BC-All and BC-Safe, not the calibrated policy. The calibrated policy avoids
    # hazards by construction, which truncates the very axis this figure
    # correlates against and attenuates the correlation downward. BC-All is
    # unfiltered and enters the hazard field; together they span the range.
    def _load_pol(fn):
        ck = torch.load(os.path.join(out, fn), map_location="cpu",
                        weights_only=False)
        m = GaussianPolicy(ck["obs_dim"], ck["act_dim"], ck["hidden_dim"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m
    POLS = [_load_pol("bc_bc_policy.pt"), _load_pol("bc_bcsafe_seed0_policy.pt")]

    import gymnasium as gym
    try:
        import dsrl
        if hasattr(dsrl, "register_envs"): dsrl.register_envs()
    except Exception:
        pass
    env = gym.make(cfg["env_name"])

    panels = []
    for sd in SEEDS:
        xs, ys, vs = [], [], []
        haz = goal = hr = None
        for ep in range(EPISODES):
            pol = POLS[ep % len(POLS)]      # alternate BC-All / BC-Safe
            obs, _ = env.reset(seed=sd)
            task = env.unwrapped.task
            if haz is None:
                haz = np.array([p[:2] for p in task.hazards.pos])
                hr = float(task.hazards.size)
                goal = np.array(task.goal.pos[:2])
            done = False
            obs_buf = []
            while not done:
                with torch.no_grad():
                    ot = torch.as_tensor(obs, dtype=torch.float32)[None]
                    a = pol(ot); a = a[0] if isinstance(a, tuple) else a
                p = np.array(task.data.qpos[:2])
                xs.append(p[0]); ys.append(p[1]); obs_buf.append(obs)
                act = a.squeeze(0).numpy()
                act = act + 0.20 * np.random.randn(*act.shape)  # light jitter for coverage
                obs, _, term, trunc, _ = env.step(np.clip(act, -1, 1))
                done = term or trunc
            with torch.no_grad():
                ob = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
                vs.extend(ens.forward_all(ob).mean(0).numpy().tolist())
        xs, ys, vs = map(np.asarray, (xs, ys, vs))
        edges = np.linspace(-EXT, EXT, NBIN + 1)
        ix = np.clip(np.digitize(xs, edges[1:-1]), 0, NBIN - 1)
        iy = np.clip(np.digitize(ys, edges[1:-1]), 0, NBIN - 1)
        grid = np.full((NBIN, NBIN), np.nan)
        cx, cy, cv = [], [], []
        for a in range(NBIN):
            for b in range(NBIN):
                m = (ix == a) & (iy == b)
                if m.sum() >= 25:
                    grid[b, a] = vs[m].mean()
                    cx.append(0.5*(edges[a]+edges[a+1])); cy.append(0.5*(edges[b]+edges[b+1]))
                    cv.append(vs[m].mean())
        cx, cy, cv = map(np.asarray, (cx, cy, cv))
        dmin = np.min(np.linalg.norm(np.stack([cx, cy], 1)[:, None] - haz[None], axis=2), 1)
        rho = float(spearmanr(dmin, cv).statistic)
        d_state = np.min(np.linalg.norm(np.stack([xs, ys], 1)[:, None] - haz[None], axis=2), 1)
        rho_state = float(spearmanr(d_state, vs).statistic)
        panels.append(dict(seed=sd, grid=grid, haz=haz, hr=hr, goal=goal,
                           d=dmin, v=cv, rho=rho, rho_state=rho_state, edges=edges))
        print(f"  layout {sd}: {len(cv)} cells from {len(vs)} states, "
              f"cell rho = {rho:.3f}, state rho = {rho_state:.3f}", flush=True)

    pickle.dump(panels, open(CACHE, "wb"))
    render(panels)


def render(panels):
    EXT = 1.8
    fig = plt.figure(figsize=(13.6, 3.1))
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    outer = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.32], wspace=0.20,
                             left=0.045, right=0.985)
    left = outer[0, 0].subgridspec(1, 3, wspace=0.09)
    axes = [fig.add_subplot(left[0, i]) for i in (0, 1, 2)]
    axes.append(fig.add_subplot(outer[0, 1]))
    vmin = min(np.nanmin(p["grid"]) for p in panels)
    vmax = max(np.nanmax(p["grid"]) for p in panels)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("0.82")
    for ax, p in zip(axes[:3], panels):
        im = ax.imshow(np.ma.masked_invalid(p["grid"]), origin="lower",
                       extent=[-EXT, EXT, -EXT, EXT], cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        for h in p["haz"]:
            ax.add_patch(plt.Circle(h, p["hr"], fill=False, color="w",
                                    ls="--", lw=1.0))
        ax.add_patch(plt.Circle(p["goal"], 0.18, fill=False, color="w", lw=1.4))
        ax.text(*p["goal"], "G", color="w", ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.set_title(f"layout seed {p['seed']}", fontsize=9)
        ax.set_xlabel("x", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
    axes[0].set_ylabel("y", fontsize=8)
    for ax in axes[1:3]: ax.set_yticklabels([])
    cax = inset_axes(axes[2], width="4.5%", height="100%", loc="lower left",
                     bbox_to_anchor=(1.055, 0.0, 1, 1),
                     bbox_transform=axes[2].transAxes, borderpad=0)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"mean ensemble $\bar V(s)$", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    ax = axes[3]
    for p, ls in zip(panels, ("-", "--", ":")):
        b = np.quantile(p["d"], np.linspace(0, 1, 13)); b[-1] += 1e-9
        k = np.clip(np.digitize(p["d"], b[1:-1]), 0, 11)
        bd = np.array([p["d"][k == i].mean() for i in range(12) if (k == i).any()])
        bv = np.array([p["v"][k == i].mean() for i in range(12) if (k == i).any()])
        ax.plot(bd, bv, ls, lw=1.5, label=f"seed {p['seed']} ($\\rho$ = {p['rho_state']:.2f})")
    ax.axvline(panels[0]["hr"], color="0.4", ls=":", lw=1)
    _lo, _hi = ax.get_ylim()
    ax.text(panels[0]["hr"], _hi - 0.06 * (_hi - _lo), " hazard radius",
            fontsize=8.5, color="0.35", va="top", rotation=90)
    ax.set_xlabel("distance to nearest hazard", fontsize=8)
    ax.set_ylabel(r"learned $\bar V(s)$", fontsize=8)
    ax.set_title("radial profiles", fontsize=9)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    dst = os.path.join(BASE, "figures", "landscape_pointgoal1_layouts")
    fig.savefig(dst + ".pdf", bbox_inches="tight")
    fig.savefig(dst + ".png", dpi=200, bbox_inches="tight")
    json.dump({str(p["seed"]): {"rho_cell": p["rho"], "rho_state": p["rho_state"],
                                "n_cells": int(len(p["v"]))}
               for p in panels},
              open(os.path.join(BASE, "data", "landscape_layouts.json"), "w"), indent=1)
    print("  saved figures/landscape_pointgoal1_layouts.{pdf,png}")


if __name__ == "__main__":
    main()
