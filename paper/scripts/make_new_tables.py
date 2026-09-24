"""Emit the LaTeX bodies for the four results added after the arXiv-ready baseline.

  policy_cert.tex     A1  Clopper-Pearson bound on Pr[cost > budget] per arm
  meancost_cert.tex   A2  Maurer-Pontil bound on E[cost] from 2000 episodes/seed
  certrate_pred.tex   B   exact predicted vs observed certification rate
  cardinal_ctrl.tex   G   binary vs cardinal supervision at matched budget

Each reads only the artifact its own script wrote, so regenerating a result and
regenerating its table are the same operation.
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
T = os.path.join(D, "tables")
NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
         "ant_velocity": "Ant", "hopper_velocity": "Hopper",
         "swimmer_velocity": "Swimmer", "cargoal1_dsrl": "CarGoal1",
         "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1",
         "pointgoal2": "PointGoal2", "pointbutton1": "PointButton1",
         "pointbutton2": "PointButton2", "carbutton1_t3": "CarButton1",
         "carbutton2": "CarButton2", "pointcircle1": "PointCircle1",
         "pointcircle2": "PointCircle2"}
ORDER = list(NAMES)


def w(fn, lines):
    with open(os.path.join(T, fn), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {fn} ({len(lines)} rows)")


def policy_cert():
    d = json.load(open(os.path.join(D, "policy_certificate.json")))
    arms = ["bc", "bcsafe", "vfilt_calsafe", "__gated__"]
    out = []
    for t in ORDER:
        e = d["per_task"].get(t, {})
        cells = []
        for a in arms:
            c = e.get(a)
            cells.append("--" if not c else f"{c['cp_upper']:.3f}")
        out.append(NAMES[t] + " & " + " & ".join(cells) + r" \\")
    w("policy_cert.tex", out)


def meancost_cert():
    d = json.load(open(os.path.join(D, "mean_cost_certificate.json")))
    out = []
    for t in ORDER:
        e = d["per_task"].get(t)
        if not e:
            out.append(NAMES[t] + " & -- & -- & -- & -- " + r"\\")
            continue
        ms = [v["mean"] for v in e["seeds"].values()]
        ub = max(v["ub"] for v in e["seeds"].values())
        cert = e["all_seeds_certified"]
        b = f"\\textbf{{{ub:.1f}}}" if cert else f"{ub:.1f}"
        out.append(f"{NAMES[t]} & {np.mean(ms):.2f} & {b} & {e['cost_limit']:.0f} & "
                   + ("yes" if cert else "no") + r" \\")
    w("meancost_cert.tex", out)


def certrate_pred():
    d = json.load(open(os.path.join(D, "cert_rate_theory.json")))
    by = {}
    for c in d["cells"]:
        by.setdefault((c["task"], c["n"]), []).append(c)
    out = []
    for t in ORDER:
        row = [NAMES[t]] if any((t, n) in by for n in (50, 100, 200, 400)) else None
        if row is None:
            continue
        for n in (50, 100, 200, 400):
            cs = by.get((t, n), [])
            if not cs:
                row.append("--")
            else:
                row.append(f"{np.mean([c['pred'] for c in cs]):.2f}/"
                           f"{np.mean([c['obs'] for c in cs]):.2f}")
        out.append(" & ".join(row) + r" \\")
    w("certrate_pred.tex", out)


def cardinal_ctrl():
    d = json.load(open(os.path.join(D, "cardinal_control.json")))
    out = []
    for t in ORDER:
        e = d["per_task"].get(t)
        if not e:
            continue
        b, c, lim = e["binary"], e["cardinal"], e["cost_limit"]
        fb = f"\\textbf{{{b['C']:.1f}}}" if b["safe"] else f"{b['C']:.1f}"
        fc = f"\\textbf{{{c['C']:.1f}}}" if c["safe"] else f"{c['C']:.1f}"
        out.append(f"{NAMES[t]} & {b['kept_unsafe']:.3f} & {fb} & "
                   f"{c['kept_unsafe']:.3f} & {fc} & {lim:.0f}" + r" \\")
    w("cardinal_ctrl.tex", out)


if __name__ == "__main__":
    policy_cert(); meancost_cert(); certrate_pred(); cardinal_ctrl()
