"""Does CARDINAL supervision buy anything over BINARY at matched label budget?

The paper used to assert that labels-only "presumes a cost function that returns a
number". It does not: 04s forms a binary over-budget indicator, the same judgment our
calibration sample consumes. That wording is corrected, but the baseline it described
is worth having, because without it the supervision hierarchy the paper now claims
(ordinal < binary < cardinal) has an untested top rung.

This arm holds everything fixed against the labels-only n=200 run it is compared with
-- same script, same 200 trajectories, same per-task selection fraction read from that
run's own meta file, same architecture, seeds, clone and evaluation -- and changes
only what the 200 labels carry: the numeric episodic cost instead of the bit.

Pre-registered prediction (in run_cardinal.sh before the first run): if safety here is
decodable from the ordering alone, cardinal buys no additional safe tasks at matched
budget, which would make the labels-only result a statement about curation rather than
about label richness.

Usage:  python iclr2027/scripts/cardinal_control.py [--write]
"""
import argparse
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(BASE), "datasets/outputs")
T15 = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
       "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
       "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
       "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]
NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
         "ant_velocity": "Ant", "hopper_velocity": "Hopper",
         "swimmer_velocity": "Swimmer", "cargoal1_dsrl": "CarGoal1",
         "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1",
         "pointgoal2": "PointGoal2", "pointbutton1": "PointButton1",
         "pointbutton2": "PointButton2", "carbutton1_t3": "CarButton1",
         "carbutton2": "CarButton2", "pointcircle1": "PointCircle1",
         "pointcircle2": "PointCircle2"}


def cells(task, tag_prefix, seeds=(0, 1, 2)):
    """(mean R, mean C, cost_limit, n_seeds) for an arm on a task."""
    Rs, Cs, lim = [], [], None
    for s in seeds:
        p = os.path.join(OUT, task, f"eval_results_{tag_prefix}_seed{s}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        Rs.append(j["avg_reward"])
        Cs.append(j["avg_cost"])
        lim = float(j["cost_limit"])
    if not Cs:
        return None
    return float(np.mean(Rs)), float(np.mean(Cs)), lim, len(Cs)


def kept_unsafe(task, tag):
    p = os.path.join(OUT, task, f"labelsonly_meta_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("kept_unsafe_rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows, rec = [], {}
    print(f"{'task':<15}{'binary C':>11}{'cardinal C':>13}{'budget':>8}"
          f"{'  binary':>9}{'  cardinal':>11}{'   kept-unsafe b/c':>20}")
    nb = nc = both = 0
    for t in T15:
        # collect_results.py:66 maps the paper's "labels_only" arm to files named
        # labelsonly_seed<N>. The similarly named labels_only_seed<N> files on the
        # three stage3b tasks are a separate un-harvested reverify run with a
        # different selection fraction, and comparing against those would compare
        # G to an arm the paper never reports.
        b = cells(t, "labelsonly")
        c = cells(t, "labels_cardinal")
        if not b or not c:
            print(f"{NAMES[t]:<15}{'--':>11}{'--':>13}  (missing arm)")
            continue
        lim = b[2]
        sb, sc = b[1] <= lim, c[1] <= lim
        nb += sb
        nc += sc
        both += 1
        # the harvested binary arm's metas are named labelsonly_seed<N>
        kb = np.mean([x for x in (kept_unsafe(t, f"labelsonly_seed{s}")
                                  for s in (0, 1, 2)) if x is not None] or [np.nan])
        kc = np.mean([x for x in (kept_unsafe(t, f"labels_cardinal_seed{s}")
                                  for s in (0, 1, 2)) if x is not None] or [np.nan])
        print(f"{NAMES[t]:<15}{b[1]:>11.2f}{c[1]:>13.2f}{lim:>8.0f}"
              f"{('safe' if sb else 'UNSAFE'):>9}{('safe' if sc else 'UNSAFE'):>11}"
              f"{kb:>11.3f}{kc:>9.3f}")
        rows.append((t, b, c, sb, sc, float(kb), float(kc)))
        rec[t] = {"binary": {"R": b[0], "C": b[1], "safe": bool(sb), "seeds": b[3],
                             "kept_unsafe": float(kb)},
                  "cardinal": {"R": c[0], "C": c[1], "safe": bool(sc), "seeds": c[3],
                               "kept_unsafe": float(kc)},
                  "cost_limit": lim}
    print(f"\n  binary (bits)      safe on {nb}/{both} tasks")
    print(f"  cardinal (numbers) safe on {nc}/{both} tasks")
    flips = [(NAMES[t], sb, sc) for t, _b, _c, sb, sc, _kb, _kc in rows if sb != sc]
    print(f"  verdict flips: {len(flips)}"
          + ("  " + ", ".join(f"{n}: {'b' if b else '-'}->{'c' if c else '-'}"
                              for n, b, c in flips) if flips else ""))
    if rows:
        du = [r[6] - r[5] for r in rows]
        print(f"  kept-unsafe rate, cardinal minus binary: mean {np.mean(du):+.4f}, "
              f"max {max(du):+.4f}, min {min(du):+.4f}")
    if args.write:
        p = os.path.join(BASE, "data", "cardinal_control.json")
        json.dump({"n_binary_safe": nb, "n_cardinal_safe": nc, "n_tasks": both,
                   "per_task": rec}, open(p, "w"), indent=2)
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
