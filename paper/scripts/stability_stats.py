"""Safe-set stability: how much the set of safe tasks moves under resampling.

stability_stats.json had NO surviving generator. It was written 2026-07-21, before
the 2026-08-16 ensemble rewrite, and the appendix quotes it, so its numbers could not
be checked until the computation was reverse-engineered. This file fixes that: the
definition below reproduces three of the four archived quantities exactly from the
current snapshot, which is what identifies it as the original computation.

Definition. A task is safe for an arm and seed when that cell's mean episodic cost is
within the task's budget. The safe-SET of a (arm, seed) is the subset of the fifteen
main tasks that are safe. Stability is the mean pairwise Jaccard of those sets across
whichever axis is being resampled:

  draw axis     -- seeds of a labels-only budget arm. 04s line 62 defaults
                   LABEL_DRAW_SEED to the run seed, so each seed IS an independent
                   label draw, which is why the seed axis measures draw sensitivity.
  learner axis  -- labels_only_fixdraw seeds 10/11/12: three initializations on ONE
                   fixed draw, isolating learner noise from draw noise.
  scorer axis   -- vfilt_calsafe seeds, i.e. independent value-ensemble seeds.

Reproduction check against the archived July file: n=100 draw 0.6099 vs 0.61,
learner 0.9487 vs 0.95, scorer 0.8462 vs 0.8462 (exact). The n=200 draw figure moved,
0.7802 -> 0.8498, because nine of the fifteen tasks were rescored after the rewrite.
"""
import itertools
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T15 = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
       "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
       "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
       "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]
LIM = {t: (20 if "velocity" in t else 25) for t in T15}
SNAP = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))


def safe_set(arm, seed):
    out = set()
    for t in T15:
        e = SNAP.get(t, {}).get(arm, {}).get(str(seed))
        if e and "C" in e and e["C"] <= LIM[t]:
            out.add(t)
    return out


def mean_jaccard(arm, seeds):
    sets = [s for s in (safe_set(arm, x) for x in seeds) if s]
    vals = [len(a & b) / len(a | b)
            for a, b in itertools.combinations(sets, 2) if a | b]
    return (float(np.mean(vals)) if vals else None), len(sets)


def main():
    res = {"definition": "mean pairwise Jaccard of safe-task sets over the 15 tasks",
           "draw_axis": {}, "learner_axis": {}, "scorer_axis": {}}
    for arm, n in (("labels_only_n50", 50), ("labels_only_n100", 100),
                   ("labels_only", 200), ("labels_only_n400", 400)):
        j, k = mean_jaccard(arm, (0, 1, 2))
        res["draw_axis"][str(n)] = {"arm": arm, "jaccard": j, "n_sets": k}
        print(f"  draw   n={n:<4} {arm:<20} J={j:.4f}")
    j, k = mean_jaccard("labels_only_fixdraw", (10, 11, 12))
    res["learner_axis"] = {"arm": "labels_only_fixdraw", "jaccard": j, "n_sets": k}
    print(f"  learner  fixed draw, 3 inits      J={j:.4f}")
    for arm in ("vfilt_calsafe", "calfilt_csf"):
        j, k = mean_jaccard(arm, (0, 1, 2))
        res["scorer_axis"][arm] = {"jaccard": j, "n_sets": k}
        print(f"  scorer   {arm:<22} J={j:.4f}")
    p = os.path.join(BASE, "data", "stability_stats.json")
    json.dump(res, open(p, "w"), indent=2)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
