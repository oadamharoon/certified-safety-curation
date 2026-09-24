"""Flag any published result whose inputs were regenerated after it was produced.

This is the gap class that the 2026-08-16 pess-ensemble rewrite created and that no
other check could see. audit_claims compares numbers to the data layer, validate_snapshot
checks table wiring, lint_pipeline checks scripts. None of them asks the question that
actually matters for reproducibility: was the thing this result READ still the same
thing afterwards?

A result is stale-input if any artifact it consumes has an mtime NEWER than the result's
own mtime. Rerunning such a cell cannot reproduce it, and worse, results that consume the
OLD and NEW versions of the same artifact are not comparable to each other even though the
paper presents them side by side.

mtime is a weak signal on its own (a copy or a touch moves it without changing content),
so a hit is a prompt to check provenance in wandb, not a verdict. A clean result is the
meaningful direction: it proves no input moved under the result.

Usage:  python runs/scripts/check_input_recency.py [--verbose]
"""

# --- paths ------------------------------------------------------------------
# Datasets, checkpoints and run output live outside the repository. Set CSC_WORKSPACE,
# or the individual roots, to point at yours. See the README.
import os as _os


def _csc_root(_p):
    """The repository root, found by the .csc-root marker rather than by depth."""
    _d = _os.path.dirname(_os.path.abspath(_p))
    while True:
        if _os.path.exists(_os.path.join(_d, ".csc-root")):
            return _d
        _up = _os.path.dirname(_d)
        if _up == _d:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(_p)))
        _d = _up


# __file__ is undefined when a script's source is exec'd in a fresh namespace, which the
# audit does to reuse the table builder's tables; fall back to the working directory, which
# the .csc-root walk resolves from anywhere inside the repository.
_self = globals().get("__file__") or _os.path.join(_os.getcwd(), "_")
CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(_self))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "datasets"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs ship with the repository, so they resolve on their own
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import os
import re
import sys

ROOT = CSC_WORKSPACE
NEW = os.path.join(ROOT, "datasets")
OUT = os.path.join(NEW, "outputs")

# Which artifacts each arm family reads. Families absent here consume no scorer:
# bc_all / bc_safe / bcsafeseg clone raw data, and the random/return/retbot controls
# select without ever calling the value, which is why they were exempt from the
# ensemble-dependency sweep earlier.
def deps_for(arm, seed):
    """Artifacts this arm actually reads. Verified against each script's env
    contract, not guessed: an over-broad map produced 1025 hits dominated by
    false positives, which is worse than no check because it buries the real one."""
    d = []
    # learned-source xlab reads the pess ensemble; oracle sources derive the
    # advantage from ground-truth cost and never open a checkpoint.
    if arm.startswith("xlab") and "oracle" not in arm:
        d.append(f"v_ensemble_pess_seed{seed}.pt")
    if arm.startswith("cf_octg"):
        d += [f"v_ensemble_octg_seed{seed}.pt", f"dyn_ensemble_seed{seed}.pt", "bc_policy.pt"]
    elif arm.startswith("cf_"):
        d += [f"v_ensemble_pess_seed{seed}.pt", f"dyn_ensemble_seed{seed}.pt", "bc_policy.pt"]
    # noise arms score with their OWN perturbed-label ensemble, not pess.
    m = re.match(r"calfilt_noise(\d+)$", arm)
    if m:
        d.append(f"v_ensemble_noise{m.group(1)}_seed{seed}.pt")
    elif arm.startswith("calfilt") or \
            arm in ("vfilt_q25", "vfilt_matchgt", "vfilt_calsafe"):
        d.append(f"v_ensemble_pess_seed{seed}.pt")
    # wbc / mwbc / bc_a* / toph / bc_echo are produced by 04s_bc_on_subset.py, which
    # has ZERO v_ensemble references: it clones a KEPT_JSON selection. Their tie to
    # the ensemble is historical (the selection was derived from it), not a runtime
    # read, so an ensemble rewrite cannot change them. Proven, not argued: the
    # resample fingerprint "eff trajectories 201/405 unique" reproduces exactly on
    # today's selection, and a full rerun with BATCH_SIZE pinned reproduced R and C
    # to 0.000000. Assigning them the ensemble produced ~110 false positives.
    # the random/return/retbot controls select without ever calling the value.
    # calfilt_pref additionally thresholds on held-out preference labels.
    if arm == "calfilt_pref" or arm.startswith("qfilt"):
        d += ["active_segments.pkl", "gt_labels.json"]
    if arm == "bcsafeseg":
        d.append("active_segments.pkl")
    return d


# Families proven to reproduce bit-exactly, so a moved mtime on their inputs is a
# copy/touch rather than a content change. mtime is a weak signal; a reproduction
# test is a strong one, and where we have the strong one it wins.
VERIFIED_EXACT = {"bcsafeseg", "labels_only", "labels_only_n50",
                  "labels_only_n100", "labels_only_n400"}

UNIVERSAL = []


def main():
    verbose = "--verbose" in sys.argv
    stale, clean, skipped = [], 0, 0
    for task in sorted(os.listdir(OUT)):
        tdir = os.path.join(OUT, task)
        if not os.path.isdir(tdir):
            continue
        for fn in sorted(os.listdir(tdir)):
            m = re.match(r"eval_results_(.+?)_seed(\d+)\.json$", fn)
            if not m:
                continue
            arm, seed = m.group(1), m.group(2)
            rp = os.path.join(tdir, fn)
            rt = os.path.getmtime(rp)
            if arm in VERIFIED_EXACT:
                skipped += 1
                continue
            names = deps_for(arm, seed) + UNIVERSAL
            if not names:
                skipped += 1
                continue
            bad = []
            for nm in names:
                p = os.path.join(tdir, nm)
                if os.path.exists(p) and os.path.getmtime(p) > rt:
                    bad.append((nm, os.path.getmtime(p) - rt))
            if bad:
                stale.append((task, arm, seed, bad))
            else:
                clean += 1

    print(f"checked {clean + len(stale)} results ({skipped} with no tracked inputs)")
    print(f"  clean (no input moved after the result): {clean}")
    print(f"  STALE-INPUT: {len(stale)}\n")
    by_dep = {}
    for task, arm, seed, bad in stale:
        for nm, _ in bad:
            key = re.sub(r"seed\d+", "seed<n>", nm)
            by_dep.setdefault(key, set()).add(task)
    for nm, tasks in sorted(by_dep.items(), key=lambda kv: -len(kv[1])):
        print(f"  {nm:34s} newer than results on {len(tasks)} task(s): "
              + ", ".join(sorted(tasks)[:6]) + (" ..." if len(tasks) > 6 else ""))
    if verbose:
        print()
        for task, arm, seed, bad in stale:
            worst = max(b[1] for b in bad) / 86400.0
            print(f"    {task}/{arm} seed{seed}: "
                  + ", ".join(n for n, _ in bad) + f"  (up to {worst:.1f}d newer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
