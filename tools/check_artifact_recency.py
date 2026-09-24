"""Flag paper DATA ARTIFACTS that predate the checkpoints they were computed from.

check_input_recency.py covers eval_results_*.json under outputs/. It does NOT cover
iclr2027/data/, and that gap hid a real defect: guarantee_stats.json was computed on
2026-07-17 from v_ensemble_pess checkpoints that were overwritten on 2026-08-16, so
the paper's certification statistics for cargoal2, pointgoal1_dsrl and pointgoal2
described ensembles that no longer existed. Five further artifacts had the same
problem, and the first sweep missed some of them because it globbed data/*.json
rather than recursing into data/review_response/.

Rule: an artifact is stale if its mtime is older than any checkpoint it plausibly
consumed, where "plausibly consumed" means the artifact's own generator loads that
checkpoint family for a task the artifact names. mtime is weak evidence on its own,
so a hit is a prompt to regenerate and diff, not a verdict. The decisive test is the
one the regenerations used here: a deterministic generator must reproduce the
unaffected tasks bit-exactly, so any control task that moves means something else
changed too.

Usage:  python runs/scripts/check_artifact_recency.py [--verbose]
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

import glob
import os
import re
import sys
import time

ROOT = CSC_WORKSPACE
DATA = os.path.join(ROOT, "iclr2027/data")
OUT = os.path.join(ROOT, "datasets/outputs")

# Checkpoint families, and the source token that proves a generator loads one.
# Assigning every family to every artifact is what made the first version useless:
# it returned 14 hits, mostly artifacts flagged because some unrelated bc policy for
# a task they merely NAME was retrained later. An artifact is only checked against a
# family its own generator actually references.
CKPT_FAMILIES = {
    "v_ensemble_pess_*.pt": "v_ensemble_pess",
    "v_ensemble_noise*.pt": "v_ensemble_noise",
    "v_ensemble_octg_*.pt": "v_ensemble_octg",
    "dyn_ensemble_*.pt": "dyn_ensemble",
    "bc_policy.pt": "bc_policy.pt",
}
SRC_DIRS = ["certified-safety-curation/paper", "certified-safety-curation/analysis",
            "iclr2027/scripts", "runs/scripts"]


# Artifacts whose writer cannot be found by reading source, because the output path is
# computed (an f-string or a .replace), or because the file is a hand-recorded transcript of
# a script that prints rather than dumps. Listing them here is what keeps them CHECKED
# instead of silently unverifiable; None means "no builder exists", which is a standing
# reproducibility gap, not a pass.
BUILDERS = {
    "guarantee_H10_2000.json": "runs/scripts/guarantee_h_sweep.py",
    "guarantee_H30_2000.json": "runs/scripts/guarantee_h_sweep.py",
    "guarantee_H50_2000.json": "runs/scripts/guarantee_h_sweep.py",
    "disagreement_matched.json": "iclr2027/scripts/disagreement_matched.py",
    "h_seed_failures.json": "runs/scripts/build_h_seed_failures.py",
    "obstacle2_stats.json": "iclr2027/scripts/obstacle2_stats.py",
    "osrl_results_mtimes.json": "iclr2027/scripts/harvest_osrl.py",
    # landscape_fig.py prints these rho values; the json is the transcript. PointGoal1 only.
    "landscape_frozen.json": "iclr2027/scripts/landscape_fig.py",
    # full-data CDT over cost targets: reads OSRL evals, consumes no value ensemble.
    "cdt_target_sweep.json": None,
    # matched-size control selections: rank by return only, so no value ensemble is consumed
    "control_selection_stats.json": None,
    "calibration_audit.json": "runs/scripts/calibration_audit.py",
    "boltz_flip_rates.json": "runs/scripts/boltz_flip_rates.py",
    # these two now write through _write_if_changed, so the basename no longer appears
    # inside an open(..., "w") call and the static scan cannot see the writer
    "cert_rate_theory.json": "iclr2027/scripts/cert_rate_theory.py",
    "selection_stability.json": "runs/scripts/selection_stability.py",
    # data/e_scores/<task>_seed<s>.npz, written per task by a computed path
    "_e_scores_npz": "iclr2027/scripts/e_cache_scores.py",
}


def _open_calls(txt):
    """(argument text, is-write) for every open(...) in the source.

    A deterministic paren scan, not a regex: the path is often built with
    os.path.join(...), so the argument list nests, and a regex that allows
    nesting backtracks catastrophically on these files.
    """
    out = []
    for m in re.finditer(r"\bopen\s*\(", txt):
        i, depth = m.end(), 1
        while i < len(txt) and depth:
            depth += (txt[i] == "(") - (txt[i] == ")")
            i += 1
        arg = txt[m.end():i - 1]
        out.append((arg, re.search(r"[\"'][wa]b?[\"']", arg) is not None))
    return out


def writes(txt, base):
    """True if this source actually opens `base` for writing.

    Merely mentioning the basename is not enough. The first version of this test
    accepted any file containing the name plus a "json.dump" anywhere, so for
    guarantee_stats_2000.json and margin_vs_yield.json it returned make_tables.py --
    a READER -- which references no checkpoint family, and both artifacts were then
    scored against nothing and reported clean while being a month out of date.
    """
    calls = _open_calls(txt)
    for arg, is_w in calls:
        if is_w and base in arg:
            return True
    # ...or the path is bound to a name that is opened for writing later.
    names = {a.split(",")[0].strip() for a, w in calls if w}
    names = {n for n in names if re.fullmatch(r"\w+", n)}
    for n in names:
        for m in re.finditer(r"^\s*" + re.escape(n) + r"\s*=", txt, re.M):
            stmt = txt[m.end():txt.find("\n\n", m.end()) if "\n\n" in txt[m.end():] else len(txt)]
            if base in stmt.split("\n")[0] or base in "".join(stmt.split("\n")[:3]):
                return True
    return False


def generator_for(artifact_path):
    """The .py that writes this artifact. None means no writer was found."""
    base = os.path.basename(artifact_path)
    if os.sep + "e_scores" + os.sep in artifact_path:
        return os.path.join(ROOT, BUILDERS["_e_scores_npz"])
    if base in BUILDERS:
        b = BUILDERS[base]
        return os.path.join(ROOT, b) if b else None
    for d in SRC_DIRS:
        for py in sorted(glob.glob(os.path.join(ROOT, d, "*.py"))):
            try:
                txt = open(py, errors="ignore").read()
            except Exception:
                continue
            if base in txt and writes(txt, base):
                return py
    return None


def families_used(py):
    """Checkpoint families this generator references in source."""
    if not py:
        return []
    try:
        txt = open(py, errors="ignore").read()
    except Exception:
        return []
    return [g for g, tok in CKPT_FAMILIES.items() if tok in txt]


def task_dirs():
    return [d for d in sorted(os.listdir(OUT)) if os.path.isdir(os.path.join(OUT, d))]


def newest_ckpt_for(task, globs):
    """(path, mtime) of the most recent checkpoint for this task in these families."""
    best = None
    for g in globs:
        for p in glob.glob(os.path.join(OUT, task, g)):
            mt = os.path.getmtime(p)
            if best is None or mt > best[1]:
                best = (p, mt)
    return best


def main():
    verbose = "--verbose" in sys.argv
    tasks = task_dirs()
    # .npz and .pkl caches count as artifacts too. The first version globbed *.json only,
    # which is how data/e_scores/<task>_seed<s>.npz -- a pre-F0 score cache that
    # certificate_triple and e_contamination_sweep both read -- stayed invisible for a week.
    files = [p for p in glob.glob(os.path.join(DATA, "**/*"), recursive=True)
             if os.path.isfile(p) and "/archive/" not in p
             and os.path.splitext(p)[1] in (".json", ".npz", ".pkl")]
    stale, clean, unowned, known_none = [], 0, [], []
    for p in sorted(files):
        mt = os.path.getmtime(p)
        if os.path.splitext(p)[1] in (".npz", ".pkl"):
            txt = os.path.basename(p)      # binary: the task is in the filename
        else:
            try:
                txt = open(p).read()
            except Exception:
                continue
        gen = generator_for(p)
        if gen is None:
            # registered with builder None = V-independent by inspection; otherwise a gap
            (known_none if os.path.basename(p) in BUILDERS else unowned).append(p)
            continue
        globs = families_used(gen)
        if not globs:
            clean += 1          # its generator consumes no checkpoint family
            continue
        hits = []
        for t in tasks:
            if not re.search(r"\b" + re.escape(t) + r"\b", txt):
                continue
            nk = newest_ckpt_for(t, globs)
            if nk and nk[1] > mt:
                hits.append((t, nk[0], nk[1] - mt))
        if hits:
            stale.append((p, hits, gen))
        else:
            clean += 1

    print(f"checked {len(files)} artifacts under iclr2027/data (archive excluded)")
    print(f"  clean: {clean}")
    print(f"  no checkpoint dependency (registered): {len(known_none)}")
    print(f"  NO WRITER FOUND (unverifiable): {len(unowned)}")
    for p in unowned:
        print(f"    {os.path.relpath(p, ROOT)}")
    print(f"  STALE (a checkpoint for a task they name is NEWER): {len(stale)}\n")
    for p, hits, gen in stale:
        rel = os.path.relpath(p, ROOT)
        worst = max(h[2] for h in hits) / 86400.0
        print(f"  {rel}   <- {os.path.relpath(gen, ROOT) if gen else '?'}")
        print(f"    {len(hits)} task(s), up to {worst:.1f}d newer: "
              + ", ".join(t for t, _, _ in hits[:6])
              + (" ..." if len(hits) > 6 else ""))
        if verbose:
            for t, ck, dt in hits:
                print(f"      {t}: {os.path.relpath(ck, OUT)} "
                      f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(ck)))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
