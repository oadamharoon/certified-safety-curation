"""Scan the tree for the specific defect classes that have bitten this project.

Each check exists because the bug it looks for actually shipped and silently
corrupted results. None of them raise at runtime, which is what makes them
dangerous.

  1 double-escaped regex   r"(\\d+)" inside a raw string is a literal
                           backslash-d. Five harvester patterns had this and
                           matched nothing for weeks.
  2 missing SEED_OVERRIDE  a training script that seeds but never reads the
                           override runs every "seed" identically.
  3 unexported bash fn     a function called inside xargs/parallel subshells
                           without export -f is undefined there; the call fails
                           silently while the surrounding logic continues.
  4 ephemeral marker dir   done-markers under /tmp or a session scratchpad
                           cannot survive, so reruns cannot resume.
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
DIRS = ["certified-safety-curation/pipeline", "certified-safety-curation/analysis",
        "certified-safety-curation/experiments", "iclr2027/scripts",
        "runs/scripts", os.path.join(CSC_WORK, "scripts")]
SKIP = ("legacy/", "_quarantine", "/.git/")

BAD_ESCAPE = re.compile(r'r"[^"]*\\\\[dwsSWDb+.*][^"]*"|r\'[^\']*\\\\[dwsSWDb+.*][^\']*\'')


def seeded_by_runner():
    """Basenames of python scripts launched with SEED_OVERRIDE set."""
    out = set()
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.endswith(".sh"):
                continue
            src = open(os.path.join(p, fn), errors="ignore").read()
            src = src.replace("\\\n", " ")      # join line continuations first
            for blk in re.findall(r"SEED_OVERRIDE=\S+[^\n]*?python\s+(\S+\.py)", src):
                out.add(os.path.basename(blk))
    return out


def files(ext):
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.endswith(ext) and not any(s in fn for s in SKIP):
                yield os.path.join(p, fn)


SEEDED_BY_RUNNER = set()


def main():
    global SEEDED_BY_RUNNER
    SEEDED_BY_RUNNER = seeded_by_runner()
    hits = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    for f in files(".py"):
        src = open(f, errors="ignore").read()
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            # dict entries use ":" not "="; the original double-escape bug
            # lived in a dict literal, which this guard used to skip.
            if BAD_ESCAPE.search(code) and any(c in code for c in ("=", "return", ":")):
                hits[1].append(f"{f}:{i}  {line.strip()[:88]}")
        # only a defect if a runner actually passes SEED_OVERRIDE to it
        if ("set_seed(" in src and "SEED_OVERRIDE" not in src
                and os.path.basename(f) in SEEDED_BY_RUNNER):
            hits[2].append(f"{f}  (launched with SEED_OVERRIDE, never reads it)")

    for f in files(".sh"):
        src = open(f, errors="ignore").read()
        defined = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*\(\)\s*\{", src, re.M))
        exported = set()
        for m in re.findall(r"export -f ([^\n]+)", src):
            exported |= {x.strip() for x in re.split(r"[;\s]+", m) if x.strip()}
        # only matters if the script actually dispatches through a subshell
        if re.search(r"xargs .*bash -c|parallel ", src):
            for fn in sorted(defined - exported):
                if re.search(rf"\b{fn}\b", src.split("export -f")[-1] if "export -f" in src else src):
                    hits[3].append(f"{f}  {fn}() defined, used in subshell, never export -f'd")
        for m in re.finditer(r"^\s*LOGDIR=(\S+)", src, re.M):
            if "/tmp" in m.group(1) or "scratchpad" in m.group(1):
                hits[4].append(f"{f}  LOGDIR={m.group(1)}")

    # 5: live tree vs release mirror. the dataset tree's scripts/ is what the
    # runners execute; certified-safety-curation is the cleaned public copy. A
    # fix applied only to the mirror does nothing, which cost a 135-job relaunch.
    LIVE = os.path.join(ROOT, os.path.join(CSC_WORK, "scripts"))
    hits[5] = []
    if os.path.isdir(LIVE):
        for fn in sorted(os.listdir(LIVE)):
            if not fn.endswith(".py"):
                continue
            # baselines/ was missing here, so a live-only edit to a baseline
            # trainer (04c) drifted from the mirror without being flagged.
            for sub in ("certified-safety-curation/pipeline",
                        "certified-safety-curation/analysis",
                        "certified-safety-curation/baselines"):
                m = os.path.join(ROOT, sub, fn)
                if os.path.exists(m):
                    a = open(os.path.join(LIVE, fn), errors="ignore").read()
                    b = open(m, errors="ignore").read()
                    if a != b:
                        hits[5].append(f"{fn}  live and mirror differ "
                                       f"({sub.split('/')[-1]})")
                    break

    # 6: concurrent runner writing a per-run artifact to a seed-less filename.
    # 04c's default policy name is shared, so running seeds under xargs -P>1
    # made both evals read whichever policy saved last. Resolves one level of
    # shell variable so a $tag that already embeds the seed is not flagged.
    for f in files(".sh"):
        raw = open(f, errors="ignore").read()
        src = " ".join(raw.split("\\\n"))
        if not re.search(r"-P\s*[2-9]", src):
            continue
        # A tag taken from a job file's positional args cannot be resolved by
        # reading the script, so a runner that ENFORCES the property at runtime
        # (refusing any tag that does not embed its seed) is accepted instead.
        if re.search(r"does not embed seed", raw):
            continue
        # collect every assignment anywhere (case branches, multi-assign
        # lines like: local task=$1 seed=$2 tag="..._seed${2}")
        assigns = {}
        for nm, val in re.findall(r'(\w+)=("[^"]*"|\S+)', raw):
            assigns.setdefault(nm, []).append(val)
        def carries_seed(val, depth=0):
            if re.search(r"seed|SEED", val, re.I):
                return True
            if depth > 2:
                return False
            for v in re.findall(r"\$\{?(\w+)\}?", val):
                for cand in assigns.get(v, []):
                    if carries_seed(cand, depth + 1):
                        return True
            return False
        for m in re.finditer(r"(POLICY_OUT|OUT_TAG|--policy_file|--results_suffix)[= ]+\"?([^\s\"\\]+)", src):
            if not carries_seed(m.group(2)):
                hits[6].append(f"{os.path.basename(f)}  {m.group(1)}={m.group(2)}  (concurrent, no seed in name)")

    # 7: script writing an artifact into a session scratchpad. The certified_h5
    # selections, build_subsets2.py, bc_on_subset.py and the cplgt orchestrator
    # all lived in /tmp/claude-*/ and were deleted with the session, leaving
    # published results unreproducible.
    for f in list(files(".py")) + list(files(".sh")):
        # the mirror's experiments/ dir is a historical record of runs already
        # executed; it necessarily references the scratchpad those runs used.
        if "certified-safety-curation/experiments" in f:
            continue
        for i, line in enumerate(open(f, errors="ignore").read().splitlines(), 1):
            code = line.split("#", 1)[0]
            # require a real path use, not prose mentioning the word
            if not re.search(r"/tmp/claude-|\$S/|\$\{S\}/|\$SCRATCH", code):
                continue
            # logs and done-markers in scratchpad are wasteful but recoverable;
            # what cost us was ARTIFACTS and executable scripts living there.
            if re.search(r"(\.py|\.hdf5|\.h5|_kept\.json|\.pkl|\.pt)\b", code) or \
               re.search(r"(--subset_h5|KEPT_JSON|SUBSET_KEPT|LABELS_OUT|POLICY_OUT|OUT=)", code):
                hits[7].append(f"{os.path.basename(f)}:{i}  {line.strip()[:80]}")

    titles = {1: "double-escaped regex in raw string",
              2: "training script ignoring SEED_OVERRIDE",
              3: "bash function used in subshell but not exported",
              4: "done-markers in an ephemeral directory",
              5: "live script and release mirror out of sync",
              6: "concurrent runner writes a seed-less per-run filename",
              7: "script writes artifacts into a session scratchpad"}
    bad = 0
    for k in (1, 2, 3, 4, 5, 6, 7):
        print(f"[{k}] {titles[k]}: {len(hits[k])}")
        for h in hits[k]:
            print(f"      {h.replace(ROOT + '/', '')}")
        bad += len(hits[k])
    print()
    print("CLEAN" if bad == 0 else f"{bad} issue(s) found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
