#!/usr/bin/env python3
"""Import the live research tree into this repository, rewriting machine paths.

The research was run from a working tree whose scripts address each other by absolute
path. That is fine for running experiments and useless to anyone else, so the repository
is produced from the working tree by this script rather than by hand: run it again after
any change in the working tree and the repository regenerates deterministically.

    python tools/import_from_worktree.py --worktree /path/to/workspace [--check]

--check reports what would change and exits non-zero if anything would, so drift between
the repository and the working tree is detectable before a release.

Path rewriting. Absolute paths under the working tree become roots resolved from the
environment, with defaults relative to the repository (located by the .csc-root marker):

    CSC_REPO       this repository
    CSC_WORKSPACE  the directory holding the working trees   (default: the repo's parent)
    CSC_WORK       datasets, checkpoints, method code        (default: $CSC_WORKSPACE/vlm-with-cpl/new_data)
    CSC_RUNS       campaign outputs: selections, logs        (default: $CSC_WORKSPACE/runs)
    CSC_OSRL       an OSRL checkout, for the full-label baselines
    CSC_PAPER      the archived evaluation records           (default: $CSC_REPO/paper)

Excluded from the repository, and why: scripts belonging to follow-on work this paper does
not report (a policy-level certificate, curriculum over certified selections, an
alpha-aware learner, certified curation for imitation), and an exploratory probe series on
readout choice, coverage, DRO and stratification whose results the paper does not report
either. The registered form of the readout question that the paper does report is in
analysis/, not in those probes. EXCLUDE lists every one by name.
"""
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

EXCLUDE_FOLLOWUP = {
    "concentrability_probe.py", "covered_comparator_probe.py", "curriculum_watch.sh",
    "probe_alpha_aware_bc.py", "probe_curriculum.py", "probe_il_certstats.py",
    "run_aa_eval.sh", "run_curriculum_eval.sh", "run_hoppermedq_probe.sh",
    "run_hoppermix_probe.sh", "run_il_clone_eval.sh",
}
EXCLUDE_UNREPORTED = {
    "probe_aggregation_coverage.py", "probe_consensus_theory.py", "probe_dro_bc.py",
    "probe_labelfree_readout.py", "probe_pref_sampling.py", "probe_readout_certification.py",
    "probe_readout_instruments.py", "probe_readout_search.py", "probe_stratified_certify.py",
    "probe_stratified_policy.py", "probe_trim_sweep.py", "run_dro.sh", "run_dro_sweep.sh",
    "run_strat_bc.sh", "run_trim_sweep.sh", "stratified_selection_probe.py",
}
# authoring infrastructure, not science: it writes into a private Overleaf tree
EXCLUDE_AUTHORING = {"sync_overleaf.sh"}
EXCLUDE = EXCLUDE_FOLLOWUP | EXCLUDE_UNREPORTED | EXCLUDE_AUTHORING

TOOLING = re.compile(r"^(verify_|lint_|smoke_|check_|audit_staleness|find_a40_missing)")
DRIVER = re.compile(r"^(stage|chain_|run_|v2_|a3_chain|cpl_retry|retry_|mon_|regen_)")
SKIP_SUFFIX = (".bak", ".pyc", "~")

ROOTS = [                      # longest prefix first
    ("vlm-with-cpl/new_data", "CSC_WORK"),
    ("iclr2027/data",         "CSC_PAPER_DATA"),
    ("iclr2027",              "CSC_PAPER"),
    ("certified-safety-curation", "CSC_REPO"),
    ("runs",                  "CSC_RUNS"),
    ("osrl",                  "CSC_OSRL"),
]

PY_PREAMBLE = '''
# --- paths ------------------------------------------------------------------
# The research tree addressed itself by absolute path; these roots replace it. Set
# CSC_WORKSPACE (or the individual roots) to point at your own trees. See the README.
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
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "vlm-with-cpl", "new_data"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs are carried by the repository, so they resolve without a working tree
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------
'''

SH_PREAMBLE = '''# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/vlm-with-cpl/new_data}"
CSC_RUNS="${CSC_RUNS:-$([ -d "$CSC_WORKSPACE/runs" ] && printf %s "$CSC_WORKSPACE/runs" || printf %s "$CSC_REPO/runs")}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
CSC_CONFIG="${CSC_CONFIG:-$CSC_REPO/configs}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------
'''


def rewrite(text, kind, worktree):
    ws = worktree.rstrip("/")
    used = set()

    def sub_one(rest):
        for sub, var in ROOTS:
            if rest == "/" + sub or rest.startswith("/" + sub + "/"):
                used.add(var)
                return var, rest[len(sub) + 1:]
        used.add("CSC_WORKSPACE")
        return "CSC_WORKSPACE", rest

    if kind == "py":
        def py_lit(m):
            var, tail = sub_one(m.group("rest") or "")
            return var if not tail else '%s + "%s"' % (var, tail)
        text = re.sub(r'"' + re.escape(ws) + r'(?P<rest>[^"]*)"', py_lit, text)
        text = re.sub(r"'" + re.escape(ws) + r"(?P<rest>[^']*)'", py_lit, text)

        def py_fstr(m):
            var, tail = sub_one(m.group("rest") or "")
            return "{%s}%s" % (var, tail)
        text = re.sub(re.escape(ws) + r"(?P<rest>[^\"'{}]*)", py_fstr, text)
    else:
        def sh_lit(m):
            var, tail = sub_one(m.group("rest") or "")
            return "${%s}%s" % (var, tail)
        text = re.sub(re.escape(ws) + r"(?P<rest>[^\s\"';:)]*)", sh_lit, text)

    # the run configs ship with the repository
    text = text.replace('CSC_WORK + "/config', 'CSC_CONFIG + "/config')
    text = text.replace("{CSC_WORK}/config", "{CSC_CONFIG}/config")
    text = text.replace("${CSC_WORK}/config", "${CSC_CONFIG}/config")
    if "CSC_CONFIG" in text:
        used.add("CSC_CONFIG")

    text = re.sub(r"/home/[A-Za-z0-9_.-]+/(?:mini|ana)conda3/envs/[A-Za-z0-9_.-]+/bin/python",
                  "${PYTHON}" if kind == "sh" else "python", text)
    text = re.sub(r"/home/[A-Za-z0-9_.-]+", "${HOME}" if kind == "sh" else "~", text)
    return text, used


def preamble_for(text, kind, used):
    if not used:
        return text
    lines = text.split("\n")
    if kind == "sh":
        i = 1 if lines and lines[0].startswith("#!") else 0
        while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
            i += 1
        return "\n".join(lines[:i] + [SH_PREAMBLE] + lines[i:])
    i = 1 if lines and lines[0].startswith("#!") else 0
    if i < len(lines) and re.match(r'\s*[ru]?"""', lines[i]):            # module docstring
        j = i + 1
        if not re.match(r'\s*[ru]?""".*"""', lines[i]):
            while j < len(lines) and '"""' not in lines[j]:
                j += 1
            j += 1
        i = j
    while i < len(lines):                     # a __future__ import must stay first
        st = lines[i].lstrip()
        if st.startswith("from __future__"):
            i += 1
            break
        if st and not st.startswith("#"):
            break
        i += 1
    return "\n".join(lines[:i] + [PY_PREAMBLE] + lines[i:])


def kind_of(path):
    if path.endswith(".py"):
        return "py"
    if path.endswith(".sh"):
        return "sh"
    if path.endswith((".yaml", ".yml")):
        return "yaml"
    return None


def convert(raw, kind, ws, path=""):
    if path.endswith(".log"):
        return raw.replace(ws, "${CSC_WORKSPACE}")
    if kind == "yaml":
        # dataset paths under somebody's home directory: the DSRL cache is ~/.dsrl
        out = re.sub(r"/home/[A-Za-z0-9_.-]+/\.dsrl", "~/.dsrl", raw)
        return out.replace(ws, "${CSC_WORKSPACE}")
    if kind and (ws in raw or "/home/" in raw):
        out, used = rewrite(raw, kind, ws)
        return preamble_for(out, kind, used)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True, help="the workspace the research ran from")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ws = os.path.abspath(a.worktree)
    plan = []

    def usable(name):
        return not name.startswith(".") and not name.endswith(SKIP_SUFFIX)

    # campaign drivers, producers and tooling
    for name in sorted(os.listdir(os.path.join(ws, "runs", "scripts"))):
        src = os.path.join(ws, "runs", "scripts", name)
        if not os.path.isfile(src) or name in EXCLUDE or not usable(name):
            continue
        d = "experiments" if DRIVER.match(name) else "tools" if TOOLING.match(name) else "analysis"
        plan.append((src, os.path.join(REPO, d, name)))

    # the paper's table, figure and audit scripts
    for name in sorted(os.listdir(os.path.join(ws, "iclr2027", "scripts"))):
        src = os.path.join(ws, "iclr2027", "scripts", name)
        if os.path.isfile(src) and usable(name):
            plan.append((src, os.path.join(REPO, "paper", "scripts", name)))

    # the paper source: the audit gates read it (the orphan guard checks that every
    # value a check claims still appears in the text at the precision it states)
    for name in ("paper.tex", "references.bib"):
        src = os.path.join(ws, "iclr2027", name)
        if os.path.exists(src):
            plan.append((src, os.path.join(REPO, "paper", name)))

    # the method. A working-tree stage enters the repository only where the repository
    # already carries it: the working tree also holds stages for other projects (D4RL
    # mixtures, the VLM query path) that this paper does not use.
    md = os.path.join(ws, "vlm-with-cpl", "new_data")
    for name in sorted(os.listdir(os.path.join(md, "scripts"))):
        src = os.path.join(md, "scripts", name)
        if not os.path.isfile(src) or not usable(name):
            continue
        for d in ("pipeline", "analysis", "baselines", "legacy"):
            if os.path.exists(os.path.join(REPO, d, name)):
                plan.append((src, os.path.join(REPO, d, name)))
                break
    for sub, dest in (("model", "src/model"), ("utils", "src/utils")):
        for name in sorted(os.listdir(os.path.join(md, sub))):
            src = os.path.join(md, sub, name)
            if os.path.isfile(src) and usable(name):
                plan.append((src, os.path.join(REPO, dest, name)))
    for name in sorted(os.listdir(md)):
        if re.fullmatch(r"config(_h\d+)?\.yaml", name):
            plan.append((os.path.join(md, name), os.path.join(REPO, "configs", name)))
    plan.append((os.path.join(md, "requirements.txt"), os.path.join(REPO, "requirements.txt")))

    # Small run-level records the paper scripts read by path. The bulk of runs/ is raw
    # training output and stays out; these are the summaries the tables depend on.
    for rel in ("selections/v2_summary.json",):
        src = os.path.join(ws, "runs", rel)
        if os.path.exists(src):
            plan.append((src, os.path.join(REPO, "runs", rel)))
    # the 2000-episode evaluation logs behind the deployment certificate: the appendix
    # tables are derived from them, so they are evidence rather than raw training output
    for sub in ("logs/policycert", "logs/policycert_v2"):
        d = os.path.join(ws, "runs", sub)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                f = os.path.join(d, name)
                if os.path.isfile(f):
                    plan.append((f, os.path.join(REPO, "runs", sub, name)))

    changed = []
    for src, dst in plan:
        raw = open(src, encoding="utf-8", errors="surrogateescape").read()
        out = convert(raw, kind_of(src), ws, src)
        old = open(dst, encoding="utf-8", errors="surrogateescape").read() if os.path.exists(dst) else None
        if old != out:
            changed.append(os.path.relpath(dst, REPO))
            if not a.check:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w", encoding="utf-8", errors="surrogateescape") as fh:
                    fh.write(out)
                if os.access(src, os.X_OK):
                    os.chmod(dst, 0o755)

    # Sweep: files written directly in the repository carry machine paths too. The
    # rewrite is idempotent, so this converges and then reports nothing.
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in files:
            f = os.path.join(root, name)
            if kind_of(f) is None or os.path.abspath(f) == os.path.abspath(__file__):
                continue
            raw = open(f, encoding="utf-8", errors="surrogateescape").read()
            if ws not in raw and "/home/" not in raw:
                continue
            out = convert(raw, kind_of(f), ws)
            if out != raw:
                changed.append(os.path.relpath(f, REPO))
                if not a.check:
                    with open(f, "w", encoding="utf-8", errors="surrogateescape") as fh:
                        fh.write(out)

    verb = "would change" if a.check else "imported"
    print("%s: %d file(s) %s of %d considered" % (os.path.basename(__file__), len(changed), verb, len(plan)))
    for c in changed[:15]:
        print("   ", c)
    if len(changed) > 15:
        print("    ... and %d more" % (len(changed) - 15))
    if a.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
