"""RL paper v2: experimental-completeness gate. Every rule is a statement about files on disk.
  R1  every (task, arm) cell that make_tables.py reads from results_snapshot.json is present, with
      at least the expected seeds (5 for the calibrated-filter and BC-safe families on the fifteen
      tasks, 3 elsewhere; SEED_CAP respected).
  R2  every (task, algo) cell read from osrl_results.json (CDT/CPQ/COptiDICE and the composability
      grids) is present with 3 seeds.
  R3  provenance on the six regenerated tasks (+ pointbutton1's BC arms): every eval_results file
      behind a snapshot cell used by a table, and every OSRL run behind an osrl cell, is newer than
      the F0 install of the 300/512 ensembles (2026-09-14 21:50), so no old-cohort result survives.
  R4  the F2 job list is fully done (done markers, minus the identical-selection SKIPPED ones).
  R6  the F0b variant value ensembles (7 variants x 3 seeds x nine analysis tasks) postdate F0.
  R7  evaluations read DIRECTLY by an artifact builder (not via the snapshot) postdate their policy.
  R8  every non-table prose number is bound to the expression that produces it (verify_constants).
  R9  every generated table postdates the artifacts its writer reads.
  R10 every appendix section is referenced from the main text; no dangling refs.
  R11 derived artifacts regenerate byte-identically from their builders.
  R12 each table caption declares the markings that table uses.
  R13 every claim in CLAIMS.md is ESTABLISHED or RESOLVED.
  R14 every figure postdates the artifacts its builder reads.
  R15 a number in a sentence that names a table is a cell of that table.
  R16 a number in a sentence that names a figure appears in that figure's data.
  R17 the changes promised in the reviewer response letter still hold.
  R18 no prose number is a value only an archived snapshot still holds.
  R5  the prose-number audit (A1) is clean.
Exit 1 on any failure. Usage: python scripts/completeness_check.py [--no-refresh]"""
import glob, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from audit_claims import prose_lines as _prose_lines
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W = os.path.dirname(BASE)
REPO = f"{W}/vlm-with-cpl/new_data"; PY = sys.executable; fails = []
F0 = time.mktime(time.strptime("2026-09-14 21:50", "%Y-%m-%d %H:%M"))
SIX = ("halfcheetah_velocity", "cargoal1_dsrl", "walker2d_velocity", "ant_velocity", "hopper_velocity", "swimmer_velocity")
def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok: fails.append(msg)

if "--no-refresh" not in sys.argv:
    subprocess.run([PY, f"{BASE}/scripts/collect_results.py"], check=True, capture_output=True)
    subprocess.run([PY, f"{BASE}/scripts/harvest_osrl.py"], check=True, capture_output=True)

# ---- replay make_tables.py with logging proxies around the two result stores
reads_snap, reads_osrl = {}, {}
class _Arm(dict):
    def __init__(self, task, cfg, d): super().__init__(d); self.task, self.cfg = task, cfg
class _Task(dict):
    def __init__(self, task, d, log): super().__init__(d); self.task, self.log = task, log
    def get(self, cfg, default=None):
        v = super().get(cfg); self.log.setdefault((self.task, cfg), len(v) if v else 0); return v if v is not None else default
    def __getitem__(self, cfg): return self.get(cfg, {})
    def __contains__(self, cfg): return super().__contains__(cfg)
class _Store(dict):
    def __init__(self, d, log): super().__init__({t: _Task(t, v, log) for t, v in d.items()}); self.log = log
    def get(self, task, default=None): return super().get(task, _Task(task, {}, self.log) if default is None else default)
    def __getitem__(self, task): return self.get(task)
src = open(f"{BASE}/scripts/make_tables.py").read()
src = src.replace("    SNAP = json.load(f)", "    SNAP = _Store(json.load(f), reads_snap)")
src = src.replace('_OSRL2 = _j.load(open(os.path.join(BASE, "data", "osrl_results.json")))', '_OSRL2 = _Store(_j.load(open(os.path.join(BASE, "data", "osrl_results.json"))), reads_osrl)')
src = src.replace("    OSRL = json.load(f)", "    OSRL = _Store(json.load(f), reads_osrl)").replace("    OSRL2 = json.load(f)", "    OSRL2 = _Store(json.load(f), reads_osrl)")
assert src.count("_Store(") == 4, "make_tables.py store loads changed; update the replay"
ns = {"_Store": _Store, "reads_snap": reads_snap, "reads_osrl": reads_osrl, "__name__": "__replay__"}
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "make_tables.py", "exec"), ns)

# ---- R1 / R2
print("R1 snapshot cells read by the tables")
FIVE = ("calfilt_csf", "calfilt_lttR50", "bcsafe", "bcsafeseg", "vfilt_calsafe", "vfilt_matchgt", "vfilt_q25", "calfilt_pref", "vfilt_random", "vfilt_return")
for (task, cfg), n in sorted(reads_snap.items()):
    need = 1 if cfg == "bc_all" else (5 if cfg in FIVE else 3)
    check(n >= need, f"{task:22s} {cfg:26s} seeds {n} (need {need})")
print("R2 osrl cells read by the tables")
for (task, algo), n in sorted(reads_osrl.items()):
    check(n >= 3, f"{task:22s} {algo:26s} seeds {n} (need 3)")

# ---- R3 provenance on the regenerated cohort: only arms that consume the pessimistic V ensembles
# (or selections made from them) must postdate the F0 install; V-independent arms (BC on labels,
# CPL on GT pairs, oracle-labeled AWR, the T3.2 V-preference AWR sweep, return filters, OSRL on
# full data) keep their original vintage, whose protocol was verified separately (B4, R5).
V_DEP = re.compile(r"^(calfilt_|vfilt_(?!retbot)|xagg_|xlab_learned|xlab_k|xlab_rank|xlab_binary|xlab_exp|vawr_from_bcsafe|cf_v|cdt_(cert|a25|a40)|cpl_gt_(cert|a25|a40)|bc_a40|wbc|toph|mwbc|bc_cplgt)")
# Old-cohort provenance keys. _a25q80 is deliberately NOT here: F2 reran the alpha=.25 CPL grid
# under the same arm name, so those files are regenerated and R3's freshness test is the right
# judge for them; matching on the name alone would fail a cell that was in fact rerun.
OLD_NAMES = re.compile(r"(_draw[23]$|selq\d+$|_a40d[123]$|_a25q55$)")
sys.path.insert(0, f"{W}/runs/scripts"); from v2_aliases import aliases; ALIAS = aliases()   # deployed keys that alias an identical grid cell
print("R3 provenance (V-dependent arms of the regenerated cohort newer than the F0 install)")
sys.path.insert(0, f"{BASE}/scripts"); import importlib.util
spec = importlib.util.spec_from_file_location("cr", f"{BASE}/scripts/collect_results.py"); cr = importlib.util.module_from_spec(spec)
csrc = open(f"{BASE}/scripts/collect_results.py").read(); cns = {}; exec(compile(csrc.split("\ndef main():")[0], "collect_results.py", "exec"), cns); PATTERNS = cns["PATTERNS"]
for (task, cfg), n in sorted(reads_snap.items()):
    if task not in SIX + ("pointbutton1",) or n == 0: continue
    if task == "pointbutton1" and cfg not in ("bc_all", "bcsafeseg"): continue
    if not V_DEP.match(cfg): continue
    if OLD_NAMES.search(cfg): check(False, f"{task:22s} {cfg:26s} OLD-COHORT KEY still read by a table (F4 rewrite pending)"); continue
    src_cfg = ALIAS.get(task, {}).get(cfg, cfg)   # an aliased deployed key is served by its grid cell's files
    pat = PATTERNS.get(src_cfg)
    if pat is None: check(False, f"{task} {cfg}: no collect_results pattern"); continue
    files = [f for f in glob.glob(f"{REPO}/outputs/{task}/eval_results_*.json") if re.match(pat, os.path.basename(f)[len("eval_results_"):-5])]
    old = [os.path.basename(f) for f in files if os.path.getmtime(f) < F0]
    check(files and not old, f"{task:22s} {cfg:26s} {len(files)} files, stale: {old[:3]}")
mt = json.load(open(f"{BASE}/data/osrl_results_mtimes.json"))
for (task, algo), n in sorted(reads_osrl.items()):
    if task not in SIX or n == 0 or not V_DEP.match(algo): continue
    if OLD_NAMES.search(algo): check(False, f"{task:22s} {algo:26s} OLD-COHORT KEY still read by a table (F4 rewrite pending)"); continue
    src_algo = ALIAS.get(task, {}).get(algo, algo)
    old = [sd for sd, t in mt.get(task, {}).get(src_algo, {}).items() if t < F0]
    if src_algo != algo and src_algo not in mt.get(task, {}): old = ["grid cell not yet harvested"]
    check(not old, f"{task:22s} {algo:26s} stale seeds: {old}")

# ---- R4 F2 job list
print("R4 F2 job lists")
L = f"{W}/runs/logs/v2f2"
if os.path.exists(f"{L}/jobs_cdt.txt"):
    todo = [l.split() for l in open(f"{L}/jobs_cdt.txt")]; miss = [f"cdt_{h5}_s{s}" for t, h5, s in todo if not os.path.exists(f"{L}/done_cdt_{h5}_s{s}")]
    check(not miss, f"CDT cells: {len(todo) - len(miss)}/{len(todo)} done, missing {len(miss)}")
    todo = [l.split() for l in open(f"{L}/jobs_bc.txt")]; miss = [k for k in todo if not os.path.exists(f"{L}/done_{k[1]}_{k[3]}_seed{k[4]}")]
    check(not miss, f"BC/CPL cells: {len(todo) - len(miss)}/{len(todo)} done, missing {len(miss)}")
# ---- R6 variant value ensembles (F0b): robustness_stats.py reads noise05/10/20/30, n100, n300 and
# boltz ensembles x 3 seeds on the nine analysis tasks; every one must postdate the F0 install
# (the 09-14 driver silently kept 103 old-protocol files on five tasks; found 2026-09-18).
print("R6 variant ensembles of the nine analysis tasks newer than the F0 install")
NINE = SIX + ("cargoal2", "pointgoal1_dsrl", "pointgoal2")
for task in NINE:
    old, missing = [], []
    for v in ("noise05", "noise10", "noise20", "noise30", "n100", "n300", "boltz"):
        for sd in range(3):
            f = f"{REPO}/outputs/{task}/v_ensemble_{v}_seed{sd}.pt"
            if not os.path.exists(f): missing.append(f"{v}_s{sd}")
            elif os.path.getmtime(f) < F0: old.append(f"{v}_s{sd}")
    check(not old and not missing, f"{task:22s} 21 variants, stale: {len(old)} {old[:3]}, missing: {len(missing)} {missing[:3]}")
# ---- R7 DIRECTLY-READ EVALUATIONS (added 2026-09-23 after the F3e hole).
# R3 can only see numbers that flow through results_snapshot.json / osrl_results.json. An
# artifact builder that opens eval_results_*.json itself is invisible to it -- which is how the
# 2000-episode policy-certificate cells (certn2k_*, deliberately named so the published-arm globs
# cannot match them) went five weeks describing policies that F0 had retrained. This rule checks
# those families directly: on the six regenerated tasks, a directly-read evaluation must postdate
# the policy checkpoint it evaluates.
# Registry: eval-tag template -> the policy checkpoint template it evaluates. Adding a builder
# that reads eval files directly means adding it here.
print("R7 directly-read evaluations postdate the policies they evaluate")
DIRECT = [
    ("eval_results_certn2k_calfilt_csf_seed{s}.json", "bc_calfilt_csf_seed{s}_policy.pt", range(5)),
    ("eval_results_certn2k_gated_seed{s}.json", "bc_calfilt_lttR50_seed{s}_policy.pt", range(5)),
]
for task in SIX + ("cargoal2", "pointgoal1_dsrl", "pointgoal2"):
    bad, missing = [], []
    for ev_t, pol_t, seeds in DIRECT:
        for sd in seeds:
            ev = f"{REPO}/outputs/{task}/" + ev_t.format(s=sd)
            pol = f"{REPO}/outputs/{task}/" + pol_t.format(s=sd)
            if not os.path.exists(pol):
                continue                      # this cell has no such policy: nothing to evaluate
            if "gated" in ev_t:
                # the gated policy is only REPORTED where that seed certifies; a leftover gated
                # evaluation on a seed that no longer certifies feeds nothing the paper shows
                mp = f"{REPO}/outputs/{task}/calfilt_meta_calfilt_csf_seed{sd}.json"
                if not (os.path.exists(mp) and json.load(open(mp)).get("certified")):
                    continue
            if not os.path.exists(ev):
                if "gated" not in ev_t:
                    missing.append(f"s{sd}:{ev_t.split('_seed')[0][len('eval_results_'):]}")
                continue
            if os.path.getmtime(ev) < os.path.getmtime(pol):
                bad.append(f"s{sd}:{ev_t.split('_seed')[0][len('eval_results_'):]}")
    check(not bad and not missing,
          f"{task:22s} stale: {len(bad)} {bad[:3]}, missing: {len(missing)} {missing[:3]}")

# ---- R15 TABLE-SCOPED PROSE (the LLM loop's stricter rule). A sentence that names a table must
# quote a cell of THAT table. R5 checks a literal against every value of the task the sentence
# names, which passes a number that is right for the task but wrong for the table being pointed at.
print("R15 numbers in a sentence that names a table are cells of that table")
_tex15 = open(f"{BASE}/paper.tex").read()
_TABFILE = {}
for _m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", _tex15, re.S):
    _lab = re.search(r"\\label\{(tab:[^}]+)\}", _m.group(1))
    _ti = re.findall(r"\\tabinput\{data/tables/([^}]+)\}", _m.group(1))
    if _lab and _ti:
        _TABFILE[_lab.group(1)] = _ti


def _cells_of(lab):
    out = set()
    for fn in _TABFILE.get(lab, []):
        fp = f"{BASE}/data/tables/{fn}"
        if not os.path.exists(fp):
            continue
        for tok in re.findall(r"-?\d+\.?\d*", re.sub(r"\\[a-zA-Z]+|\{|\}", " ", open(fp).read())):
            try:
                v = float(tok)
            except ValueError:
                continue
            out.add(round(v, 4))
    return out


_PROTO15 = re.compile(r"seeds?|draws?|episodes?|budget|tasks?|percent of|Table~|Figure~")
# Protocol constants are not table cells: alpha, delta, the discount, the clip, kappa.
# (literal, tables cited) -> the verify_constants check that binds it
_R15_DEFERRED = {
    ("0.43", "tab:certtriple"),   # "the plug-in averages 0.04 against a realized 0.43": a mean over the table
    ("11.4", "tab:operator"), ("19.6", "tab:operator"),   # CDT on the ORIGINAL CarRun selection
    ("7.0", "tab:operator"), ("11.1", "tab:operator"),    # CDT on the regenerated one
}
_lab_key = lambda labs: sorted(labs)[0] if len(labs) == 1 else ",".join(sorted(labs))
_CONST15 = {0.25, 0.4, 0.1, 0.05, 2.0, 3.0, 1.0, 0.99, 0.95, 0.5, 0.85, 0.3, 0.2}
_miss = []
for _ln, _line in _prose_lines(_tex15):
    # a sentence may point at SEVERAL tables; the union is what it is entitled to quote
    for _sent in re.split(r"(?<=[a-z0-9\)])\.\s", _line):
        labs = set(re.findall(r"\\ref\{(tab:[^}]+)\}", _sent))
        if not labs:
            continue
        cells = set()
        for _l in labs:
            cells |= _cells_of(_l)
        if not cells:
            continue
        for _lit in re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", _sent):
            x = float(_lit)
            if x in _CONST15:
                continue
            k = len(_lit.split(".")[1])
            # prose may quote magnitudes where the table stores signed cells
            # ("Spearman -0.55 to -0.93" against a column of negatives)
            if any(abs(round(c, k) - x) < 1e-9 or abs(round(abs(c), k) - x) < 1e-9 for c in cells):
                continue
            # a sentence may cite a table to CONTRAST with it. These are bound by name in
            # verify_constants instead; the entry names the check that covers them.
            if any(f in _sent for f in ("so these differ from", "instead of", "rather than the")):
                continue
            # A sentence may legitimately cite a table while quoting an AGGREGATE over it or
            # another arm's values. Those are bound by name in verify_constants (R8); each entry
            # names the check, so this is a pointer to a stronger test, not an exemption.
            if (_lit, _lab_key(labs)) in _R15_DEFERRED:
                continue
            _miss.append(f"L{_ln} {_lit} not in {sorted(labs)}")
check(not _miss, f"{len(_TABFILE)} tables checked; literals not found in the table they cite: {_miss[:4]}")

# ---- R14 FIGURE FRESHNESS (no LLM counterpart; added 2026-09-23 after three stale figures).
# Same property as R9 but for figures, and it needs its own rule because figures are built by a
# DIFFERENT set of entry points: make_figures.py gates several behind FIG= environment variables,
# so a plain run rebuilds only the ungated ones, and pareto/landscape have their own scripts that
# no driver calls. alpha_curve.pdf and interpretability.pdf were rendering data F3 had already
# regenerated, and nothing noticed.
_tex = open(f"{BASE}/paper.tex").read()
print("R14 figures postdate the artifacts their builder reads")
_SELF = ("completeness_check.py", "verify_constants.py", "audit_prose_numbers.py",
         "audit_claims.py", "validate_snapshot.py")
_FIG_SRC = [p for p in glob.glob(f"{BASE}/scripts/*.py") if os.path.basename(p) not in _SELF]
_fig_targets = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _tex))
for u in sorted(_fig_targets):
    path = next((c for c in (f"{BASE}/{u}", f"{BASE}/{u}.pdf", f"{BASE}/{u}.png") if os.path.exists(c)), None)
    if path is None:
        check(False, f"{u:44s} file not found"); continue
    stem = os.path.splitext(os.path.basename(u))[0]
    # Scope the inputs to the FUNCTION that writes this figure. make_figures.py holds every
    # figure's builder in one file, so collecting the whole file's json paths attributes all of
    # them to each figure and produces false failures.
    reads, found = set(), False
    for w in _FIG_SRC:
        src = open(w, errors="ignore").read()
        if stem not in src:
            continue
        blocks = re.split(r"\ndef ", src)
        for b in blocks:
            if stem in b:
                found = True
                reads |= set(re.findall(r"[\"']((?:data/)?[\w/]+\.(?:json|npz))[\"']", b))
    if not found:
        check(True, f"{stem:30s} no builder in scripts/ (static asset)"); continue
    newest, src = 0.0, ""
    for r in reads:
        for cand in (f"{BASE}/{r}", f"{BASE}/data/{os.path.basename(r)}"):
            if os.path.exists(cand) and os.path.getmtime(cand) > newest:
                newest, src = os.path.getmtime(cand), os.path.basename(cand)
            if os.path.exists(cand):
                break
    check(not newest or os.path.getmtime(path) >= newest - 1,
          f"{stem:30s} figure {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(path)))}"
          + (f" vs {src} {time.strftime('%m-%d %H:%M', time.localtime(newest))}" if newest else " (no data inputs)"))

# ---- R16 FIGURE-SCOPED PROSE, the R15 rule for figures. A sentence naming a figure must quote
# values that appear in THAT figure's data artifacts. Figures have no cell file, so the artifact
# its builder reads is the referent.
print("R16 numbers in a sentence that names a figure appear in that figure's data")
_FIGLAB = {}
for _m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", _tex, re.S):
    _lab = re.search(r"\\label\{(fig:[^}]+)\}", _m.group(1))
    _gr = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _m.group(1))
    if _lab and _gr:
        _FIGLAB[_lab.group(1)] = [os.path.splitext(os.path.basename(g))[0] for g in _gr]


def _fig_values(lab):
    vals = set()
    for stem in _FIGLAB.get(lab, []):
        reads = set()
        for w in _FIG_SRC:
            src = open(w, errors="ignore").read()
            if stem not in src:
                continue
            for b in re.split(r"\ndef ", src):
                if stem in b:
                    reads |= set(re.findall(r"[\"']((?:data/)?[\w/]+\.json)[\"']", b))
        for r in reads:
            for cand in (f"{BASE}/{r}", f"{BASE}/data/{os.path.basename(r)}"):
                if os.path.exists(cand):
                    for tok in re.findall(r"-?\d+\.?\d*", open(cand, errors="ignore").read()):
                        try:
                            vals.add(round(float(tok), 4))
                        except ValueError:
                            pass
                    break
    return vals


# (literal, figure) -> bound in verify_constants under the named check
_R16_DEFERRED = {
    ("0.078", "fig:coverage"), ("0.067", "fig:coverage"), ("0.091", "fig:coverage"),  # worst cell + CP, 2000-draw
    ("0.06", "fig:alphacurve"), ("0.050", "fig:alphacurve"), ("0.071", "fig:alphacurve"),  # Bullet worst + CP
    # the operating curve quotes SEED MEANS; the artifact stores per-seed rates, so the mean is
    # not a stored token. Bound in verify_constants under "App bullet: the operating curve ...".
    ("0.87", "fig:alphacurve"), ("0.20", "fig:alphacurve"), ("0.995", "fig:alphacurve"),
}
_fmiss = []
for _ln, _line in _prose_lines(_tex):
    for _sent in re.split(r"(?<=[a-z0-9\)])\.\s", _line):
        labs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", _sent))
        if not labs:
            continue
        vals = set()
        for _l in labs:
            vals |= _fig_values(_l)
        if not vals:
            continue
        for _lit in re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", _sent):
            x = float(_lit)
            if x in _CONST15:
                continue
            k = len(_lit.split(".")[1])
            if any(abs(round(c, k) - x) < 1e-9 or abs(round(abs(c), k) - x) < 1e-9 for c in vals):
                continue
            # A figure is often cited as CONTEXT beside a number drawn from a sibling artifact
            # of the same family (the 2000-draw validation beside the 200-draw coverage figure).
            # Those are bound by name in verify_constants; the entry names the covering check.
            if (_lit, sorted(labs)[0]) in _R16_DEFERRED:
                continue
            _fmiss.append(f"L{_ln} {_lit} not in {sorted(labs)}")
check(not _fmiss, f"{len(_FIGLAB)} figures checked; literals not in the figure's data: {_fmiss[:4]}")

# ---- R17 REVIEW COMMITMENTS (added 2026-09-23, checklist item 17). The 38 review
# comments on the 2026-08-14 draft each carry a documented change in
# paper/comments/response-letter.md. The F4 prose pass rewrote several of the sections those
# comments anchor to, so the commitments are re-checked mechanically where they are decidable.
# It caught one real regression on its first run: "power" had gone back to bare use in the main
# text, defined only in the appendix proof.
print("R17 the changes promised in the response letter still hold")
out = subprocess.run([PY, f"{BASE}/scripts/verify_review_commitments.py"], capture_output=True, text=True)
check(out.returncode == 0, "verify_review_commitments exit 0" + ("" if out.returncode == 0 else
      f" ({[l.strip() for l in out.stdout.splitlines() if l.strip().startswith('FAIL')][:2]})"))

# ---- R13 CLAIMS LEDGER (LLM gate R6). Nothing enters the paper as a plain statement unless
# CLAIMS.md records it as ESTABLISHED or RESOLVED. A PROVISIONAL or PENDING row means the paper
# is making a claim its evidence does not yet carry.
print("R13 every claim in CLAIMS.md is ESTABLISHED or RESOLVED")
_cl = open(f"{BASE}/CLAIMS.md").read().splitlines()
_rows = [l for l in _cl if re.match(r"^\|\s*C\d+\s*\|", l)]
check(bool(_rows), f"CLAIMS.md has claim rows ({len(_rows)} found)")
_bad = []
for r in _rows:
    cid = re.match(r"^\|\s*(C\d+)", r).group(1)
    cells = [c.strip() for c in r.split("|")]
    status = cells[4] if len(cells) > 4 else ""
    if not re.match(r"^(ESTABLISHED|RESOLVED)", status):
        _bad.append(f"{cid}:{status.split(',')[0][:34]}")
check(not _bad, f"{len(_rows)} claims, not yet established: {_bad}")

# ---- R10 APPENDIX REACHABILITY (LLM gate R4). Every appendix section must be referenced from
# the MAIN text, so nothing in the appendix is unreachable from the paper proper.
print("R10 every appendix section is referenced from the main text")
_cut = re.search(r"\\appendix|\\section\*?\{Appendix", _tex).start()
_main_refs = set(re.findall(r"\\(?:ref|autoref|Cref|cref)\{([^}]+)\}", _tex[:_cut]))
_app_secs = re.findall(r"\\label\{(app:[^}]+)\}", _tex[_cut:])
_unreach = [l for l in _app_secs if l not in _main_refs]
check(not _unreach, f"{len(_app_secs)} appendix sections, unreferenced: {_unreach}")
_labels = set(re.findall(r"\\label\{([^}]+)\}", _tex))
_refs = set(re.findall(r"\\(?:ref|autoref|eqref|Cref|cref)\{([^}]+)\}", _tex))
check(not (_refs - _labels), f"dangling refs: {sorted(_refs - _labels)[:5]}")

# ---- R11 DERIVED ARTIFACTS REGENERATE (LLM gate R11). Re-run each cheap deterministic builder
# into a scratch copy and require byte-identical output. This is the decisive reproducibility
# test and the one that catches an artifact whose INPUT artifact moved -- the class that hid
# cert_rate_theory (never written by F3 at all, and run before the guarantee_stats it reads).
# SEMANTICS: the builder writes in place, so a FAIL means the stored artifact WAS stale and this
# run has just refreshed it -- re-read any prose quoting it, then the next run passes. A PASS
# therefore means "reproducible as of now", not "was never stale"; the FAIL is the signal.
print("R11 derived artifacts regenerate to the stored values")
# Fast, deterministic builders run every time. The three that need the trajectory pickles
# (operator_selection_alpha, control_selection_stats, e_cache_scores) run only under --full,
# because each loads several gigabytes; they are listed so the set is explicit rather than
# whatever happened to be cheap.
_REGEN = [("data/calibration_audit.json", f"{W}/runs/scripts/calibration_audit.py", []),
          ("data/review_response/selection_stability.json", f"{W}/runs/scripts/selection_stability.py", []),
          ("data/h_seed_failures.json", f"{W}/runs/scripts/build_h_seed_failures.py", []),
          ("data/cert_rate_theory.json", f"{BASE}/scripts/cert_rate_theory.py", ["--write"])]
_REGEN_FULL = [("data/operator_alpha_hat.json", f"{W}/runs/scripts/operator_selection_alpha.py", []),
               ("data/control_selection_stats.json", f"{W}/runs/scripts/control_selection_stats.py", [])]
if "--full" in sys.argv:
    _REGEN += _REGEN_FULL
else:
    print(f"  (skipping {len(_REGEN_FULL)} pickle-loading builders; --full includes them)")
for rel, builder, extra in _REGEN:
    cur = f"{BASE}/{rel}"
    if not (os.path.exists(cur) and os.path.exists(builder)):
        check(False, f"{os.path.basename(rel):32s} missing artifact or builder"); continue
    before = open(cur, "rb").read()
    r = subprocess.run([PY, builder] + extra, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{W}/vlm-with-cpl/new_data"})
    after = open(cur, "rb").read()
    check(r.returncode == 0 and before == after,
          f"{os.path.basename(rel):32s} regenerates identically"
          + ("" if before == after else " (CONTENT MOVED: its inputs changed since it was written)"))

# ---- R12 CAPTION MARKINGS (LLM gate R12). A table that bolds cells must say what bold means.
print("R12 each table caption declares the markings that table uses")
for m in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", _tex, re.S):
    blk = m.group(1)
    cap = re.search(r"\\caption\{(.*?)\}\s*\n", blk, re.S)
    ti = re.findall(r"\\tabinput\{data/tables/([^}]+)\}", blk)
    if not (cap and ti):
        continue
    body = "".join(open(f"{BASE}/data/tables/{t}", errors="ignore").read()
                   for t in ti if os.path.exists(f"{BASE}/data/tables/{t}"))
    if "\\textbf" in body:
        check("bold" in cap.group(1).lower(), f"{ti[0]:24s} bolds cells; caption explains bold")

# ---- R9 TABLE FRESHNESS (added 2026-09-23). A generated table must not predate the artifacts
# its writer reads. Four appendix tables (certrate_pred, meancost_cert, policy_cert,
# cardinal_ctrl) sat at their 2026-09-02 vintage because their writer, make_new_tables.py, is not
# called by make_tables or by any driver, so every regeneration pass silently skipped them; the
# published certification-rate table was showing Walker2d at 0.02 where the current cohort gives
# 0.50. Same shape as make_figures' FIG=-gated figures and cert_rate_theory's unpassed --write.
print("R9 generated tables postdate the artifacts their writer reads")
_TSRC = [p for p in glob.glob(f"{BASE}/scripts/*.py") if os.path.basename(p) not in _SELF]
for tex in sorted(glob.glob(f"{BASE}/data/tables/*.tex")):
    base = os.path.basename(tex)
    writers = [p for p in _TSRC if base in open(p, errors="ignore").read() and "open(" in open(p, errors="ignore").read()]
    if not writers:
        check(False, f"{base:24s} no writer found among scripts/*.py")
        continue
    reads = set()
    for w in writers:
        reads |= {m for m in re.findall(r"[\"']((?:data/)?[\w/]+\.json)[\"']", open(w, errors="ignore").read())}
    newest = 0.0
    for r in reads:
        for cand in (f"{BASE}/{r}", f"{BASE}/data/{os.path.basename(r)}"):
            if os.path.exists(cand):
                newest = max(newest, os.path.getmtime(cand)); break
    check(not newest or os.path.getmtime(tex) >= newest - 1,
          f"{base:24s} table {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(tex)))}"
          f" vs newest input {time.strftime('%m-%d %H:%M', time.localtime(newest)) if newest else 'n/a'}")

# ---- R8 SOURCE TRACE (added 2026-09-23). audit_prose_numbers (R5) is task-scoped EXISTENCE
# matching: a literal passes if it equals some value of the task its sentence names. That cannot
# catch a derived statistic -- a Spearman, an R^2, a slope, a count over a set, a range -- that was
# never recomputed when the cohort moved, which is the class that actually goes stale. R8 binds each
# non-table prose number to the one expression that produces it. It found four on its first run: a
# slope of -1.62 that had become -1.61, "nine of the twelve" that was eight, a Spearman printed at a
# precision that no longer rounded, and an R^2 of 0.50 that had become 0.49.
print("R8 source trace of every non-table prose number")
out = subprocess.run([PY, f"{BASE}/scripts/verify_constants.py"], capture_output=True, text=True)
check(out.returncode == 0, "verify_constants exit 0" + ("" if out.returncode == 0 else
      f" ({[l for l in out.stdout.splitlines() if l.strip().startswith('FAIL')][:2]})"))

# ---- R5
print("R5 prose-number audit")
out = subprocess.run([PY, f"{BASE}/scripts/audit_prose_numbers.py"], capture_output=True, text=True)
check(out.returncode == 0, "audit_prose_numbers exit 0" + ("" if out.returncode == 0 else f" (tail: {out.stdout.strip().splitlines()[-1][:120] if out.stdout.strip() else out.stderr[-200:]})"))
# ---- R18 STALE ARCHIVED VALUES (added 2026-09-23). A prose number must not be a value that
# only an ARCHIVED snapshot still holds. The paper said the ungated top-half sub-selection on
# Hopper reached mean cost 148, worst seed 487; those are results_snapshot_pre_cardinal's
# values for calfilt_lttR50. The arm was regenerated (current: 82 and 289) and the prose was
# not updated. Every existence-style audit passed it, because 148 and 487 do exist in the data
# layer -- in the archive. Only a current-vs-archive comparison sees it.
print("R18 prose numbers are not stale archived-snapshot values")
import statistics as _st
_NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
          "ant_velocity": "Ant", "hopper_velocity": "Hopper", "swimmer_velocity": "Swimmer",
          "cargoal1_dsrl": "CarGoal1", "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1",
          "pointgoal2": "PointGoal2", "pointbutton1": "PointButton1",
          "pointbutton2": "PointButton2", "carbutton1_t3": "CarButton1",
          "carbutton2": "CarButton2", "pointcircle1": "PointCircle1",
          "pointcircle2": "PointCircle2"}


def _agg(D):
    out = {}
    for t, arms in D.items():
        if not isinstance(arms, dict):
            continue
        for a, cells in arms.items():
            if not isinstance(cells, dict):
                continue
            for key in ("C", "R"):
                vals = [v[key] for v in cells.values() if isinstance(v, dict) and key in v]
                if len(vals) > 1:
                    out[(t, a, key, "mean")] = _st.mean(vals)
                    out[(t, a, key, "max")] = max(vals)
    return out


_prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ",
                open(f"{BASE}/paper.tex").read(), flags=re.S)
_prose = re.sub(r"(?<!\\)%.*", "", _prose)
_arcs = glob.glob(f"{BASE}/data/archive/results_snapshot*.json")
_cur = _agg(json.load(open(f"{BASE}/data/results_snapshot.json")))
# Collisions are unavoidable: the archive holds thousands of aggregates, so some archived
# value will equal an unrelated prose number near the same task name. Each entry below was
# traced by hand to the CURRENT quantity the prose actually means.
_R18_VERIFIED = {
    # prose 169 is ant_velocity/xlab_oracle_step C-mean (169.25), identical in both snapshots
    "halfcheetah_velocity/vfilt_return C-max",
    # prose 60 is hopper_velocity/vfilt_q25 C-mean (59.88) in the CURRENT snapshot
    "hopper_velocity/calfilt_ltt C-max",
    # prose 82 is hopper_velocity/calfilt_lttR50 C-mean (81.89) in the CURRENT snapshot
    "hopper_velocity/vawr_from_bcsafe C-mean",
    "hopper_velocity/xlab_rank R-mean",
}
_stale = set()
for _ap in _arcs:
    try:
        _A = _agg(json.load(open(_ap)))
    except Exception:
        continue
    for _k, _av in _A.items():
        _t, _a, _key, _how = _k
        _cv = _cur.get(_k)
        # only distinctive magnitudes: small integers collide with unrelated prose constantly
        if _cv is None or abs(_av) < 40 or abs(_av - _cv) < max(0.05, 0.002 * abs(_av)):
            continue
        _nm = _NAMES.get(_t)
        if not _nm:
            continue
        for _d in (0, 1):
            _lit, _cl = f"{_av:.{_d}f}", f"{_cv:.{_d}f}"
            for _m in re.finditer(rf"(?<![\d.]){re.escape(_lit)}(?![\d.])", _prose):
                _ctx = _prose[max(0, _m.start() - 320):_m.start() + 200]
                if _nm in _ctx and not re.search(rf"(?<![\d.]){re.escape(_cl)}(?![\d.])", _ctx):
                    if f"{_t}/{_a} {_key}-{_how}" not in _R18_VERIFIED:
                        _stale.add(f"{_t}/{_a} {_key}-{_how}: prose {_lit}, "
                                   f"archive {_av:.2f} -> current {_cv:.2f}")
                break
check(not _stale, f"{len(_cur)} current aggregates vs {len(_arcs)} archived snapshot(s); "
                  f"stale prose values: {sorted(_stale)}")

print(f"\nCOMPLETENESS: {'PASS' if not fails else f'{len(fails)} FAIL(s)'}")
sys.exit(1 if fails else 0)
