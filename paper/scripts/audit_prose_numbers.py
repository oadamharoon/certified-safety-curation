"""A1 (V2_REMEDIATION): task-scoped agreement and freshness audit of every number in the prose.

audit_claims.py asks whether a prose literal exists ANYWHERE in the data layer. That is the
check that certified v1 and missed twelve stale numbers: a Swimmer value that had moved still
matched some other task's cell, so it never surfaced. This audit is stricter on three axes.

  1. TASK SCOPE. A literal in a sentence that names a task must be supported by that task's
     own values (snapshot cells and seed means, OSRL harvest, every data/*.json subtree keyed
     by the task, every generated-table row that starts with the task) at the precision the
     prose states. A sentence naming several tasks is checked against their union; a
     sentence naming none, against everything (the old check).
  2. AGREEMENT. Literals are bucketed by (task, arm keyword, statistic keyword) taken from the
     sentence; a bucket holding more than one distinct value is a candidate disagreement
     (Table 1 says 5, App H says 4). Buckets are printed for review, not judged.
  3. FRESHNESS OF QUOTED ARTIFACTS. Every artifact that supports a task-scoped prose literal
     is checked with check_artifact_recency's rule (artifact mtime vs the newest checkpoint
     of a family its generator loads, for the task the sentence names). The recency pass of
     2026-09-01 covered artifacts read by make_tables only; mt_procedures.json was quoted by
     hand and stale.
  4. COUNT PHRASES ("four of the five", "three of fifteen") are listed with computed counts
     from the tables beside them.

Exit status 1 when any task-scoped orphan or any stale quoted artifact remains: that is the
v2 gate. Usage: python scripts/audit_prose_numbers.py [--verbose]
"""
import glob, json, os, re, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "runs", "scripts"))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import check_artifact_recency as rec          # noqa: E402
from audit_claims import prose_lines            # noqa: E402

NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
         "ant_velocity": "Ant", "hopper_velocity": "Hopper", "swimmer_velocity": "Swimmer",
         "cargoal1_dsrl": "CarGoal1", "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1",
         "pointgoal2": "PointGoal2", "pointbutton1": "PointButton1", "pointbutton2": "PointButton2",
         "carbutton1_t3": "CarButton1", "carbutton2": "CarButton2", "pointcircle1": "PointCircle1",
         "pointcircle2": "PointCircle2", "ballrun_b": "BallRun", "ballcircle_b": "BallCircle",
         "carcircle_b": "CarCircle", "carrun_b": "CarRun", "dronerun_b": "DroneRun"}
ARMS = ["BC-All", "BC-Safe-Seg", "BC-Safe", "calibrated", "certified", "V-filter", "CDT", "CPL",
        "CPQ", "COptiDICE", "random", "return", "oracle", "clone", "gated", "labels-only",
        "held-out", "fallback", "top-quartile", "matched", "Ours", "our"]
STATS = ["certification rate", "false-cert", "precision", "contamination", "purity", "unsafe",
         "cost", "reward", "return", "margin", "rate", "AUC", "correlation", "labels", "kept"]
CONST = {0.05, 0.1, 0.25, 0.4, 0.5, 0.9, 0.95, 1.0, 2.0, 3.0, 0.85, 0.8, 0.3, 0.99}
PROTO = re.compile(r"draws?|pairs?|episodes?|transitions?|horizon|epochs?|batch|width|length|"
                   r"seeds?|trajectories|Adam|\\times|10\^|percent of the pool|quantile")


def _nums(o, acc):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        acc.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            _nums(v, acc)
    elif isinstance(o, list):
        for v in o:
            _nums(v, acc)


def _means(o, acc):
    """Per-arm means over seed dicts, the quantity prose actually quotes."""
    if isinstance(o, dict):
        kids = [v for v in o.values() if isinstance(v, dict)]
        if len(kids) > 1:
            keys = set(kids[0])
            for k in kids[1:]:
                keys &= set(k)
            for key in keys:
                xs = [k[key] for k in kids if isinstance(k.get(key), (int, float))
                      and not isinstance(k.get(key), bool)]
                if len(xs) == len(kids):
                    acc.append(sum(xs) / len(xs))
        for v in o.values():
            _means(v, acc)
    elif isinstance(o, list):
        for v in o:
            _means(v, acc)


def task_values():
    """task -> {artifact: [values]}; '' holds task-free artifacts."""
    tv = {t: {} for t in NAMES}
    tv[""] = {}
    files = [p for p in glob.glob(os.path.join(BASE, "data", "**", "*.json"), recursive=True)
             if "/archive/" not in p and "/_superseded/" not in p]
    for p in files:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rel = os.path.relpath(p, BASE)
        found = False

        def walk(o):
            nonlocal found
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in tv and k:
                        acc = []
                        _nums(v, acc); _means(v, acc)
                        tv[k].setdefault(rel, []).extend(acc); found = True
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
        acc = []
        _nums(d, acc); _means(d, acc)
        tv[""].setdefault(rel, []).extend(acc)
        if not found:
            # per-task lists keyed by a "task" field
            def walk2(o):
                if isinstance(o, dict):
                    t = o.get("task")
                    if t in tv and t:
                        acc = []; _nums(o, acc); tv[t].setdefault(rel, []).extend(acc)
                    for v in o.values():
                        walk2(v)
                elif isinstance(o, list):
                    for v in o:
                        walk2(v)
            walk2(d)
    tdir = os.path.join(BASE, "data", "tables")
    for fn in sorted(os.listdir(tdir)):
        fp = os.path.join(tdir, fn)
        if not os.path.isfile(fp):
            continue
        rel = "data/tables/" + fn
        for line in open(fp):
            ns = [float(x) for x in re.findall(r"-?\d+\.?\d*", re.sub(r"\\[a-zA-Z]+", " ", line))]
            if not ns:
                continue
            head = line.split("&")[0].strip()
            hit = [t for t, n in NAMES.items() if head.startswith(n)]
            for t in hit:
                tv[t].setdefault(rel, []).extend(ns)
            tv[""].setdefault(rel, []).extend(ns)
    return tv


_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
          "fifteen|twenty|all|none")
_COUNT_RE = re.compile(rf"\b((?:{_WORDS})|\d+) of (?:the |its |our |all )?((?:{_WORDS})|\d+)\b")


# Literals this heuristic cannot support but that ARE bound, by name, in
# scripts/verify_constants.py (gate rule R8). Each entry names the check that covers it, so an
# entry is a pointer to a stronger test rather than an exemption. The pattern is always the same:
# a range computed over several tasks, attributed by the task-scoping rule to the one task its
# sentence happens to name.
DEFERRED = [
    ("39.9", "kept-set mean trajectory cost",
     "App extended: HalfCheetah's bottom-return kept set has mean trajectory cost 39.9"),
    ("32", "Random selection at the identical matched fractions",
     "App bullet: random selection is unsafe on four of five Bullet tasks, costs 32 to 79"),
    ("79", "Random selection at the identical matched fractions",
     "App bullet: random selection is unsafe on four of five Bullet tasks, costs 32 to 79"),
    ("65", "Cloning everything is unsafe on four of five",
     "App bullet: cloning everything is unsafe on four of five, costs 24 to 65"),
    # the Boltzmann error range spans all nine tasks; 5.3 is Hopper's, and the sentence names
    # Swimmer only as the task with the worst PRECISION drop
    ("5.3", "Boltzmann labeler with temperature",
     "App extended: its realized error rates are 5.3 to 14.6 percent"),
]


def deferred_to_r8(lit, sent):
    return next((c for l, frag, c in DEFERRED if l == lit and frag in sent), None)


def count_phrase_numerals(sent):
    """The N and M of every 'N of M' in this sentence.

    Section 4 already lists these beside the computed counts, so reporting them again as
    task-scoped orphans double-counts them: "39 of the 40 within-specification cells" is a
    tally over the operator grid, not a value of CarRun, which the sentence merely names.
    """
    out = set()
    for m in _COUNT_RE.finditer(sent):
        for g in m.groups():
            if g.isdigit():
                out.add(g)
    return out


def supports(lit, vals):
    k = len(lit.split(".")[1]) if "." in lit else 0
    x = float(lit)
    if any(abs(round(v, k) - x) < 1e-9 or abs(round(abs(v), k) - x) < 1e-9 for v in vals):
        return True
    # the prose may state a percentage where the artifact stores the fraction
    y = x / 100.0
    return any(abs(round(v, k + 2) - y) < 1e-9 for v in vals)


def sentences(tex):
    for ln, line in prose_lines(tex):
        clean = re.sub(r"\\(cite|ref|eqref|citep|citet|label)\{[^}]*\}", " ", line)
        clean = re.sub(r"\$\\pm\$|\\pm", " +- ", clean)
        clean = re.sub(r"\\[a-zA-Z]+\*?", " ", clean)
        clean = clean.replace("{", " ").replace("}", " ")
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z\\(])", clean):
            if re.search(r"\d", sent):
                yield ln, sent.strip()


def literals(sent):
    out = []
    for m in re.finditer(r"(?<![\w.\\-])(\d+\.\d+|\d{2,5})(?![\w.])", sent):
        ctx = sent[max(0, m.start() - 70):m.end() + 40]
        if PROTO.search(ctx):
            continue
        v = float(m.group(1))
        if v in CONST:
            continue
        if "." not in m.group(1) and (v < 10 or 1900 < v < 2100):
            continue
        out.append(m.group(1))
    return out


def main():
    verbose = "--verbose" in sys.argv
    tex = open(os.path.join(BASE, "paper.tex")).read()
    tv = task_values()
    everything = [v for art in tv[""].values() for v in art]
    orphans, deferred, buckets, quoted = [], [], {}, {}
    for ln, sent in sentences(tex):
        lits = literals(sent)
        if not lits:
            continue
        tasks = [t for t, n in NAMES.items() if re.search(r"\b" + n + r"\b", sent)]
        arm = next((a for a in ARMS if a in sent), "")
        stat = next((s for s in STATS if s in sent.lower()), "")
        pool, srcs = [], {}
        if tasks:
            for t in tasks:
                for art, vals in tv[t].items():
                    pool += vals; srcs.setdefault(art, []).extend(vals)
        else:
            pool = everything; srcs = tv[""]
        _cp = count_phrase_numerals(sent)
        for lit in lits:
            if lit in _cp:
                continue          # a tally, checked in section 4 against the computed counts
            if not supports(lit, pool):
                cov = deferred_to_r8(lit, sent)
                (deferred if cov else orphans).append((ln, lit, tasks, sent[:110], cov))
                continue
            if tasks:
                for art, vals in srcs.items():
                    if supports(lit, vals):
                        quoted.setdefault(art, set()).update(tasks)
                key = (tuple(tasks), arm, stat)
                buckets.setdefault(key, {}).setdefault(lit, []).append(ln)

    print(f"== 1. TASK-SCOPED ORPHANS: {len(orphans)} literals unsupported by the values of the "
          f"task(s) their sentence names (or by anything, when it names none)\n")
    for ln, lit, tasks, sent, _ in orphans:
        print(f"  L{ln:<5} {lit:<8} [{','.join(NAMES[t] for t in tasks) or 'no task'}]  ...{sent}...")

    dis = {k: v for k, v in buckets.items() if len(v) > 1 and (k[1] or k[2])}
    print(f"\n== 2. AGREEMENT BUCKETS with >1 distinct value (task, arm, statistic): {len(dis)}"
          f"  (review; a bucket may legitimately hold several quantities)\n")
    for (tasks, arm, stat), vals in sorted(dis.items()):
        if len(vals) > 6 and not verbose:
            continue
        print(f"  [{','.join(NAMES[t] for t in tasks)} | {arm or '-'} | {stat or '-'}]  "
              + "  ".join(f"{lit}@L{','.join(map(str, ls))}" for lit, ls in sorted(vals.items())))

    if deferred:
        print(f"\n  ({len(deferred)} further literal(s) are not supported by this heuristic but are "
              f"bound by name in verify_constants, gate rule R8:)")
        for ln, lit, tasks, sent, cov in deferred:
            print(f"    L{ln:<5} {lit:<8} -> {cov}")
    print(f"\n== 3. FRESHNESS of the {len(quoted)} artifacts that support a task-scoped prose literal\n")
    stale = []
    for art, tasks in sorted(quoted.items()):
        p = os.path.join(BASE, art)
        if not art.endswith(".json"):
            continue
        gen = rec.generator_for(p)
        globs = rec.families_used(gen)
        if not globs:
            continue
        mt = os.path.getmtime(p)
        for t in sorted(tasks):
            nk = rec.newest_ckpt_for(t, globs)
            if nk and nk[1] > mt:
                stale.append((art, t, os.path.relpath(nk[0], rec.OUT), (nk[1] - mt) / 86400))
    for art, t, ck, days in stale:
        print(f"  STALE {art}: quoted for {NAMES[t]}, {ck} is {days:.1f} d newer")
    if not stale:
        print("  none stale")

    print("\n== 4. COUNT PHRASES in prose (compare against the computed counts below)\n")
    words = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|all|none"
    for ln, line in prose_lines(tex):
        for m in re.finditer(rf"\b((?:{words})|\d+) of (?:the |its |our |all )?((?:{words})|\d+)\b[^.;]{{0,60}}", line):
            print(f"  L{ln:<5} {m.group(0)[:90]}")
    try:
        snap = json.load(open(os.path.join(BASE, "data", "results_snapshot.json")))
        lim = {t: (20 if "velocity" in t else 10 if t.endswith("_b") else 25) for t in NAMES}
        print("\n  computed: tasks whose seed-mean cost is within budget, per arm (15 DSRL + 5 Bullet):")
        for cfg in ("bc_all", "bcsafe", "bcsafeseg", "vfilt_calsafe", "calfilt_csf", "calfilt_lttR50",
                    "calfilt_pref", "cpl_gt", "vfilt_matchgt", "vfilt_random", "vfilt_return"):
            ok = [NAMES[t] for t in NAMES if t in snap and cfg in snap[t]
                  and sum(e["C"] for e in snap[t][cfg].values()) / len(snap[t][cfg]) <= lim[t]]
            n = sum(1 for t in NAMES if t in snap and cfg in snap[t])
            print(f"    {cfg:16s} {len(ok):2d} of {n:2d}: {', '.join(ok)}")
        cert = [NAMES[t] for t in NAMES if t in snap and any(e.get("meta", {}).get("certified")
                for e in snap[t].get("calfilt_csf", {}).values())]
        print(f"    certified (any csf seed) {len(cert)}: {', '.join(cert)}")
    except Exception as e:  # pragma: no cover
        print("  (count summary unavailable:", e, ")")
    bad = len(orphans) + len(stale)
    print(f"\nGATE: {'CLEAN' if bad == 0 else f'{len(orphans)} orphan(s), {len(stale)} stale quoted artifact(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
