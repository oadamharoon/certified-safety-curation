"""Cross-check every number in the paper's prose against the data layer.

Tables and figures are generated, so they cannot drift. Prose numbers are typed
by hand and silently go stale whenever a harvest changes a cell. This pulls
every numeric literal out of the prose and asks whether the data layer contains
that value anywhere; anything unmatched is a candidate for a stale claim.

Unmatched does not mean wrong. Derived quantities, counts and percentages will
not appear verbatim. It is a shortlist to check by hand, not a verdict.
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_values():
    """Every numeric value in the snapshot, osrl results and generated tables,
    plus the per-arm MEAN over seeds. Prose routinely quotes a mean, which is
    computed and never stored, so a flat walk of the files alone reports a
    supported number as unsupported: pointgoal2/vfilt_retbot (43.7) and
    pointcircle2/calfilt_tier2 (112.5) were both flagged that way while
    check_groups, which does average, accepted them."""
    vals = set()
    for _name, _sc in subtrees():
        if _name.endswith("#mean"):
            for _v in _sc:
                for _d in (0, 1, 2, 3):
                    vals.add(round(float(_v), _d))

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            for d in (0, 1, 2, 3):
                vals.add(round(float(o), d))

    for fn in os.listdir(os.path.join(BASE, "data")):
        if fn.endswith(".json"):
            try:
                walk(json.load(open(os.path.join(BASE, "data", fn))))
            except Exception:
                pass
    tdir = os.path.join(BASE, "data", "tables")
    if os.path.isdir(tdir):
        for fn in os.listdir(tdir):
            _fp = os.path.join(tdir, fn)
            if not os.path.isfile(_fp):      # skip data/tables/_superseded/
                continue
            for m in re.findall(r"-?\d+\.?\d*", open(_fp).read()):
                try:
                    for d in (0, 1, 2, 3):
                        vals.add(round(float(m), d))
                except ValueError:
                    pass
    return vals



def subtrees():
    """(name, scalars) for every json subtree and every generated-table row.

    Support is judged per subtree rather than per file. A group of numbers cited
    in one sentence should come from one place in the data layer; the smallest
    subtree containing all of them names that place, and a group with no small
    subtree is almost certainly stale.
    """
    out = []

    def walk(o, name):
        if isinstance(o, dict):
            acc = []
            for k, v in o.items():
                acc += walk(v, name + "/" + str(k))
            out.append((name, acc))
            # Prose quotes per-arm MEANS over seeds, which are not stored: only
            # the per-seed cells are. Without this, every mean in the paper read
            # as unsupported and the real stale numbers hid in that noise.
            kids = [v for v in o.values() if isinstance(v, dict)]
            if len(kids) > 1:
                keys = set(kids[0])
                for kd in kids[1:]:
                    keys &= set(kd)
                means = []
                for key in sorted(keys):
                    xs = [kd[key] for kd in kids
                          if isinstance(kd.get(key), (int, float))
                          and not isinstance(kd.get(key), bool)]
                    if len(xs) == len(kids):
                        means.append(sum(xs) / len(xs))
                if means:
                    out.append((name + "#mean", means))
            return acc
        if isinstance(o, list):
            acc = []
            for i, v in enumerate(o):
                acc += walk(v, name + "[%d]" % i)
            out.append((name, acc))
            return acc
        if isinstance(o, (int, float)) and not isinstance(o, bool):
            return [float(o)]
        return []

    ddir = os.path.join(BASE, "data")
    for fn in os.listdir(ddir):
        if fn.endswith(".json"):
            try:
                walk(json.load(open(os.path.join(ddir, fn))), fn)
            except Exception:
                pass
    tdir = os.path.join(ddir, "tables")
    if os.path.isdir(tdir):
        for fn in os.listdir(tdir):
            fp = os.path.join(tdir, fn)
            if not os.path.isfile(fp):
                continue
            rows = []
            for i, line in enumerate(open(fp), 1):
                ns = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                if ns:
                    out.append(("tables/%s:%d" % (fn, i), ns))
                    rows += ns
            if rows:
                out.append(("tables/" + fn, rows))
    return out


def _matches(lit, vals):
    """A value supports a literal only at the precision the literal states.

    Rounding to fewer places was the flaw in the flat check: at one decimal
    "0.43" is satisfied by any 0.4 in the data layer, which is nearly always
    present, so stale numbers matched something and never surfaced.
    """
    k = len(lit.split(".")[1]) if "." in lit else 0
    x = float(lit)
    return any(abs(round(v, k) - x) < 1e-9 or abs(round(abs(v), k) - x) < 1e-9
               for v in vals)


def check_groups(tex, subs, limit=80):
    """Attribute each 3+-number sentence to the smallest subtree supporting it."""
    res = []
    for ln, line in prose_lines(tex):
        clean = re.sub(r"\\(cite|ref|eqref|citep|citet)\{[^}]*\}", " ", line)
        clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
        for sent in re.split(r"(?<=[.;])\s+", clean):
            lits = re.findall(r"(?<![\w.\\])(\d+\.\d+)(?![\w.])", sent)
            lits = [l for l in lits
                    if float(l) not in (0.05, 0.1, 0.25, 0.4, 0.5, 0.9, 0.95, 1.0)]
            if len(lits) < 3:
                continue
            best = None
            for nm, vals in subs:
                if len(vals) > limit:
                    continue
                if all(_matches(l, vals) for l in lits):
                    if best is None or len(vals) < best[1]:
                        best = (nm, len(vals))
            res.append((ln, lits, best, sent.strip()[:100]))
    return res


def prose_lines(tex):
    """Body paragraphs and captions, skipping preamble and tabular bodies."""
    out, in_tab = [], False
    for i, line in enumerate(tex.splitlines(), 1):
        s = line.strip()
        if s.startswith(r"\begin{tabular}"):
            in_tab = True
        if s.startswith(r"\end{tabular}"):
            in_tab = False
            continue
        if in_tab or s.startswith("%") or not s:
            continue
        if s.startswith(("\\usepackage", "\\newcommand", "\\documentclass",
                         "\\label", "\\input", "\\tabinput", "\\includegraphics")):
            continue
        out.append((i, line))
    return out


def main():
    tex = open(os.path.join(BASE, "paper.tex")).read()
    vals = data_values()
    print(f"data layer holds {len(vals)} distinct numeric values\n")
    flagged = []
    for ln, line in prose_lines(tex):
        # strip latex commands and refs, then take standalone numbers
        clean = re.sub(r"\\(cite|ref|eqref|citep|citet)\{[^}]*\}", " ", line)
        clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
        # integers were never audited, which let a stale "2172" stand in the
        # prose for two paragraphs; 3-5 digit integers are checked too now
        for m in re.finditer(r"(?<![\w.\\])(\d+\.\d+|\d{3,5})(?![\w.])", clean):
            # protocol constants (draw counts, pair counts, epochs, horizons) are
            # stated parameters, not measured results, so they will never appear in
            # the data layer; auditing them buries the real hits in noise
            _ctx = clean[max(0, m.start() - 90):m.end() + 60]
            if re.search(r"draws?|pairs?|episodes?|transitions?|labels?|horizon|epochs?|batch|width|length|seeds?|trajectories", _ctx):
                continue
            v = float(m.group(1))
            if round(v, 3) in vals or round(v, 2) in vals or round(v, 1) in vals:
                continue
            if v in (0.05, 0.1, 0.25, 0.4, 0.5, 0.9, 0.95, 1.0, 2.0, 3.0):
                continue  # alpha/delta/kappa constants
            ctx = clean.strip()
            i = ctx.find(m.group(1))
            flagged.append((ln, m.group(1), ctx[max(0, i - 55):i + 40].strip()))
    print(f"{len(flagged)} decimal literals in prose with no match in the data layer:\n")
    for ln, v, ctx in flagged:
        print(f"  L{ln:<5} {v:<9} ...{ctx}...")

    subs = subtrees()
    groups = check_groups(tex, subs)
    every = []
    for _nm, _v in subs:
        every += _v

    # Primary signal: a literal supported nowhere at its stated precision.
    # Sentences legitimately cite several tasks at once, so "no single subtree
    # covers this group" is noisy; "this number exists nowhere" is not.
    orphan = []
    for ln, lits, best, sent in groups:
        miss = [l for l in lits if not _matches(l, every)]
        if miss:
            orphan.append((ln, miss, lits, sent))
    print(f"\n{len(groups)} number groups (3+ decimals in one sentence); "
          f"{len(orphan)} contain a literal absent from the data layer.\n")
    for ln, miss, lits, sent in orphan:
        print(f"  L{ln:<5} MISSING {miss}  of {lits}\n         ...{sent}...")

    # Secondary: groups a single small subtree explains, and their source.
    single = [g for g in groups if g[2] is not None]
    print(f"\n{len(single)} of {len(groups)} groups trace to one subtree "
          f"(check that the named source is plausible):\n")
    for ln, lits, best, sent in single:
        print(f"  L{ln:<5} {lits}\n         src={best[0]} ({best[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
