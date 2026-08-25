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
    """Every numeric value in the snapshot, osrl results and generated tables."""
    vals = set()

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
            for m in re.findall(r"-?\d+\.?\d*", open(os.path.join(tdir, fn)).read()):
                try:
                    for d in (0, 1, 2, 3):
                        vals.add(round(float(m), d))
                except ValueError:
                    pass
    return vals


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
        for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w])", clean):
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
