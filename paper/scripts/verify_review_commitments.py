"""the advisor's 38 Overleaf comments: the textual commitments made in the response letter, checked
against the CURRENT paper.

The letter (paper/comments/response-letter.md) records what was changed for each comment. A later
rewrite can silently undo one of those changes, which is what checklist item 17 is about: the F4
prose pass rewrote several of the very sections his comments anchor to. Each check below names the
comment it guards and the commitment it verifies. Only mechanically decidable commitments are
here; the rest are listed at the bottom as requiring a read, so the count is honest about what a
PASS covers.

Usage: python scripts/verify_review_commitments.py   (exit 1 on any regression)
"""
import os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = open(f"{BASE}/paper.tex").read()
CUT = re.search(r"\\appendix|\\section\*?\{Appendix", TEX).start()
MAIN, APP = TEX[:CUT], TEX[CUT:]


def body(s):
    """Prose only: drop comments, math and the tabular bodies."""
    s = re.sub(r"(?<!\\)%.*", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", s, flags=re.S)
    return s


BM, BA, BT = body(MAIN), body(APP), body(TEX)


def _bold_unexplained():
    """Tables whose generated body uses \\textbf while the caption never says what bold marks."""
    out = []
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", TEX, re.S):
        blk = m.group(1)
        lab = re.search(r"\\label\{(tab:[^}]+)\}", blk)
        cap = re.search(r"\\caption\{", blk)
        if not (lab and cap):
            continue
        i = cap.end(); d = 1; j = i
        while d and j < len(blk):
            if blk[j] == "{":
                d += 1
            elif blk[j] == "}":
                d -= 1
            j += 1
        caption = blk[i:j - 1]
        ti = re.search(r"\\tabinput\{([^}]+)\}", blk)
        body = open(f"{BASE}/{ti.group(1)}").read() if ti and os.path.exists(f"{BASE}/{ti.group(1)}") else blk
        if "\\textbf" in body and not re.search(r"bold", caption, re.I):
            out.append(lab.group(1))
    return out



def _binom_defined_first():
    """The FIRST occurrence of \\binom must be the one inside the gloss itself, so that no
    use of the stacked parenthetical precedes its definition."""
    f = TEX.find("\\binom")
    if f == -1:
        return False
    return "denotes the binomial coefficient" in TEX[f:f + 60]


CHECKS = [
    ("C001/C008 the phrase 'decoded from state alone' is gone",
     lambda: "decoded from state alone" not in BT),
    ("C001 the abstract describes a STATE-ONLY VALUE learned from segment comparisons",
     lambda: re.search(r"state-only value[^.]{0,80}segment comparison", BM) is not None),
    ("C007 the word 'instrumentation' is gone",
     lambda: "instrumentation" not in BT.lower()),
    ("C021 the word 'multiplicity' is gone",
     lambda: "multiplicity" not in BT.lower()),
    # C026 concerned the UNDEFINED main-text use, which is gone. The appendix construction that
    # separates the two classes may name them, and now glosses them at first use.
    ("C026 'operator class' is gone from the main text and glossed at first use in the appendix",
     lambda: "operator class" not in body(MAIN).lower()
             and "per-transition reweighting and whole-trajectory selection" in BA),
    ("C012 the logistic function is defined at first use",
     lambda: "\\mathrm{logistic}(x) = 1/(1+e^{-x})" in TEX),
    # the check must ALSO deny the main text, or a regression that re-pins K passes silently.
    # It did: the rendered page 3 showed "an ensemble of $K = 3$" in the method section.
    ("C013 the method carries K as a symbol; the value 3 lives only in the appendix",
     lambda: "$K = 3$" in APP and "$K = 3$" not in MAIN),
    ("C034 Learn-then-Test is spelled out where threshold selection is introduced",
     lambda: "Learn-then-Test" in BM),
    ("C017 'power' is defined at first use, not used bare",
     lambda: "monotonicity affects only \\emph{power}, that is, how deep into the grid" in TEX),
    ("C031 constraint satisfaction is the primary criterion, stated once",
     lambda: re.search(r"within budget means[^.]{0,90}", BT) is not None),
    ("C037 a normalized-cost table exists in the appendix with a pointer from the main table",
     lambda: os.path.exists(f"{BASE}/data/tables/normalized_main.tex") and "normalized" in BA.lower()),
    ("C032 the full-label baselines point at their appendix where first named",
     lambda: re.search(r"COptiDICE[^.]{0,200}\\ref\{app:baselines\}", MAIN) is not None),
    ("C029 saliency is defined where it is used, not left bare",
     lambda: "with respect to each observation coordinate" in BM),
    ("C003 'composes' is glossed where the main text first uses it",
     lambda: re.search(r"selection composes, meaning", BM) is not None),
    ("C030 the landscape appendix names three independently sampled layouts and their reset seeds",
     lambda: re.search(r"reset seeds 7, 11, 23", BA) is not None),
    # C004 is a REGRESSION BUDGET, not zero. The colons left are explanatory ("yield is governed
    # by the purity margin, not raw safe mass: Swimmer has less safe mass than Ant yet certifies
    # more often"), where the right side elaborates the left. the advisor's tell was two independent
    # statements joined by a colon. Titles and captions are excluded by the response letter.
    ("C004 explanatory colons in prose stay at or below the audited count of four",
     lambda: len([m for m in re.finditer(r"(?<![A-Za-z])[a-z]{3,}: [A-Z][a-z]", BT)
                  if not re.search(r"(curation|Method|Guarantees|Curation): ", m.group(0))]) <= 4),
    ("C022 the reward-vs-cost labeling-asymmetry claim is gone from abstract and gating",
     lambda: not re.search(r"(cheap|cheaper|expensive)[^.]{0,60}(cost label|labeling)", BM)),
    # C035 mechanized 2026-09-23: a bolded table must say in its caption what bold marks.
    ("C035 every table whose body is bolded explains bold in its caption",
     lambda: _bold_unexplained() == []),
    # C027 mechanized: the granularity section opens by naming the rejected alternative
    # rather than by describing our own method.
    ("C027 the granularity section opens by naming the rejected alternative",
     lambda: re.search(r"The standard alternative reweights individual transitions instead; "
                       r"everything analyzed in this section is that alternative, not a "
                       r"component of our method\.", TEX) is not None),
    # Added by the 2026-09-23 audit. The introduction's gap statement said prior work lacks a
    # guarantee "relating supervision to the safety of what is learned", which is a policy-level
    # guarantee this paper explicitly does not provide (the abstract and Section 7 say the
    # policy's cost is measured, not bounded). The paper must not promise one anywhere.
    ("SCOPE: no sentence promises a guarantee on the learned policy's safety",
     lambda: not re.search(r"guarantee[^.]{0,60}(safety of what is learned|on the learned policy"
                           r"|that the policy is safe)", BT)),
    # The labels-only parity is asserted in four places (Sec 1, 6.4, 7 and App L). Three were
    # corrected on 2026-09-23 and the fourth was missed, leaving the appendix contradicting the
    # main text. No place may say preferences buy NO additional safety: at matched budget they
    # buy one task, twelve against eleven.
    ("CONSISTENCY: nowhere claims preferences buy no additional safety",
     lambda: not re.search(r"buy neither additional safety|buy no additional safety", BT)),
    # C010, C019 and C020 mechanized 2026-09-23 after the audit found all three had been
    # trimmed away by a later prose pass, which is exactly the regression R17 exists to catch.
    ("C010 episodic cost is defined where it is first used",
     lambda: "episodic cost, the sum of its per-step" in BT),
    ("C019 a calibration run is identified with the pseudocode",
     lambda: re.search(r"per calibration run, one pass of the Appendix~\\ref\{app:algo\}", TEX) is not None),
    ("C020 a task is defined as one dataset and environment pair",
     lambda: "one dataset and environment pair per task" in BM),
    ("C025 the preference-pair labeling protocol is stated, ties included",
     lambda: "labeled by the two segments' own summed costs, ties skipped" in BA),
    # Advisor review, 2026-09-24: the stacked parenthetical could be read as a column vector, so the
    # binomial coefficient must be named before its first use in the proofs.
    ("ADVISOR: the binomial coefficient is defined before its first use",
     lambda: _binom_defined_first()),
    # Same sweep as the advisor's binomial note: one name for the hypergeometric distribution, its
    # argument convention stated, and the two non-standard spellings glossed.
    ("NOTATION: the hypergeometric distribution has one name and a stated argument order",
     lambda: "\\mathrm{Hypergeom}" not in TEX
             and "is the hypergeometric\ndistribution, its arguments the population size" in TEX),
    ("NOTATION: clip and stochastic dominance are glossed where they appear",
     lambda: "the clip holds it in" in BT and "is stochastic dominance" in BA),
    ("RELEASE POLICY: the paper carries no repository URL",
     lambda: not re.search(r"github\.com|anonymous\.4open", TEX, re.I)),
]

NEEDS_A_READ = [
    "C002/C003 abstract voice and defined terms", "C005/C006 introduction supervision hierarchy",
    "C009/C011 preliminaries definitions", "C014 ensemble justification",
    "C015/C016/C018 calibration wording", "C023/C024 scale and read-out wording",
    "C028 'correctly' restated", "C029 saliency restated as top-one recovery",
    "C033 setup rewritten, diagnostics demoted",
    "C036 term defined at first use", "C038 explanation plus the operator result",
]

fails = 0
for name, fn in CHECKS:
    try:
        ok = bool(fn())
    except Exception as e:
        ok = False
        name += f"  [checker error: {e}]"
    if not ok:
        fails += 1
    print(("  PASS  " if ok else "  FAIL  ") + name)
print(f"\n  {len(NEEDS_A_READ)} further commitments are not mechanically decidable and are verified by reading:")
for n in NEEDS_A_READ:
    print(f"    - {n}")
print(f"\nREVIEW COMMITMENTS: {'PASS' if not fails else str(fails) + ' REGRESSION(S)'} "
      f"({len(CHECKS)} mechanical, {len(NEEDS_A_READ)} by read)")
sys.exit(1 if fails else 0)
