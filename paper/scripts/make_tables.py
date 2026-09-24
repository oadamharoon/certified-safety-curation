"""Generate every LaTeX table body for the paper from the results snapshot.

Emits mean +- 95% bootstrap CI half-widths over seeds. Output files land in
data/tables/*.tex so paper tables regenerate with one command.
"""

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

import json, os
import numpy as np

BASE = CSC_PAPER
LIMITS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
          "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25,
          "cargoal2": 25, "pointgoal1_dsrl": 25, "pointgoal2": 25,
          "pointbutton1": 25, "pointbutton2": 25, "carbutton1_t3": 25,
          "carbutton2": 25, "pointcircle1": 25, "pointcircle2": 25}
NAMES = {"halfcheetah_velocity": "HalfCheetah", "walker2d_velocity": "Walker2d",
         "ant_velocity": "Ant", "hopper_velocity": "Hopper",
         "swimmer_velocity": "Swimmer", "cargoal1_dsrl": "CarGoal1",
         "cargoal2": "CarGoal2", "pointgoal1_dsrl": "PointGoal1",
         "pointgoal2": "PointGoal2", "pointbutton1": "PointButton1",
         "pointbutton2": "PointButton2", "carbutton1_t3": "CarButton1",
         "carbutton2": "CarButton2", "pointcircle1": "PointCircle1",
         "pointcircle2": "PointCircle2"}
ORDER = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
         "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
         "pointgoal1_dsrl", "pointgoal2", "pointbutton1", "pointbutton2",
         "carbutton1_t3", "carbutton2", "pointcircle1", "pointcircle2"]
# analysis-set emitters (oracle/extraction tables) cover the 9 core tasks
ANALYSIS_ORDER = ORDER[:9]

with open(os.path.join(BASE, "data", "results_snapshot.json")) as f:
    SNAP = json.load(f)


def ci(vals, n_boot=10000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (hi - lo) / 2


# vawr_5seed (2026-06-08) could not be re-derived: only an eval log survives
# and its June training script is unidentifiable. It was also ABSENT on
# pointgoal2, where a FALLBACK silently substituted xlab_exp_k1 -- so one row
# of tab:oracle mixed a June arm on eight tasks with a July arm on the ninth.
# It is replaced by xlab_exp_k1 regenerated on all nine analysis tasks at the
# recovered config (mode=exp, k=1, AWR_EPOCHS=50, BATCH_SIZE=1024,
# AWR_BETA=0.1). No fallback: every cell is the same arm and the same vintage.
FALLBACK = {}
# tab:oracle averages three seeds per cell for its other columns
# (xlab_oracle_step / xlab_oracle_ctg); pointgoal2 carries five for
# xlab_exp_k1, so restrict that column to 0-2 to keep the row uniform.
SEED_CAP = {"xlab_exp_k1": {"0", "1", "2"}}


def cell(task, config, metric, digits=1, bold_if_safe=False):
    entries = SNAP.get(task, {}).get(config, {})
    if not entries and config in FALLBACK:
        entries = SNAP.get(task, {}).get(FALLBACK[config], {})
    if config in SEED_CAP:
        entries = {k: v for k, v in entries.items() if k in SEED_CAP[config]}
    if not entries:
        return "--"
    vals = [e[metric] for e in entries.values()]
    m = np.mean(vals)
    if config == "bc_all" or len(vals) < 2:
        txt = f"{m:.{digits}f}"          # deterministic reference: no CI
    else:
        h = ci(vals)
        txt = f"{m:.{digits}f}\\,\\scriptsize$\\pm${h:.{digits}f}"
    if bold_if_safe and metric == "C" and m <= LIMITS[task]:
        txt = f"\\textbf{{{txt}}}"
    return txt


def emit(fname, configs, digits_map=None, skip_empty=False, order=None):
    lines = []
    for task in (order or ORDER):
        if skip_empty and not any(SNAP.get(task, {}).get(c) for c in configs):
            continue
        d = digits_map.get(task, 1) if digits_map else (
            0 if "velocity" in task else (1 if "circle" in task else 2))
        row = NAMES[task]
        for cfg in configs:
            row += f" & {cell(task, cfg, 'R', d)} & {cell(task, cfg, 'C', d, True)}"
        lines.append(row + r" \\")
    out = os.path.join(BASE, "data", "tables", fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


# Main results: BC-All, BC-Safe, V-filter(best of matchgt/q25 per task is a
# judgment call -> emit both variants; paper table uses vfilt best manually),
# SUPERSEDED (superseded by main_gated.tex (calfilt_csf); built on calfilt_ltt); output in data/tables/_superseded/
# emit("main.tex", ["bc_all", "bcsafe", "vfilt_matchgt", "calfilt_ltt",
#                   "calfilt_lttR50"])
# SUPERSEDED (superseded by controls.tex); output in data/tables/_superseded/
# emit("vfilt_variants.tex", ["vfilt_matchgt", "vfilt_q25"])

# T3.2 extraction temperature/clip sweep and T2.6 score aggregators, on the
# nine analysis tasks. Pre-registered 2026-08-10; the first pass was void
# because 04f ignored SEED_OVERRIDE and produced identical seeds.
# PROVENANCE. The nine alpha=0.25 selections the paper reports are, per task,
# three DISTINCT quantile thresholds of the same standardized V ensemble:
#   halfcheetah (V seed 1): q85 cdt_cert .219 | q80 cdt_cert_draw2 .287
#                           | q75 cdt_a25selq75 .351
#   cargoal1    (V seed 3): q85 cdt_cert .171 | q80 cdt_cert_draw2 .212
#                           | q75 cdt_a25selq75 .266
#   pointgoal1  (V seed 0): q85 cdt_cert .122 | q55 cdt_cert_draw2 .238
#                           | q80 cdt_a25selq80 .141
# Verified 2026-08-25 against the paper's own quoted values: 27 of 27 safe,
# max per-task cost span 5.17 ("at most 5.2"), exactly three out-of-spec
# selections (.287, .351, .266). The a25selq arms ARE part of the nine; an
# earlier note here claimed they were not, which would drop the genuine third
# selection for every task.
# cdt_cert_draw3 is NOT a tenth selection. That draw landed on the same
# threshold as draw1 (d1=d3, see analysis/build_distinct.py), which is why the
# q75/q80 arms were built to supply a distinct third; for cargoal1 its results
# are bit-identical to cdt_cert. Do not count or plot it.
# Also unused: pointgoal1 cdt_a25selq65/70 (contamination .208/.183), run but
# not part of the reported nine.
ANALYSIS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
            "hopper_velocity", "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
            "pointgoal1_dsrl", "pointgoal2"]
emit("t32_sweep.tex",
     ["vawr_b01", "vawr_b03", "vawr_b3", "vawr_c5", "vawr_c100"],
     order=ANALYSIS)
emit("t26_agg.tex", ["xagg_min", "xagg_p10", "vfilt_calsafe"], order=ANALYSIS)

# tab:obstacle2, regenerated from obstacle2_stats.json. Previously hardcoded
# with a lost source: its accuracy row omitted a task and its per-transition
# rows were HalfCheetah seed-pair statistics presented as cross-task ranges.
_o2 = json.load(open(os.path.join(BASE, "data", "obstacle2_stats.json")))
_acc = [np.mean(v["accuracy"]) for v in _o2.values()]
_rho = [r for v in _o2.values() for r in v["rho"]]
_jac = [j for v in _o2.values() for j in v["jaccard"]]
with open(os.path.join(BASE, "data", "tables", "obstacle2.tex"), "w") as f:
    f.write(
        f"Trajectory & held-out segment-pair accuracy & "
        f"${min(_acc):.3f}$ to ${max(_acc):.3f}$ \\\\\n"
        f"Per-transition & cross-seed advantage rank correlation $\\rho$ & "
        f"${min(_rho):.2f}$ to ${max(_rho):.2f}$ \\\\\n"
        f"Per-transition & top-1-percent transition overlap (Jaccard) & "
        f"${min(_jac):.2f}$ to ${max(_jac):.2f}$ \\\\\n")
print("wrote obstacle2.tex")

# tab:labelcomplexity, previously hardcoded. n50 comes from the resampling fit
# in label_complexity.json; margins from margin_vs_yield.json (all 14 tasks);
# ">3200" marks tasks whose certification rate never reaches half at the
# largest budget swept, per label_complexity_ext.json.
_lc = json.load(open(os.path.join(BASE, "data", "review_response", "label_complexity.json")))
_ext = json.load(open(os.path.join(BASE, "data", "review_response", "label_complexity_ext.json")))
_mvy = {r["task"]: r for r in json.load(
    open(os.path.join(BASE, "data", "review_response", "margin_vs_yield.json")))}
_LCNAME = {"carrun_b": "CarRun", "pointgoal1_dsrl": "PointGoal1",
           "cargoal1_dsrl": "CarGoal1", "dronerun_b": "DroneRun",
           "cargoal2": "CarGoal2", "halfcheetah_velocity": "HalfCheetah",
           "swimmer_velocity": "Swimmer", "walker2d_velocity": "Walker2d"}
_rows, _pos = [], []
for _t, _nm in _LCNAME.items():
    _m = _mvy.get(_t, {}).get("margin")
    if _m is None:
        _m = _ext.get(_t, {}).get("margin")
    _n = _lc["n50"].get(_t)
    _cell = f"${round(_n)}$" if _n is not None else r"$> 3200$"
    _pos.append((_m, _nm, _cell))
for _m, _nm, _cell in sorted(_pos, key=lambda x: -x[0]):
    _rows.append(f"{_nm} & ${_m:.3f}$ & {_cell} \\\\")
_neg = [r["margin"] for r in _mvy.values() if r["margin"] <= 0]
_rows.append(r"\midrule")
_rows.append(f"{len(_neg)} tasks with margin $\\le 0$ & ${max(_neg):.3f}$ to "
             f"${min(_neg):.3f}$ & never \\\\")
with open(os.path.join(BASE, "data", "tables", "labelcomplexity.tex"), "w") as f:
    f.write("\n".join(_rows) + "\n")
print("wrote labelcomplexity.tex")
# Controls ablation
emit("controls.tex", ["vfilt_random", "vfilt_return", "vfilt_retbot", "vfilt_matchgt"])
# Alpha sweep policies
# SUPERSEDED (superseded by Figure fig:alphacurve; built on calfilt_ltt); output in data/tables/_superseded/
# emit("alpha.tex", ["calfilt_a10", "calfilt_ltt", "calfilt_a40"], order=ANALYSIS_ORDER)
# Extraction sweep appendix (oracle + variants; 3 seeds)
emit("extraction.tex", ["xlab_rank", "xlab_binary", "cf_v2", "cf_octg"],
     skip_empty=True, order=ANALYSIS_ORDER)
# Noise policies (4 tasks only, harmless dashes elsewhere)
# SUPERSEDED (noise ablation lives in prose + app:extended; built on calfilt_ltt); output in data/tables/_superseded/
# emit("noise_policy.tex", ["calfilt_noise20", "calfilt_ltt"], order=ANALYSIS_ORDER)


# Combined main table body: BC-All | BC-Safe | V-filter | Calibrated | Gated
def main_gated():
    gated = {}
    for task in ORDER:
        # calfilt_csf is the deployed procedure: Learn-then-Test with the
        # Clopper-Pearson fallback the paper describes. calfilt_ltt is a
        # pre-standardization tag whose fallback rule the code no longer has.
        base = SNAP.get(task, {}).get("calfilt_csf", {})
        r50 = SNAP.get(task, {}).get("calfilt_lttR50", {})
        Rs, Cs = [], []
        for seed, e in base.items():
            cert = e.get("meta", {}).get("certified", False)
            src = r50.get(seed, e) if cert else e
            Rs.append(src["R"]); Cs.append(src["C"])
        gated[task] = (Rs, Cs)
    lines = []
    for task in ORDER:
        d = 0 if "velocity" in task else (1 if "circle" in task else 2)
        row = NAMES[task]
        for cfg in ("bc_all", "bcsafe", "vfilt_calsafe"):
            row += f" & {cell(task, cfg, 'R', d)} & {cell(task, cfg, 'C', d, True)}"
        # calibrated column: calsafe fallback on uncertified tasks, gated (dagger)
        # values on the certified tasks (four in the regenerated cohort)
        # derived from the runs themselves: a task is certified when at least one
        # calibration run returns a certificate. Hardcoding this list let it drift.
        certified = any(e.get("meta", {}).get("certified", False)
                        for e in SNAP.get(task, {}).get("calfilt_csf", {}).values())
        Rs, Cs = gated[task]
        mR, hR, mC, hC = np.mean(Rs), ci(Rs), np.mean(Cs), ci(Cs)
        dag = r"$^\dagger$" if certified else ""
        ctxt = f"{mC:.{d}f}\\,\\scriptsize$\\pm${hC:.{d}f}"
        if mC <= LIMITS[task]:
            ctxt = f"\\textbf{{{ctxt}}}"
        row += f" & {mR:.{d}f}\\,\\scriptsize$\\pm${hR:.{d}f}{dag} & {ctxt}"
        lines.append(row + " \\\\")
    with open(os.path.join(BASE, "data", "tables", "main_gated.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote main_gated.tex")
    return gated


GATED = main_gated()   # C11 (V2_REMEDIATION): the normalized table must report the same deployed (gated) arm as Table 1

# Full-benchmark oracle ablation (Table 1): all 8 tasks
emit("oracle_all.tex", ["bc_all", "xlab_exp_k1", "xlab_oracle_step",
                        "xlab_oracle_ctg"], order=ANALYSIS_ORDER)

# Normalized-cost scan across ALL methods (main text): C_n = C / budget.
import pickle, yaml
import json as _j
REPO = CSC_WORK
_OSRL2 = _j.load(open(os.path.join(BASE, "data", "osrl_results.json")))
lines = []
for task in ORDER:
    lim = LIMITS[task]
    row = NAMES[task]
    def _cn(vals):
        if not vals:
            return " & --"
        cn = np.mean(vals) / lim
        t = f"{cn:.2f}"
        return " & " + (f"\\textbf{{{t}}}" if cn <= 1.0 else t)
    for cfgname in ("bc_all", "bcsafe", "bcsafeseg", "vfilt_calsafe"):
        ent = SNAP.get(task, {}).get(cfgname, {})
        row += _cn([e["C"] for e in ent.values()])
    # calibrated column: identical to Table 1's deployed arm (gated on the certified tasks)
    row += _cn(GATED[task][1])
    for algo in ("cdt", "cpq", "coptidice"):
        e = _OSRL2.get(task, {}).get(algo, {})
        row += _cn([v["C"] for v in e.values()])
    ent = SNAP.get(task, {}).get("cpl_gt", {})
    row += _cn([e["C"] for e in ent.values()])
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "normalized_main.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote normalized_main.tex")
print("done")


# External baselines table: full-label OSRL (CDT at its most conservative
# evaluated target; CPQ/COptiDICE single eval) + same-supervision CPL.
with open(os.path.join(BASE, "data", "osrl_results.json")) as f:
    OSRL = json.load(f)


def osrl_cell(task, algo, metric, d):
    e = OSRL.get(task, {}).get(algo, {})
    if not e:
        return "--"
    vals = [v[metric] for v in e.values()]
    m = np.mean(vals)
    if len(vals) < 2:
        return f"{m:.{d}f}"
    h = ci(vals)
    txt = f"{m:.{d}f}\\,\\scriptsize$\\pm${h:.{d}f}"
    if metric == "C" and m <= LIMITS[task]:
        txt = f"\\textbf{{{txt}}}"
    return txt


lines = []
for task in ORDER:
    d = 0 if "velocity" in task else (1 if "circle" in task else 2)
    row = NAMES[task]
    for algo in ("cdt", "cpq", "coptidice"):
        row += f" & {osrl_cell(task, algo, 'R', d)} & {osrl_cell(task, algo, 'C', d)}"
    row += f" & {cell(task, 'cpl_gt', 'R', d)} & {cell(task, 'cpl_gt', 'C', d, True)}"
    row += f" & {cell(task, 'bcsafeseg', 'R', d)} & {cell(task, 'bcsafeseg', 'C', d, True)}"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "baselines.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote baselines.tex")

# Labels-only control across label budgets (appendix)
emit("labelsonly.tex", ["labels_only_n50", "labels_only_n100",
                        "labels_only", "labels_only_n400"])

# Composability tables (from OSRL harvest + snapshot)
with open(os.path.join(BASE, "data", "osrl_results.json")) as f:
    OSRL2 = json.load(f)


def _mc(entries, metric, d, bold_lim=None):
    if not entries:
        return "--"
    vals = [v[metric] for v in entries.values()]
    m = np.mean(vals)
    txt = f"{m:.{d}f}" if len(vals) < 2 else f"{m:.{d}f}\\,\\scriptsize$\\pm${ci(vals):.{d}f}"
    if bold_lim is not None and metric == "C" and m <= bold_lim:
        txt = f"\\textbf{{{txt}}}"
    return txt


lines = []
# Walker2d belongs here: it is one of the four tasks certifying at alpha = 0.25, the main text
# counts four, and it is the row where CPL violates. It was omitted because the operator column
# was hard-coded to wbc_q85 and Walker2d's deepest certified quantile is 0.75. The rule below is
# each task's MOST SELECTIVE distinct certified selection, which reproduces q85 for the other
# three and leaves every previously reported cell unchanged.
def _deepest_wbc(task):
    arms = [a for a in SNAP.get(task, {}) if a.startswith("wbc_q")]
    return max(arms, key=lambda a: int(a.split("q")[1])) if arms else "wbc_q85"


for task in ("halfcheetah_velocity", "walker2d_velocity", "cargoal1_dsrl", "pointgoal1_dsrl"):
    d = 0 if "velocity" in task else 2
    lim = LIMITS[task]
    row = NAMES[task]
    for src in ("cdt", "cdt_noaug", "cdt_cert"):
        e = OSRL2.get(task, {}).get(src, {})
        row += f" & {_mc(e, 'R', d)} & {_mc(e, 'C', d, lim)}"
    # the clone column is the deployed procedure (calfilt_csf). calfilt_ltt is a
    # pre-standardization tag whose fallback rule the code no longer has, and on
    # HalfCheetah it averaged a certified seed with two minimum-selection seeds.
    row += f" & {cell(task, 'calfilt_csf', 'R', d)} & {cell(task, 'calfilt_csf', 'C', d, True)}"
    _op = _deepest_wbc(task)
    row += f" & {cell(task, _op, 'R', d)} & {cell(task, _op, 'C', d, True)}"
    # CPL is a composability arm too: it consumes the same certified selection.
    # Presenting CDT on it without CPL was the asymmetry this column removes.
    row += f" & {cell(task, 'cpl_gt_cert', 'R', d)} & {cell(task, 'cpl_gt_cert', 'C', d, True)}"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "compose.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote compose.tex")

# alpha=0.25 distinct-selection grid (three per certifying task; four tasks since
# Walker2d began certifying at this level, so twelve selections). The claim was
# carried in prose only, while the weaker
# alpha=0.40 result had a table; this tabulates it. Membership is documented in
# the PROVENANCE block above. Reward is shown alongside cost so the safety
# result is not read without its return cost.
# (cdt_key, quantile, realized contamination, cpl_key). CPL consumes the same
# selection as CDT, so it is reported beside it rather than in a separate table.
# --- C8: the composability selections of the six regenerated tasks are built from
# runs/selections/v2_summary.json rather than typed, so the key names and the realized
# contaminations come from the selections themselves. Tasks absent from the summary keep
# the entries below, which were never part of the regenerated cohort.
_V2 = json.load(open(CSC_RUNS + "/selections/v2_summary.json"))


def _v2_rows(level):
    """task -> [(q label, realized contamination)] for the three distinct selections."""
    out = {}
    for task, rec in _V2.items():
        d = rec.get(level)
        if d:
            out[task] = [(x["name"].split("_")[-1], x["contamination"]) for x in d["distinct"]]
    return out


_A25SEL = {
    "halfcheetah_velocity": [("cdt_cert", "q85", 0.219, "cpl_gt_cert"),
                             ("cdt_cert_draw2", "q80", 0.287, "cpl_gt_a25q80"),
                             ("cdt_a25selq75", "q75", 0.351, "cpl_gt_a25q75")],
    "cargoal1_dsrl": [("cdt_cert", "q85", 0.171, "cpl_gt_cert"),
                      ("cdt_cert_draw2", "q80", 0.212, "cpl_gt_a25q80"),
                      ("cdt_a25selq75", "q75", 0.266, "cpl_gt_a25q75")],
    "pointgoal1_dsrl": [("cdt_cert", "q85", 0.122, "cpl_gt_cert"),
                        ("cdt_a25selq80", "q80", 0.141, "cpl_gt_a25q80"),
                        ("cdt_cert_draw2", "q55", 0.238, "cpl_gt_a25q55")],
}

for _task, _rows in _v2_rows("a25").items():
    _A25SEL[_task] = [(f"cdt_a25new_{q}", q, c, f"cpl_gt_a25{q}") for q, c in _rows]
_A25ORDER = tuple(k for k in ("halfcheetah_velocity", "walker2d_velocity", "cargoal1_dsrl",
                              "pointgoal1_dsrl") if k in _A25SEL)
_lines, _nsafe, _ntot = [], 0, 0
for task in _A25ORDER:
    d = 0 if "velocity" in task else 2
    dr = 0 if "velocity" in task else 1
    row = NAMES[task]
    for algo, q, ku, cplk in _A25SEL[task]:
        e = OSRL.get(task, {}).get(algo, {})
        _ntot += len(e)
        _nsafe += sum(1 for v in e.values() if v["C"] <= LIMITS[task])
        row += (f" & {ku:.3f} & {osrl_cell(task, algo, 'R', dr)}"
                f" & {osrl_cell(task, algo, 'C', d)}"
                f" & {cell(task, cplk, 'R', dr)}"
                f" & {cell(task, cplk, 'C', d, True)}")
    _lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "a25_draws.tex"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"wrote a25_draws.tex ({len(_A25ORDER) * 3} distinct selections; {_nsafe} of {_ntot} runs within budget)")

lines = []
for task in ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
             "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
             "pointgoal1_dsrl", "pointgoal2"):
    d = 0 if "velocity" in task else 2
    lim = LIMITS[task]
    row = NAMES[task]
    for src_key in ("cdt", "cdt_noaug", "cdt_a40"):
        e = OSRL2.get(task, {}).get(src_key, {})
        row += f" & {_mc(e, 'R', d)} & {_mc(e, 'C', d, lim)}"
    row += f" & {cell(task, 'calfilt_a40', 'R', d)} & {cell(task, 'calfilt_a40', 'C', d, True)}"
    # CPL on the same alpha=0.40 deployed selection, so both external learners
    # appear wherever CDT does rather than CDT alone.
    row += f" & {cell(task, 'cpl_gt_a40', 'R', d)} & {cell(task, 'cpl_gt_a40', 'C', d, True)}"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "compose_a40.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote compose_a40.tex")

# Main-text oracle table: six diagnostic rows, learned + two oracles
# SUPERSEDED (superseded by oracle_all.tex (all nine analysis tasks)); output in data/tables/_superseded/
# emit("oracle_main.tex", ["vawr_5seed", "xlab_oracle_step", "xlab_oracle_ctg"],
#      order=["halfcheetah_velocity", "ant_velocity", "hopper_velocity",
#             "cargoal2", "pointgoal1_dsrl", "pointgoal2"])

# Score-cost / score-return Spearman correlations (review response)
_corr_p = os.path.join(BASE, "data", "review_response", "score_correlations.json")
if os.path.exists(_corr_p):
    with open(_corr_p) as f:
        _corr = json.load(f)
    lines = []
    for task in ORDER:
        v = _corr[task]
        lines.append(f"{NAMES[task]} & {v['mean_rho_cost']:+.2f} & "
                     f"{v['mean_rho_return']:+.2f} & "
                     f"{v['rho_cost_return_of_costs']:+.2f} \\\\")
    with open(os.path.join(BASE, "data", "tables", "score_corr.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote score_corr.tex")

# a40 distinct-selection grid: three distinct certified selections per task
# (ordered by realized contamination), CDT and BC cost on identical selections.
# (cdt_key, bc_snap_key, realized_contamination)
# CarGoal2 / PointGoal1 / PointGoal2 read the REGENERATED a40new selections:
# their BC and CPL arms were harvested earlier but CDT was missing, so these
# rows had been showing CDT on the OLD selections beside BC/CPL on the new
# ones. The CDT block (33 cells) closed that; contamination values come from
# runs/selections/a40new_summary.json, ordered ascending as the grid expects.
_A40SEL = {
    "halfcheetah_velocity": [("cdt_a40_draw3", "bc_a40d3", 0.219),
                             ("cdt_a40_draw2", "bc_a40d2", 0.287),
                             ("cdt_a40", "bc_a40d1", 0.351)],
    "walker2d_velocity": [("cdt_a40", "bc_a40d1", 0.276),
                          ("cdt_a40_draw2", "bc_a40d2", 0.325),
                          ("cdt_a40selq55", "bc_a40selq55", 0.393)],
    "ant_velocity": [("cdt_a40selq85", "bc_a40selq85", 0.343),
                     ("cdt_a40_draw2", "bc_a40d2", 0.360),
                     ("cdt_a40", "bc_a40d1", 0.406)],
    "swimmer_velocity": [("cdt_a40", "bc_a40d1", 0.221),
                         ("cdt_a40selq80", "bc_a40selq80", 0.382),
                         ("cdt_a40selq75", "bc_a40selq75", 0.462)],
    "cargoal1_dsrl": [("cdt_a40", "bc_a40d1", 0.187),
                      ("cdt_a40_draw3", "bc_a40d3", 0.353),
                      ("cdt_a40_draw2", "bc_a40d2", 0.375)],
    "cargoal2": [("cdt_a40new_q75", "bc_a40new_q75", 0.278),
                 ("cdt_a40new_q70", "bc_a40new_q70", 0.310),
                 ("cdt_a40new_q65", "bc_a40new_q65", 0.349)],
    "pointgoal1_dsrl": [("cdt_a40new_q40", "bc_a40new_q40", 0.284),
                        ("cdt_a40new_q35", "bc_a40new_q35", 0.298),
                        ("cdt_a40new_q30", "bc_a40new_q30", 0.320)],
    "pointgoal2": [("cdt_a40new_q85", "bc_a40new_q85", 0.257),
                   ("cdt_a40new_q80", "bc_a40new_q80", 0.319),
                   ("cdt_a40new_q75", "bc_a40new_q75", 0.372)],
}
# CPL arm per grid cell. The deployed selection of each of the five
# originally-run tasks is one of its three grid cells and already carries CPL
# as cpl_gt_a40; the other ten cells were run by stage_cpl_a40grid.sh.
_A40CPL = {
    "halfcheetah_velocity": ["cpl_gt_a40d3", "cpl_gt_a40d2", "cpl_gt_a40"],
    "walker2d_velocity":    ["cpl_gt_a40", "cpl_gt_a40d2", "cpl_gt_a40selq55"],
    "ant_velocity":         ["cpl_gt_a40selq85", "cpl_gt_a40d2", "cpl_gt_a40"],
    "swimmer_velocity":     ["cpl_gt_a40", "cpl_gt_a40selq80", "cpl_gt_a40selq75"],
    "cargoal1_dsrl":        ["cpl_gt_a40", "cpl_gt_a40d3", "cpl_gt_a40d2"],
    "cargoal2":             ["cpl_gt_a40new_q75", "cpl_gt_a40new_q70", "cpl_gt_a40new_q65"],
    "pointgoal1_dsrl":      ["cpl_gt_a40new_q40", "cpl_gt_a40new_q35", "cpl_gt_a40new_q30"],
    "pointgoal2":           ["cpl_gt_a40new_q85", "cpl_gt_a40new_q80", "cpl_gt_a40new_q75"],
}
for _task, _rows in _v2_rows("a40").items():
    _A40SEL[_task] = [(f"cdt_a40new_{q}", f"bc_a40new_{q}", c) for q, c in _rows]
    _A40CPL[_task] = [f"cpl_gt_a40new_{q}" for q, _ in _rows]

_A40ORDER = ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
             "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
             "pointgoal1_dsrl", "pointgoal2")

def _a40grid(second_key_of, fname, label):
    """Grid rows: alpha-hat, CDT, and one further learner per selection."""
    lines = []
    for task in _A40ORDER:
        d = 0 if "velocity" in task else 2
        dr = 0 if "velocity" in task else 1
        row = NAMES[task]
        for i, (cdt_key, bc_key, ku) in enumerate(_A40SEL[task]):
            row += f" & {ku:.2f}"
            row += f" & {osrl_cell(task, cdt_key, 'R', dr)}"
            row += f" & {osrl_cell(task, cdt_key, 'C', d)}"
            k = second_key_of(task, i, bc_key)
            if k and SNAP.get(task, {}).get(k):
                row += f" & {cell(task, k, 'R', dr)}"
                row += f" & {cell(task, k, 'C', d, True)}"
            else:
                row += " & -- & --"
        lines.append(row + r" \\")
    with open(os.path.join(BASE, "data", "tables", fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {fname} ({label})")

_a40grid(lambda task, i, bc: _A40CPL[task][i], "a40_draws.tex",
         "distinct-selection grid, CDT and CPL")
# Clone-only contamination-tolerance grid: alpha-hat, R, C per selection. CDT
# is omitted because the adjacent composability grid already reports it.
lines = []
for task in _A40ORDER:
    d = 0 if "velocity" in task else 2
    dr = 0 if "velocity" in task else 1
    row = NAMES[task]
    for cdt_key, bc_key, ku in _A40SEL[task]:
        row += f" & {ku:.2f}"
        if SNAP.get(task, {}).get(bc_key):
            row += f" & {cell(task, bc_key, 'R', dr)}"
            row += f" & {cell(task, bc_key, 'C', d, True)}"
        else:
            row += " & -- & --"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "a40_draws_bc.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote a40_draws_bc.tex (clone-only contamination-tolerance grid)")

# BulletGym transfer table
_BUL = [("ballrun_b", "BallRun"), ("ballcircle_b", "BallCircle"),
        ("carcircle_b", "CarCircle"), ("carrun_b", "CarRun"),
        ("dronerun_b", "DroneRun")]
_BLIM = 10
lines = []
for task, name in _BUL:
    row = name
    _bcal = "calfilt_ltt" if task == "carrun_b" else "calfilt_csf"
    for cfgname in ("bc_all", "bcsafe", "vfilt_calsafe", _bcal):
        ent = SNAP.get(task, {}).get(cfgname, {})
        if not ent:
            row += " & -- & --"
            continue
        R = np.mean([e["R"] for e in ent.values()])
        C = np.mean([e["C"] for e in ent.values()])
        cs = [e["C"] for e in ent.values()]
        rtxt = f"{R:.0f}"
        ctxt = f"{C:.1f}"
        if len(cs) > 1:
            rtxt += f"\\,\\scriptsize$\\pm${ci([e['R'] for e in ent.values()]):.0f}"
            ctxt += f"\\,\\scriptsize$\\pm${ci(cs):.1f}"
        if C <= _BLIM:
            ctxt = f"\\textbf{{{ctxt}}}"
        row += f" & {rtxt} & {ctxt}"
    e = OSRL.get(task, {}).get("cdt", {})
    if e:
        R = np.mean([v["R"] for v in e.values()])
        C = np.mean([v["C"] for v in e.values()])
        ctxt = f"{C:.1f}\\,\\scriptsize$\\pm${ci([v['C'] for v in e.values()]):.1f}"
        if C <= _BLIM:
            ctxt = f"\\textbf{{{ctxt}}}"
        row += f" & {R:.0f}\\,\\scriptsize$\\pm${ci([v['R'] for v in e.values()]):.0f} & {ctxt}"
    else:
        row += " & -- & --"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "bullet_main.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote bullet_main.tex")

# Certified return-weighted operator grid (distinct alpha=0.25 selections + CarRun echo).
# Realized contamination of each operator selection, computed from its own membership file
# by runs/scripts/operator_selection_alpha.py. This column previously came from
# profile_by_H's H30 u_profile_mean, the contamination at that quantile AVERAGED OVER VALUE
# ENSEMBLE SEEDS, which is a different quantity from the realized contamination the
# composability grids report -- so the two tables disagreed on the same selections
# (Walker2d 0.24/0.24/0.27 against 0.139/0.177/0.248) while the prose called them the same.
_OPAH = json.load(open(os.path.join(BASE, "data", "operator_alpha_hat.json")))


def _alpha_hat(task, sel, fallback=None):
    e = _OPAH.get(task, {}).get(sel)
    return e["contamination"] if e else fallback


_OPSEL = [
    ("halfcheetah_velocity", "HalfCheetah", ["q85", "q80", "q75"]),
    ("walker2d_velocity", "Walker2d", ["q75", "q70", "q65"]),   # certifies at .25 in the regenerated cohort
    ("cargoal1_dsrl", "CarGoal1", ["q85", "q80", "q75"]),
    ("pointgoal1_dsrl", "PointGoal1", ["q85", "q70", "q65"]),
    # CarRun echo now runs on the REGENERATED certified selection: its original
    # selection (ahat=0.138, ~half the pool) was a lost scratchpad h5 that no
    # current (V seed, quantile) reproduces, so it was re-derived by the same
    # certification procedure and yields q85 at ahat=0.051 over 98 trajectories.
    ("carrun_b", "CarRun (echo)", ["echonew"]),
]
_ECHO = {"carrun_b": 0.051}   # regenerated selection; see note in _OPSEL
lines = []
for task, name, sels in _OPSEL:
    lim = LIMITS.get(task, 10)
    rd = 0 if ("velocity" in task or task == "carrun_b") else 1
    for sel in sels:
        ku = _alpha_hat(task, sel, _ECHO.get(task))
        # three decimals: at two, CarGoal1's out-of-specification 0.251 and Walker2d's
        # in-specification 0.248 both print as 0.25 and the 39-of-40 partition is unreadable
        row = f"{name} & {ku:.3f}"
        for var in ("wbc1", "wbc", "wbc3", "toph"):
            ent = SNAP.get(task, {}).get(f"{var}_{sel}", {})
            if not ent:
                row += " & -- & --"
                continue
            R = np.mean([e["R"] for e in ent.values()])
            C = np.mean([e["C"] for e in ent.values()])
            rtxt = f"{R:.{rd}f}\\,\\scriptsize$\\pm${ci([e['R'] for e in ent.values()]):.{rd}f}"
            ctxt = f"{C:.1f}\\,\\scriptsize$\\pm${ci([e['C'] for e in ent.values()]):.1f}"
            if C <= lim:
                ctxt = f"\\textbf{{{ctxt}}}"
            row += f" & {rtxt} & {ctxt}"
        lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "operator_grid.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote operator_grid.tex")


# Conditional false-certification table (T1.4; from archived 2000-draw stats)
import json as _json
_gv = _json.load(open(os.path.join(BASE, "data", "review_response", "guarantee_stats_2000.json")))
_bv = _json.load(open(os.path.join(BASE, "data", "review_response", "bullet_guarantee_2000.json")))
lines = []
_seen = set()
for src_d in (_gv, _bv):
    for task, td in src_d.items():
        if task in _seen:   # _gv already covers Bullet; _bv is a stale duplicate
            continue
        _seen.add(task)
        cells = [s["200"] for s in td["seeds"].values() if "200" in s]
        if not cells:
            continue
        cr = np.mean([c["cert_rate"] for c in cells])
        un = np.mean([c["false_cert_rate_uncond"] for c in cells])
        cvs = [c["cond_viol_rate"] for c in cells if c["cond_viol_rate"] is not None]
        ncert = int(round(cr * 2000 * len(cells)))
        if cvs:
            cv = f"{np.mean(cvs):.2f}"
        else:
            cv = "--"
        _BNAMES = {"ballrun_b": "BallRun", "ballcircle_b": "BallCircle",
           "carcircle_b": "CarCircle", "carrun_b": "CarRun",
           "dronerun_b": "DroneRun"}
        lines.append(f"{NAMES.get(task, _BNAMES.get(task, task))} & {cr:.2f} & {un:.3f} & {cv} & {ncert} " + r"\\")
with open(os.path.join(BASE, "data", "tables", "cond_viol.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote cond_viol.tex")

# CDT cost-target sweep tables (T1.1): DSRL (targets 5/10/15/budget) + Bullet (2/5/10)
_CDTS = json.load(open(os.path.join(BASE, "data", "review_response", "cdt_target_sweep.json")))
_BN = {"ballrun_b": "BallRun", "ballcircle_b": "BallCircle", "carcircle_b": "CarCircle",
       "carrun_b": "CarRun", "dronerun_b": "DroneRun"}


def _cdt_cell(d, tg, lim, dg):
    if tg not in d:
        return " & --"
    C = d[tg]["C"]
    txt = f"{d[tg]['R']:.{dg}f}/{C:.1f}"
    return " & " + (f"\\textbf{{{txt}}}" if C <= lim else txt)


lines = []
for task in ORDER:
    d = _CDTS.get(task)
    if not d:
        continue
    lim = LIMITS[task]
    dg = 0 if "velocity" in task else (1 if "circle" in task else 2)
    row = NAMES[task]
    for tg in ("5", "10", "15"):
        row += _cdt_cell(d, tg, lim, dg)
    row += _cdt_cell(d, str(int(lim)), lim, dg)
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "cdt_targets.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote cdt_targets.tex")

lines = []
for task in ("ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b"):
    d = _CDTS.get(task)
    if not d:
        continue
    row = _BN[task]
    for tg in ("2", "5", "10"):
        row += _cdt_cell(d, tg, 10, 1)
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "cdt_targets_bullet.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote cdt_targets_bullet.tex")
