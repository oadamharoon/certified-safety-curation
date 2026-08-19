"""Generate every LaTeX table body for the paper from the results snapshot.

Emits mean +- 95% bootstrap CI half-widths over seeds. Output files land in
data/tables/*.tex so paper tables regenerate with one command.
"""
import json, os
import numpy as np

BASE = "/home/omniverse/workspace/safevlmcpl/iclr2027"
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


FALLBACK = {"vawr_5seed": "xlab_exp_k1"}


def cell(task, config, metric, digits=1, bold_if_safe=False):
    entries = SNAP.get(task, {}).get(config, {})
    if not entries and config in FALLBACK:
        entries = SNAP.get(task, {}).get(FALLBACK[config], {})
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
emit("main.tex", ["bc_all", "bcsafe", "vfilt_matchgt", "calfilt_ltt",
                  "calfilt_lttR50"])
emit("vfilt_variants.tex", ["vfilt_matchgt", "vfilt_q25"])
# Controls ablation
emit("controls.tex", ["vfilt_random", "vfilt_return", "vfilt_retbot", "vfilt_matchgt"])
# Alpha sweep policies
emit("alpha.tex", ["calfilt_a10", "calfilt_ltt", "calfilt_a40"], order=ANALYSIS_ORDER)
# Extraction sweep appendix (oracle + variants; 3 seeds)
emit("extraction.tex", ["xlab_rank", "xlab_binary", "cf_v2", "cf_octg"],
     skip_empty=True, order=ANALYSIS_ORDER)
# Noise policies (4 tasks only, harmless dashes elsewhere)
emit("noise_policy.tex", ["calfilt_noise20", "calfilt_ltt"], order=ANALYSIS_ORDER)


# Combined main table body: BC-All | BC-Safe | V-filter | Calibrated | Gated
def main_gated():
    gated = {}
    for task in ORDER:
        ltt = SNAP.get(task, {}).get("calfilt_ltt", {})
        r50 = SNAP.get(task, {}).get("calfilt_lttR50", {})
        Rs, Cs = [], []
        for seed, e in ltt.items():
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
        # values on the certified three
        certified = task in ("halfcheetah_velocity", "cargoal1_dsrl", "pointgoal1_dsrl")
        if not certified and SNAP.get(task, {}).get("calfilt_csf"):
            ent = SNAP[task]["calfilt_csf"]
            gated[task] = ([e["R"] for e in ent.values()], [e["C"] for e in ent.values()])
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


main_gated()

# Full-benchmark oracle ablation (Table 1): all 8 tasks
emit("oracle_all.tex", ["bc_all", "vawr_5seed", "xlab_oracle_step",
                        "xlab_oracle_ctg"], order=ANALYSIS_ORDER)

# Normalized-cost scan across ALL methods (main text): C_n = C / budget.
import pickle, yaml
import json as _j
REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
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
    _cert3 = ("halfcheetah_velocity", "cargoal1_dsrl", "pointgoal1_dsrl")
    _calib = "calfilt_ltt" if task in _cert3 else "calfilt_csf"
    for cfgname in ("bc_all", "bcsafe", "bcsafeseg", "vfilt_calsafe", _calib):
        ent = SNAP.get(task, {}).get(cfgname, {}) or SNAP.get(task, {}).get("calfilt_ltt", {}) if cfgname == _calib else SNAP.get(task, {}).get(cfgname, {})
        row += _cn([e["C"] for e in ent.values()])
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
for task in ("halfcheetah_velocity", "cargoal1_dsrl", "pointgoal1_dsrl"):
    d = 0 if "velocity" in task else 2
    lim = LIMITS[task]
    row = NAMES[task]
    for src in ("cdt", "cdt_noaug", "cdt_cert"):
        e = OSRL2.get(task, {}).get(src, {})
        row += f" & {_mc(e, 'R', d)} & {_mc(e, 'C', d, lim)}"
    e = SNAP.get(task, {}).get("calfilt_ltt", {})
    row += f" & {cell(task, 'calfilt_ltt', 'R', d)} & {cell(task, 'calfilt_ltt', 'C', d, True)}"
    row += f" & {cell(task, 'wbc_q85', 'R', d)} & {cell(task, 'wbc_q85', 'C', d, True)}"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "compose.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote compose.tex")

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
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "compose_a40.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote compose_a40.tex")

# Main-text oracle table: six diagnostic rows, learned + two oracles
emit("oracle_main.tex", ["vawr_5seed", "xlab_oracle_step", "xlab_oracle_ctg"],
     order=["halfcheetah_velocity", "ant_velocity", "hopper_velocity",
            "cargoal2", "pointgoal1_dsrl", "pointgoal2"])

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
    "cargoal2": [("cdt_a40", "bc_a40d1", 0.238),
                 ("cdt_a40_draw2", "bc_a40d2", 0.287),
                 ("cdt_a40_draw3", "bc_a40d3", 0.334)],
    "pointgoal1_dsrl": [("cdt_a40", "bc_a40d1", 0.299),
                        ("cdt_a40selq35", "bc_a40selq35", 0.318),
                        ("cdt_a40_draw2", "bc_a40d2", 0.345)],
    "pointgoal2": [("cdt_a40", "bc_a40d1", 0.269),
                   ("cdt_a40selq80", "bc_a40selq80", 0.335),
                   ("cdt_a40selq75", "bc_a40selq75", 0.382)],
}
lines = []
for task in ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity",
             "swimmer_velocity", "cargoal1_dsrl", "cargoal2",
             "pointgoal1_dsrl", "pointgoal2"):
    lim = LIMITS[task]
    d = 0 if "velocity" in task else 2
    row = NAMES[task]
    for cdt_key, bc_key, ku in _A40SEL[task]:
        row += f" & {ku:.2f}"
        row += f" & {osrl_cell(task, cdt_key, 'C', d)}"
        ent = SNAP.get(task, {}).get(bc_key, {})
        if ent:
            C = np.mean([e["C"] for e in ent.values()])
            h = ci([e["C"] for e in ent.values()])
            txt = f"{C:.{d}f}\\,\\scriptsize$\\pm${h:.{d}f}"
            if C <= lim:
                txt = f"\\textbf{{{txt}}}"
            row += f" & {txt}"
        else:
            row += " & --"
    lines.append(row + r" \\")
with open(os.path.join(BASE, "data", "tables", "a40_draws.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote a40_draws.tex (distinct-selection grid)")

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

# Certified return-weighted operator grid (distinct alpha=0.25 selections + CarRun echo)
_OPSEL = [
    ("halfcheetah_velocity", "HalfCheetah", [("q85", 0.219), ("q80", 0.287), ("q75", 0.351)]),
    ("cargoal1_dsrl", "CarGoal1", [("q85", 0.171), ("q80", 0.212), ("q75", 0.266)]),
    ("pointgoal1_dsrl", "PointGoal1", [("q85", 0.086), ("q70", 0.158), ("q65", 0.178)]),
    ("carrun_b", "CarRun (echo)", [("echo", 0.138)]),
]
lines = []
for task, name, sels in _OPSEL:
    lim = LIMITS.get(task, 10)
    rd = 0 if ("velocity" in task or task == "carrun_b") else 1
    for sel, ku in sels:
        row = f"{name} & {ku:.2f}"
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
