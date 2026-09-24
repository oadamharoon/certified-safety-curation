"""Source trace for every number in the paper's prose that is NOT a cell of a generated table.

audit_prose_numbers.py checks that a literal is supported by SOME value of the task its sentence
names. That is task-scoped existence matching, and it cannot catch a derived statistic that was
never recomputed: a Spearman, an R^2, a slope, a count over a set, a range. Those are exactly the
class that goes stale, because nothing regenerates them when the cohort moves. This file binds each
such number to the one expression that produces it, so a number that drifts from its source fails
here. Every check names the sentence it guards.

Usage: python scripts/verify_constants.py   (exit 1 on any mismatch)
"""
import json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(BASE)
L = lambda p: json.load(open(f"{BASE}/{p}"))


def rnd(x, k):
    """Round half away from zero, the convention the prose uses.

    Python's round() is banker's rounding, so round(0.0135, 3) gives 0.013 where the paper
    writes 0.014. Using it inside a checker makes the checker disagree with correct prose.
    """
    q = Decimal(1).scaleb(-k)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))

R = lambda p: json.load(open(f"{W}/{p}"))

G2K = L("data/review_response/guarantee_stats_2000.json")
PROF = L("data/review_response/profile_by_H.json")["arms"]
MVY = {r["task"]: r["margin"] for r in L("data/review_response/margin_vs_yield.json")}
LC, LCI = L("data/review_response/label_complexity.json"), L("data/review_response/label_complexity_ci.json")
MEXP = L("data/review_response/margin_expand.json")
STAB = L("data/review_response/selection_stability.json")
DIS = L("data/review_response/disagreement_matched.json")
OPAH = L("data/operator_alpha_hat.json")
CTRI = L("data/certificate_triple.json")
CTRL = L("data/control_selection_stats.json")
CAUD = L("data/calibration_audit.json")
CRT = L("data/cert_rate_theory.json")
LBP = L("data/label_budget_plan.json")
HSF = L("data/h_seed_failures.json")
MCC = L("data/mean_cost_certificate.json")
PCT = L("data/policy_certificate.json")
MTP = L("data/review_response/mt_procedures.json")
ROB = L("data/robustness_stats.json")
BFR = L("data/boltz_flip_rates.json")
OBS = L("data/obstacle2_stats.json")
_o2jac = [float(np.mean(OBS[t]["jaccard"])) for t in OBS if isinstance(OBS[t], dict)]
_o2seed = [float(v) for t in OBS if isinstance(OBS[t], dict) for v in OBS[t]["jaccard"]]


def _norm_main():
    """Read the GENERATED normalized-cost table: C_n = C/budget, safe when C_n <= 1.
    Reading the artifact the paper prints keeps this trace independent of the arm keys."""
    import re as _re
    r = {}
    for ln in open(f"{BASE}/data/tables/normalized_main.tex"):
        p = [_re.sub(r"\\textbf\{|\}|\\\\", "", x).strip() for x in ln.split("&")]
        if len(p) >= 10:
            r[p[0]] = {"bcsafe": float(p[2]), "vfilt": float(p[4]), "cdt": float(p[6])}
    return r


_NM = _norm_main()
_vb = [v for v in _NM.values() if v["bcsafe"] <= 1 and v["vfilt"] <= 1]
_vb_both = len(_vb)
_vb_low = sum(v["vfilt"] < v["bcsafe"] for v in _vb)
_vb_mn_ours = float(np.mean([v["vfilt"] for v in _vb]))
_vb_mn_bcs = float(np.mean([v["bcsafe"] for v in _vb]))
_cdt_matched = sum(v["cdt"] <= 1 for v in _NM.values())
# CPQ and COptiDICE columns of the same generated table, plus CPL and budget-matched CDT.
def _fc_mean(task, n=200):
    sd = G2K[task]["seeds"]
    return float(np.mean([sd[k][str(n)]["false_cert_rate_uncond"] for k in sd]))


_fc200 = tuple(rnd(_fc_mean(t), 3) for t in
               ("ant_velocity", "pointgoal2", "hopper_velocity"))


def _labels_only():
    """Safe-task count per calibration budget, read off the generated table."""
    out = {}
    for _i, _n in enumerate((50, 100, 200, 400)):
        _c = 0
        for _ln in open(f"{BASE}/data/tables/labelsonly.tex"):
            _p = [x.strip() for x in _ln.split("&")]
            if len(_p) >= 9 and "textbf" in _p[2 + 2 * _i]:
                _c += 1
        out[_n] = _c
    return out


_lo = _labels_only()
_optbased = [_cdt_matched]
for _col, _nm in ((7, "cpq"), (8, "copt"), (9, "cpl")):
    import re as _re2
    _n = 0
    for _ln in open(f"{BASE}/data/tables/normalized_main.tex"):
        _p = [_re2.sub(r"\\textbf\{|\}|\\\\", "", x).strip() for x in _ln.split("&")]
        if len(_p) >= 10 and float(_p[_col]) <= 1:
            _n += 1
    _optbased.append(_n)


def _pareto_counts():
    """Safe-task count per method, from the same getters the Pareto figure uses."""
    import subprocess
    out = subprocess.run([sys.executable, f"{BASE}/scripts/pareto_figure.py"],
                         capture_output=True, text=True).stdout
    d = {}
    for m in re.finditer(r"^(\S.*?)\s+x=\s*[-\d.]+\s+safe=(\d+)/15", out, re.M):
        d[m.group(1).strip()] = int(m.group(2))
    if d:
        return d
    # The figure script needs the trajectory pickles. Where they are not available, read
    # the counts the last run of it archived.
    return {k: v["safe"] for k, v in L("data/pareto_counts.json").items()}


_pareto = _pareto_counts()
_filterarms = [_pareto[k] for k in ("BC-Safe", "V-filter", "Calibrated",
                                    "Labels-only 200", "Labels-only 400", "Labels-split")]

BG = L("data/review_response/bullet_guarantee_2000.json")
BAC = L("data/review_response/bullet_alpha_curve.json")
SNAP, OSRL = L("data/results_snapshot.json"), L("data/osrl_results.json")
_hopR50 = [v["C"] for v in SNAP["hopper_velocity"]["calfilt_lttR50"].values()]
def _extraction_col(task, col):
    for ln in open(f"{BASE}/data/tables/extraction.tex"):
        p = [c.strip() for c in ln.split("&")]
        if p and p[0] == task:
            m = re.search(r"-?\d+\.?\d*", p[col])
            return float(m.group(0))
    raise KeyError(task)


_actcond = tuple(rnd(_extraction_col(t, 8), 1)
                 for t in ("PointGoal1", "CarGoal2", "PointGoal2"))
_ALL20 = ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
          "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2",
          "pointbutton1", "pointbutton2", "carbutton1_t3", "carbutton2", "pointcircle1",
          "pointcircle2", "ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b")


def _cert_seeds(t):
    return sum(bool(v.get("meta", {}).get("certified"))
               for v in SNAP.get(t, {}).get("calfilt_csf", {}).values())


def _cdtfull(t):
    lim = 20 if "velocity" in t else 25
    return float(np.mean([v["C"] for v in OSRL[t]["cdt"].values()])) / lim


def _poolsafe():
    """Pool safe fraction per analysis task; the paper's own "Hopper ... 11 percent" fixes
    the semantics of base_rate, and k_ablation and aggregation_ablation agree on every task."""
    d = {}
    for r in KAB:
        d.setdefault(r["task"], []).append(r["base_rate"])
    return {t: float(np.mean(v)) for t, v in d.items()}


def _controls_counts():
    """(tasks where return beats random on reward, tasks where it costs more)."""
    import re as _r
    def _n(c):
        c = _r.sub(r"\\textbf\{|\}|\\,\\scriptsize\$\\pm\$.*", "", c)
        m = _r.search(r"-?\d+\.?\d*", c)
        return float(m.group(0)) if m else None
    rew = cost = 0
    for ln in open(f"{BASE}/data/tables/controls.tex"):
        if "&" not in ln:
            continue
        p_ = [c.strip() for c in ln.split("&")]
        rr, rc, tr, tc = _n(p_[1]), _n(p_[2]), _n(p_[3]), _n(p_[4])
        if None in (rr, rc, tr, tc):
            continue
        rew += tr > rr
        cost += tc > rc
    return (rew, cost)


def _hc_deep():
    """(R, C) of CDT on HalfCheetah's deepest distinct certified selection, from Table 11."""
    for ln in open(f"{BASE}/data/tables/a25_draws.tex"):
        if ln.startswith("HalfCheetah"):
            col = [c.strip() for c in ln.split("&")]
            import re as _r
            g = lambda c: float(_r.search(r"-?\d+\.?\d*", _r.sub(r"\\textbf\{|\}|\\,\\scriptsize.*", "", c)).group(0))
            return (g(col[12]), g(col[13]))
    raise KeyError("HalfCheetah")


def _label_share():
    """200 calibration labels as a percentage of each main-study pool, min and max."""
    out = []
    for t, arms in SNAP.items():
        if not isinstance(arms, dict) or "calfilt_csf" not in arms:
            continue
        if t.endswith("_b"):          # Bullet suite is quoted separately
            continue
        for v in arms["calfilt_csf"].values():
            m = v.get("meta", {})
            if m.get("n_kept") and m.get("kept_frac"):
                out.append(200.0 / (m["n_kept"] / m["kept_frac"]) * 100)
                break
    return (rnd(min(out), 1), rnd(max(out), 1))


def _pt_within():
    """Per-transition arms within budget. The trajectory-scale aggregators (xagg_*) and the
    BC-Safe-seeded control are excluded: they are not per-transition reweighting."""
    out = []
    for t in ("pointgoal1_dsrl", "cargoal2", "pointgoal2"):
        n = 0
        for a, cells in SNAP[t].items():
            if not a.startswith(("xlab_", "vawr")) or a == "vawr_from_bcsafe":
                continue
            if not isinstance(cells, dict) or not cells:
                continue
            if np.mean([v["C"] for v in cells.values()]) <= 25:
                n += 1
        out.append(n)
    return tuple(out)


def _a40_deployed():
    """(clone-safe, CDT-certified-safe) rows of the generated alpha=0.40 composability table."""
    cl = cd = 0
    for ln in open(f"{BASE}/data/tables/compose_a40.tex"):
        if "&" not in ln:
            continue
        col = [c.strip() for c in ln.split("&")]
        cd += "textbf" in col[6]
        cl += "textbf" in col[8]
    return (cl, cd)


def _tier2_counts():
    ts = [t for t in SNAP if isinstance(SNAP[t], dict) and "calfilt_tier2" in SNAP[t]]
    up = sum(np.mean([v["C"] for v in SNAP[t]["calfilt_tier2"].values()])
             > np.mean([v["C"] for v in SNAP[t]["calfilt_csf"].values()]) for t in ts)
    return (len(ts), int(up))


def _scorecorr_rows():
    out = []
    for ln in open(f"{BASE}/data/tables/score_corr.tex"):
        p_ = [c.strip().replace("$", "") for c in ln.split("&")]
        if len(p_) < 4:
            continue
        try:
            out.append((float(p_[1]), float(p_[2]), float(p_[3].replace("\\\\", "").strip())))
        except ValueError:
            pass
    return out


def _scorecorr_r():
    rows = _scorecorr_rows()
    xs = [c for _, _, c in rows]
    ys = [-b for _, b, _ in rows]
    return float(np.corrcoef(xs, ys)[0, 1])


def _scorecorr_neg():
    rows = _scorecorr_rows()
    return (sum(b < 0 for _, b, _ in rows), len(rows))


def _a25_counts():
    """(selections, CDT-safe, CPL-safe) from the generated alpha=0.25 draws table."""
    tot = c = p_ = 0
    for ln in open(f"{BASE}/data/tables/a25_draws.tex"):
        if "&" not in ln:
            continue
        col = [x.strip() for x in ln.split("&")]
        for i in range(3):
            b = 1 + 5 * i
            if b + 4 >= len(col):
                continue
            tot += 1
            c += "textbf" in col[b + 2]
            p_ += "textbf" in col[b + 4]
    return tot, c, p_


_a25 = _a25_counts()


def _noaug():
    """(velocity safe, navigation unsafe, min nav cost, max nav cost) for full-data CDT
    without frontier augmentation."""
    vel = nav = 0
    costs = []
    for t, arms in OSRL.items():
        if "cdt_noaug" not in arms:
            continue
        c = float(np.mean([v["C"] for v in arms["cdt_noaug"].values()]))
        if "velocity" in t:
            vel += c <= 20
        else:
            nav += c > 25
            costs.append(c)
    return (vel, nav, rnd(min(costs), 1), rnd(max(costs), 1))


def _hyp_pmf(N, K, n, k):
    import math
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def _hyp_cdf(N, K, n, k):
    return sum(_hyp_pmf(N, K, n, i) for i in range(0, k + 1))


def _superunif_violations(delta=0.1):
    """Lemma 2: with p = F_hyp(k; Nj, Uj*, mj) and k ~ Hyp(Nj, Uj, mj), the null Uj >= Uj*
    must give Pr[p <= delta] <= delta. Counts any (Nj, Uj*, Uj, mj) where it does not."""
    bad = 0
    for Nj in (20, 40):
        for Ujstar in range(1, Nj):
            for Uj in range(Ujstar, Nj + 1):
                for mj in (5, 10):
                    if mj > Nj:
                        continue
                    pr = sum(_hyp_pmf(Nj, Uj, mj, k) for k in range(mj + 1)
                             if _hyp_cdf(Nj, Ujstar, mj, k) <= delta + 1e-15)
                    if pr > delta + 1e-12:
                        bad += 1
    return bad


def _falsecert_violations(alpha=0.25, delta=0.1):
    """Propositions 1 and 3: on a selection whose unsafe fraction exceeds alpha, the closed-form
    certification probability must not exceed delta."""
    import math
    bad = 0
    for N, N1, n in ((400, 60, 50), (300, 45, 60), (200, 40, 40)):
        kstar = math.floor(alpha * N1) + 1
        for U1 in range(N1 + 1):
            if U1 / N1 <= alpha:
                continue
            tot = 0.0
            for m in range(max(0, n - (N - N1)), min(n, N1) + 1):
                pm = _hyp_pmf(N, N1, n, m)
                if pm <= 0:
                    continue
                tot += pm * sum(_hyp_pmf(N1, U1, m, k) for k in range(m + 1)
                                if _hyp_cdf(N1, kstar, m, k) <= delta + 1e-15)
            if tot > delta + 1e-12:
                bad += 1
    return bad


def _prop4():
    """(fixed-point mismatches, weight-ratio mismatches, pbar-sufficiency mismatches)."""
    import math
    bad = [0, 0, 0]
    for q in (0.05, 0.2, 0.5, 0.8):
        for ps in (0.1, 0.3, 0.7, 0.9):
            pbar = q + (1 - q) * ps
            for W in (1.0, 5.0, 100.0):
                pi = pbar * W / (pbar * W + (1 - pbar))          # maximiser of sum c_a log pi_a
                if abs(pi - pbar * W / (pbar * W + 1 - pbar)) > 1e-12:
                    bad[0] += 1
            for T in (10, 50, 200, 1000):
                for eps in (0.01, 0.1):
                    r = (1 - eps) ** (1.0 / T)
                    exact = (1 - pbar) / pbar * r / (1 - r)
                    paper = (1 - pbar) / pbar * ((1 - eps) ** (-1.0 / T) - 1) ** -1
                    if abs(exact - paper) / exact > 1e-9:
                        bad[1] += 1
                    # attained all-safe probability at that ratio is exactly 1 - eps
                    pi = pbar * exact / (pbar * exact + 1 - pbar)
                    if abs(pi ** T - (1 - eps)) > 1e-9:
                        bad[1] += 1
            # (q, p_s) pairs sharing pbar must give the same answer: q is not separately visible
            for q2 in (0.05, 0.2, 0.5, 0.8):
                if q2 == q or q2 >= 1:
                    continue
                ps2 = (pbar - q2) / (1 - q2)
                if not 0 <= ps2 <= 1:
                    continue
                if abs((q2 + (1 - q2) * ps2) - pbar) > 1e-12:
                    bad[2] += 1
    return tuple(bad)


def _resample_bound(kappa, alpha):
    e = float(np.exp(2 * kappa))
    return e * alpha / (e * alpha + 1 - alpha)


_NAVI = ("cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2")


def _sweep(t):
    lim = 20 if "velocity" in t else 25
    return [float(np.mean([v["C"] for v in SNAP[t][a].values()])) / lim
            for a in ("calfilt_a10", "calfilt_ltt", "calfilt_a40")]


def _bcs(t):
    lim = 20 if "velocity" in t else 25
    return float(np.mean([v["C"] for v in SNAP[t]["bcsafe"].values()])) / lim


INTERP = L("data/interpretability.json")
ECON = L("data/e_contamination.json")
KAB = L("data/k_ablation.json")
_k_bytask = {}
for _r in KAB:
    _k_bytask.setdefault(_r["task"], []).append(_r)
_k_gap = max(abs(np.mean([r["precision_K1_mean"] for r in rs])
                 - np.mean([r["precision_K3"] for r in rs])) for rs in _k_bytask.values())
_k_spread = max(np.mean([max(r["precision_K1_members"]) - min(r["precision_K1_members"])
                         for r in rs]) for rs in _k_bytask.values())
CARD = L("data/cardinal_control.json")
_card_d = [CARD["per_task"][t]["cardinal"]["kept_unsafe"] - CARD["per_task"][t]["binary"]["kept_unsafe"]
           for t in CARD["per_task"]]
_card_less = sum(x > 0 for x in _card_d)
_card_gap = float(np.mean(_card_d))
_AGG_TASKS = ["halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
              "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2"]


def _aggC(arm, task="hopper_velocity"):
    return float(np.mean([v["C"] for v in SNAP[task][arm].values()]))


_agg_safe = tuple(
    sum(_aggC(a, t) <= (20 if "velocity" in t else 25) for t in _AGG_TASKS)
    for a in ("xagg_min", "xagg_p10", "vfilt_calsafe"))

V2 = R("runs/selections/v2_summary.json")

LIM = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
       "hopper_velocity": 20, "swimmer_velocity": 20, "cargoal1_dsrl": 25, "cargoal2": 25,
       "pointgoal1_dsrl": 25, "pointgoal2": 25,
       "carrun_b": 10, "ballrun_b": 10, "ballcircle_b": 10, "carcircle_b": 10, "dronerun_b": 10}
NINE = ("halfcheetah_velocity", "walker2d_velocity", "ant_velocity", "hopper_velocity",
        "swimmer_velocity", "cargoal1_dsrl", "cargoal2", "pointgoal1_dsrl", "pointgoal2")

# ---- guarantee validation: the worst unconditional false-certification cell over all cells
_w = max(((c["false_cert_rate_uncond"], t, s, n, c["false_cert_cp95"])
          for t, e in G2K.items() for s, sd in e["seeds"].items() for n, c in sd.items()),
         key=lambda x: x[0])
worst_rate, worst_task, worst_seed, worst_n, worst_cp = _w
_pg1 = G2K["pointgoal1_dsrl"]["seeds"]
pg1_200 = np.mean([sd["200"]["cert_rate"] for sd in _pg1.values()])
pg1_400 = np.mean([sd["400"]["cert_rate"] for sd in _pg1.values()])
# rate tracks margin across the fourteen validated tasks, at the deployed n = 200
_rt = [(MVY[t], np.mean([sd["200"]["cert_rate"] for sd in G2K[t]["seeds"].values()]))
       for t in G2K if t in MVY and G2K[t]["seeds"]]
rho_tasks = spearmanr([a for a, _ in _rt], [b for _, b in _rt])[0]

# ---- segment length: the 27 task-length cells (nine tasks x three H arms)
_cells = []
for arm in ("H10", "H30", "H50"):
    g = L(f"data/review_response/guarantee_{arm}_2000.json")
    for t, e in g.items():
        rs = [sd["200"]["cert_rate"] for sd in e["seeds"].values()]
        if not rs:
            continue
        _cells.append((arm, t, float(np.mean(rs)), PROF[arm][t]["margin_mean"],
                       max(sd["200"]["false_cert_rate_uncond"] for sd in e["seeds"].values())))
armrate = {a: float(np.mean([c[2] for c in _cells if c[0] == a])) for a in ("H10", "H30", "H50")}
armmarg = {a: float(np.mean([c[3] for c in _cells if c[0] == a])) for a in ("H10", "H30", "H50")}
armworst = {a: max(c[4] for c in _cells if c[0] == a) for a in ("H10", "H30", "H50")}
rho_cells = spearmanr([c[3] for c in _cells], [c[2] for c in _cells])[0]
neg = [c[2] for c in _cells if c[3] < 0]
pos = [c[2] for c in _cells if c[3] >= 0]
pg1_h10 = np.mean([sd["200"]["cert_rate"]
                   for sd in L("data/review_response/guarantee_H10_2000.json")["pointgoal1_dsrl"]["seeds"].values()])


def _safe_count(suffix, arm, seeds=None):
    n = 0
    for t in NINE:
        e = SNAP.get(t + suffix, {}).get(arm, {})
        cs = [v["C"] for s, v in e.items() if seeds is None or s in seeds]
        if cs and np.mean(cs) <= LIM[t]:
            n += 1
    return n


h_safe = {"H30": _safe_count("", "calfilt_csf", {"0", "1", "2"}),
          "H10": _safe_count("_h10", "calfilt_hcsf"),
          "H50": _safe_count("_h50", "calfilt_hcsf")}

# ---- composability grids, counted from the stores the tables read
_src = open(f"{BASE}/scripts/make_tables.py").read()
_ns = {}
exec(compile(_src[:_src.index("def _a40grid")], "make_tables.py", "exec"), _ns)
A25, A40, A40C = _ns["_A25SEL"], _ns["_A40SEL"], _ns["_A40CPL"]


def _runs(task, key, store):
    return [v["C"] for v in store.get(task, {}).get(key, {}).values()]


a25_cdt = [(t, r[0], c) for t, rows in A25.items() for r in rows for c in _runs(t, r[0], OSRL)]
a25_cpl = [(t, r[3], c) for t, rows in A25.items() for r in rows for c in _runs(t, r[3], SNAP)]
a25_cdt_ok = sum(c <= LIM[t] for t, _, c in a25_cdt)
a25_cpl_ok = sum(c <= LIM[t] for t, _, c in a25_cpl)
a25_cpl_fail_tasks = {t for t, _, c in a25_cpl if c > LIM[t]}
a40_sel = [(t, i, r) for t, rows in A40.items() for i, r in enumerate(rows)]
def _sel_ok(t, k, store):
    cs = _runs(t, k, store)
    return bool(cs) and float(np.mean(cs)) <= LIM[t]   # an explicit test: a mean of 0.0 is safe


a40_cdt_sel = sum(_sel_ok(t, r[0], OSRL) for t, i, r in a40_sel)
a40_cpl_sel = sum(_sel_ok(t, A40C[t][i], SNAP) for t, i, r in a40_sel)
a40_bc_sel = sum(_sel_ok(t, r[1], SNAP) for t, i, r in a40_sel)

# ---- operator grid
_OPSEL = [("halfcheetah_velocity", ["q85", "q80", "q75"]), ("walker2d_velocity", ["q75", "q70", "q65"]),
          ("cargoal1_dsrl", ["q85", "q80", "q75"]), ("pointgoal1_dsrl", ["q85", "q70", "q65"]),
          ("carrun_b", ["echonew"])]
op_in = op_in_ok = op_out = op_out_ok = 0
for t, sels in _OPSEL:
    for sel in sels:
        ah = OPAH[t][sel]["contamination"]
        for var in ("wbc1", "wbc", "wbc3", "toph"):
            cs = [v["C"] for v in SNAP.get(t, {}).get(f"{var}_{sel}", {}).values()]
            if not cs:
                continue
            ok = np.mean(cs) <= LIM[t]
            if ah <= 0.25:
                op_in += 1; op_in_ok += ok
            else:
                op_out += 1; op_out_ok += ok
_hc75 = np.mean([v["C"] for v in SNAP["halfcheetah_velocity"]["wbc3_q75"].values()])
_hc85 = SNAP["halfcheetah_velocity"]["wbc_q85"]
op_deployed_R = np.mean([v["R"] for v in _hc85.values()])
op_deployed_C = np.mean([v["C"] for v in _hc85.values()])

# ---- headline safe counts on the fifteen DSRL tasks (the gated arm is Table 1's column)
ORDER, LIMITS, GATED = _ns["ORDER"], _ns["LIMITS"], _ns["GATED"]
NAMES = _ns["NAMES"]
gated_safe = sum(np.mean(GATED[t][1]) <= LIMITS[t] for t in ORDER)
arm_safe = lambda a: sum(1 for t in ORDER
                         if SNAP.get(t, {}).get(a) and
                         np.mean([v["C"] for v in SNAP[t][a].values()]) <= LIMITS[t])
n_certified = sum(any(e.get("meta", {}).get("certified") for e in SNAP[t]["calfilt_csf"].values())
                  for t in ORDER)

# ---- matched-supervision disagreement
_both = [t for t in DIS if t != "_summary" and "pess" in DIS[t] and "octg" in DIS[t]]
octg_worse = sum(DIS[t]["octg"]["mean_rho"] < DIS[t]["pess"]["mean_rho"] for t in _both)
pess_rho = [DIS[t]["pess"]["mean_rho"] for t in _both]
octg_rho = [DIS[t]["octg"]["mean_rho"] for t in _both]
pess_jac = [DIS[t]["pess"]["mean_top1_jaccard"] for t in _both]
octg_jac = [DIS[t]["octg"]["mean_top1_jaccard"] for t in _both]

# ---- the certificate triple (D1)
_tri = [(t, c["realized_cert_rate"], c["est_cert_rate_mean"], c["realized_cond_viol"],
         c["bound_mean"], c["plugin_mean"])
        for t, sd in CTRI.items() for c in sd.values() if c["realized_cond_viol"] is not None]
tri_cov = sum(b >= v for _, _, _, v, b, _ in _tri)
_tri_hi = [r for r in _tri if r[1] >= 0.10]
tri_cov_hi = sum(b >= v for _, _, _, v, b, _ in _tri_hi)
tri_fail_maxrate = max((r[1] for r in _tri if r[4] < r[3]), default=0.0)
tri_plug = float(np.mean([r[5] for r in _tri]))
tri_real = float(np.mean([r[3] for r in _tri]))
_tm = lambda t, k: float(np.mean([c[k] for c in CTRI[t].values()]))

def _pc1():
    r = sorted((x for x in HSF["cells"] if x["task"] == "pointcircle1"), key=lambda x: -x["cost2k"])
    bad = [round(x["cost2k"], 1) for x in r[:2]]
    ok = sorted(round(x["cost2k"], 1) for x in r[2:])
    return bad + [ok[0], ok[-1]]


def _median_over():
    import sys as _s
    _s.path.insert(0, f"{BASE}/scripts")
    import mean_cost_certificate as _M
    fails = [(x["task"], x["seed"], x["budget"]) for x in HSF["cells"] if x["cost2k"] > x["budget"]]
    n = 0
    for t, sd, b in fails:
        v = _M.episode_costs(t, sd)
        if v is not None and float(np.median(v)) > b:
            n += 1
    return [n, len(fails)]


def _bac(task, alpha):
    v = BAC[task]
    return rnd(np.mean([v[s][alpha]["cert_rate"] for s in v]), 2)


def _bac3(task, alpha):
    v = BAC[task]
    return rnd(np.mean([v[s][alpha]["cert_rate"] for s in v]), 3)


def _bullet_worst():
    w = max(((c["false_cert_rate_uncond"], c["false_cert_cp95"])
             for e in BG.values() for sd in e["seeds"].values() for c in sd.values()),
            key=lambda x: x[0])
    return [rnd(w[0], 3), rnd(w[1][0], 3), rnd(w[1][1], 3)]


def _carrun(arm):
    return [round(float(v["C"]), 2) for v in OSRL["carrun_b"][arm].values()]


def _toph_span():
    """min and max mean cost of the gated (top-half) cells, per certified task."""
    out = []
    for t in ("halfcheetah_velocity", "walker2d_velocity", "cargoal1_dsrl", "pointgoal1_dsrl"):
        v = []
        for r in A25[t]:
            e = SNAP.get(t, {}).get(f"toph_{r[1]}", {})
            if e:
                v.append(float(np.mean([x["C"] for x in e.values()])))
        out += [min(v), max(v)]
    return out


def _armC(task, arm):
    return rnd(float(np.mean([v["C"] for v in SNAP[task][arm].values()])), 1)


def _swim_ku():
    v = [e.get("meta", {}).get("kept_unsafe_rate") for e in SNAP["swimmer_velocity"]["calfilt_csf"].values()]
    return rnd(float(np.mean([x for x in v if x is not None])), 3)


def _robp(task, arm):
    return rnd(float(np.mean([x["precision"] for x in ROB[task][arm]])), 3)


def _robr(task, arm):
    return rnd(float(np.mean([x["cert_rate"] for x in ROB[task][arm]])), 2)


def _rob_worst(arm=None):
    arms = [arm] if arm else ["clean", "noise05", "noise10", "noise20", "noise30", "n100", "n300", "boltz"]
    return rnd(max(x["false_cert_rate"] for t in ROB for a in arms for x in ROB[t][a]), 3)


_BUL5 = ("ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b")


def _osrl_names(algo):
    return [NAMES[t] for t in ORDER
            if OSRL.get(t, {}).get(algo)
            and np.mean([v["C"] for v in OSRL[t][algo].values()]) <= LIMITS[t]]


def _osrl_safe(algo):
    return len(_osrl_names(algo))


def _cert20():
    tasks = list(ORDER) + list(_BUL5)
    n = sum(any(v.get("meta", {}).get("certified") for v in SNAP.get(t, {}).get("calfilt_csf", {}).values())
            for t in tasks)
    return (n, len(tasks))


def _mtp(task, proc):
    v = MTP[task]
    return rnd(float(np.mean([v[s][proc]["cert_rate"] for s in v])), 3)


def _mtp_worst():
    return rnd(max(v[s][p]["false_rate"] for v in MTP.values() for s in v
                   for p in ("fixedseq", "bonferroni", "holm")), 3)


def _rho_split():
    """(safe, total) among tasks with rho >= 0.85, then among those below."""
    hi = lo = hs = ls = 0
    for t, r in OBS.items():
        rho = float(np.mean(r["rho"]))
        e = SNAP.get(t, {}).get("calfilt_csf", {})
        if not e:
            continue
        ok = float(np.mean([v["C"] for v in e.values()])) <= LIM[t]
        if rho >= 0.85:
            hi += 1; hs += ok
        else:
            lo += 1; ls += ok
    return hs, hi, ls, lo


_BULLET = ("ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b")


def _bull(arm):
    """Mean cost of the Bullet cells this arm leaves UNSAFE, which is what the range quotes."""
    out = []
    for t in _BULLET:
        e = SNAP.get(t, {}).get(arm, {})
        if not e:
            continue
        c = float(np.mean([v["C"] for v in e.values()]))
        if c > LIM[t]:
            out.append(c)
    return out


# ---- pass-21 source trace: prose numbers that no table cell carried
_DNF = L("data/review_response/donotfilter.json")
_BULLET_ORDER = ("ballrun_b", "ballcircle_b", "carcircle_b", "carrun_b", "dronerun_b")


def _bullet_safe_fracs():
    return tuple(rnd((1 - _DNF[t]["pool_unsafe_frac"]) * 100, 1) for t in _BULLET_ORDER)


def _dronerun_pool():
    e = _DNF["dronerun_b"]
    return rnd(e["pool_mean_cost"], 1), rnd(e["pool_unsafe_frac"] * 100, 0)


_BGN = {t: e["n_trajs"] for t, e in L("data/review_response/bullet_guarantee_2000.json").items()}
# The paper source is not redistributed with the code archive. Where it is absent the two
# rules that read it announce a skip; every other check binds a number to its producer and
# runs unchanged.
HAVE_TEX = os.path.exists(f"{BASE}/paper.tex")
_PAPER_SRC = open(f"{BASE}/paper.tex").read() if HAVE_TEX else ""
_SEGCOV = L("data/review_response/segment_coverage.json")


def _segcov():
    """Coverage, overlap and multiplicity of the 1000 sampled pairs, over the nine tasks."""
    cov = [e["coverage_frac"] * 100 for e in _SEGCOV.values()]
    ov = [e["overlap_frac"] * 100 for e in _SEGCOV.values()]
    mult = [e["mean_multiplicity"] for e in _SEGCOV.values()]
    return (rnd(min(cov), 1), rnd(max(cov), 1),
            int(np.floor(min(ov))), int(np.ceil(max(ov))),
            int(round(100 - float(np.mean(cov)))), rnd(max(mult), 1))


def _armC_seeds(task, arm):
    return sorted((round(v["C"], 2) for v in SNAP[task][arm].values()), reverse=True)


CHECKS = [
    # Sec 5 guarantee validation
    ("Sec 5: worst unconditional false-certification is 0.078", 0.078, worst_rate, 0.0005),
    ("Sec 5: the worst cell is CarGoal1 seed 1 at n = 400", (1, 1, 1),
     (worst_task == "cargoal1_dsrl", worst_seed == "1", worst_n == "400"), 0),
    ("Sec 5: its Clopper-Pearson interval is 0.067 to 0.091", (0.067, 0.091), tuple(worst_cp), 0.0005),
    ("Sec 5: PointGoal1 certifies in about two thirds of draws at n = 200", 0.648, pg1_200, 0.005),
    ("Sec 5: and on more than nine tenths by n = 400", 1, pg1_400 > 0.9, 0),
    ("Sec 5: rate tracks margin across the validated tasks (Spearman 0.996)", 0.996, round(rho_tasks, 3), 0.0005),
    ("Sec 5 and App extended: Spearman 0.994 over the 27 task-length cells", 0.994, rho_cells, 0.0005),
    # App extended, segment length
    ("App extended: mean certification rate 0.204 at H = 50", 0.204, armrate["H50"], 0.0005),
    ("App extended: 0.224 at H = 30", 0.224, armrate["H30"], 0.0005),
    ("App extended: 0.102 at H = 10", 0.102, armrate["H10"], 0.0005),
    ("App extended: mean purity margin -0.002 at H = 50", -0.002, armmarg["H50"], 0.0005),
    ("App extended: +0.003 at H = 30", 0.003, armmarg["H30"], 0.0005),
    ("App extended: -0.072 at H = 10", -0.072, armmarg["H10"], 0.0005),
    ("App extended: deployed safe count 8 of 9 at H = 50 and H = 30, 7 of 9 at H = 10",
     (8, 8, 7), (h_safe["H50"], h_safe["H30"], h_safe["H10"]), 0),
    ("App extended: negative-margin cells certify at mean rate 0.015", 0.015, float(np.mean(neg)), 0.0005),
    ("App extended: and never above 0.057", 0.057, float(max(neg)), 0.0005),
    ("App extended: positive-margin cells average 0.327", 0.327, float(np.mean(pos)), 0.0005),
    ("App extended: worst unconditional false-certification 0.044, 0.061, 0.054",
     (0.044, 0.061, 0.054), tuple(round(armworst[a], 3) for a in ("H10", "H30", "H50")), 0.0005),
    ("App extended: PointGoal1's true H = 10 rate is 0.69", 0.69, pg1_h10, 0.005),
    # App extended, label complexity
    ("App extended: log n50 = -1.61 log(margin) + 1.66", (-1.61, 1.66),
     (round(LC["slope"], 2), round(LC["intercept"], 2)), 0.0005),
    ("App extended: with R^2 = 0.997", 0.997, LC["r2"], 0.0005),
    ("App extended: the fit is over the seven tasks that cross", 7, LCI["n_tasks"], 0),
    ("App extended: bootstrap exponent 1.51 to 1.67", (1.51, 1.67), tuple(LCI["boot_ci95"]), 0.005),
    ("App extended: below the 1/margin^2 rate in 99.98 percent of resamples", 0.9998, LCI["p_below_2"], 0.00005),
    ("App extended: Swimmer's margin is 0.014", 0.014, MVY["swimmer_velocity"], 0.0005),
    # App composability, selection stability
    ("App composability: the resample covers twelve task-level settings", 12, STAB["n_settings"], 0),
    ("App composability: the modal selection takes 21 to 83 percent of certified draws",
     (0.21, 0.83), (STAB["modal_share_min"], STAB["modal_share_max"]), 0.005),
    ("App composability: the three reported selections take at least 53 percent", 0.53, STAB["top3_share_min"], 0.005),
    ("App composability: more than nine tenths on eight of the twelve", 8,
     sum(r["top3_share_of_certified"] > 0.9 for r in STAB["settings"]), 0),
    # App composability, the grids
    ("Sec 4 and App composability: CDT is safe on 36 of 36 alpha=0.25 runs",
     (36, 36), (a25_cdt_ok, len(a25_cdt)), 0),
    ("App composability: CPL is within budget on 28 of the same 36 runs",
     (28, 36), (a25_cpl_ok, len(a25_cpl)), 0),
    ("App composability: every CPL failure is on Walker2d", 1,
     a25_cpl_fail_tasks == {"walker2d_velocity"}, 0),
    ("App composability: at alpha=0.40 CDT is safe on 14 of the 24 selections", (14, 24), (a40_cdt_sel, len(a40_sel)), 0),
    ("Sec 4 and App composability: CPL on 17 of 24, stated identically in both", 17, a40_cpl_sel, 0),
    ("Sec 4 and App composability: the clone's 22 of 24 is stated identically in both",
     2, len(re.findall(r"22 of 24", _PAPER_SRC)), 0),
    ("Sec 4 and App composability: the clone on 22 of 24", 22, a40_bc_sel, 0),
    ("App composability: the out-of-specification alpha=0.25 selections are 0.251, 0.263, 0.337",
     (0.251, 0.263, 0.337),
     tuple(sorted(round(r[2], 3) for rows in A25.values() for r in rows if r[2] > 0.25)), 0.0005),
    # App operator
    ("Sec 4 and App operator: 39 of 40 within-specification cells are safe", (39, 40), (op_in_ok, op_in), 0),
    ("App operator: 11 of the 12 out-of-specification cells are safe", (11, 12), (op_out_ok, op_out), 0),
    ("App operator: HalfCheetah's 0.337 selection violates at kappa=3 with cost 25.1", 25.1, _hc75, 0.05),
    ("Sec 4: on HalfCheetah's deployed selection the operator reaches 2271 at cost 0.03",
     (2271, 0.03), (op_deployed_R, op_deployed_C), 0.5),
    # headline counts
    ("Abstract and Sec 4: the calibrated procedure is safe on twelve of fifteen", 12, gated_safe, 0),
    ("Abstract and Sec 4: BC-Safe is also safe on twelve", 12, arm_safe("bcsafe"), 0),
    ("Sec 4: the V-filter at the estimated fraction is safe on twelve", 12, arm_safe("vfilt_calsafe"), 0),
    ("Sec 1: BC-Safe-Seg is safe on two of fifteen", 2, arm_safe("bcsafeseg"), 0),
    ("Sec 4: the certificate fires on four tasks at alpha = 0.25", 4, n_certified, 0),
    # App contamination-cost mapping
    ("App composability: 235 selection-clone pairs", 235, MEXP["n_points"], 0),
    ("App composability: contamination alone explains R^2 = 0.41", 0.41, MEXP["r2_contam"], 0.005),
    ("App composability: adding margin statistics raises R^2 to 0.49", 0.49, round(MEXP["r2_full"], 2), 0.0005),
    # App extended, selection-signal controls and the Bullet suite. These four are the literals
    # audit_prose_numbers reports as task-scoped orphans: a multi-task range attributed to the one
    # task its sentence names, and a selection-composition number that is in no results file.
    ("App extended: HalfCheetah's bottom-return kept set has mean trajectory cost 39.9",
     39.9, CTRL["halfcheetah_velocity"]["retbot_kept_mean_traj_cost"], 0.05),
    ("App extended: against the pool's 115.4",
     115.4, CTRL["halfcheetah_velocity"]["pool_mean_traj_cost"], 0.05),
    ("App bullet: random selection is unsafe on four of five Bullet tasks, costs 32 to 79",
     (32, 79), (round(min(_bull("vfilt_random"))), round(max(_bull("vfilt_random")))), 0.5),
    ("App bullet: cloning everything is unsafe on four of five, costs 24 to 65",
     (24, 65), (round(min(_bull("bc_all"))), round(max(_bull("bc_all")))), 0.5),
    # App extended: the 2000-episode policy-level section
    ("App extended: E[cost] certified within budget on ten of the fifteen tasks",
     (10, 15), (sum(v["all_seeds_certified"] for v in MCC["per_task"].values()), len(MCC["per_task"])), 0),
    ("App extended: the demeaned Spearmans are 0.32, 0.36 and 0.30",
     (0.32, 0.36, 0.30),
     tuple(rnd(HSF["demeaned_spearman_vs_cost2k"][k]["spearman_demeaned"], 2)
           for k in ("n_kept", "kept_frac", "contamination")), 0.0005),
    ("App extended: each is significant at p < 0.01", 1,
     all(HSF["demeaned_spearman_vs_cost2k"][k]["p"] < 0.01
         for k in ("n_kept", "kept_frac", "contamination")), 0),
    ("App extended: 75 calibrated cells", 75, HSF["n_cells"], 0),
    ("App extended: PointCircle1's two broken seeds at 59.1 and 27.6, others 4.7 to 7.3",
     (59.1, 27.6, 4.7, 7.3), tuple(_pc1()), 0.05),
    ("App extended: eight of the fifteen failing cells have a median episode over budget",
     (8, 15), tuple(_median_over()), 0),
    # Sec 6: what the three failing tasks share (checklist item 24b)
    ("Sec 6: BC-Safe already violates on PointButton2 and PointCircle2, 32.9 and 44.2",
     (32.9, 44.2), (_armC("pointbutton2", "bcsafe"), _armC("pointcircle2", "bcsafe")), 0.05),
    ("Sec 6: Swimmer's deployed selection is 14.5 percent unsafe yet its clone runs at 58.9",
     (0.145, 58.9), (_swim_ku(), _armC("swimmer_velocity", "calfilt_csf")), 0.0005),
    ("Sec 6: the uncalibrated V-filter on Swimmer lands at 14.6",
     14.6, _armC("swimmer_velocity", "vfilt_calsafe"), 0.05),
    # App extended: multiple-testing procedures over the grid (C7)
    ("App extended: PointGoal1 0.295 under Bonferroni/Holm against fixed-sequence 0.654",
     (0.295, 0.654), (_mtp("pointgoal1_dsrl", "bonferroni"), _mtp("pointgoal1_dsrl", "fixedseq")), 0.0005),
    ("App extended: Walker2d 0.295 against 0.511",
     (0.295, 0.511), (_mtp("walker2d_velocity", "bonferroni"), _mtp("walker2d_velocity", "fixedseq")), 0.0005),
    ("App extended: worst false-certification across the three procedures is 0.063",
     0.063, _mtp_worst(), 0.0005),
    # App extended: preference noise and budget (C20)
    ("App extended: the Boltzmann labeler is T_lab = 3 on every task",
     [3.0], BFR["T_lab_values"], 0),
    ("App extended: its realized error rates are 5.3 to 14.6 percent",
     (5.3, 14.6), (rnd(100 * BFR["rate_min"], 1), rnd(100 * BFR["rate_max"], 1)), 0.05),
    ("App extended: worst false-certification over every noise and budget arm is exactly delta",
     0.100, _rob_worst(), 0.0005),
    ("App extended: the Boltzmann arm's worst false-certification is 0.08",
     0.08, _rob_worst("boltz"), 0.0005),
    ("App extended: Walker2d precision 0.834 to 0.855 and rate 0.46 to 0.77 at five percent noise",
     (0.834, 0.855, 0.46, 0.77),
     (_robp("walker2d_velocity", "clean"), _robp("walker2d_velocity", "noise05"),
      _robr("walker2d_velocity", "clean"), _robr("walker2d_velocity", "noise05")), 0.005),
    ("App extended: Swimmer loses 0.33 under the Boltzmann labeler",
     -0.33, rnd(_robp("swimmer_velocity", "boltz") - _robp("swimmer_velocity", "clean"), 2), 0.0005),
    # Sec 6: external baselines and the certification count across both suites
    ("Sec 6: CDT keeps five of fifteen, CPQ four, COptiDICE two",
     (5, 4, 2), (_osrl_safe("cdt"), _osrl_safe("cpq"), _osrl_safe("coptidice")), 0),
    ("Sec 6: CDT's safe set is four velocity tasks plus PointCircle1", 1,
     set(_osrl_names("cdt")) == {"HalfCheetah", "Walker2d", "Ant", "Hopper", "PointCircle1"}, 0),
    ("Sec 6: the 200-bit filter keeps twelve", 12, gated_safe, 0),
    ("Sec 6: the certificate fires on 6 of 20 tasks at alpha = 0.25", (6, 20), _cert20(), 0),
    ("Sec 6: every trajectory-scale filter-then-clone arm beats every optimization-based method",
     1, min(arm_safe("bcsafe"), arm_safe("vfilt_calsafe"), arm_safe("vfilt_matchgt"), gated_safe)
        > max(_osrl_safe("cdt"), _osrl_safe("cpq"), _osrl_safe("coptidice"), arm_safe("cpl_gt")), 0),
    # Sec 6.3: the deployment certificate, pooled over episodes
    ("Sec 6.3: pooled Pr[cost > budget] 0.224 calibrated, 0.627 BC-All, 0.243 BC-Safe",
     (0.224, 0.627, 0.243),
     (rnd(PCT["arms"]["__gated__"]["cp_upper"], 3), rnd(PCT["arms"]["bc"]["cp_upper"], 3),
      rnd(PCT["arms"]["bcsafe"]["cp_upper"], 3)), 0.0005),
    ("Sec 6.3: the mean-cost bound certifies ten of fifteen, against twelve safe",
     (10, 12), (sum(v["all_seeds_certified"] for v in MCC["per_task"].values()),
                PCT["arms"]["__gated__"]["mean_cost_safe_tasks"]), 0),
    ("Sec 7 and App operator: the worst out-of-spec selection is safe at kappa 1, 2 and top-half and violates at kappa 3",
     (1, 1, 1, 0),
     tuple(int(np.mean([v["C"] for v in SNAP["halfcheetah_velocity"][f"{a}_q75"].values()]) <= 20)
           for a in ("wbc1", "wbc", "toph", "wbc3")), 0),
    # App extraction: the value at two scales (obstacle2 diagnostics vs the deployed policy)
    ("App extraction: Swimmer has the strongest cross-seed agreement, rho 0.99 and overlap 0.90",
     (0.99, 0.90), (rnd(np.mean(OBS["swimmer_velocity"]["rho"]), 2),
                    rnd(np.mean(OBS["swimmer_velocity"]["jaccard"]), 2)), 0.0005),
    ("App extraction: Swimmer's held-out accuracy 0.996 with its clone at cost 58.9",
     (0.996, 58.9), (rnd(np.mean(OBS["swimmer_velocity"]["accuracy"]), 3),
                     rnd(np.mean([v["C"] for v in SNAP["swimmer_velocity"]["calfilt_csf"].values()]), 1)), 0.05),
    ("App extraction: PointGoal2 has the weakest agreement, rho 0.79, and is safe at 20.8",
     (0.79, 20.8), (rnd(np.mean(OBS["pointgoal2"]["rho"]), 2),
                    rnd(np.mean([v["C"] for v in SNAP["pointgoal2"]["calfilt_csf"].values()]), 1)), 0.05),
    ("App extraction: CarGoal2 has the lowest held-out accuracy 0.798 and is safe at 19.3",
     (0.798, 19.3), (rnd(np.mean(OBS["cargoal2"]["accuracy"]), 3),
                     rnd(np.mean([v["C"] for v in SNAP["cargoal2"]["calfilt_csf"].values()]), 1)), 0.05),
    ("App extraction: safe on six of the seven tasks with rho >= 0.85 and both below it",
     (6, 7, 2, 2), tuple(_rho_split()), 0),
    # App extended: gated sub-selection per cell, the four certified tasks
    ("App extended: gated cells span HalfCheetah 0.2-7.2, Walker2d 1.3-18.7, CarGoal1 6.6-9.4, PointGoal1 5.1-6.6",
     (0.2, 7.2, 1.3, 18.7, 6.6, 9.4, 5.1, 6.6), tuple(_toph_span()), 0.05),
    ("App bullet: worst unconditional false-certification 0.06, Clopper-Pearson 0.050 to 0.071",
     (0.06, 0.050, 0.071), tuple(_bullet_worst()), 0.0005),
    ("App bullet: the operating curve, DroneRun 0.20 at alpha=.25 to 0.87 at .40",
     (0.20, 0.87), (_bac("dronerun_b", "0.25"), _bac("dronerun_b", "0.4")), 0.0005),
    ("App bullet: BallRun 0.00 to 0.06 over the same range",
     (0.00, 0.06), (_bac("ballrun_b", "0.25"), _bac("ballrun_b", "0.4")), 0.0005),
    ("App bullet: CarRun certifies almost always, 0.995 at alpha=.25",
     0.995, _bac3("carrun_b", "0.25"), 0.0005),
    ("App bullet: BallCircle and CarCircle never certify",
     (0.0, 0.0), (_bac("ballcircle_b", "0.4"), _bac("carcircle_b", "0.4")), 0.0005),
    # App bullet: the CarRun composability echo, cited beside tab:operator but not a cell of it
    ("App bullet: CDT on the ORIGINAL CarRun certified selection, 11.4 to 19.6 across three seeds",
     (11.4, 19.6), (min(_carrun("cdt_cert")), max(_carrun("cdt_cert"))), 0.05),
    ("App bullet: on the regenerated selection it violates on one seed of three, 0.0, 7.0, 11.1",
     (0.0, 7.0, 11.1), tuple(sorted(_carrun("cdt_echonew_q85"))), 0.05),
    # App proofs: Proposition (rate) against the resampled rates
    ("App proofs: across all 100 cells the mean absolute error is 0.019",
     (100, 0.019), (len(CRT["cells"]), rnd(CRT["mean_abs_err"], 3)), 0.0005),
    # Sec 6.3 quotes TWO resamples in consecutive sentences. It called them "the same
    # resample"; they are not, and conflating them would misattribute the 0.019.
    ("Sec 6.3: no theory-vs-observed cell is anything but a 200-draw resample",
     0, sum(c["n_draws"] != 200 for c in CRT["cells"]), 0),
    # fig:coverage caption. It quoted Ant's per-seed MAX (0.006) beside PointGoal2's seed
    # MEAN (0.047), while the panel itself plots seed means.
    ("Fig coverage caption: at n=200 the seed-mean false-certification rates are "
     "Ant 0.004, PointGoal2 0.047, Hopper 0.000",
     (0.004, 0.047, 0.000), _fc200, 0.0005),
    # App proofs, label budget. The realized rate at the deployed budget was stated as 0.73;
    # the plan's own closed form gives 0.66, which is also what the resample shows (0.65).
    ("App proofs: PointGoal1 needs n >= 360 for a rate of 0.9 and realizes 0.66 at n = 200",
     (360, 0.66), (LBP["tasks"]["pointgoal1_dsrl"]["n_for_target"],
                   rnd(LBP["tasks"]["pointgoal1_dsrl"]["rate_at_deployed_n"], 2)), 0.0005),
    ("Sec 6.3: no conditional-rate cell is anything but a 2000-draw resample",
     0, sum(r.get("n_draws") != 2000
            for t in G2K if isinstance(G2K[t], dict) and "seeds" in G2K[t]
            for sd in G2K[t]["seeds"].values() for r in sd.values() if isinstance(r, dict)), 0),
    ("App proofs: and the Pearson correlation 0.994", 0.994, rnd(CRT["pearson_r"], 3), 0.0005),
    # App hyper: the accepted hypergeometric test behind each deployed certificate
    ("App hyper: m = 28, 42, 31, 28 on the deployed certified draws",
     (28, 42, 31, 28),
     tuple(CAUD[t][sd]["accepted"]["m"] for t, sd in
           (("halfcheetah_velocity", "4"), ("walker2d_velocity", "2"),
            ("cargoal1_dsrl", "3"), ("pointgoal1_dsrl", "0"))), 0),
    ("App hyper: giving p = 0.014, 0.027, 0.068, 0.044",
     (0.014, 0.027, 0.068, 0.044),
     tuple(rnd(CAUD[t][sd]["accepted"]["p"], 3) for t, sd in
           (("halfcheetah_velocity", "4"), ("walker2d_velocity", "2"),
            ("cargoal1_dsrl", "3"), ("pointgoal1_dsrl", "0"))), 0.0005),
    ("App hyper: Walker2d accepts at the third grid threshold, the others at the first",
     (0.85, 0.75, 0.85, 0.85),
     tuple(CAUD[t][sd]["accepted"]["quantile"] for t, sd in
           (("halfcheetah_velocity", "4"), ("walker2d_velocity", "2"),
            ("cargoal1_dsrl", "3"), ("pointgoal1_dsrl", "0"))), 0.0005),
    # App extended, the certificate as a triple (D1)
    ("App extended: the bound covers in 25 of the 40 cells with a defined conditional rate",
     (25, 40), (tri_cov, len(_tri)), 0),
    ("App extended: and in all 24 cells certifying in at least a tenth of draws",
     (24, 24), (tri_cov_hi, len(_tri_hi)), 0),
    ("App extended: failures are confined to cells certifying in at most 7 percent of draws",
     0.07, round(tri_fail_maxrate, 2), 0.005),
    ("App extended: Walker2d's estimate 0.51 against a realized 0.52",
     (0.51, 0.52), (round(_tm("walker2d_velocity", "est_cert_rate_mean"), 2),
                    round(_tm("walker2d_velocity", "realized_cert_rate"), 2)), 0.0005),
    ("App extended: PointGoal1's 0.59 against 0.65",
     (0.59, 0.65), (round(_tm("pointgoal1_dsrl", "est_cert_rate_mean"), 2),
                    round(_tm("pointgoal1_dsrl", "realized_cert_rate"), 2)), 0.0005),
    ("App extended: Ant's 0.06 against 0.01",
     (0.06, 0.01), (round(_tm("ant_velocity", "est_cert_rate_mean"), 2),
                    round(_tm("ant_velocity", "realized_cert_rate"), 2)), 0.0005),
    ("App extended: the plug-in averages 0.04 against a realized 0.43",
     (0.04, 0.43), (round(tri_plug, 2), round(tri_real, 2)), 0.0005),
    # App extraction, matched-supervision control
    ("App extraction: the supervised ensembles agree less on all nine tasks", (9, 9), (octg_worse, len(_both)), 0),
    ("App extraction: task-mean rho 0.43 to 0.82 for octg", (0.43, 0.82), (min(octg_rho), max(octg_rho)), 0.005),
    ("App extraction: against 0.79 to 0.99 for the preference ensembles",
     (0.79, 0.99), (min(pess_rho), max(pess_rho)), 0.005),
    ("App extraction: top-one-percent overlap 0.31 to 0.67 for octg", (0.31, 0.67), (min(octg_jac), max(octg_jac)), 0.005),
    ("App extraction: against 0.42 to 0.90 for the preference ensembles",
     (0.42, 0.90), (min(pess_jac), max(pess_jac)), 0.005),
    # Two aggregations of the same quantity live in this paper and must not be swapped:
    # tab:obstacle2 (and the prose citing it) ranges over SEED PAIRS, 0.39 to 0.92, while the
    # matched-supervision control quotes TASK MEANS, 0.42 to 0.90. The prose called the
    # seed-pair range "across tasks", which is what made them look contradictory.
    ("Sec 5 and tab:obstacle2: seed-pair top-one-percent overlap is 0.39 to 0.92",
     (0.39, 0.92), (min(_o2seed), max(_o2seed)), 0.005),
    ("App extraction: task-mean top-one-percent overlap is 0.42 to 0.90",
     (0.42, 0.90), (min(_o2jac), max(_o2jac)), 0.005),

    # Sec 6.2 V-filter vs BC-Safe. This sentence had NO source trace and carried three wrong
    # numbers (eight / 0.48 / 0.54) that the existence-style prose audit could not catch.
    ("Sec 6.2: V-filter and BC-Safe are both safe on eleven tasks", 11, _vb_both, 0),
    ("Sec 6.2: ours is lower-cost on nine of them", 9, _vb_low, 0),
    ("Sec 6.2: mean normalized cost 0.46 for ours against 0.53 for BC-Safe",
     (0.46, 0.53), (rnd(_vb_mn_ours, 2), rnd(_vb_mn_bcs, 2)), 0.0005),
    # App baselines / Fig 2: counts read off the pareto builder's own arms.
    ("App baselines: the calibrated filter is safe on twelve of fifteen", 12, _pareto["Calibrated"], 0),
    ("App baselines: CDT reaches seven of fifteen at its best target", 7, _pareto["CDT"], 0),
    ("App baselines: BC-Safe reaches twelve on ground-truth labels", 12, _pareto["BC-Safe"], 0),
    # tab:normcost caption: budget-matched CDT is safe on five, best-target on seven,
    # so the best target buys TWO further safe tasks, not one.
    ("Tab normcost caption: best-target CDT gains two further safe tasks",
     2, _pareto["CDT"] - _cdt_matched, 0),
    # Sec 6.2 "eleven or twelve against two to five": the optimization-based methods at the
    # budget-matched configuration the sentence compares against.
    ("Sec 6.2: optimization-based methods span two to five safe tasks",
     (2, 5), (min(_optbased), max(_optbased)), 0),
    # ...and the trajectory-scale filter-then-clone arms they are compared against. The text
    # said "eleven or twelve", which omitted labels-only at n = 400, plotted at 13 in Figure 6.
    ("Sec 6.2: trajectory-scale filter arms span eleven to thirteen safe tasks",
     (11, 13), (min(_filterarms), max(_filterarms)), 0),
    # Sec 1, 6.4 and 7 compare the labels-only classifier with the calibrated filter. All
    # three said "matches"; at the matched budget the generated table prints eleven against
    # the filter's twelve, so the parity claim contradicted the paper's own Table 14.
    # Sec 6.2 and App extended: the ungated top-half sub-selection on Hopper. The paper
    # quoted 148 / 487, which are the PRE-CARDINAL ARCHIVE's values for this arm; the arm was
    # regenerated and the prose was not. The current snapshot gives 82 / 289.
    ("Sec 6.2 and App extended: ungated Hopper R50 mean cost 82, worst seed 289",
     (82, 289), (rnd(np.mean(_hopR50), 0), rnd(max(_hopR50), 0)), 1.0),
    # App t32t26 score aggregation. The paper said the tenth percentile is "safe on all nine"
    # and rescues Hopper at cost 8 against the mean's 26. The current snapshot: eight of nine
    # (it loses Swimmer), Hopper 19.7 against 33.0. The 8/26 pair came from a pre-refresh table.
    ("App t32t26: min/10th-pct/mean are safe on 4, 8 and 8 of the nine analysis tasks",
     (4, 8, 8), _agg_safe, 0),
    # App extraction, oracle ablation. The paper said Swimmer's learned value costs 9; that is
    # the PRE-REFRESH table's value for xlab_exp_k1 (8.87). The current snapshot gives 1.03.
    ("App extraction: Swimmer learned value 1 against oracles 19 and 56",
     (1, 19, 56), (rnd(_aggC("xlab_exp_k1", "swimmer_velocity"), 0),
                   rnd(_aggC("xlab_oracle_step", "swimmer_velocity"), 0),
                   rnd(_aggC("xlab_oracle_ctg", "swimmer_velocity"), 0)), 0.51),
    ("App extraction: no oracle beats the learned value on any of the nine analysis tasks",
     0, sum(min(_aggC("xlab_oracle_step", t), _aggC("xlab_oracle_ctg", t))
            < _aggC("xlab_exp_k1", t) for t in _AGG_TASKS), 0),
    # ---- traces added by the 2026-09-23 audit for prose numbers that had no producer.
    # App extended, bottom-return control
    ("App extended: bottom-return PointGoal1 20.1 against the V-filter's 10.7",
     (20.1, 10.7), (rnd(_aggC("vfilt_retbot", "pointgoal1_dsrl"), 1),
                    rnd(_aggC("vfilt_matchgt", "pointgoal1_dsrl"), 1)), 0.05),
    ("App extended: bottom-return breaks PointGoal2 at 43.7 where the V-filter is 19.8",
     (43.7, 19.8), (rnd(_aggC("vfilt_retbot", "pointgoal2"), 1),
                    rnd(_aggC("vfilt_matchgt", "pointgoal2"), 1)), 0.05),
    ("App extended: bottom-return is safe by collapse on Swimmer, cost 1",
     1, rnd(_aggC("vfilt_retbot", "swimmer_velocity"), 0), 0.51),
    # App extended, the hard top-half rule's two flipped verdicts
    ("App extended: the top-half rule converts Swimmer 58.9 to 18.0",
     (58.9, 18.0), (rnd(_aggC("calfilt_csf", "swimmer_velocity"), 1),
                    rnd(_aggC("calfilt_tier2", "swimmer_velocity"), 1)), 0.05),
    ("App extended: and breaks Hopper 3.4 to 43.8",
     (3.4, 43.8), (rnd(_aggC("calfilt_csf", "hopper_velocity"), 1),
                   rnd(_aggC("calfilt_tier2", "hopper_velocity"), 1)), 0.05),
    ("App extended: PointCircle2 87.2 to 112.5",
     (87.2, 112.5), (rnd(_aggC("calfilt_csf", "pointcircle2"), 1),
                     rnd(_aggC("calfilt_tier2", "pointcircle2"), 1)), 0.05),
    # App extended, cardinal-supervision control
    ("App extended: the cardinal selection is less pure on eleven of fifteen by 0.024",
     (11, 0.024), (_card_less, rnd(_card_gap, 3)), 0.0005),
    ("App extended: cardinal clone safe on twelve against the bit route's eleven",
     (12, 11), (CARD["n_cardinal_safe"], CARD["n_binary_safe"]), 0),
    # App extraction: the action-conditioned oracle, the strongest per-transition configuration.
    # Read off the generated table, since the prose rounds its cells.
    ("App extraction: the action-conditioned oracle costs 31.5, 39.5 and 49.6 against budget 25",
     (31.5, 39.5, 49.6), _actcond, 0.051),
    # The appendix had said the action-conditioned oracle "reaches safety there", contradicting
    # the main text. It is over budget on all three failing navigation tasks, and its reward is
    # the highest of the four per-transition columns, not a severe cost.
    # tab:a25draws caption said "all 27 CDT runs and all 27 CPL runs are within budget". The
    # table holds four tasks x three selections = 12 selections, 36 runs at three seeds. CDT is
    # safe on all 36; CPL fails all three Walker2d selections. The 27 is stale from before
    # Walker2d began certifying at this level.
    # App bullet: the four composability cases. "BC-All costs 34 to 86" was the PRE-CARDINAL
    # archive's range; the current snapshot gives 106, 22.4 and 36.9, and CarGoal1's full pool is
    # now within budget, so "all three main-suite tasks" no longer have an unsafe pool.
    ("App bullet: full-data CDT is safe on HalfCheetah and at 1.47x budget on both navigation tasks",
     (1, 1.47, 1.47), (int(_cdtfull("halfcheetah_velocity") <= 1.0),
                       rnd(_cdtfull("cargoal1_dsrl"), 2), rnd(_cdtfull("pointgoal1_dsrl"), 2)), 0.005),
    ("App composability: the alpha=0.25 grid is 12 selections, CDT safe on all of them",
     (12, 12), (_a25[0], _a25[1]), 0),
    ("App composability: CPL is unsafe on exactly the three Walker2d selections",
     3, _a25[0] - _a25[2], 0),
    ("App extraction: the action-conditioned oracle is over budget on all three",
     3, sum(_extraction_col(t, 8) > 25 for t in ("PointGoal1", "CarGoal2", "PointGoal2")), 0),
    ("App extraction: and its reward beats every weight transform on those three",
     3, sum(_extraction_col(t, 7) > max(_extraction_col(t, c) for c in (1, 3, 5))
            for t in ("PointGoal1", "CarGoal2", "PointGoal2")), 0),
    # App proofs, controlled contamination. The paper said the worst rate at a non-positive
    # margin is 0.075 and the mean absolute error 0.011; the sweep's own summary gives 0.08
    # and 0.0097. Both were untraced.
    ("App proofs: 198 constructed pools, worst non-positive-margin rate 0.08",
     (198, 0.08), (ECON["summary"]["n_cells"],
                   rnd(ECON["summary"]["worst_rate_negative_margin"], 3)), 0.0005),
    ("App proofs: contamination sweep mean absolute error 0.010, correlation 0.999",
     (0.010, 0.999), (rnd(ECON["summary"]["mae"], 3),
                      rnd(ECON["summary"]["pearson_r"], 3)), 0.0005),
    # tab:certratepred caption said "all 108 cells"; the artifact holds 100, the eight missing
    # being hopper seeds 1 and 2, which have no first-threshold record at any budget.
    ("App proofs: the rate table covers 100 cells, hopper seeds 1 and 2 excluded",
     (100, 2), (len(CRT["cells"]),
                len({(t, sd) for t in {c["task"] for c in CRT["cells"]} | {"hopper_velocity"}
                     for sd in ("0", "1", "2")}
                    - {(c["task"], c["seed"]) for c in CRT["cells"]})), 0),
    # App hyper, ensemble size. Both statistics are per-task, averaged over the three seeds.
    ("App hyper: a single network matches K=3 precision within 0.006",
     0.006, rnd(_k_gap, 3), 0.0005),
    ("App hyper: individual members vary by up to 0.075",
     0.075, rnd(_k_spread, 3), 0.0005),
    # tab:compose covers three of the four alpha=0.25 tasks. The caption now says why: the
    # operator column is the 0.85 selection and Walker2d's deepest certified quantile is 0.75.
    # tab:compose now carries all four alpha=0.25 tasks. The operator column is each task's most
    # selective distinct certified selection: q85 for three, q75 for Walker2d, whose deepest it
    # is. Walker2d is also the row where CPL violates, which is why omitting it was a
    # presentation problem and not only a completeness one.
    # App composability opener. "8.4 against 9.0 on CarGoal1" was the archive's clone reward
    # (9.02); current is 8.54 against CDT-certified 7.46. The PointGoal1 pair was already current.
    ("App composability: CDT on the certified selection trails the clone, 7.5 vs 8.5 and 5.2 vs 7.5",
     (7.5, 8.5, 5.2, 7.5),
     (rnd(float(np.mean([v["R"] for v in OSRL["cargoal1_dsrl"]["cdt_cert"].values()])), 1),
      rnd(float(np.mean([v["R"] for v in SNAP["cargoal1_dsrl"]["calfilt_csf"].values()])), 1),
      rnd(float(np.mean([v["R"] for v in OSRL["pointgoal1_dsrl"]["cdt_cert"].values()])), 1),
      rnd(float(np.mean([v["R"] for v in SNAP["pointgoal1_dsrl"]["calfilt_csf"].values()])), 1)), 0.05),
    # Sec 6.4 said the score "anti-correlates with return only on navigation, where return and
    # cost are themselves positively correlated". Table 24 shows it anti-correlates on 13 of 15
    # including three velocity tasks, and cost-return is positive on all 15. What IS true is
    # that the anti-correlation tracks the coupling.
    ("Sec 6.4: score-return anti-correlation tracks cost-return coupling at Pearson 0.72",
     0.72, rnd(_scorecorr_r(), 2), 0.005),
    ("Sec 6.4: it anti-correlates with return on 13 of 15, so not only on navigation",
     (13, 15), _scorecorr_neg(), 0),
    # App certprops, two-tier fallback: nine tasks, not the nine ANALYSIS tasks, and it raises
    # mean cost on five of them.
    ("App certprops: the two-tier fallback covers nine tasks and raises cost on five",
     (9, 5), _tier2_counts(), 0),
    # App proofs: "the benchmarks supply safe trajectories at 9 to 46 percent of each pool".
    # 9 to 46 is the BULLET range alone; PointGoal1's pool is 51.8 percent safe, so the range
    # across both suites tops out at 52.
    ("App proofs: DSRL pool safe fractions run to 51.8 percent, above the Bullet maximum",
     (0.113, 0.518), (rnd(min(_poolsafe().values()), 3), rnd(max(_poolsafe().values()), 3)), 0.0005),
    # App composability, alpha=0.40 deployed selections: the appendix said "Our clone is safe on
    # all eight". Table 12 shows Swimmer at 24 against budget 20, unbolded. Seven of eight.
    ("App composability: at alpha=0.40 the clone is safe on seven of eight, CDT on six",
     (7, 6), _a40_deployed(), 0),
    # App extended, selection-signal controls: return selection costs more than random on
    # TWELVE of fifteen, not thirteen (Walker2d, Ant and Swimmer go the other way), and it
    # buys reward on all fifteen.
    ("App extended: return selection buys reward on 15 and costs more than random on 12",
     (15, 12), _controls_counts(), 0),
    # App bullet: on HalfCheetah and Walker2d full-data CDT is already within budget and
    # curation LOWERS its reward on the deployed selection (2756->2048, 2712->2686). An earlier
    # audit edit wrongly said it "lifts reward"; the reward recovery is on HalfCheetah's
    # DEEPEST selection (Table 11), not the deployed one.
    ("App bullet: curation does not lift CDT reward on the deployed HalfCheetah or Walker2d selection",
     0, sum(np.mean([v["R"] for v in OSRL[t]["cdt_cert"].values()])
            > np.mean([v["R"] for v in OSRL[t]["cdt"].values()])
            for t in ("halfcheetah_velocity", "walker2d_velocity")), 0),
    ("Sec 4: on HalfCheetah's deepest selection CDT does reach full-data reward at cost below 1",
     (1, 1), (int(_hc_deep()[0] >= np.mean([v["R"] for v in OSRL["halfcheetah_velocity"]["cdt"].values()])),
              int(_hc_deep()[1] < 1.0)), 0),
    # App certprops: "cuts ... by roughly two thirds and lands within a point of the full-label
    # oracle". 0.627 -> 0.224 is a 64 percent cut, but 0.224 against 0.243 is 1.9 points, not
    # within one.
    ("App certprops: the cut is 64 percent and the gap to the oracle is 1.9 points",
     (64.3, 1.9), (rnd((0.627 - 0.224) / 0.627 * 100, 1), rnd(abs(0.243 - 0.224) * 100, 1)), 0.05),
    # Sec 5: "on PointGoal1 five of the eleven per-transition configurations reach the budget,
    # while on CarGoal2 and PointGoal2 none of nine does". The denominators depend on which
    # variants count as a configuration; the substance -- five, none, none -- is bound here.
    ("Sec 5: five per-transition configurations reach budget on PointGoal1, none on the other two",
     (5, 0, 0), _pt_within(), 0),
    # App bullet: "the 200 calibration labels are 10 to 31 percent of these smaller pools,
    # extending the main study's 5-to-22-percent range". The six Button/Circle pool sizes are
    # not stored directly; they are recoverable as n_kept / kept_frac from the deployed run.
    ("App bullet: 200 labels are 5 to 22 percent of the main-study pools",
     (4.9, 22.3), _label_share(), 0.05),
    ("App composability: tab:compose covers all four certifying tasks",
     4, len([l for l in open(f"{BASE}/data/tables/compose.tex") if "&" in l]), 0),
    ("App composability: the operator column is the deepest certified quantile per task, q75 only on Walker2d",
     (4, 1), (sum(max((a for a in SNAP[t] if a.startswith("wbc_q")),
                      key=lambda a: int(a.split("q")[1])) == exp
                  for t, exp in (("halfcheetah_velocity", "wbc_q85"),
                                 ("walker2d_velocity", "wbc_q75"),
                                 ("cargoal1_dsrl", "wbc_q85"),
                                 ("pointgoal1_dsrl", "wbc_q85"))),
              int("wbc_q85" not in SNAP["walker2d_velocity"])), 0),
    ("App composability: CPL violates on Walker2d's certified selection at 50.9 against 20",
     50.9, rnd(float(np.mean([v["C"] for v in SNAP["walker2d_velocity"]["cpl_gt_cert"].values()])), 1), 0.05),
    # fig:landscape trace caption
    ("App landscape: the illustrated unsafe trajectory has episodic cost 40",
     40, INTERP["halfcheetah_velocity"]["traces"]["unsafe"]["traj_cost"], 0.01),
    # fig:paretotargets caption claimed the alpha sweep is at or inside BC-Safe's cost on
    # EVERY navigation task. CarGoal2 at alpha=0.40 is 0.71 against BC-Safe's 0.66.
    ("App baselines: the alpha sweep is within budget on all four navigation tasks",
     4, sum(max(_sweep(t)) <= 1.0 for t in _NAVI), 0),
    ("App baselines: and inside BC-Safe's cost on three of them, CarGoal2 excepted",
     (3, 1), (sum(max(_sweep(t)) <= _bcs(t) for t in _NAVI),
              int(max(_sweep("cargoal2")) > _bcs("cargoal2"))), 0),
    # tab:bullet caption said every task but CarRun refuses on every seed. DroneRun certifies
    # on one of its five deployed seeds, which is also what makes the main text's "6 of 20"
    # right: 4 DSRL tasks + CarRun + DroneRun.
    ("App bullet: CarRun certifies on five deployed seeds, DroneRun on one, others none",
     (5, 1, 0), (_cert_seeds("carrun_b"), _cert_seeds("dronerun_b"),
                 sum(_cert_seeds(t) for t in ("ballrun_b", "ballcircle_b", "carcircle_b"))), 0),
    ("Sec 6.2: the certificate fires on 6 of the 20 tasks across both suites",
     6, sum(_cert_seeds(t) > 0 for t in _ALL20), 0),
    # App proofs, resampling corollary: e^{2k} a / (e^{2k} a + 1 - a) at alpha = 0.25.
    # The paper says the bound already exceeds one half by kappa = 1.
    # Proposition 4, re-derived 2026-09-24. Three identities, checked over a grid of
    # (q, p_s, T, eps): the product-policy fixed point, the weight ratio needed for all-safe
    # probability 1-eps, and that q enters (i) and (ii) only through pbar = q + (1-q) p_s.
    # Lemma 2 and Propositions 1/3, re-derived 2026-09-24 by exact enumeration (math.comb, no
    # new dependency). Lemma 1 is the conditioning step both rest on and was checked by
    # simulation; what is decidable exactly is stated here.
    ("Lemma 2: p_j is super-uniform under the null, Pr[p_j <= delta] <= delta",
     0, _superunif_violations(), 0),
    ("Prop 1 and 3: certifying an out-of-specification selection has probability at most delta",
     0, _falsecert_violations(), 0),
    # App composability, no-augmentation control: safe on all four velocity tasks, unsafe on all
    # four navigation tasks at 47 to 109.
    ("App composability: CDT without augmentation is safe on 4 velocity, unsafe on 4 navigation at 47 to 109",
     (4, 4, 47.0, 108.9), _noaug(), 0.05),
    ("Prop 4: product-policy fixed point pbar W / (pbar W + 1 - pbar)", 0, _prop4()[0], 0),
    ("Prop 4: the required weight ratio matches the paper's closed form", 0, _prop4()[1], 0),
    ("Prop 4: q enters parts (i) and (ii) only through pbar", 0, _prop4()[2], 0),
    ("App proofs: the resampling bound exceeds one half by kappa = 1",
     1, int(_resample_bound(1.0, 0.25) > 0.5), 0),
    ("App proofs: and is still below one half at kappa = 0.5",
     1, int(_resample_bound(0.5, 0.25) <= 0.5), 0),
    ("App t32t26: on Hopper the tenth percentile is 19.7 against the mean's 33.0",
     (19.7, 33.0), (rnd(_aggC("xagg_p10"), 1), rnd(_aggC("vfilt_calsafe"), 1)), 0.05),
    ("Sec 6.4: labels-only is safe on eleven at the matched budget n = 200", 11, _lo[200], 0),
    ("Sec 6.4: labels-only spans eleven to thirteen across budgets",
     (11, 13), (min(_lo.values()), max(_lo.values())), 0),
    # pass 21: the source trace found these quoted from archived records, not from a table cell
    ("App bullet: ground-truth safe fractions 9.0, 9.7, 11.4, 46.2 and 28.0 percent",
     (9.0, 9.7, 11.4, 46.2, 28.0), _bullet_safe_fracs(), 0.05),
    ("App bullet: DroneRun's pool has mean cost 50.5 and 72 percent unsafe trajectories",
     (50.5, 72), _dronerun_pool(), 0.05),
    ("App extraction: multi-step advantages move HalfCheetah from 0.3 at h = 1 to 15.6 at h = 5",
     (0.3, 15.6), (rnd(_aggC("xlab_exp_k1", "halfcheetah_velocity"), 1),
                   rnd(_aggC("xlab_k5", "halfcheetah_velocity"), 1)), 0.05),
    ("App bullet: CarRun's hard top-half rule violates on one seed of three at cost 30.0",
     (1, 30.0), (sum(c > LIM["carrun_b"] for c in _armC_seeds("carrun_b", "toph_echonew")),
                 _armC_seeds("carrun_b", "toph_echonew")[0]), 0.05),
    # seeds compared as a sorted multiset: the prose lists them in run order, which no
    # archived record fixes, so only the values and their mean are the claim
    ("App ablation: Q on Hopper costs 14.2, 35.8 and 37.2 with mean 29.1",
     (14.2, 35.8, 37.2, 29.1),
     tuple(rnd(x, 1) for x in sorted(_armC_seeds("hopper_velocity", "qfilt")))
     + (rnd(float(np.mean([v["C"] for v in SNAP["hopper_velocity"]["qfilt"].values()])), 1),), 0.05),
    ("App extended: the 1000 pairs constrain 1.4 to 4.2 percent of each pool's states, the "
     "drawn segments overlapping on 1 to 10 percent of their slots, leaving roughly 97 percent "
     "unconstrained at about one segment sum per covered state",
     (1.4, 4.2, 1, 10, 97, 1.1), _segcov(), 0.05),
    ("App proofs: 10^3 labels inside a 15 percent selection is about 6700 labels",
     6700, round(1000 / 0.15, -2), 50),
    ("App bullet: pools of 651 to 1990 trajectories, where 200 labels are 10 to 31 percent",
     (651, 1990, 10, 31),
     (min(_BGN.values()), max(_BGN.values()),
      int(np.floor(min(200 / n * 100 for n in _BGN.values()))),
      int(np.ceil(max(200 / n * 100 for n in _BGN.values())))), 0),
]

# ---- ORPHANED-CHECK GUARD. A check asserts "the paper says X and the source gives X", but it
# never verified the paper still SAYS X. After a trim removes a sentence, such a check keeps
# passing while guarding nothing. This reports every claimed value that no longer appears in
# paper.tex at the precision the check states, so a trim cannot silently orphan a guard.
_PAPER = _PAPER_SRC


def _in_paper(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return True
    if float(v).is_integer() and abs(v) < 10000:
        # an integral value may still be SPELLED with a decimal ("33.0" beside "19.7"), and
        # the bare-integer pattern rejects that because of the trailing dot
        if re.search(rf"(?<![\d.]){int(v)}(?![\d.])", _PAPER):
            return True
        return re.search(rf"(?<![\d.]){int(v)}\.0(?![\d])", _PAPER) is not None
    # A value counts as present only at ITS OWN precision. Matching any rounding made the
    # guard nearly vacuous: 0.067 was "found" because f"{0.067:.1f}" is "0.1", which occurs
    # in the paper as delta = 0.1, so two checks guarded prose that had never been written.
    for k in (0, 1, 2, 3, 4):
        if abs(round(abs(v), k) - abs(v)) < 1e-12 and f"{abs(v):.{k}f}" in _PAPER:
            return True
        # the paper often states a fraction as a percentage ("14.5 percent" for 0.145).
        # The literal must actually be marked as a percentage, or 0.091 would be "found"
        # by any unrelated 9.1 in the text.
        pc = abs(v) * 100
        if abs(round(pc, k) - pc) < 1e-12 and re.search(
                rf"(?<![\d.]){re.escape(f'{pc:.{k}f}')}(?![\d.])\$?(?:\s*to\s*\$?[\d.]+\$?)?\s*(percent|\\%)", _PAPER):
            return True
    return False


# A few checks bind a computed quantity to a claim the paper states in words rather than
# digits. The guard cannot find a literal for those, so each is exempted by name with the
# phrase it stands behind; anything not listed here must appear in the prose.
VERBAL = {
    "Sec 5: PointGoal1 certifies in about two thirds of draws at n = 200": "about two thirds",
    "App hyper: Walker2d accepts at the third grid threshold, the others at the first": "the third grid threshold",
    "App bullet: full-data CDT is safe on HalfCheetah and at 1.47x budget on both navigation tasks":
        "nearly 1.5 times budget",
    # reported as a cell of tab:compose, not in prose; the prose says only that the certified
    # selections "push it beyond" the budget
    "App composability: CPL violates on Walker2d's certified selection at 50.9 against 20":
        "push it beyond",
    # the paper states these as percentages, "11 percent" and "9 to 52 percent"
    "App proofs: DSRL pool safe fractions run to 51.8 percent, above the Bullet maximum":
        "9 to 52 percent of each pool",
    # the paper states these in words, "roughly two thirds" and "within two points"
    "App certprops: the cut is 64 percent and the gap to the oracle is 1.9 points":
        "roughly two thirds and lands within two points",
    # the paper rounds these to "5-to-22-percent"
    "App bullet: 200 labels are 5 to 22 percent of the main-study pools":
        "5-to-22-percent range",
    # the paper rounds the upper end to "109"
    "App composability: CDT without augmentation is safe on 4 velocity, unsafe on 4 navigation at 47 to 109":
        "costs 47 to 109",
}

orphans = []
for name, claimed, actual, tol in CHECKS:
    if name in VERBAL or not HAVE_TEX:
        continue
    vals = claimed if isinstance(claimed, (tuple, list)) else [claimed]
    missing = [v for v in vals if not _in_paper(v)]
    if missing:
        orphans.append((name, missing))

# checks whose "source" is the paper's own text rather than a data artifact
TEX_DEPENDENT = {"Sec 4 and App composability: the clone's 22 of 24 is stated identically in both"}

fails = 0
for name, claimed, actual, tol in CHECKS:
    if name in TEX_DEPENDENT and not HAVE_TEX:
        print(f"  SKIP  {name}")
        continue
    if isinstance(claimed, (tuple, list)):
        ok = len(claimed) == len(actual) and all(abs(c - a) <= tol for c, a in zip(claimed, actual))
    else:
        ok = abs(claimed - actual) <= tol
    if not ok:
        fails += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        paper says {claimed}, source gives {actual}"))
if orphans:
    print(f"\n  {len(orphans)} check(s) whose claimed value no longer appears in paper.tex "
          f"(the guarded sentence may have been trimmed):")
    for n, m in orphans:
        print(f"    {n}  -> missing {m}")
if not HAVE_TEX:
    print("\n  SKIP  paper.tex is not in this archive, so the orphan guard and the "
          "cross-section wording check did not run")
print(f"\nCONSTANTS: {'PASS' if not fails else f'{fails} MISMATCH'} "
      f"({len(CHECKS)} checks, "
      f"{len(orphans) if HAVE_TEX else 'orphan guard skipped, '}"
      f"{' orphaned' if HAVE_TEX else 'no paper.tex'})")
sys.exit(1 if fails else 0)
