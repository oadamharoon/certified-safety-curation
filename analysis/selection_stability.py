"""Selection-stability statistics for App. composability's opening sentence.

The claim "independent calibration draws frequently select the identical threshold" was
quoted from numbers computed by hand ("five of the eight tasks", "85 percent of certified
draws on Swimmer's top quantile") with nothing in the repo producing them, so they could
not be rechecked when the cohort was regenerated. This binds each to an expression.

Both selection summaries record, per (task, level), the certification rate over 500
resampled n = 200 calibration draws and the share of ALL draws landing on each of the three
most probable distinct certified thresholds (runs/scripts/build_v2_selections.py, distinct()).
The quantity the sentence is about is the share among CERTIFIED draws, so each prob is
renormalized by the certification rate.

Writes iclr2027/data/review_response/selection_stability.json
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


def _write_if_changed(path, text):
    """Write only when the content differs, so an identical regeneration keeps its mtime.

    The completeness gate's R11 re-runs these builders on every invocation to prove they
    reproduce. An unconditional write bumps the artifact's mtime, which R9 and R14 then read as
    "the table/figure is older than its input" -- the freshness rules would eat each other.
    """
    import os as _os
    if _os.path.exists(path) and open(path).read() == text:
        return False
    with open(path, "w") as _fh:
        _fh.write(text)
    return True


W = CSC_WORKSPACE
SEL = f"{W}/runs/selections"
DST = f"{W}/iclr2027/data/review_response/selection_stability.json"

rows = []
v2 = json.load(open(f"{SEL}/v2_summary.json"))
for task, rec in v2.items():
    for lvl in ("a25", "a40"):
        e = rec.get(lvl)
        if not e or not e.get("distinct"):
            continue
        rate = e["cert_rate_500"]
        probs = sorted((d["prob"] for d in e["distinct"]), reverse=True)
        dep = e.get("deployed", {})
        rows.append({
            "task": task, "level": lvl, "cert_rate_500": rate,
            "modal_share_of_certified": probs[0] / rate if rate else None,
            "top3_share_of_certified": sum(probs) / rate if rate else None,
            "n_distinct_reported": len(e["distinct"]),
            "deployed_n": dep.get("n"),
            "deployed_is_a_grid_selection": any(d["n"] == dep.get("n") for d in e["distinct"]),
        })
a40 = json.load(open(f"{SEL}/a40new_summary.json"))
for task, rec in a40.items():
    if task in v2:
        continue
    rate = rec["cert_rate"]
    probs = sorted((d["prob"] for d in rec["selections"]), reverse=True)
    rows.append({
        "task": task, "level": "a40", "cert_rate_500": rate,
        "modal_share_of_certified": probs[0] / rate if rate else None,
        "top3_share_of_certified": sum(probs) / rate if rate else None,
        "n_distinct_reported": len(rec["selections"]),
        "deployed_n": None, "deployed_is_a_grid_selection": None,
    })

rows.sort(key=lambda r: (r["level"], r["task"]))
shares = [r["modal_share_of_certified"] for r in rows if r["modal_share_of_certified"] is not None]
best = max(rows, key=lambda r: r["modal_share_of_certified"])
payload = {
    "n_settings": len(rows),
    "modal_share_min": min(shares), "modal_share_max": max(shares),
    "n_settings_modal_share_above_half": sum(s > 0.5 for s in shares),
    "most_concentrated": {"task": best["task"], "level": best["level"],
                          "modal_share_of_certified": best["modal_share_of_certified"]},
    "top3_share_min": min(r["top3_share_of_certified"] for r in rows),
    "settings": rows,
}
_write_if_changed(DST, json.dumps(payload, indent=1))
for r in rows:
    print(f"  {r['task']:22s} {r['level']}  cert {r['cert_rate_500']:.3f}  "
          f"modal share of certified {r['modal_share_of_certified']:.2f}  "
          f"top3 {r['top3_share_of_certified']:.2f}")
print(f"\n  {payload['n_settings_modal_share_above_half']} of {len(rows)} settings above half; "
      f"range {payload['modal_share_min']:.2f} to {payload['modal_share_max']:.2f}; "
      f"most concentrated {best['task']} {best['level']} at {best['modal_share_of_certified']:.2f}")
print(f"  wrote {DST}")
