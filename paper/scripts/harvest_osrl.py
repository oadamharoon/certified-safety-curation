"""Harvest OSRL baseline results (CDT/CPQ/COptiDICE) from wandb run dirs.

CDT is target-conditioned; we report its most conservative evaluated target
(cost target 20, at or below every task budget). CPQ/COptiDICE report their
single final eval. Output: data/osrl_results.json
  {task_key: {algo: {seed: {"R": r, "C": c}}}}
"""

# --- paths ------------------------------------------------------------------
# Datasets, checkpoints and run output live outside the repository. Set CSC_WORKSPACE,
# or the individual roots, to point at yours. See the README.
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
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "datasets"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs ship with the repository, so they resolve on their own
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import glob
import json
import os
import re

OSRL = CSC_OSRL
OUT = CSC_PAPER_DATA + "/osrl_results.json"

ENV2TASK = {
    "OfflineHalfCheetahVelocityGymnasium-v1": "halfcheetah_velocity",
    "OfflineWalker2dVelocityGymnasium-v1": "walker2d_velocity",
    "OfflineAntVelocityGymnasium-v1": "ant_velocity",
    "OfflineHopperVelocityGymnasium-v1": "hopper_velocity",
    "OfflineSwimmerVelocityGymnasium-v1": "swimmer_velocity",
    "OfflineCarGoal1Gymnasium-v0": "cargoal1_dsrl",
    "OfflineCarGoal2Gymnasium-v0": "cargoal2",
    "OfflinePointGoal1Gymnasium-v0": "pointgoal1_dsrl",
    "OfflinePointGoal2Gymnasium-v0": "pointgoal2",
    "OfflinePointButton1Gymnasium-v0": "pointbutton1",
    "OfflinePointButton2Gymnasium-v0": "pointbutton2",
    "OfflineCarButton1Gymnasium-v0": "carbutton1_t3",
    "OfflineCarButton2Gymnasium-v0": "carbutton2",
    "OfflinePointCircle1Gymnasium-v0": "pointcircle1",
    "OfflinePointCircle2Gymnasium-v0": "pointcircle2",
    "OfflineBallRun-v0": "ballrun_b",
    "OfflineBallCircle-v0": "ballcircle_b",
    "OfflineCarCircle-v0": "carcircle_b",
    "OfflineCarRun-v0": "carrun_b",
    "OfflineDroneRun-v0": "dronerun_b",
}



def _write_if_changed(path, text):
    """Rewrite only when the content differs.

    The completeness gate re-runs this harvester every time, and an unconditional write bumps
    the mtime of a file that R9 and R14 use as the freshness reference for every table and
    figure. That made both rules fail forever on artifacts nothing had actually changed.
    """
    import os
    if os.path.exists(path) and open(path).read() == text:
        return False
    with open(path, "w") as fh:
        fh.write(text)
    return True


def main():
    out = {}
    n = 0
    for d in glob.glob(os.path.join(OSRL, "wandb", "run-*")):
        meta_p = os.path.join(d, "files", "wandb-metadata.json")
        summ_p = os.path.join(d, "files", "wandb-summary.json")
        if not (os.path.exists(meta_p) and os.path.exists(summ_p)):
            continue
        try:
            meta = json.load(open(meta_p))
            s = json.load(open(summ_p))
        except Exception:
            continue
        args = " ".join(meta.get("args", []))
        prog = meta.get("program", "")
        m_algo = re.search(r"train_(cdt|cpq|coptidice)\.py", prog + " " + args)
        variant = ""
        if "--subset_h5" in args:
            import re as _re
            _m = _re.search(r"_(a25|a40)selq(\d+)_", args)
            # A3 wrote its selections as <task>_a25_q65.hdf5 rather than the
            # older _a25selq65_ form. Without this branch those runs fall
            # through to the bare "_a40"/"_cert" cases below and overwrite
            # unrelated results, since "_a40_q85" contains "_a40_".
            _m3 = _re.search(r"_(a25|a40)_q(\d+)\.hdf5", args)
            # A3's regenerated grids are <task>_a40new_q30.hdf5 and the CarRun
            # echo rerun is <task>_echonew_q80.hdf5. Neither contains "_a40_"
            # or "_a25_q", so both fell through EVERY branch below and landed in
            # the final else, i.e. they were recorded as cdt_cert -- the arm the
            # "27 of 27" claim rests on. Same failure mode the _m3 comment above
            # describes, one naming generation later. Match them explicitly and
            # keep them ahead of the bare-prefix tests.
            _m4 = _re.search(r"_(a40new|a25new)_q(\d+)\.hdf5", args)
            _m5 = _re.search(r"_(echonew)_q(\d+)\.hdf5", args)
            if _m4:
                variant = f"_{_m4.group(1)}_q{_m4.group(2)}"
            elif _m5:
                variant = f"_{_m5.group(1)}_q{_m5.group(2)}"
            elif _m:
                variant = f"_{_m.group(1)}selq{_m.group(2)}"
            elif _m3:
                variant = f"_{_m3.group(1)}selq{_m3.group(2)}"
            elif "_a40d3_" in args:
                variant = "_a40_draw3"
            elif "_a40d2_" in args:
                variant = "_a40_draw2"
            elif "_a40_" in args:
                variant = "_a40"
            elif "_draw3_" in args:
                variant = "_cert_draw3"
            elif "_draw2_" in args:
                variant = "_cert_draw2"
            else:
                variant = "_cert"
        elif "--augment_percent 0.0" in args or "--augment_percent=0.0" in args:
            variant = "_noaug"
        m_env = None
        for env in ENV2TASK:
            if env in args:
                m_env = env
                break
        m_seed = re.search(r"--seed[= ](\d+)", args)
        if not (m_algo and m_env and m_seed):
            continue
        algo, task, seed = m_algo.group(1) + variant, ENV2TASK[m_env], m_seed.group(1)

        if algo.startswith("cdt"):
            # most conservative evaluated target at or below the task budget:
            # DSRL budgets 20/25 -> target 20; Bullet budget 10 -> target 10
            tgt = "10" if task.endswith("_b") else "20"
            ck = [k for k in s if k.startswith(f"cost/c_{tgt}")]
            rk = [k for k in s if k.startswith(f"ret/c_{tgt}")]
            if not (ck and rk):
                continue
            R, C = float(s[rk[0]]), float(s[ck[0]])
        else:
            # single-eval algos: OSRL logs eval/Reward and eval/Cost
            rk = [k for k in s if k in ("eval/Reward", "eval/reward", "ret/reward")] or \
                 [k for k in s if k.startswith("ret/")]
            ck = [k for k in s if k in ("eval/Cost", "eval/cost", "ret/cost")] or \
                 [k for k in s if k.startswith("cost/")]
            if not (rk and ck):
                continue
            R, C = float(s[rk[0]]), float(s[ck[0]])
        # keep the LATEST run per (task, algo, seed): overwrite by mtime order
        out.setdefault(task, {}).setdefault(algo, {})
        prev = out[task][algo].get(seed)
        mt = os.path.getmtime(summ_p)
        if prev is None or mt > prev["_mtime"]:
            out[task][algo][seed] = {"R": R, "C": C, "_mtime": mt}
            n += 1
    mt = {t: {a: {sd: out[t][a][sd]["_mtime"] for sd in out[t][a]} for a in out[t]} for t in out}
    json.dump(mt, open(OUT.replace(".json", "_mtimes.json"), "w"), indent=1)   # provenance for completeness_check.py
    for task in out:
        for algo in out[task]:
            for seed in out[task][algo]:
                out[task][algo][seed].pop("_mtime", None)
    # V2 remediation (F2 dedupe): where a regenerated task's deployed selection is the same kept
    # set as one of its grid selections, the deployed CDT arm was not trained a second time; its
    # key is the identical grid cell (runs/scripts/v2_aliases.py). Any older run under the deployed
    # key belongs to the archived cohort and is replaced.
    import sys
    sys.path.insert(0, CSC_RUNS + "/scripts")
    from v2_aliases import aliases
    for task, al in aliases().items():
        for dep, grid in al.items():
            if dep.startswith("cdt") and grid in out.get(task, {}):
                out[task][dep] = dict(out[task][grid])
    _write_if_changed(OUT, json.dumps(out, indent=1))
    cov = {a: sum(len(out[t].get(a, {})) for t in out)
           for a in ("cdt", "cdt_cert", "cdt_a40", "cdt_noaug", "cpq", "coptidice")}
    print(f"harvested -> {OUT} | coverage: {cov}")


if __name__ == "__main__":
    main()
