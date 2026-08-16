"""Harvest OSRL baseline results (CDT/CPQ/COptiDICE) from wandb run dirs.

CDT is target-conditioned; we report its most conservative evaluated target
(cost target 20, at or below every task budget). CPQ/COptiDICE report their
single final eval. Output: data/osrl_results.json
  {task_key: {algo: {seed: {"R": r, "C": c}}}}
"""
import glob
import json
import os
import re

OSRL = "/home/omniverse/workspace/safevlmcpl/osrl"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data/osrl_results.json"

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
            if _m:
                variant = f"_{_m.group(1)}selq{_m.group(2)}"
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
    for task in out:
        for algo in out[task]:
            for seed in out[task][algo]:
                out[task][algo][seed].pop("_mtime", None)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    cov = {a: sum(len(out[t].get(a, {})) for t in out)
           for a in ("cdt", "cdt_cert", "cdt_a40", "cdt_noaug", "cpq", "coptidice")}
    print(f"harvested -> {OUT} | coverage: {cov}")


if __name__ == "__main__":
    main()
