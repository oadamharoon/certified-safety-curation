#!/bin/bash
# Conditional composability echo (pre-registered decision rule):
# runs ONLY if full-data CDT mean cost on CarRun exceeds 10.
set -u
S=/tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad
LOGDIR=$S/echo_logs
mkdir -p "$LOGDIR"
log_run () { echo "[$(date +%m/%d-%H:%M:%S)] $1" | tee -a "$LOGDIR/progress.log"; }

while ! grep -q "ROUND-2 QUEUE ALL DONE" $S/round2_logs/progress.log 2>/dev/null; do
  sleep 900
done
log_run "ROUND-2 COMPLETE - EVALUATING ECHO RULE"

DEC=$(cd /home/omniverse/workspace/safevlmcpl/iclr2027 && conda run -n safevlmcpl --no-capture-output python - <<'PYEOF'
import json
d = json.load(open("data/osrl_results.json"))
e = d.get("carrun_b", {}).get("cdt", {})
if not e:
    print("NODATA"); raise SystemExit
m = sum(v["C"] for v in e.values())/len(e)
print("TRIGGER" if m > 10 else f"NULL:{m:.2f}")
PYEOF
)
log_run "echo rule decision: $DEC"
case "$DEC" in
  TRIGGER)
    log_run "BUILDING CARRUN CERTIFIED SUBSET"
    cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
    env CUDA_VISIBLE_DEVICES="" conda run -n safevlmcpl --no-capture-output python - > "$LOGDIR/build.log" 2>&1 <<'PYEOF'
import json, pickle, sys, os
import numpy as np, torch, h5py
sys.path.insert(0, ".")
from model.policy import VEnsemble
import yaml
cfg = yaml.safe_load(open("config.yaml"))
snap = json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/results_snapshot.json"))
meta = snap["carrun_b"]["calfilt_ltt"]["0"]["meta"]
assert meta["certified"]
tau = float(meta["tau"])
trajs = pickle.load(open(cfg["tasks"]["carrun_b"]["data_pickle"], "rb"))
obs_dim = trajs[0]["observations"].shape[1]
ens = VEnsemble(obs_dim, 256, K=3)
ck = torch.load("outputs/carrun_b/v_ensemble_pess_seed0.pt", map_location="cpu", weights_only=False)
ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
ens.eval()
g = np.zeros(len(trajs))
with torch.no_grad():
    for i, t in enumerate(trajs):
        o = torch.as_tensor(t["observations"], dtype=torch.float32)
        g[i] = ens(o).mean().item()
kept = g >= tau
kf, ku = float(kept.mean()), float((np.array([t["costs"].sum() for t in trajs])[kept] > 10).mean())
assert abs(kf - meta["kept_frac"]) < 0.005 and abs(ku - meta["kept_unsafe_rate"]) < 0.005, "verify fail"
import gymnasium as gym
import dsrl
if hasattr(dsrl, "register_envs"): dsrl.register_envs()
env = gym.make("OfflineCarRun-v0"); d = env.get_dataset(); env.close()
term = np.asarray(d["terminals"], dtype=bool); tout = np.asarray(d["timeouts"], dtype=bool)
end = np.logical_or(term, tout)
if not end[-1]: end[-1] = True
ends = np.where(end)[0]; starts = np.concatenate([[0], ends[:-1]+1])
spans = [(s, e) for s, e in zip(starts, ends) if e+1-s >= 2]
assert len(spans) == len(trajs)
idx = np.concatenate([np.arange(s, e+1) for (s, e), k in zip(spans, kept) if k])
OUT = "SCRATCH/certified_h5/carrun_b_echocert_seed0.hdf5".replace("SCRATCH", os.environ["S"])
with h5py.File(OUT, "w") as f:
    for key in ("observations", "actions", "rewards", "costs", "next_observations"):
        if key in d: f.create_dataset(key, data=np.asarray(d[key])[idx])
    f.create_dataset("terminals", data=term[idx]); f.create_dataset("timeouts", data=tout[idx])
json.dump({"kept": np.where(kept)[0].tolist()}, open(OUT.replace(".hdf5", "_kept.json"), "w"))
print("built", OUT, "VERIFIED")
PYEOF
    grep -q VERIFIED "$LOGDIR/build.log" || { log_run "BUILD FAILED"; exit 1; }
    for seed in 0 1 2; do
      tag="cdtecho_carrun_s${seed}"
      [ -f "$LOGDIR/done_${tag}" ] && continue
      cd /home/omniverse/workspace/safevlmcpl/osrl
      env PYTHONNOUSERSITE=1 PYTHONPATH=/home/omniverse/workspace/safevlmcpl/osrl \
        conda run -n safevlmcpl --no-capture-output \
        python examples/train/train_cdt.py --task OfflineCarRun-v0 --seed $seed \
        --cost_limit 10 --device cuda --augment_percent 0.0 --random_aug 0.0 \
        --subset_h5 "$S/certified_h5/carrun_b_echocert_seed0.hdf5" \
        --logdir "$LOGDIR/runs" > "$LOGDIR/${tag}.log" 2>&1 \
        && { touch "$LOGDIR/done_${tag}"; log_run "DONE $tag"; } || log_run "FAIL $tag"
    done
    for seed in 0 1 2; do
      tag="bcecho_carrun_seed${seed}"
      key="carrun_b_${tag}"
      [ -f "$LOGDIR/done_${key}" ] && continue
      cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
      env SAFETY_VLM_TASK=carrun_b KEPT_JSON="$S/certified_h5/carrun_b_echocert_seed0_kept.json" \
        SEED_OVERRIDE=$seed OUT_TAG="$tag" WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
        conda run -n safevlmcpl --no-capture-output python $S/bc_on_subset.py > "$LOGDIR/${key}.log" 2>&1 \
        && env SAFETY_VLM_TASK=carrun_b WANDB_MODE=disabled OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" \
           conda run -n safevlmcpl --no-capture-output python scripts/05_evaluate.py \
           --policy_file "bc_${tag}_policy.pt" --results_suffix "$tag" >> "$LOGDIR/${key}.log" 2>&1 \
        && { touch "$LOGDIR/done_${key}"; log_run "DONE $key"; } || log_run "FAIL $key"
    done
    cd /home/omniverse/workspace/safevlmcpl/iclr2027
    conda run -n safevlmcpl --no-capture-output python scripts/harvest_osrl.py >> "$LOGDIR/progress.log" 2>&1
    conda run -n safevlmcpl --no-capture-output python scripts/collect_results.py >> "$LOGDIR/progress.log" 2>&1
    log_run "ECHO COMPLETE"
    ;;
  NULL:*)
    log_run "ECHO NOT TRIGGERED (full-data CDT safe on CarRun, ${DEC#NULL:}); null recorded per prereg"
    ;;
  *)
    log_run "NO CARRUN CDT DATA YET - check manually"
    ;;
esac
