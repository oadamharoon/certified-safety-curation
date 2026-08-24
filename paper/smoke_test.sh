#!/bin/bash
# Quick pre-flight: does this script run to completion?
#
# Usage: smoke_test.sh <script.py> <task> [EXTRA_ENV=v ...]
#   e.g. smoke_test.sh 04f_train_v_awr.py cargoal2 AWR_BETA=0.1 AWR_WEIGHT_CLIP=20
#
# Runs the target once against a shrunken config (1 epoch, 20 preference pairs,
# 2 eval episodes) via SAFETY_VLM_CONFIG, which load_cfg genuinely honours.
# Do NOT shrink via env vars like V_EPOCHS: several scripts read those knobs
# from the config only, so an env override is silently ignored and you end up
# running the full job.
#
# Scope: this answers "does it complete", nothing more. Semantic defects are
# other tools' jobs. lint_pipeline.py catches a trainer that ignores
# SEED_OVERRIDE; validate_snapshot.py catches results that never land.
set -u
SCRIPT=${1:?usage: smoke_test.sh <script.py> <task> [ENV=v ...]}
TASK=${2:?usage: smoke_test.sh <script.py> <task> [ENV=v ...]}
shift 2
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
CFG=$TMP/smoke_config.yaml

/home/omniverse/miniconda3/envs/safevlmcpl/bin/python - "$D/config.yaml" "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.update(cpl_epochs=1, v_epochs=1, awr_epochs=1, bc_epochs=1,
           bc_only_epochs=1, num_pairs=20, eval_episodes=2)
yaml.safe_dump(cfg, open(sys.argv[2], "w"))
PY

echo "smoke: $SCRIPT on $TASK (1 epoch, 20 pairs, 2 eval episodes)"
t0=$SECONDS
cd "$D"
env SAFETY_VLM_TASK=$TASK SAFETY_VLM_CONFIG=$CFG WANDB_MODE=disabled \
    OMP_NUM_THREADS=2 SEED_OVERRIDE=0 OUT_TAG=smoke "$@" \
    conda run -n safevlmcpl --no-capture-output python scripts/$SCRIPT \
    > "$TMP/run.log" 2>&1
rc=$?
dt=$((SECONDS - t0))
if [ $rc -ne 0 ]; then
  echo "  FAIL after ${dt}s (exit $rc), last lines:"
  tail -15 "$TMP/run.log" | sed 's/^/    /'
else
  echo "  PASS in ${dt}s"
fi
find "$D/outputs/$TASK" -name "*smoke*" -delete 2>/dev/null
exit $rc
