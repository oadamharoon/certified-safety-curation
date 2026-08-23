#!/bin/bash
# Cheap pre-flight for any training stage. Run this BEFORE launching a sweep.
#
# Usage: smoke_test.sh <script.py> <task> [EXTRA_ENV=v ...]
#   e.g. smoke_test.sh 04f_train_v_awr.py cargoal2 AWR_BETA=0.1 AWR_WEIGHT_CLIP=20
#
# Runs the target twice at 2 epochs with SEED_OVERRIDE=0 and =1, then compares
# the checkpoints. Identical weights mean the script ignores the seed, which is
# exactly the defect that voided the T3.2 sweep: 135 jobs and 34 hours spent
# producing three bit-identical copies of every configuration. This catches it
# in about a minute.
set -u
SCRIPT=${1:?usage: smoke_test.sh <script.py> <task> [ENV=v ...]}
TASK=${2:?usage: smoke_test.sh <script.py> <task> [ENV=v ...]}
shift 2
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
OUT=$D/outputs/$TASK
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "smoke: $SCRIPT on $TASK, 2 epochs, seeds 0 and 1"
for s in 0 1; do
  cd $D
  env SAFETY_VLM_TASK=$TASK WANDB_MODE=disabled OMP_NUM_THREADS=3 \
      V_EPOCHS=2 AWR_EPOCHS=2 BC_EPOCHS=2 \
      SEED_OVERRIDE=$s OUT_TAG=smoke_seed$s "$@" \
      conda run -n safevlmcpl --no-capture-output python scripts/$SCRIPT \
      > "$TMP/seed$s.log" 2>&1 \
    || { echo "  FAIL: seed $s exited nonzero"; tail -15 "$TMP/seed$s.log"; exit 1; }
  echo "  seed $s ok"
done

# find whatever policy artifacts the two runs produced
a=$(ls -t "$OUT"/*smoke_seed0*.pt 2>/dev/null | head -1)
b=$(ls -t "$OUT"/*smoke_seed1*.pt 2>/dev/null | head -1)
if [ -z "$a" ] || [ -z "$b" ]; then
  echo "  WARN: no checkpoint written, cannot compare seeds"
  exit 0
fi
/home/omniverse/miniconda3/envs/safevlmcpl/bin/python - "$a" "$b" <<'PY'
import sys, torch
x, y = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
sx, sy = x.get("state_dict", x), y.get("state_dict", y)
same = all(torch.equal(sx[k], sy[k]) for k in sx)
print("  FAIL: seeds 0 and 1 produced IDENTICAL weights, the script ignores"
      " SEED_OVERRIDE" if same else "  PASS: seeds produce different weights")
sys.exit(1 if same else 0)
PY
rc=$?
rm -f "$OUT"/*smoke_seed*  2>/dev/null
exit $rc
