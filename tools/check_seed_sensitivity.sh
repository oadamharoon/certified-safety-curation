#!/bin/bash
# Does this trainer actually respond to SEED_OVERRIDE?
#
# Usage: check_seed_sensitivity.sh <script.py> <task> [ENV=v ...]
#
# Runs the target twice at 1 epoch (via SAFETY_VLM_CONFIG, the only shrink
# mechanism load_cfg reliably honours) with SEED_OVERRIDE=0 and =1, then
# compares the checkpoints. Identical weights mean the seed is ignored and any
# multi-seed sweep built on it is worthless.
#
# This is the check that matters before a sweep. smoke_test.sh only proves the
# script completes; a script that completes perfectly can still produce three
# identical copies of every configuration, which is exactly what happened to
# T3.2 and cost 34 hours, then nearly cost 3 days more.
# --- paths: set CSC_WORKSPACE or the individual roots; see the README ---
_csc_root () { local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do [ -e "$d/.csc-root" ] && { printf %s "$d"; return; }; d="$(dirname "$d")"; done
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); }
CSC_REPO="${CSC_REPO:-$(_csc_root)}"
CSC_WORKSPACE="${CSC_WORKSPACE:-$(dirname "$CSC_REPO")}"
CSC_WORK="${CSC_WORK:-$CSC_WORKSPACE/vlm-with-cpl/new_data}"
CSC_RUNS="${CSC_RUNS:-$([ -d "$CSC_WORKSPACE/runs" ] && printf %s "$CSC_WORKSPACE/runs" || printf %s "$CSC_REPO/runs")}"
CSC_OSRL="${CSC_OSRL:-$CSC_WORKSPACE/osrl}"
CSC_PAPER="${CSC_PAPER:-$CSC_REPO/paper}"
CSC_PAPER_DATA="${CSC_PAPER_DATA:-$CSC_PAPER/data}"
CSC_CONFIG="${CSC_CONFIG:-$CSC_REPO/configs}"
PYTHON="${PYTHON:-python}"
# ------------------------------------------------------------------------

set -u
SCRIPT=${1:?usage: check_seed_sensitivity.sh <script.py> <task> [ENV=v ...]}
TASK=${2:?usage: check_seed_sensitivity.sh <script.py> <task> [ENV=v ...]}
shift 2
D=${CSC_WORK}   # the LIVE tree
TMP=$(mktemp -d); trap 'rm -rf "$TMP"; find "$D/outputs/$TASK" -name "*seedchk*" -delete 2>/dev/null' EXIT
${PYTHON} - "$D/config.yaml" "$TMP/cfg.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.update(cpl_epochs=1, v_epochs=1, awr_epochs=1, bc_epochs=1, bc_only_epochs=1,
         num_pairs=20, eval_episodes=2)
yaml.safe_dump(c, open(sys.argv[2], "w"))
PY
echo "seed check: $SCRIPT on $TASK"
for s in 0 1; do
  (cd "$D" && env SAFETY_VLM_TASK=$TASK SAFETY_VLM_CONFIG=$TMP/cfg.yaml WANDB_MODE=disabled \
     OMP_NUM_THREADS=2 SEED_OVERRIDE=$s OUT_TAG=seedchk_s$s "$@" \
     conda run -n safevlmcpl --no-capture-output python scripts/$SCRIPT > "$TMP/s$s.log" 2>&1) \
    || { echo "  FAIL: seed $s exited nonzero"; tail -12 "$TMP/s$s.log" | sed 's/^/    /'; exit 1; }
  echo "  seed $s done"
done
${PYTHON} - "$D/outputs/$TASK" <<'PY'
import sys, glob, torch
d = sys.argv[1]
f0 = sorted(glob.glob(f"{d}/*seedchk_s0*.pt")); f1 = sorted(glob.glob(f"{d}/*seedchk_s1*.pt"))
if not (f0 and f1):
    print("  INCONCLUSIVE: no checkpoints written"); sys.exit(2)
a = torch.load(f0[0], map_location="cpu", weights_only=False)
b = torch.load(f1[0], map_location="cpu", weights_only=False)
sa, sb = a.get("state_dict", a), b.get("state_dict", b)
md = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
if md == 0.0:
    print("  FAIL: seeds 0 and 1 gave IDENTICAL weights, SEED_OVERRIDE is ignored")
    sys.exit(1)
print(f"  PASS: seeds diverge, max abs weight diff {md:.4e}")
PY
