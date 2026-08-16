#!/bin/bash
set -u
cd /home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
env OMP_NUM_THREADS=4 conda run -n safevlmcpl --no-capture-output python /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/margin_expand.py > /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/margin_expand.log 2>&1 || echo "FAIL margin_expand" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe2_progress.log
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 conda run -n safevlmcpl --no-capture-output python /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/pool_scaling.py > /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/pool_scaling.log 2>&1 || echo "FAIL pool_scaling" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe2_progress.log
echo "PROBE2 ALL DONE" >> /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/scratchpad/probe2_progress.log
