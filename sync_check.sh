#!/bin/bash
# Verify the repo matches the live working tree, and optionally sync.
#   ./sync_check.sh          check only (exit 1 if drift)
#   ./sync_check.sh --sync   copy live -> repo, then report what changed
R="$(cd "$(dirname "$0")" && pwd)"
D=/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data
SYNC=0; [ "${1:-}" = "--sync" ] && SYNC=1
drift=0
check () { # repo_path live_path
  if ! diff -q "$1" "$2" >/dev/null 2>&1; then
    echo "DRIFT: ${1#$R/}"
    [ $SYNC -eq 1 ] && cp "$2" "$1" && echo "  synced from live tree"
    drift=$((drift+1))
  fi
}
for f in config.yaml config_h10.yaml config_h50.yaml; do
  [ -f "$D/$f" ] && check "$R/configs/$f" "$D/$f"
done
for dir in pipeline analysis baselines legacy; do
  for f in "$R"/$dir/*.py; do
    b=$(basename "$f"); [ -f "$D/scripts/$b" ] && check "$f" "$D/scripts/$b"
  done
done
for f in "$R"/src/model/*.py; do b=$(basename "$f"); [ -f "$D/model/$b" ] && check "$f" "$D/model/$b"; done
for f in "$R"/src/utils/*.py; do b=$(basename "$f"); [ -f "$D/utils/$b" ] && check "$f" "$D/utils/$b"; done
if [ $drift -eq 0 ]; then echo "IN SYNC: repo matches the live tree"; else
  echo "$drift file(s) drifted"; [ $SYNC -eq 0 ] && exit 1; fi
