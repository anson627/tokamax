#!/bin/bash
# Lays out /workspace as in PROJECT.md section 4, then runs the PoC checks.
# Usage (inside the pod, from the copied poc/ dir): bash run_in_pod.sh <step>
set -euo pipefail
POC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WS=/workspace PYTHON=/opt/venv/bin/python
setup() {
  rm -rf $WS && mkdir -p $WS/submission $WS/probes
  cp -r "$POC/workspace/moe_ffn" $WS/moe_ffn
  cp -r "$POC/workspace/validation" $WS/validation
  cp -r $WS/moe_ffn $WS/validation/starter          # frozen starter copy
  for p in "$POC"/probes/*; do
    n=$(basename "$p"); mkdir -p $WS/probes/$n
    cp -r "$p/moe_ffn" $WS/probes/$n/moe_ffn
  done
  for f in common.py kernel_sm90.py mgpu_utils.py; do
    cp $WS/moe_ffn/$f $WS/probes/tokamax_tuned/moe_ffn/
  done
  bash "$POC/baseline.sh"
}
case "${1:-setup}" in
  setup) setup ;;
  *) shift; bash $WS/validation/val.sh "$@" ;;
esac
