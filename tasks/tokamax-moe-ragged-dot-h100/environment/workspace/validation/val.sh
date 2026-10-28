#!/bin/bash
# Development evaluator for /workspace/submission.
#   val.sh quick [--submission DIR]   2 workloads, short timing, no edge cases
#   val.sh full  [--submission DIR]   all 20 workloads + edge cases + reward
#   --only SUBSTR...  restrict to matching workloads (no reward file)
#   --out FILE        also write the detailed JSON result
# A completed full run writes /logs/verifier/reward.json:
#   {"reward": float, "invalid": int, "speedup_geomean": float,
#    "latency_us_geomean": float, "roofline_fraction_geomean": float,
#    "max_normalized_error": float}
# Evaluator or machine failures exit nonzero and write no reward file.
set -euo pipefail
if [[ ! -x /opt/venv/bin/python ]]; then
  echo "Evaluator error: /opt/venv/bin/python is missing; no reward was computed." >&2
  exit 2
fi
export OMP_NUM_THREADS=4
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/venv/bin/python -s -E "$ROOT/evaluate.py" "$@"
