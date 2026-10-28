#!/bin/bash
# Hidden evaluation of /workspace/submission. Takes no options.
# Writes /logs/verifier/reward.json:
#   {"reward": float, "invalid": int, "speedup_geomean": float,
#    "latency_us_geomean": float, "roofline_fraction_geomean": float,
#    "max_normalized_error": float}
# Evaluator or machine failures exit nonzero and write no reward file.
set -euo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/result.json
if [[ ! -x /opt/venv/bin/python ]]; then
  echo "Evaluator error: /opt/venv/bin/python is missing; no reward was computed." >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
  printf '{"reason": "GPU is not an H100: %s"}\n' \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -n1)" \
    > /logs/verifier/environment_unsupported.json
  exit 1
fi
export OMP_NUM_THREADS=4
exec /opt/venv/bin/python -s -E /tests/evaluate.py full
