#!/bin/bash
# In the pod: lay out /workspace and /tests as the two Dockerfiles do.
set -euo pipefail
T=${TASK_SRC:-/src/tokamax-moe-ragged-dot-h100}
id -u 2000 >/dev/null 2>&1 || useradd --uid 2000 --create-home candidate
mkdir -p /workspace/submission /tests /logs/verifier /opt/moe_eval
cp -R $T/environment/workspace/. /workspace/
mkdir -p /workspace/validation/starter
cp -R /workspace/moe_ffn/. /workspace/validation/starter/moe_ffn/ 2>/dev/null || cp -R /workspace/moe_ffn /workspace/validation/starter/moe_ffn
cp /workspace/validation/worker.py /workspace/validation/reference.py /opt/moe_eval/
chmod -R a+rX,go-w /workspace/validation /opt/moe_eval
cp -R $T/tests/. /tests/
chmod -R go-rwx /tests
