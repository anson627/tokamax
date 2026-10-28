#!/bin/bash
# Copies the shared evaluator files and the starter into tests/ (no symlinks allowed).
set -euo pipefail
T=tasks/tokamax-moe-ragged-dot-h100; V=$T/environment/workspace/validation
for f in evaluate.py harness.py worker.py reference.py; do cp $V/$f $T/tests/$f; done
mkdir -p $T/tests/starter/moe_ffn
cp $T/environment/workspace/moe_ffn/*.py $T/tests/starter/moe_ffn/
