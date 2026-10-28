#!/bin/bash
# Copies the reference solution's deliverables into /workspace/submission.
set -euo pipefail
SRC="$(dirname "$(realpath "$0")")/reference"
mkdir -p /workspace/submission
rm -rf /workspace/submission/moe_ffn
cp -R "$SRC/moe_ffn" /workspace/submission/moe_ffn
cp "$SRC/summary.md" /workspace/submission/summary.md
