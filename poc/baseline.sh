#!/bin/bash
# Copies the unchanged starter into the submission folder (PROJECT.md section 7).
set -euo pipefail
WS="${WS:-/workspace}"
rm -rf "$WS/submission/moe_ffn"
mkdir -p "$WS/submission"
cp -r "$WS/moe_ffn" "$WS/submission/moe_ffn"
cat > "$WS/submission/summary.md" <<'MD'
## Experiments
None. This is the unchanged starter.

## Submitted solution
Tokamax sm90 ragged dot with the default H100 config (64x64x128, 2 stages),
called twice with an XLA SwiGLU in between. Reproduce: `bash /workspace/validation/val.sh full`.
MD
