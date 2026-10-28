#!/bin/bash
# Packages the unchanged starter as the baseline submission.
set -euo pipefail
mkdir -p /workspace/submission
rm -rf /workspace/submission/moe_ffn
cp -R /workspace/moe_ffn /workspace/submission/moe_ffn
find /workspace/submission/moe_ffn -name __pycache__ -prune -exec rm -rf {} +
cat > /workspace/submission/summary.md <<'SUMMARY'
## Experiments
No optimization experiments. This packages the supplied starter as the baseline.

## Submitted solution
The unchanged starter from /workspace/moe_ffn: the Tokamax sm90 Pallas Mosaic GPU
ragged-dot kernel with its default H100 config (64x64x128 tiles, 2 stages,
non-persistent), called for the gate/up projection and the down projection with
an XLA SwiGLU in between. `prepare` returns the weights unchanged. No build step.
Reproduce with `bash /workspace/validation/val.sh full`.
SUMMARY
