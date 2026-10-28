#!/bin/bash
# Usage: val.sh quick|full [--clean] [--submission DIR] [--out FILE] [--only SUBSTR...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
exec "$PY" "$HERE/evaluate.py" "$@"
