#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_exec="$PYTHON_BIN"
else
    python_exec=python
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$python_exec" "$SCRIPT_DIR/src/evaluate_img_all.py" "$@"
