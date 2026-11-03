#!/usr/bin/env bash
# Unified inference: use configs/ir.yaml defaults; --tile_size 0 selects whole-image inference.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_exec="$PYTHON_BIN"
else
    python_exec=python
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$python_exec" "$SCRIPT_DIR/src/inference_s3diff.py" "$@"
