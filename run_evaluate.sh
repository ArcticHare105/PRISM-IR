#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_exec="$PYTHON_BIN"
else
    python_exec=python
fi
exec "$python_exec" "$SCRIPT_DIR/evaluate.py" "$@"
