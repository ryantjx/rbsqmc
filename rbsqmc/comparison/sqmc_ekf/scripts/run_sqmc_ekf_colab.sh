#!/usr/bin/env bash
# Activate the desired virtual environment first; colab is discovered on PATH.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/run_sqmc_ekf_local.py" "$@"
