#!/usr/bin/env bash
# Activate the desired virtual environment first; colab is discovered on PATH.
# Falls back to python3 when no venv is active (plain `python` may not exist).
#
# Usage:
#   run_sqmc_ekf_colab.sh [--config overrides.json]   # launch a new run
#   run_sqmc_ekf_colab.sh --resume <output-dir>       # reconnect to a detached run
#
# The Colab session is preserved when local monitoring disconnects; reconnect
# with --resume to collect the results without restarting training.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PYTHON:-}" ]; then
    INTERPRETER="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    INTERPRETER="python"
else
    INTERPRETER="python3"
fi
exec "$INTERPRETER" "$SCRIPT_DIR/run_sqmc_ekf_local.py" "$@"
