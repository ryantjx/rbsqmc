#!/usr/bin/env bash
set -euo pipefail

# Run smoothing.py on Colab GPU (test mode), download outputs, and verify.
#
# Usage:  ./run_colab_smoothing_test.sh

SESSION="rbsqmc"
REMOTE_OUTPUTS="/content/rbsqmc/rbpf/outputs"
LOCAL_OUTPUTS="./rbpf/outputs_test"
SCRIPT="run_smoothing_gpu_test.py"

echo "============================================================"
echo "  COLAB SMOOTHING GPU TEST — STARTING"
echo "============================================================"

# --- Step 1: Run the bootstrap script on a GPU VM ---
echo "[1/4] Launching Colab GPU session and running ${SCRIPT}..."
colab run --gpu T4 --keep "${SCRIPT}"

# --- Step 2: Download outputs ---
echo "[2/4] Downloading outputs from VM..."
mkdir -p "${LOCAL_OUTPUTS}"
colab download -s "${SESSION}" "${REMOTE_OUTPUTS}" "${LOCAL_OUTPUTS}"

# --- Step 3: Verify outputs ---
echo "[3/4] Checking outputs..."
echo ""
echo "--- Output files ---"
ls -lhR "${LOCAL_OUTPUTS}"
echo ""

# Print per-epoch parameter summaries
for f in "${LOCAL_OUTPUTS}"/em_params_epoch_*.json; do
    if [ -f "$f" ]; then
        epoch=$(basename "$f" | grep -oE '[0-9]+')
        echo "--- Epoch ${epoch} ---"
        python3 -c "
import json, sys
with open('$f') as fh:
    p = json.load(fh)
print(f\"  κ = {p['init_kappa']}\")
print(f\"  α = {p['init_alpha']}\")
print(f\"  β = {p['init_beta']}\")
print(f\"  friendly_scale = {p['init_friendly_scale']}\")
print(f\"  B = {p['init_B']}\")
print(f\"  Γ_0 shape = {len(p['init_gamma'])}x{len(p['init_gamma'][0])}\")
print(f\"  μ_0 shape = {len(p['init_mean'])}x{len(p['init_mean'][0])}\")
"
        echo ""
    fi
done

# Print final parameters
FINAL="${LOCAL_OUTPUTS}/em_params_final.json"
if [ -f "$FINAL" ]; then
    echo "--- Final parameters ---"
    python3 -c "
import json
with open('$FINAL') as fh:
    p = json.load(fh)
print(f\"  κ = {p['init_kappa']}\")
print(f\"  α = {p['init_alpha']}\")
print(f\"  β = {p['init_beta']}\")
print(f\"  friendly_scale = {p['init_friendly_scale']}\")
print(f\"  B = {p['init_B']}\")
print(f\"  Γ_0 shape = {len(p['init_gamma'])}x{len(p['init_gamma'][0])}\")
print(f\"  μ_0 shape = {len(p['init_mean'])}x{len(p['init_mean'][0])}\")
"
    echo ""
else
    echo "WARNING: em_params_final.json not found!"
fi

# --- Step 4: Tear down ---
echo "[4/4] Tearing down VM..."
colab stop -s "${SESSION}"

echo "============================================================"
echo "  DONE — outputs saved to ${LOCAL_OUTPUTS}"
echo "============================================================"