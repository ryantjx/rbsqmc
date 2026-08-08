#!/usr/bin/env bash
set -euo pipefail

# Run smoothing.py on Colab GPU (production), download outputs, and verify.
#
# Uses the full parameters from smoothing.py:
#   N = 1000 particles, start_date = 2000-01-01, n_epochs = 10
#
# Usage:  ./run_colab_smoothing.sh

REMOTE_OUTPUTS="/content/rbsqmc/rbpf/outputs_gpu"
LOCAL_OUTPUTS="./rbpf/outputs_gpu"
SCRIPT="run_smoothing_gpu.py"
SESSION="rbsqmc"

# Production run has 10 epochs — generate the expected file list
OUTPUT_FILES=(
    em_params_init.json
    em_params_epoch_0.json
    em_params_epoch_1.json
    em_params_epoch_2.json
    em_params_epoch_3.json
    em_params_epoch_4.json
    em_params_epoch_5.json
    em_params_epoch_6.json
    em_params_epoch_7.json
    em_params_epoch_8.json
    em_params_epoch_9.json
    em_params_final.json
)

echo "============================================================"
echo "  COLAB SMOOTHING GPU — PRODUCTION RUN"
echo "  N=1000, start_date=2000-01-01, n_epochs=10"
echo "============================================================"

# --- Step 1: Run the bootstrap script on a GPU VM ---
# GPU_N controls particle count (default 100, fits T4 16GB VRAM)
# Override with: GPU_N=200 ./run_colab_smoothing.sh
echo "[1/5] Launching Colab GPU session and running ${SCRIPT} (GPU_N=${GPU_N:-100})..."
GPU_N="${GPU_N:-100}" colab run --gpu T4 --keep --timeout 3600 --session "${SESSION}" "${SCRIPT}"

# --- Step 2: Check active sessions ---
echo "[2/5] Checking active Colab sessions..."
colab sessions

# --- Step 3: Download outputs (individual files, not directory) ---
echo "[3/5] Downloading outputs from VM..."
mkdir -p "${LOCAL_OUTPUTS}"
for f in "${OUTPUT_FILES[@]}"; do
    echo "  Downloading ${f}..."
    colab download -s "${SESSION}" "${REMOTE_OUTPUTS}/${f}" "${LOCAL_OUTPUTS}/${f}" || \
        echo "  WARNING: ${f} not found on VM"
done

# --- Step 4: Verify outputs ---
echo "[4/5] Checking outputs..."
echo ""
echo "--- Output files ---"
ls -lh "${LOCAL_OUTPUTS}"/ 2>/dev/null || echo "  No output files found"
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
    echo "WARNING: em_params_final.json not found\!"
fi

# --- Step 5: Tear down ---
echo "[5/5] Tearing down VM..."
colab stop -s "${SESSION}"

echo "============================================================"
echo "  DONE — outputs saved to ${LOCAL_OUTPUTS}"
echo "============================================================"
