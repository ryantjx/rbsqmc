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
SESSION="${SESSION:-rbsqmc}"

# Only download init and final — skip intermediate epochs
OUTPUT_FILES=(
    em_params_init.json
    em_params_final.json
)

echo "============================================================"
echo "  COLAB SMOOTHING GPU — PRODUCTION RUN"
echo "  N=1000, start_date=2000-01-01, n_epochs=15, teams=WorldCup2026"
echo "============================================================"

# Ensure the session is stopped even if a step fails
cleanup() {
    echo "Cleaning up: stopping Colab session '${SESSION}'..."
    colab stop -s "${SESSION}" 2>/dev/null || true
}
trap cleanup EXIT

# --- Step 1: Run the bootstrap script on a GPU VM ---
# GPU_TYPE controls the accelerator (default T4 with 16GB)
# GPU_N optionally overrides particle count (default: 1000, which fits a T4)
# With 48 teams (WorldCup2026 filter), N=1000 fits in T4 16GB VRAM.
# Override with: GPU_N=500 GPU_TYPE=T4 ./run_colab_smoothing.sh
GPU_TYPE="${GPU_TYPE:-T4}"
GPU_N="${GPU_N:-1000}"

echo "[1/5] Launching Colab GPU session and running ${SCRIPT} (GPU=${GPU_TYPE})..."
colab run --gpu "${GPU_TYPE}" --keep --timeout 3600 --session "${SESSION}" "${SCRIPT}" "${GPU_N}"

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

# --- Step 5: Run model_trained.py locally with the trained parameters ---
echo "[5/6] Running model_trained.py with trained parameters..."
cd rbpf
python3 -u model_trained.py --params-path ./outputs_gpu/em_params_final.json --output-dir ./outputs_gpu/trained || \
    echo "  WARNING: model_trained.py failed"
cd ..

# --- Step 6: Tear down ---
echo "[6/6] Tearing down VM..."
colab stop -s "${SESSION}"
# Disable the trap so it doesn't fire again
trap - EXIT

echo "============================================================"
echo "  DONE — EM outputs saved to ${LOCAL_OUTPUTS}"
echo "  Filter outputs saved to ${LOCAL_OUTPUTS}/trained/"
echo "============================================================"
