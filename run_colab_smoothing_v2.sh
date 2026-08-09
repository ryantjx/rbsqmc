#!/usr/bin/env bash
set -euo pipefail

# Run smoothing_v2.py on Colab GPU (V2: fixed κ=1.0, B=I₂).
# Estimates Γ₀ (IW prior), α, β, μ₀.
#
# Usage:  ./run_colab_smoothing_v2.sh

REMOTE_OUTPUTS="/content/rbsqmc/rbpf/outputs_gpu_v2"
LOCAL_OUTPUTS="./rbpf/outputs_gpu_v2"
SCRIPT="run_smoothing_gpu_v2.py"
SESSION="${SESSION:-rbsqmc-v2}"

# Only download init and final — skip intermediate epochs
OUTPUT_FILES=(
    em_params_init.json
    em_params_final.json
)

echo "============================================================"
echo "  COLAB SMOOTHING V2 GPU — PRODUCTION RUN"
echo "  Fixed κ=1.0, B=I₂. Estimates Γ₀, α, β, μ₀."
echo "  N=1000, start_date=2000-01-01, n_epochs=15, teams=WorldCup2026"
echo "============================================================"

cleanup() {
    echo "Cleaning up: stopping Colab session '${SESSION}'..."
    colab stop -s "${SESSION}" 2>/dev/null || true
}
trap cleanup EXIT

GPU_TYPE="${GPU_TYPE:-T4}"
GPU_N="${GPU_N:-1000}"

echo "[1/5] Launching Colab GPU session and running ${SCRIPT} (GPU=${GPU_TYPE}, N=${GPU_N})..."
colab run --gpu "${GPU_TYPE}" --keep --timeout 3600 --session "${SESSION}" "${SCRIPT}" "${GPU_N}"

echo "[2/5] Checking active Colab sessions..."
colab sessions

echo "[3/5] Downloading outputs from VM..."
mkdir -p "${LOCAL_OUTPUTS}"
for f in "${OUTPUT_FILES[@]}"; do
    echo "  Downloading ${f}..."
    colab download -s "${SESSION}" "${REMOTE_OUTPUTS}/${f}" "${LOCAL_OUTPUTS}/${f}" || \
        echo "  WARNING: ${f} not found on VM"
done

echo "[4/5] Checking outputs..."
echo ""
echo "--- Output files ---"
ls -lh "${LOCAL_OUTPUTS}"/ 2>/dev/null || echo "  No output files found"
echo ""

for f in "${LOCAL_OUTPUTS}"/em_params_epoch_*.json; do
    if [ -f "$f" ]; then
        epoch=$(basename "$f" | grep -oE '[0-9]+')
        echo "--- Epoch ${epoch} ---"
        python3 -c "
import json, sys
with open('$f') as fh:
    p = json.load(fh)
print(f\"  κ = {p['init_kappa']} [FIXED]\")
print(f\"  α = {p['init_alpha']}\")
print(f\"  β = {p['init_beta']}\")
print(f\"  friendly_scale = {p['init_friendly_scale']}\")
print(f\"  B = {p['init_B']} [FIXED]\")
print(f\"  Γ_0 shape = {len(p['init_gamma'])}x{len(p['init_gamma'][0])}\")
print(f\"  μ_0 shape = {len(p['init_mean'])}x{len(p['init_mean'][0])}\")
"
        echo ""
    fi
done

FINAL="${LOCAL_OUTPUTS}/em_params_final.json"
if [ -f "$FINAL" ]; then
    echo "--- Final parameters ---"
    python3 -c "
import json
with open('$FINAL') as fh:
    p = json.load(fh)
print(f\"  κ = {p['init_kappa']} [FIXED]\")
print(f\"  α = {p['init_alpha']}\")
print(f\"  β = {p['init_beta']}\")
print(f\"  friendly_scale = {p['init_friendly_scale']}\")
print(f\"  B = {p['init_B']} [FIXED]\")
print(f\"  Γ_0 shape = {len(p['init_gamma'])}x{len(p['init_gamma'][0])}\")
print(f\"  μ_0 shape = {len(p['init_mean'])}x{len(p['init_mean'][0])}\")
"
    echo ""
else
    echo "WARNING: em_params_final.json not found\!"
fi

# --- Step 5: Run model_trained_v2.py locally with the trained parameters ---
echo "[5/6] Running model_trained_v2.py with trained parameters..."
cd rbpf
python3 -u model_trained_v2.py --params-path ./outputs_gpu_v2/em_params_final.json --output-dir ./outputs_gpu_v2/trained || \
    echo "  WARNING: model_trained_v2.py failed"
cd ..

# --- Step 6: Tear down ---
echo "[6/6] Tearing down VM..."
colab stop -s "${SESSION}"
trap - EXIT

echo "============================================================"
echo "  DONE — EM outputs saved to ${LOCAL_OUTPUTS}"
echo "  Filter outputs saved to ${LOCAL_OUTPUTS}/trained/"
echo "============================================================"
