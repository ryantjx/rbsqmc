#!/usr/bin/env bash
set -euo pipefail

# Run rbpf/test/smoothing_gpu.py on a Colab GPU.
#
# Reuses the EM machinery from rbpf/test/src/smoothing.py (parameter set:
# estimates sigma_0, gamma_Q, B_Q, alpha, beta; mean_0 fixed) but runs with a
# GPU-oriented configuration.
#
# Usage:  ./run_smoothing_colab.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/smoothing_gpu_config.json"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# --- Load shared configuration (single source of truth) ---
# Parse the JSON config with python3 (jq may not be installed on macOS).
read_config() {
    python3 -c "
import json
with open('${CONFIG}') as f:
    c = json.load(f)
print(c.get('$1', ''))
"
}

# config output_dir is repo-root-relative (e.g. "rbpf/test/outputs_gpu").
REMOTE_OUTPUTS="/content/rbsqmc/$(read_config output_dir)"
LOCAL_OUTPUTS="${REPO_ROOT}/$(read_config output_dir)"
# Full path to the script so `colab run` can upload it regardless of cwd.
SCRIPT="${HERE}/smoothing_gpu.py"
SESSION="${SESSION:-rbsqmc-test-gpu}"

# Download init, final, and log marginal history
OUTPUT_FILES=(
    em_params_init.json
    em_params_final.json
    em_log_marginal_history.json
    em_mstep_diagnostics.json
)

GPU_TYPE="${GPU_TYPE:-$(read_config gpu_type)}"
TPU_TYPE="${TPU_TYPE:-$(read_config tpu_type)}"
HARDWARE="${HARDWARE:-$(read_config hardware)}"  # gpu | tpu
GPU_N="${GPU_N:-$(read_config N)}"

# Build the accelerator flag: GPU uses --gpu, TPU uses --tpu.
if [ "${HARDWARE}" = "tpu" ]; then
    ACCEL_FLAG="--tpu ${TPU_TYPE}"
    ACCEL_LABEL="TPU=${TPU_TYPE}"
else
    ACCEL_FLAG="--gpu ${GPU_TYPE}"
    ACCEL_LABEL="GPU=${GPU_TYPE}"
fi

echo "============================================================"
echo "  COLAB SMOOTHING GPU — RANDOM-WALK MODEL TEST RUN"
echo "  Reuses rbpf/test/src/smoothing.py EM (sigma_0, gamma_Q, B_Q, alpha, beta; mean_0 fixed)."
echo "  ${ACCEL_LABEL}, N=${GPU_N}, start_date=$(read_config start_date), n_epochs=$(read_config n_epochs), teams=$(read_config teams)"
if [ "$(read_config high_ram)" = "true" ]; then
    echo "  High-RAM: ENABLED (note: set in the Colab UI — the CLI exposes no --high-ram flag)"
fi
echo "============================================================"

cleanup() {
    echo "Cleaning up: stopping Colab session '${SESSION}'..."
    colab stop -s "${SESSION}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/5] Launching Colab session (${ACCEL_LABEL}) and running ${SCRIPT} (N=${GPU_N})..."
colab run ${ACCEL_FLAG} --keep --timeout $(read_config colab_timeout) --session "${SESSION}" "${SCRIPT}" "${GPU_N}"

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

echo "[5/5] Running filter with trained params and producing graphics..."

# Graphics go into the same outputs_gpu dir, under a `trained/` subfolder.
GRAPHIC_OUTDIR="${GRAPHIC_OUTDIR:-${LOCAL_OUTPUTS}/trained}"

# The Colab runner writes em_params_final.json into the GPU output dir. Use it
# as the trained params unless the caller overrides PARAMS_PATH.
TRAINED_PARAMS="${PARAMS_PATH:-${LOCAL_OUTPUTS}/em_params_final.json}"

if [ -f "$TRAINED_PARAMS" ]; then
    echo "  Trained params: ${TRAINED_PARAMS}"
    echo "  Graphic output: ${GRAPHIC_OUTDIR}"
    # Run through the project venv (uv) so `jax`/project deps are available;
    # bare `python3` on macOS lacks them.
    ( cd "${REPO_ROOT}" && uv run python -u "${HERE}/model_trained.py" \
        --params-path "${TRAINED_PARAMS}" \
        --output-dir "${GRAPHIC_OUTDIR}" ) || \
        echo "  WARNING: model_trained.py failed"
    echo "  Graphics written under ${GRAPHIC_OUTDIR}/"
else
    echo "  WARNING: ${TRAINED_PARAMS} not found; skipping filter/graphics step"
fi

# --- Final status report ---
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE — STATUS"
echo "============================================================"
echo "  Session:        ${SESSION} (${ACCEL_LABEL})"
echo "  EM params:      ${LOCAL_OUTPUTS}/em_params_{init,final}.json"
echo "  Log marginal:   ${LOCAL_OUTPUTS}/em_log_marginal_history.json"
echo ""
echo "  EM output files:"
for f in "${OUTPUT_FILES[@]}"; do
    if [ -f "${LOCAL_OUTPUTS}/${f}" ]; then
        echo "    [OK]   ${f}"
    else
        echo "    [MISS] ${f}"
    fi
done
echo ""
echo "  Graphics:"
if [ -d "${GRAPHIC_OUTDIR}" ]; then
    for img in "${GRAPHIC_OUTDIR}"/*.png; do
        [ -f "$img" ] && echo "    [OK]   $(basename "${img}")"
    done
else
    echo "    [MISS] no graphics in ${GRAPHIC_OUTDIR}"
fi
echo "============================================================"
