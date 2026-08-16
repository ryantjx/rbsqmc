#!/usr/bin/env bash
set -euo pipefail

# Launch rbpf_v2/scripts/run_smoothing_gpu.py on a Colab GPU and download all
# training/evaluation artifacts into rbpf_v2/outputs/smoothing.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
CONFIG="${HERE}/smoothing_gpu_config.json"
SCRIPT="${HERE}/scripts/run_smoothing_gpu.py"
LOCAL_OUTPUTS="${REPO_ROOT}/rbpf_v2/outputs/smoothing"
REMOTE_OUTPUTS="/content/rbsqmc/rbpf_v2/outputs/smoothing"
SESSION="${SESSION:-rbsqmc-rbpf-v2-smoothing}"

read_config() {
    python3 -c "import json; print(json.load(open('${CONFIG}')).get('$1', ''))"
}

GPU_TYPE="${GPU_TYPE:-$(read_config gpu_type)}"
TIMEOUT="${COLAB_TIMEOUT:-$(read_config colab_timeout)}"

OUTPUT_FILES=(
    em_initial_params.json
    em_final_params.json
    training_arrays.npz
    training_summary.json
    evaluation_summary.json
    baseline_comparison.json
    objective_terms_by_epoch.png
    transition_normalization_vs_quadratic.png
    covariance_eigenvalues_and_condition.png
    ou_half_life_and_parameters.png
    transition_mahalanobis_by_time.png
    backward_ess_entropy_and_unique_indices.png
    smoothed_team_trajectories_with_intervals.png
    heldout_log_score_by_date.png
    result_calibration.png
    goal_marginal_calibration.png
)

cleanup() {
    echo "Stopping Colab session '${SESSION}'..."
    colab stop -s "${SESSION}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Launching RBPF v2 smoothing on Colab GPU=${GPU_TYPE}"
echo "Configuration: ${CONFIG}"
colab run --gpu "${GPU_TYPE}" --keep --timeout "${TIMEOUT}" \
    --session "${SESSION}" "${SCRIPT}"

mkdir -p "${LOCAL_OUTPUTS}"
for file in "${OUTPUT_FILES[@]}"; do
    echo "Downloading ${file}"
    colab download -s "${SESSION}" \
        "${REMOTE_OUTPUTS}/${file}" "${LOCAL_OUTPUTS}/${file}"
done

echo "Downloaded RBPF v2 outputs to ${LOCAL_OUTPUTS}"
ls -lh "${LOCAL_OUTPUTS}"
