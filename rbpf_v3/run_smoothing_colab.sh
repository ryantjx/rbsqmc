#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
CONFIG="${HERE}/smoothing_gpu_config.json"
BOOTSTRAP="${HERE}/scripts/run_smoothing_gpu.py"
VALIDATOR="${HERE}/scripts/validate_outputs.py"
LOCAL_OUTPUTS="${REPO_ROOT}/rbpf_v3/outputs/smoothing"
REMOTE_OUTPUTS="/content/rbsqmc/rbpf_v3/outputs/smoothing"
SESSION="${SESSION:-rbsqmc-rbpf-v3-smoothing}"
SESSION_LAUNCHED=0
DRY_RUN=0

progress() {
    local stream="$1"
    shift
    local outer inner
    outer="$(date '+%Y-%m-%d %H:%M:%S')"
    inner="$(date '+%H:%M:%S')"
    printf '[%s] %s: [%s] %s\n' "${outer}" "${stream}" "${inner}" "$*"
}

read_config() {
    python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); key=sys.argv[2]; value=data[key]; print(value)' "${CONFIG}" "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        progress ERR "required command not found: $1"
        return 1
    }
}

download_required() {
    local file="$1"
    progress OUT "downloading required artifact ${file}"
    mkdir -p "$(dirname "${LOCAL_OUTPUTS}/${file}")"
    colab download -s "${SESSION}" "${REMOTE_OUTPUTS}/${file}" "${LOCAL_OUTPUTS}/${file}"
    [[ -s "${LOCAL_OUTPUTS}/${file}" ]] || {
        progress ERR "required artifact is unavailable or empty: ${file}"
        return 1
    }
}

download_optional() {
    local file="$1"
    if ! colab download -s "${SESSION}" "${REMOTE_OUTPUTS}/${file}" "${LOCAL_OUTPUTS}/${file}"; then
        progress OUT "optional artifact unavailable: ${file}"
    fi
}

validate_local_outputs() {
    python3 "${VALIDATOR}" "${LOCAL_OUTPUTS}"
}

cleanup() {
    local status=$?
    trap - EXIT
    if [[ "${SESSION_LAUNCHED}" -eq 1 ]]; then
        progress OUT "stopping Colab session ${SESSION}"
        colab stop -s "${SESSION}" >/dev/null 2>&1 || true
    fi
    exit "${status}"
}

main() {
    if [[ "${1:-}" == "--dry-run" ]]; then
        DRY_RUN=1
    elif [[ $# -gt 0 ]]; then
        progress ERR "unsupported argument: $1"
        return 2
    fi

    require_command python3
    [[ -s "${CONFIG}" ]] || { progress ERR "missing config: ${CONFIG}"; return 1; }
    [[ -s "${BOOTSTRAP}" ]] || { progress ERR "missing bootstrap: ${BOOTSTRAP}"; return 1; }
    [[ -s "${VALIDATOR}" ]] || { progress ERR "missing validator: ${VALIDATOR}"; return 1; }

    local gpu_type timeout output_dir
    gpu_type="${GPU_TYPE:-$(read_config gpu_type)}"
    timeout="${COLAB_TIMEOUT:-$(read_config colab_timeout)}"
    output_dir="$(read_config output_dir)"
    [[ "${output_dir}" == "rbpf_v3/outputs/smoothing" ]] || {
        progress ERR "config output_dir must be rbpf_v3/outputs/smoothing"
        return 1
    }
    python3 "${BOOTSTRAP}" --config "${CONFIG}" --dry-run >/dev/null

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf 'colab run --gpu %q --keep --timeout %q --session %q %q\n' \
            "${gpu_type}" "${timeout}" "${SESSION}" "${BOOTSTRAP}"
        return 0
    fi

    require_command colab
    trap cleanup EXIT
    progress OUT "launching RBPF v3 Cuthbert smoothing on Colab GPU=${gpu_type}"
    SESSION_LAUNCHED=1
    colab run --gpu "${gpu_type}" --keep --timeout "${timeout}" \
        --session "${SESSION}" "${BOOTSTRAP}"
    colab status -s "${SESSION}"

    mkdir -p "${LOCAL_OUTPUTS}"
    local required=(
        progress.log em_initial_params.json em_final_params.json
        training_summary.json performance_summary.json evaluation_summary.json
        baseline_comparison.json objective_terms_by_epoch.png
        optimal_filter/filter_states.npz optimal_filter/optimal_filter_summary.json
        optimal_filter/top_strengths.png optimal_filter/timeseries_states.png
        optimal_filter/correlation_matrix.png optimal_filter/log_normalizing_constant.png
        transition_normalization_vs_quadratic.png covariance_eigenvalues_and_condition.png
        ou_half_life_and_parameters.png transition_mahalanobis_by_time.png
        backward_ess_entropy_and_unique_indices.png
        smoothed_team_trajectories_with_intervals.png heldout_log_score_by_date.png
        result_calibration.png goal_marginal_calibration.png
    )
    local file
    for file in "${required[@]}"; do
        download_required "${file}"
    done
    download_optional device_memory.prof
    validate_local_outputs
    progress OUT "RBPF v3 Colab acceptance completed successfully"
}

main "$@"
