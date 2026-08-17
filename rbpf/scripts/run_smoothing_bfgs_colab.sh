#!/usr/bin/env bash
# Orchestrator: launch the BFGS smoothing EM pipeline on a Colab GPU.
#
# This script runs LOCALLY. It:
#   1. Validates the config and bootstrap script.
#   2. Launches run_smoothing_gpu.py on a Colab GPU session via `colab run`.
#   3. Downloads all required output artifacts from the session.
#   4. Validates the downloaded outputs.
#
# Usage:
#   bash rbpf/scripts/run_smoothing_bfgs_colab.sh
#   bash rbpf/scripts/run_smoothing_bfgs_colab.sh --dry-run
#
# Environment overrides:
#   GPU_TYPE       e.g. A100, T4 (default: from config)
#   COLAB_TIMEOUT  seconds (default: from config)
#   SESSION        colab session name (default: rbsqmc-rbpf-bfgs-smoothing)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
CONFIG="${HERE}/config/smoothing_bfgs_gpu_config.json"
BOOTSTRAP="${HERE}/run_smoothing_bfgs_gpu.py"
VALIDATOR="${HERE}/validate_outputs.py"
DRY_RUN=0
SESSION_LAUNCHED=0

progress() {
    local stream="$1"; shift
    local outer inner
    outer="$(date '+%Y-%m-%d %H:%M:%S')"
    inner="$(date '+%H:%M:%S')"
    printf '[%s] %s: [%s] %s\n' "${outer}" "${stream}" "${inner}" "$*"
}

read_config() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${CONFIG}" "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        progress ERR "required command not found: $1"
        return 1
    }
}

require_file() {
    local file="$1" label="$2"
    [[ -s "${file}" ]] || {
        progress ERR "missing ${label}: ${file}"
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    if [[ "${1:-}" == "--dry-run" ]]; then
        DRY_RUN=1
    elif [[ $# -gt 0 ]]; then
        progress ERR "unsupported argument: $1"
        return 2
    fi

    require_command python3
    require_file "${CONFIG}" "config"
    require_file "${BOOTSTRAP}" "bootstrap script"
    require_file "${VALIDATOR}" "validator script"

    # Activate the local virtualenv so python3 / colab use the right packages.
    if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
        progress OUT "activating virtualenv: ${REPO_ROOT}/.venv"
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/.venv/bin/activate"
    else
        progress OUT "no .venv found at ${REPO_ROOT}/.venv — using system python3"
    fi

    local m_step gpu_type timeout output_dir
    m_step="$(read_config m_step)"
    gpu_type="${GPU_TYPE:-$(read_config gpu_type)}"
    timeout="${COLAB_TIMEOUT:-$(read_config colab_timeout)}"
    output_dir="$(read_config output_dir)"
    SESSION="${SESSION:-rbsqmc-rbpf-${m_step}-smoothing}"

    LOCAL_OUTPUTS="${REPO_ROOT}/${output_dir}"
    REMOTE_OUTPUTS="/content/rbsqmc/${output_dir}"

    # Dry-run validation of the bootstrap (use local path for validation only)
    python3 "${BOOTSTRAP}" --config "${CONFIG}" --dry-run >/dev/null

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf 'colab run --gpu %q --keep --timeout %q --session %q %q --config %q\n' \
            "${gpu_type}" "${timeout}" "${SESSION}" "${BOOTSTRAP}" "$(basename "${CONFIG}")"
        return 0
    fi

    require_command colab
    trap cleanup EXIT

    progress OUT "launching RBPF ${m_step} smoothing on Colab GPU=${gpu_type}"
    SESSION_LAUNCHED=1
    # Pass just the config filename — the bootstrap resolves it inside the
    # cloned repo on the VM (the local absolute path doesn't exist there).
    colab run --gpu "${gpu_type}" --keep --timeout "${timeout}" \
        --session "${SESSION}" "${BOOTSTRAP}" --config "$(basename "${CONFIG}")"
    colab status -s "${SESSION}"

    mkdir -p "${LOCAL_OUTPUTS}"

    local required=(
        optimized_params.json
        run_config.json
        em_log_marginal_likelihood_curve.png
        filter/top_strengths.png
        filter/timeseries_states.png
        filter/correlation_matrix.png
        filter/initial_correlation_matrix.png
        filter/filter_states.npz
        filter/log_normalizing_constant.png
        smoothing/smoothed_trajectories.png
        smoothing/smoothed_uncertainty.png
    )
    local file
    for file in "${required[@]}"; do
        download_required "${file}"
    done

    validate_local_outputs
    progress OUT "RBPF ${m_step} Colab acceptance completed successfully"
}

main "$@"