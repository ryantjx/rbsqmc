#!/usr/bin/env python3
"""Orchestrate a Colab GPU run for the RBPF smoothing EM pipeline.

This is a pure-Python replacement for ``run_smoothing_colab.sh``. It:
  1. Validates the config and bootstrap script.
  2. Launches ``run_smoothing_gpu.py`` on a Colab GPU session via ``colab run``.
  3. Downloads all required output artifacts from the session.
  4. Validates the downloaded outputs.

Supports both ``adam`` and ``bfgs`` M-step variants via the config's ``m_step``
key. The config JSON determines which variant runs and where outputs are stored.

Usage:
    python rbpf/scripts/run_smoothing_colab.py --config rbpf/scripts/config/smoothing_gpu_config.json
    python rbpf/scripts/run_smoothing_colab.py --config rbpf/scripts/config/smoothing_bfgs_gpu_config.json
    python rbpf/scripts/run_smoothing_colab.py --config ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
BOOTSTRAP = HERE / "run_smoothing_gpu.py"
VALIDATOR = HERE / "validate_outputs.py"


def log(message: str, *, stream: str = "OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    print(f"[{outer}] {stream}: [{inner}] {message}", flush=True)


def read_config(config_path: Path, key: str) -> str:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data[key]


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        log(f"missing {label}: {path}", stream="ERR")
        raise FileNotFoundError(f"missing {label}: {path}")


def require_command(cmd: str) -> None:
    if not shutil_which(cmd):
        log(f"required command not found: {cmd}", stream="ERR")
        raise FileNotFoundError(f"command not found: {cmd}")


def shutil_which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


def download_required(session: str, remote: str, local: Path) -> None:
    log(f"downloading required artifact {remote}")
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["colab", "download", "-s", session, remote, str(local)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not local.is_file() or local.stat().st_size == 0:
        log(f"required artifact unavailable or empty: {remote}", stream="ERR")
        raise RuntimeError(f"download failed: {remote}")
    log(f"downloaded {remote} -> {local}")


def download_optional(session: str, remote: str, local: Path) -> None:
    log(f"downloading optional artifact {remote}")
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["colab", "download", "-s", session, remote, str(local)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"optional artifact unavailable: {remote}")


def validate_outputs(local_outputs: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(local_outputs)],
    )
    if result.returncode != 0:
        raise RuntimeError("output validation failed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate a Colab GPU run for RBPF smoothing EM"
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print the colab command without running")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    require_file(config_path, "config")
    require_file(BOOTSTRAP, "bootstrap script")
    require_file(VALIDATOR, "validator script")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    m_step = config.get("m_step", "adam")
    output_dir = config["output_dir"]
    gpu_type = os.environ.get("GPU_TYPE", config.get("gpu_type", "A100"))
    timeout = int(os.environ.get("COLAB_TIMEOUT", config.get("colab_timeout", 7200)))
    session = os.environ.get(
        "SESSION", f"rbsqmc-rbpf-{m_step}-smoothing"
    )

    local_outputs = REPO_ROOT / output_dir
    remote_outputs = f"/content/rbsqmc/{output_dir}"

    # Dry-run validation of the bootstrap
    dry_result = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--config", str(config_path), "--dry-run"],
        capture_output=True, text=True,
    )
    if dry_result.returncode != 0:
        log(f"bootstrap dry-run failed:\n{dry_result.stderr}", stream="ERR")
        return 1

    if args.dry_run:
        print(
            f"colab run --gpu {gpu_type} --keep --timeout {timeout} "
            f"--session {session} {BOOTSTRAP} --config {config_path}"
        )
        return 0

    require_command("colab")

    log(f"launching RBPF {m_step} smoothing on Colab GPU={gpu_type}")
    session_launched = True
    try:
        run_result = subprocess.run(
            [
                "colab", "run", "--gpu", gpu_type,
                "--keep", "--timeout", str(timeout),
                "--session", session,
                str(BOOTSTRAP), "--config", str(config_path),
            ],
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"colab run failed with exit code {run_result.returncode}")

        subprocess.run(["colab", "status", "-s", session])

        local_outputs.mkdir(parents=True, exist_ok=True)

        required_artifacts = [
            "optimized_params.json",
            "run_config.json",
            "em_log_marginal_likelihood_curve.png",
            "filter/top_strengths.png",
            "filter/timeseries_states.png",
            "filter/correlation_matrix.png",
            "filter/log_normalizing_constant.png",
            "smoothing/smoothed_trajectories.png",
            "smoothing/smoothed_uncertainty.png",
        ]
        for artifact in required_artifacts:
            download_required(
                session,
                f"{remote_outputs}/{artifact}",
                local_outputs / artifact,
            )

        validate_outputs(local_outputs)
        log(f"RBPF {m_step} Colab acceptance completed successfully")

    except Exception:
        raise
    finally:
        if session_launched:
            log(f"stopping Colab session {session}")
            subprocess.run(
                ["colab", "stop", "-s", session],
                capture_output=True,
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}", stream="ERR")
        raise