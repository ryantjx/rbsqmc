"""Bootstrap a Colab GPU and run the model_unbiased (log marginal) pipeline.

This script runs inside the Colab VM (uploaded by ``colab run``). It:
  1. Clones/updates the repository (or uses an uploaded copy with --no-clone).
  2. Installs runtime dependencies.
  3. Loads the config JSON (model_unbiased_gpu_config.json).
  4. Imports and calls ``main`` from ``rbpf.src.model_unbiased`` directly.
  5. Saves outputs and generates plots (filter + predictions).

Usage (local dry-run):
    python rbpf/scripts/run_model_unbiased_gpu.py \
        --config rbpf/scripts/config/model_unbiased_gpu_config.json --dry-run

Usage (Colab, via the orchestrator):
    bash rbpf/scripts/run_model_unbiased_colab.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path("/content/rbsqmc")
DEFAULT_CONFIG = {
    "start_date": "1900-01-01",
    "end_date": "2026-01-01",
    "n_particles": 2500,
    "max_goals": 8,
    "seed": 0,
    "n_epochs": 200,
    "learning_rate": 0.001,
    "n_reps": 4,
    "include_friendly": False,
    "output_dir": "rbpf/outputs/filter_unbiased",
    "gpu_type": "A100",
    "colab_timeout": 14400,
    "repo_url": "https://github.com/ryantjx/rbsqmc.git",
}


def log(message: str, *, stream: str = "OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    print(f"[{outer}] {stream}: [{inner}] {message}", flush=True)


def run(command, *, cwd=None, forward_raw=False) -> None:
    log("Running: " + " ".join(map(str, command)))
    process = subprocess.Popen(
        command, cwd=cwd, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        if forward_raw:
            print(line, end="", flush=True)
        else:
            log(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}:\n"
            + " ".join(map(str, command))
        )


def is_colab() -> bool:
    return Path("/content").exists()


# ---------------------------------------------------------------------------
# Bootstrap: clone repo + install deps
# ---------------------------------------------------------------------------
def bootstrap(no_clone: bool = False) -> Path:
    """Clone/update the repository and install runtime dependencies.

    Uses Colab's pre-installed Python environment (which already ships JAX,
    NumPy, SciPy, Matplotlib, etc.) rather than creating an isolated uv venv.
    Any missing packages are installed via pip into the active environment.

    Returns the repository root.
    """
    if no_clone:
        if not (REPO_DIR / "rbpf").is_dir():
            raise FileNotFoundError(
                f"--no-clone: expected repo at {REPO_DIR} but rbpf/ is missing"
            )
        log(f"Using uploaded repo at {REPO_DIR} (no clone)")
    elif (REPO_DIR / ".git").is_dir():
        log(f"Using existing checkout at {REPO_DIR}")
        run(["git", "-c", "http.version=HTTP/1.1", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        last_error = None
        for attempt in range(1, 4):
            if REPO_DIR.exists():
                shutil.rmtree(REPO_DIR)
            try:
                run([
                    "git", "-c", "http.version=HTTP/1.1", "clone",
                    "--depth", "1", "--single-branch", "--branch", "main",
                    DEFAULT_CONFIG["repo_url"], str(REPO_DIR),
                ])
                last_error = None
                break
            except RuntimeError as error:
                last_error = error
                log(f"clone attempt {attempt}/3 failed")
        if last_error is not None:
            raise last_error
    # Install runtime dependencies into the active (Colab) environment.
    run([
        sys.executable, "-m", "pip", "install",
        "-r", str(REPO_DIR / "rbpf" / "requirements.txt"),
    ])
    return REPO_DIR


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(repo_root: Path, override: str | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if override:
        path = Path(override)
        if not path.is_absolute() and not path.parent.parts:
            path = repo_root / "rbpf/scripts/config" / path
    else:
        path = repo_root / "rbpf/scripts/config/model_unbiased_gpu_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing model_unbiased configuration: {path}")
    config.update(json.loads(path.read_text(encoding="utf-8")))

    positive_integer = (
        "n_particles", "max_goals", "n_epochs", "colab_timeout",
    )
    for key in positive_integer:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config.get("learning_rate", 0)) <= 0:
        raise ValueError("learning_rate must be positive")
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Colab GPU bootstrap for the model_unbiased (log marginal) pipeline"
    )
    p.add_argument("--config", help="Optional config JSON override")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-clone", action="store_true",
                   help="Skip git clone; use the repo already uploaded to /content/rbsqmc")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.dry_run:
        config = load_config(Path.cwd(), args.config)
        print(f"model_unbiased: n_particles={config['n_particles']}, "
              f"n_epochs={config['n_epochs']}, lr={config['learning_rate']}, "
              f"n_reps={config['n_reps']}")
        return 0

    repo_root = bootstrap(no_clone=args.no_clone) if is_colab() else Path(__file__).resolve().parents[2]
    config = load_config(repo_root, args.config)

    # Verify GPU using the active (Colab) Python.
    run([
        sys.executable, "-c",
        "import jax; print('JAX devices:', jax.devices()); "
        "assert jax.default_backend() == 'gpu', 'Colab GPU is not active'",
    ], cwd=repo_root)

    log(f"output_dir={config['output_dir']}")
    log(f"Writing remote outputs to {repo_root / config['output_dir']}")

    config_path = Path(args.config) if args.config else repo_root / "rbpf/scripts/config/model_unbiased_gpu_config.json"
    if not config_path.is_absolute():
        config_path = repo_root / "rbpf/scripts/config" / config_path
    os.environ["RBSQMC_CONFIG"] = str(config_path)
    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_matplotlib")
    os.environ["MPLBACKEND"] = "Agg"

    run([
        sys.executable, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from rbpf.src.model_unbiased import main; main()",
    ], cwd=repo_root, forward_raw=True)

    log("GPU model_unbiased completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}", stream="ERR")
        raise
