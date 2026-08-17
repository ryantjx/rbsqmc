"""Bootstrap a Colab GPU and invoke the RBPF smoothing EM pipeline.

This file deliberately imports no project or JAX modules before bootstrapping,
so ``colab run`` can upload it into a fresh VM. The config JSON is the source
of truth for all run parameters.

Supports two M-step variants via the ``m_step`` config key:
  - ``"adam"``: runs ``rbpf.src.smoothing`` (Adam gradient M-step)
  - ``"bfgs"``: runs ``rbpf.src.smoothing_bfgs`` (BFGS quasi-Newton M-step)

Usage (local dry-run):
    python rbpf/scripts/run_smoothing_gpu.py --config rbpf/scripts/config/smoothing_gpu_config.json --dry-run

Usage (Colab, via the orchestrator):
    python rbpf/scripts/run_smoothing_colab.py --config rbpf/scripts/config/smoothing_gpu_config.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime


REPO_DIR = Path("/content/rbsqmc")
DEFAULT_CONFIG = {
    "start_date": "2000-01-01",
    "end_date": "2025-12-31",
    "n_particles": 1000,
    "n_smoother_paths": 1000,
    "n_epochs": 3,
    "learning_rate": 0.0001,
    "max_goals": 8,
    "seed": 0,
    "initial_params": "",
    "output_dir": "rbpf/outputs/smoothing",
    "m_step": "adam",
    "gpu_type": "A100",
    "colab_timeout": 7200,
    "repo_url": "https://github.com/ryantjx/rbsqmc.git",
}

# Map m_step -> Python module to run
MODULE_MAP = {
    "adam": "rbpf.src.smoothing",
    "bfgs": "rbpf.src.smoothing_bfgs",
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


def bootstrap(no_clone: bool = False) -> Path:
    """Clone/update the repository and install runtime dependencies.

    When ``no_clone`` is True, skip the git clone/pull step and assume the
    repository has already been uploaded to ``REPO_DIR`` (e.g. via
    ``colab run --upload``). This avoids any dependency on GitHub.
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
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "jax[cuda12]==0.11.0", "cuthbert==0.0.14", "optax==0.2.8",
        "pandas", "pyarrow", "matplotlib", "scipy",
    ])
    return REPO_DIR


def load_config(repo_root: Path, override: str | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if override:
        path = Path(override)
        # If the override is just a filename (no path separators), resolve it
        # relative to the config directory inside the repo. This lets the
        # orchestrator pass e.g. "smoothing_bfgs_gpu_config.json" without
        # knowing the repo root path on the VM.
        if not path.is_absolute() and not path.parent.parts:
            path = repo_root / "rbpf/scripts/config" / path
    else:
        path = repo_root / "rbpf/scripts/config/smoothing_gpu_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing smoothing configuration: {path}")
    config.update(json.loads(path.read_text(encoding="utf-8")))

    if config["m_step"] not in MODULE_MAP:
        raise ValueError(f"m_step must be one of {list(MODULE_MAP)}, got {config['m_step']!r}")

    positive_integer = (
        "n_particles", "n_smoother_paths", "n_epochs", "max_goals", "colab_timeout",
    )
    for key in positive_integer:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")

    if config["m_step"] == "adam":
        if float(config.get("learning_rate", 0)) <= 0:
            raise ValueError("learning_rate must be positive for adam m_step")

    return config


def training_command(config: dict) -> list[str]:
    module = MODULE_MAP[config["m_step"]]
    return [sys.executable, "-u", "-m", module]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab GPU bootstrap for RBPF smoothing EM")
    p.add_argument("--config", help="Optional config JSON override")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-clone", action="store_true",
                   help="Skip git clone; use the repo already uploaded to /content/rbsqmc")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        config = load_config(Path.cwd(), args.config)
        print(" ".join(training_command(config)))
        return 0

    repo_root = bootstrap(no_clone=args.no_clone) if is_colab() else Path(__file__).resolve().parents[2]
    config = load_config(repo_root, args.config)
    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_matplotlib")
    progress_path = repo_root / config["output_dir"] / "progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["RBSQMC_PROGRESS_LOG"] = str(progress_path)
    if config.get("initial_params") and not (repo_root / config["initial_params"]).is_file():
        raise FileNotFoundError(
            f"remote initial parameter file does not exist: {config['initial_params']}"
        )

    run([
        sys.executable, "-c",
        "import jax; print('JAX devices:', jax.devices()); "
        "assert jax.default_backend() == 'gpu', 'Colab GPU is not active'",
    ], cwd=repo_root)
    log(f"m_step={config['m_step']}, output_dir={config['output_dir']}")
    log(f"Writing remote outputs to {repo_root / config['output_dir']}")
    log(f"Progress log: {progress_path}")
    run(training_command(config), cwd=repo_root, forward_raw=True)
    log("GPU smoothing completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}", stream="ERR")
        raise