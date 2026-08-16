"""Bootstrap a Colab GPU and invoke :mod:`rbpf_v2.scripts.run_smoothing`.

This file deliberately imports no project or JAX modules before bootstrapping,
so ``colab run`` can upload it into a fresh VM. The cloned repository's
``rbpf_v2/smoothing_gpu_config.json`` is the source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_DIR = Path("/content/rbsqmc")
DEFAULT_CONFIG = {
    "start_date": "2000-01-01",
    "end_date": "2026-01-01",
    "n_particles": 50,
    "n_smoother_paths": 50,
    "n_epochs": 5,
    "n_gradient_steps": 20,
    "learning_rate": 0.001,
    "max_goals": 8,
    "holdout_days": 1,
    "seed": 42,
    "initial_params": "",
    "output_dir": "rbpf_v2/outputs/smoothing",
    "repo_url": "https://github.com/ryantjx/rbsqmc.git",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(command, *, cwd=None) -> None:
    log("Running: " + " ".join(map(str, command)))
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True
    )
    if completed.stdout:
        for line in completed.stdout.rstrip().splitlines():
            log("OUT: " + line)
    if completed.stderr:
        for line in completed.stderr.rstrip().splitlines():
            log("ERR: " + line)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}:\n"
            + " ".join(map(str, command))
        )


def is_colab() -> bool:
    return Path("/content").exists()


def bootstrap() -> Path:
    """Clone/update the repository and install runtime dependencies."""
    if REPO_DIR.exists():
        log(f"Using existing checkout at {REPO_DIR}")
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        run(["git", "clone", DEFAULT_CONFIG["repo_url"], str(REPO_DIR)])
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "cuthbert", "optax", "pandas", "pyarrow", "matplotlib", "scipy",
    ])
    return REPO_DIR


def load_config(repo_root: Path, override: str | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    path = Path(override) if override else repo_root / "rbpf_v2/smoothing_gpu_config.json"
    if path.exists():
        config.update(json.loads(path.read_text()))
    return config


def training_command(config: dict) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "rbpf_v2.scripts.run_smoothing",
        "--start-date", str(config["start_date"]),
        "--end-date", str(config["end_date"]),
        "--output-dir", str(config["output_dir"]),
        "--seed", str(config["seed"]),
        "--n-particles", str(config["n_particles"]),
        "--n-smoother-paths", str(config["n_smoother_paths"]),
        "--n-epochs", str(config["n_epochs"]),
        "--n-gradient-steps", str(config["n_gradient_steps"]),
        "--learning-rate", str(config["learning_rate"]),
        "--max-goals", str(config["max_goals"]),
        "--holdout-days", str(config["holdout_days"]),
    ]
    if config.get("initial_params"):
        command.extend(["--initial-params", str(config["initial_params"])])
    return command


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab GPU bootstrap for RBPF v2 smoothing")
    p.add_argument("--config", help="Optional config JSON override")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        config = load_config(Path.cwd(), args.config)
        print(" ".join(training_command(config)))
        return 0

    repo_root = bootstrap() if is_colab() else Path(__file__).resolve().parents[2]
    config = load_config(repo_root, args.config)
    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_v2_matplotlib")
    progress_path = repo_root / config["output_dir"] / "progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["RBSQMC_PROGRESS_LOG"] = str(progress_path)

    run([
        sys.executable, "-c",
        "import jax; print('JAX devices:', jax.devices()); "
        "assert jax.default_backend() == 'gpu', 'Colab GPU is not active'",
    ], cwd=repo_root)
    log(f"Writing remote outputs to {repo_root / config['output_dir']}")
    log(f"Progress log: {progress_path}")
    run(training_command(config), cwd=repo_root)
    log("GPU smoothing completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
