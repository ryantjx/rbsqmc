"""Clone rbsqmc and run the unified comparison suite on Colab.

The local orchestrator submits only this entry point and its compact JSON
configuration. This script shallow-clones the public repository, installs
missing non-JAX dependencies, verifies that JAX is using a GPU, runs the
unified comparison suite (QMC, Hilbert sort, SQMC-vs-SMC), and records
metadata.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path("/content/rbsqmc")
REPO_URL = "https://github.com/ryantjx/rbsqmc.git"
DEFAULT_CONFIG_NAME = "comparison_config.json"
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "numba": "numba",
    "cuthbert": "cuthbert",
    "cuthbertlib": "cuthbertlib",
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    log("Running: " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )


def install_missing_packages() -> None:
    """Install only unavailable benchmark dependencies; preserve Colab JAX."""
    try:
        importlib.import_module("jax")
    except Exception as error:
        raise RuntimeError(
            "JAX is unavailable. A Colab GPU runtime with its preinstalled "
            "CUDA-enabled JAX is required; this script will not replace it."
        ) from error

    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)

    if missing:
        log(f"Installing missing packages: {missing}")
        run([sys.executable, "-m", "pip", "install", *missing])
    else:
        log("All benchmark dependencies are already available")


def clone_repository(repo_url: str, branch: str) -> Path:
    """Create a shallow sparse checkout containing only ``sqmc/``."""
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--single-branch",
            "--branch",
            branch,
            repo_url,
            str(REPO_DIR),
        ]
    )
    run(
        ["git", "sparse-checkout", "set", "sqmc"],
        cwd=REPO_DIR,
    )
    return REPO_DIR


def validate_config(config: dict) -> dict:
    """Validate and normalize the comparison configuration."""
    for section in ("qmc", "hilbert_sort", "sqmc_smc"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")

    qmc = config["qmc"]
    if not 1 <= int(qmc["dimension"]) <= 10_000:
        raise ValueError("qmc.dimension must be in [1, 10000].")
    if not qmc["n_values"] or any(int(n) <= 0 for n in qmc["n_values"]):
        raise ValueError("qmc.n_values must contain positive integers.")
    if int(qmc["repeats"]) <= 0:
        raise ValueError("qmc.repeats must be positive.")

    hilbert = config["hilbert_sort"]
    if not 2 <= int(hilbert["dimension"]) <= 62:
        raise ValueError("hilbert_sort.dimension must be in [2, 62].")
    if not hilbert["n_values"] or any(int(n) <= 0 for n in hilbert["n_values"]):
        raise ValueError("hilbert_sort.n_values must contain positive integers.")
    if int(hilbert["repeats"]) <= 0:
        raise ValueError("hilbert_sort.repeats must be positive.")

    sqmc = config["sqmc_smc"]
    if int(sqmc["n_steps"]) <= 0:
        raise ValueError("sqmc_smc.n_steps must be positive")
    if int(sqmc["n_reps"]) <= 0:
        raise ValueError("sqmc_smc.n_reps must be positive")
    if int(sqmc.get("accuracy_reps", 8)) <= 0:
        raise ValueError("sqmc_smc.accuracy_reps must be positive")
    if int(sqmc.get("warmups", 0)) < 0:
        raise ValueError("sqmc_smc.warmups must be non-negative")
    if not sqmc["particle_counts"] or any(
        int(n) <= 1 for n in sqmc["particle_counts"]
    ):
        raise ValueError(
            "sqmc_smc.particle_counts must contain integers greater than one"
        )
    dimensions = [int(d) for d in sqmc["dimensions"]]
    if not dimensions or any(d <= 0 or d > 62 for d in dimensions):
        raise ValueError("sqmc_smc.dimensions must contain integers in [1, 62]")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("sqmc_smc.dimensions must not contain duplicates")
    return config


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return validate_config(json.load(source))


def comparison_command(config: dict, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sqmc.comparison.benchmark_comparison",
        "--output-dir",
        str(output_dir),
    ]


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run the unified comparison suite on a Colab GPU."
    )
    argument_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_NAME,
        help="Config filename under sqmc/comparison/config after cloning.",
    )
    argument_parser.add_argument(
        "--config-json",
        help="Compact config supplied by the local orchestrator.",
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without requiring a GPU.",
    )
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.config_json:
        config = validate_config(json.loads(arguments.config_json))
    else:
        config_path = Path(arguments.config)
        if not config_path.is_absolute():
            local_candidate = Path(__file__).resolve().parent / "config" / config_path
            config_path = (
                local_candidate
                if local_candidate.exists()
                else Path.cwd() / arguments.config
            )
        config = load_config(config_path)

    output_dir = Path(config.get("remote_output_dir", "sqmc/comparison/outputs"))

    if arguments.dry_run:
        print(" ".join(comparison_command(config, output_dir)))
        return 0

    install_missing_packages()
    repo_root = clone_repository(
        config.get("repo_url", REPO_URL),
        config.get("repo_branch", "main"),
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    # The compressed table is intentionally ignored by Git. Recreate it from
    # the checked-in Joe--Kuo source before importing sqmc.qmc.qmc.
    direction_numbers = repo_root / "sqmc/qmc/_sobol_direction_numbers.npz"
    if not direction_numbers.exists():
        run(
            [
                sys.executable,
                "sqmc/qmc/_generate_sobol_data.py",
                "--verify-scipy",
            ],
            cwd=repo_root,
        )

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sqmc-matplotlib")

    import jax
    import matplotlib
    import numba
    import numpy
    import scipy

    gpu_devices = list(jax.devices("gpu"))
    if not gpu_devices:
        raise RuntimeError(f"No JAX GPU is active; available devices: {jax.devices()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"JAX devices: {jax.devices()}")

    run(comparison_command(config, output_dir), cwd=repo_root)

    required_outputs = (
        "qmc_benchmark.csv",
        "qmc_benchmark_gpu.png",
        "hilbert_sort_benchmark.csv",
        "hilbert_sort_benchmark_gpu.png",
        "sqmc_smc_gpu_runtime.png",
        "sqmc_smc_gpu_diversity.png",
        "sqmc_smc_gpu_efficiency.png",
        "sqmc_smc_gpu_results.json",
        "run_config.json",
        "run_metadata.json",
    )
    missing_outputs = [
        name for name in required_outputs if not (output_dir / name).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Comparison suite did not create: {missing_outputs}")

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_name": str(jax.devices("gpu")[0]) if jax.devices("gpu") else None,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "numba": numba.__version__,
        "config": config,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Comparison suite completed; outputs are in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())