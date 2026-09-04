"""Unified comparison suite: QMC, Hilbert sort, and SQMC-vs-SMC benchmarks.

Runs the three comparisons required by the dissertation empirical evaluation:

1. QMC point generation: our JAX implementation vs SciPy, on CPU and GPU.
2. Hilbert sort: our JAX implementation vs the ``particles`` reference, on
   CPU and GPU.
3. SQMC vs SMC on GPU, across particle counts and state dimensions.

Each sub-benchmark is invoked as a subprocess so that JAX platform selection
(CPU vs GPU) is isolated per run. Results are aggregated into
``sqmc/comparison/outputs``; ``run_config.json`` and ``run_metadata.json``
are additionally copied to ``sqmc/sqmc/scripts/outputs/sqmc_gpu``.

Usage:
    python -m sqmc.comparison.benchmark_comparison [--config CONFIG.json]
        [--skip-qmc] [--skip-hilbert] [--skip-sqmc-smc] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

DEFAULT_OUTPUT_DIR = "sqmc/comparison/outputs"
SQMC_GPU_OUTPUT_DIR = "sqmc/sqmc/scripts/outputs/sqmc_gpu"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "scripts" / "config" / "comparison_config.json"
)

# Artifacts produced by each sub-benchmark that must exist after its run.
# The by-algorithm charts are produced by the older orchestrator
# (sqmc/sqmc/scripts/run_qmc_benchmarks_gpu.py), not by the benchmark modules.
QMC_OUTPUTS = (
    "qmc_benchmark.csv",
    "qmc_benchmark_cpu.png",
    "qmc_benchmark_gpu.png",
)
HILBERT_OUTPUTS = (
    "hilbert_sort_benchmark.csv",
    "hilbert_sort_benchmark_gpu.png",
)
SQMC_SMC_OUTPUTS = (
    "sqmc_smc_gpu_runtime.png",
    "sqmc_smc_gpu_diversity.png",
    "sqmc_smc_gpu_efficiency.png",
    "sqmc_smc_gpu_results.json",
    "run_config.json",
)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run the unified QMC / Hilbert-sort / SQMC-vs-SMC comparison suite."
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the comparison configuration JSON.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for aggregated outputs.",
    )
    argument_parser.add_argument(
        "--sqmc-gpu-output-dir",
        type=Path,
        default=Path(SQMC_GPU_OUTPUT_DIR),
        help="Secondary output directory for run_config/run_metadata JSON.",
    )
    argument_parser.add_argument("--skip-qmc", action="store_true")
    argument_parser.add_argument("--skip-hilbert", action="store_true")
    argument_parser.add_argument("--skip-sqmc-smc", action="store_true")
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sub-benchmark commands without running them.",
    )
    return argument_parser


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _run(command: list[str], env: dict | None = None) -> None:
    log("Running: " + " ".join(command))
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(command, env=merged_env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def _check_outputs(output_dir: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (output_dir / name).is_file()]


def _gpu_available() -> bool:
    """Return True when JAX reports a GPU device."""
    try:
        import jax

        return bool(list(jax.devices("gpu")))
    except (ImportError, RuntimeError, ValueError):
        return False


def run_qmc(config: dict, output_dir: Path) -> dict:
    """Run the QMC benchmark (our JAX implementation vs SciPy, CPU and GPU)."""
    section = config["qmc"]
    command = [
        sys.executable,
        "-m",
        "sqmc.qmc.benchmark_qmc",
        "--dimension",
        str(section["dimension"]),
        "--n-values",
        *(str(n) for n in section["n_values"]),
        "--repeats",
        str(section["repeats"]),
        "--warmups",
        str(section["warmups"]),
        "--seed",
        str(section["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    _run(command)
    # The GPU charts are only produced when a JAX GPU is available.
    required = (
        QMC_OUTPUTS if _gpu_available() else ("qmc_benchmark.csv", "qmc_benchmark_cpu.png")
    )
    missing = _check_outputs(output_dir, required)
    if missing:
        raise RuntimeError(f"QMC benchmark did not create: {missing}")
    return {"command": command, "outputs": list(required)}


def run_hilbert(config: dict, output_dir: Path) -> dict:
    """Run the Hilbert-sort benchmark (our JAX implementation vs particles)."""
    section = config["hilbert_sort"]
    command = [
        sys.executable,
        "-m",
        "sqmc.hilbert_sort.benchmark_hilbert_sort",
        "--dimension",
        str(section["dimension"]),
        "--n-values",
        *(str(n) for n in section["n_values"]),
        "--repeats",
        str(section["repeats"]),
        "--warmups",
        str(section["warmups"]),
        "--seed",
        str(section["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    _run(command)
    # The GPU charts are only produced when a JAX GPU is available.
    required = (
        HILBERT_OUTPUTS
        if _gpu_available()
        else ("hilbert_sort_benchmark.csv",)
    )
    missing = _check_outputs(output_dir, required)
    if missing:
        raise RuntimeError(f"Hilbert-sort benchmark did not create: {missing}")
    return {"command": command, "outputs": list(required)}


def run_sqmc_smc(config: dict, output_dir: Path) -> dict:
    """Run the paired SMC/SQMC benchmark on GPU."""
    section = config["sqmc_smc"]
    command = [
        sys.executable,
        "-m",
        "sqmc.sqmc.benchmark_sqmc_smc",
        "--n-steps",
        str(section["n_steps"]),
        "--n-reps",
        str(section["n_reps"]),
        "--accuracy-reps",
        str(section["accuracy_reps"]),
        "--warmups",
        str(section["warmups"]),
        "--particle-counts",
        *(str(n) for n in section["particle_counts"]),
        "--dimensions",
        *(str(d) for d in section["dimensions"]),
        "--seed",
        str(section["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    _run(command)
    missing = _check_outputs(output_dir, SQMC_SMC_OUTPUTS)
    if missing:
        raise RuntimeError(f"SQMC-vs-SMC benchmark did not create: {missing}")
    return {"command": command, "outputs": list(SQMC_SMC_OUTPUTS)}


def write_metadata(output_dir: Path, config: dict, runs: dict) -> None:
    """Write run_config.json and run_metadata.json to the output directory."""
    run_config = {
        "config": config,
        "runs": {name: {"command": run["command"]} for name, run in runs.items()},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": config,
        "runs": list(runs),
    }
    try:
        import jax

        metadata["jax"] = jax.__version__
        metadata["jax_devices"] = [str(device) for device in jax.devices()]
    except ImportError:
        pass
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    config = load_config(arguments.config)
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}

    if not arguments.skip_qmc:
        log("=== QMC benchmark (JAX vs SciPy, CPU and GPU) ===")
        runs["qmc"] = run_qmc(config, output_dir)

    if not arguments.skip_hilbert:
        log("=== Hilbert-sort benchmark (JAX vs particles, CPU and GPU) ===")
        runs["hilbert_sort"] = run_hilbert(config, output_dir)

    if not arguments.skip_sqmc_smc:
        log("=== SQMC-vs-SMC benchmark (GPU) ===")
        runs["sqmc_smc"] = run_sqmc_smc(config, output_dir)

    write_metadata(output_dir, config, runs)

    # Mirror run_config.json and run_metadata.json into the SQMC GPU output
    # directory so the existing dissertation workflow finds them there too.
    sqmc_gpu_dir = arguments.sqmc_gpu_output_dir
    if not sqmc_gpu_dir.is_absolute():
        sqmc_gpu_dir = _REPOSITORY_ROOT / sqmc_gpu_dir
    sqmc_gpu_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run_config.json", "run_metadata.json"):
        shutil.copy2(output_dir / name, sqmc_gpu_dir / name)
    log(f"Copied run_config.json and run_metadata.json to {sqmc_gpu_dir}")

    log(f"Comparison suite complete; outputs are in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())