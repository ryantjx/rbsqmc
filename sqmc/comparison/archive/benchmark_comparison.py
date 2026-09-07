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

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Run the QMC benchmark for each configured dimension.

    The benchmark module handles one dimension per invocation, so the suite
    runs it once per dimension and merges the CSV rows. The per-dimension
    rows are what the speedup-by-dimension figure plots.
    """
    section = config["qmc"]
    dimensions = section.get("dimensions", [section["dimension"]])
    commands = []
    for dimension in dimensions:
        command = [
            sys.executable,
            "-m",
            "sqmc.qmc.benchmark_qmc",
            "--dimension",
            str(dimension),
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
        commands.append(command)
        # benchmark_qmc overwrites qmc_benchmark.csv on every invocation, so
        # snapshot the rows for this dimension before the next run replaces
        # them; _merge_qmc_csvs reassembles the full table from these copies.
        snapshot = output_dir / f"qmc_benchmark_d{dimension}.csv"
        main_csv = output_dir / "qmc_benchmark.csv"
        if main_csv.exists():
            shutil.copyfile(main_csv, snapshot)

    # Merge the per-dimension CSVs (each run overwrites qmc_benchmark.csv).
    _merge_qmc_csvs(output_dir, dimensions)

    # The GPU charts are only produced when a JAX GPU is available.
    required = (
        ("qmc_benchmark.csv", "qmc_benchmark_cpu.png", "qmc_benchmark_gpu.png")
        if _gpu_available()
        else ("qmc_benchmark.csv", "qmc_benchmark_cpu.png")
    )
    missing = _check_outputs(output_dir, required)
    if missing:
        raise RuntimeError(f"QMC benchmark did not create: {missing}")

    # Speedup-by-dimension figure from the merged CSV.
    _plot_qmc_speedup_by_dimension(output_dir, dimensions)
    return {"command": commands[-1], "outputs": list(required)}


def _merge_qmc_csvs(output_dir: Path, dimensions: list[int]) -> None:
    """Merge the per-dimension QMC CSV rows into one CSV.

    Each benchmark invocation overwrites ``qmc_benchmark.csv``, so the rows
    for earlier dimensions are recovered from the per-dimension copies.
    """
    import pandas as pd

    frames = []
    for dimension in dimensions:
        per_dim = output_dir / f"qmc_benchmark_d{dimension}.csv"
        main_csv = output_dir / "qmc_benchmark.csv"
        if per_dim.exists():
            frames.append(pd.read_csv(per_dim))
        elif main_csv.exists():
            frame = pd.read_csv(main_csv)
            rows = frame[frame["dimension"] == dimension]
            if not rows.empty:
                rows.to_csv(per_dim, index=False)
                frames.append(rows)
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["backend", "sequence", "implementation", "dimension",
                "scramble", "number_of_samples"],
        keep="last",
    )
    merged.to_csv(output_dir / "qmc_benchmark.csv", index=False)


def _plot_qmc_speedup_by_dimension(output_dir: Path,
                                   dimensions: list[int]) -> None:
    """Plot the GPU-vs-SciPy-CPU speedup of the JAX generators per dimension."""
    import pandas as pd

    csv_path = output_dir / "qmc_benchmark.csv"
    if not csv_path.exists():
        return
    frame = pd.read_csv(csv_path)
    frame = frame[frame["backend"] == "gpu"]
    if frame.empty:
        return  # No GPU rows (CPU-only run); nothing to plot.

    sequences = sorted(frame["sequence"].unique())
    figure, axes = plt.subplots(
        1, len(sequences), figsize=(5 * len(sequences), 4), squeeze=False
    )
    for ax, sequence in zip(axes[0], sequences):
        sub = frame[frame["sequence"] == sequence]
        for dimension in dimensions:
            dim_rows = sub[sub["dimension"] == dimension]
            jax_rows = dim_rows[dim_rows["implementation"] == "QMC JAX"]
            scipy_rows = dim_rows[dim_rows["implementation"] == "SciPy CPU"]
            if jax_rows.empty or scipy_rows.empty:
                continue
            merged = jax_rows.merge(
                scipy_rows, on="number_of_samples", suffixes=("_jax", "_scipy")
            )
            speedup = merged["median_seconds_scipy"] / merged["median_seconds_jax"]
            ax.plot(merged["number_of_samples"], speedup, marker="o",
                    label=f"$d={dimension}$")
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Samples $N$")
        ax.set_title(sequence)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("Speedup: SciPy CPU / JAX GPU")
    figure.suptitle(
        "QMC point generation on GPU: speedup over the SciPy CPU reference"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "qmc_speedup_by_dimension.png", dpi=200)
    plt.close(figure)


def run_hilbert(config: dict, output_dir: Path) -> dict:
    """Run the Hilbert-sort benchmark for each configured dimension.

    The benchmark module does not record the dimension in its CSV, so each
    dimension is written to its own subdirectory and the merged CSV gains an
    explicit dimension column.
    """
    import pandas as pd

    section = config["hilbert_sort"]
    dimensions = section.get("dimensions", [section["dimension"]])
    commands = []
    frames = []
    for dimension in dimensions:
        dim_dir = output_dir / f"hilbert_d{dimension}"
        dim_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "sqmc.hilbert_sort.benchmark_hilbert_sort",
            "--dimension",
            str(dimension),
            "--n-values",
            *(str(n) for n in section["n_values"]),
            "--repeats",
            str(section["repeats"]),
            "--warmups",
            str(section["warmups"]),
            "--seed",
            str(section["seed"]),
            "--output-dir",
            str(dim_dir),
        ]
        _run(command)
        commands.append(command)
        frame = pd.read_csv(dim_dir / "hilbert_sort_benchmark.csv")
        frame.insert(1, "dimension", dimension)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_dir / "hilbert_sort_benchmark.csv", index=False)

    # The GPU chart is only produced when a JAX GPU is available. Each
    # per-dimension run writes its own chart into its subdirectory; copy the
    # last one to the top level so the single-chart artifact exists there too.
    required = (
        ("hilbert_sort_benchmark.csv", "hilbert_sort_benchmark_gpu.png")
        if _gpu_available()
        else ("hilbert_sort_benchmark.csv",)
    )
    if _gpu_available():
        last_dim_dir = output_dir / f"hilbert_d{dimensions[-1]}"
        gpu_chart = last_dim_dir / "hilbert_sort_benchmark_gpu.png"
        if gpu_chart.exists():
            shutil.copy2(gpu_chart, output_dir / "hilbert_sort_benchmark_gpu.png")
    missing = _check_outputs(output_dir, required)
    if missing:
        raise RuntimeError(f"Hilbert-sort benchmark did not create: {missing}")
    _plot_hilbert_speedup_by_dimension(output_dir, dimensions)
    return {"command": commands[-1], "outputs": list(required)}


def _plot_hilbert_speedup_by_dimension(output_dir: Path,
                                       dimensions: list[int]) -> None:
    """Plot the GPU-vs-particles speedup of the Hilbert sort per dimension."""
    import pandas as pd

    csv_path = output_dir / "hilbert_sort_benchmark.csv"
    if not csv_path.exists():
        return
    frame = pd.read_csv(csv_path)
    gpu = frame[frame["backend"] == "gpu"]
    if gpu.empty:
        return

    figure, ax = plt.subplots(figsize=(6, 4))
    for dimension in dimensions:
        dim_rows = gpu[gpu["dimension"] == dimension] if "dimension" in gpu else None
        if dim_rows is None or dim_rows.empty:
            continue
        jax_rows = dim_rows[dim_rows["method"] == "Optimized JAX"]
        particles_rows = dim_rows[dim_rows["method"] == "Particles Numba (CPU)"]
        if jax_rows.empty or particles_rows.empty:
            continue
        merged = jax_rows.merge(
            particles_rows, on="number_of_particles", suffixes=("_jax", "_parts")
        )
        speedup = merged["median_seconds_parts"] / merged["median_seconds_jax"]
        ax.plot(merged["number_of_particles"], speedup, marker="o",
                label=f"$d={dimension}$")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log")
    ax.set_xlabel("Particles $N$")
    ax.set_ylabel("Speedup: particles (CPU) / JAX (GPU)")
    ax.set_title("Hilbert sort on GPU: speedup over the particles reference")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "hilbert_speedup_by_dimension.png", dpi=200)
    plt.close(figure)


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