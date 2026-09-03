"""Benchmark the SQMC and SciPy Halton and Sobol sequence generators.

The benchmark always measures the custom JAX implementation on CPU against
SciPy's CPU implementation. If JAX exposes a GPU, it also creates a GPU chart
that compares the custom implementation on that GPU with the same SciPy CPU
reference. JAX compilation, engine initialization, and host/device transfers
are excluded from timed regions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# Keep Matplotlib's cache in a writable location on sandboxed systems.
_MPL_CACHE = Path(tempfile.gettempdir()) / "sqmc-matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc as scipy_qmc

from sqmc.qmc.qmc import Halton, Sobol, _MAXBITS, _sobol_sample_batched


_DEFAULT_N_VALUES = (128, 512, 2_048, 8_192, 32_768, 131_072)
_DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
_DEFAULT_SCRAMBLE = True


@dataclass(frozen=True)
class Timing:
    """Summary of repeated runtime measurements."""

    median_seconds: float
    lower_quartile_seconds: float
    upper_quartile_seconds: float


@dataclass(frozen=True)
class BenchmarkRow:
    """One sampler's timing at one sample count and execution backend."""

    backend: str
    sequence: str
    implementation: str
    number_of_samples: int
    dimension: int
    scramble: bool
    timing: Timing


def _gpu_devices() -> list[jax.Device]:
    """Return available JAX GPU devices without failing on CPU-only installs."""
    try:
        return list(jax.devices("gpu"))
    except (RuntimeError, ValueError):
        return []


def _summarize(samples: Sequence[float]) -> Timing:
    values = np.asarray(samples, dtype=np.float64)
    return Timing(
        median_seconds=float(np.median(values)),
        lower_quartile_seconds=float(np.quantile(values, 0.25)),
        upper_quartile_seconds=float(np.quantile(values, 0.75)),
    )


def _time_sqmc_engine(
    engine_factory: Callable[[], Halton | Sobol],
    n: int,
    device: jax.Device,
    warmups: int,
    repeats: int,
) -> Timing:
    """Time a compiled sampling kernel with output kept on-device."""
    with jax.default_device(device):
        engine = engine_factory()
        first_index = jax.device_put(
            jnp.asarray(0, dtype=jnp.uint32),
            device=device,
        )

        if isinstance(engine, Halton):

            @jax.jit
            def sample_at(index: jax.Array) -> jax.Array:
                indices = index + jnp.arange(n, dtype=jnp.uint32)
                coordinates = []
                for dim, (base, num_digits) in enumerate(
                    zip(engine._bases, engine._digits_per_dim)
                ):
                    if engine.scramble:
                        coordinate = engine._scrambled_radical_inverse(
                            indices=indices,
                            base=base,
                            permutations=engine._permutations[dim],
                            tail_correction=engine._tail_corrections[dim],
                        )
                    else:
                        coordinate = engine._radical_inverse(
                            indices=indices,
                            base=base,
                            num_digits=num_digits,
                        )
                    coordinates.append(coordinate)
                return jnp.stack(coordinates, axis=-1).astype(engine.dtype)

        else:

            def sample_at(index: jax.Array) -> jax.Array:
                return _sobol_sample_batched(
                    first_index=index,
                    n=n,
                    direction_integers=engine._direction_integers,
                    digital_shift=engine._digital_shift,
                    num_bits=_MAXBITS,
                    dtype=engine.dtype,
                )

        # Compile for this output shape and finish deferred initialization work.
        result = sample_at(first_index)
        if result.shape != (n, engine.d):
            raise RuntimeError("SQMC sampler returned an invalid shape.")
        result.block_until_ready()

        for _ in range(warmups):
            sample_at(first_index).block_until_ready()

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            sample_at(first_index).block_until_ready()
            samples.append(time.perf_counter() - start)

    return _summarize(samples)


def _time_scipy_engine(
    engine_factory: Callable[[], scipy_qmc.QMCEngine],
    n: int,
    warmups: int,
    repeats: int,
) -> Timing:
    """Time SciPy sampling while excluding engine construction."""
    engine = engine_factory()

    result = engine.random(n)
    if result.shape != (n, engine.d):
        raise RuntimeError("SciPy sampler returned an invalid shape.")

    for _ in range(warmups):
        engine.reset()
        result = engine.random(n)
        if result.shape != (n, engine.d):
            raise RuntimeError("SciPy sampler returned an invalid shape.")

    samples = []
    for _ in range(repeats):
        engine.reset()
        start = time.perf_counter()
        result = engine.random(n)
        elapsed = time.perf_counter() - start
        if result.shape != (n, engine.d):
            raise RuntimeError("SciPy sampler returned an invalid shape.")
        samples.append(elapsed)

    return _summarize(samples)


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _scipy_factories(
    dimension: int,
    scramble: bool,
    seed: int,
) -> dict[str, Callable[[], scipy_qmc.QMCEngine]]:
    return {
        "Halton": lambda: scipy_qmc.Halton(
            d=dimension,
            scramble=scramble,
            rng=np.random.default_rng(seed),
        ),
        "Sobol": lambda: scipy_qmc.Sobol(
            d=dimension,
            scramble=scramble,
            bits=_MAXBITS,
            rng=np.random.default_rng(seed),
        ),
    }


def _sqmc_factories(
    dimension: int,
    scramble: bool,
    seed: int,
    device: jax.Device,
) -> dict[str, Callable[[], Halton | Sobol]]:
    def key() -> jax.Array:
        return jax.device_put(jax.random.PRNGKey(seed), device=device)

    return {
        "Halton": lambda: Halton(
            d=dimension,
            scramble=scramble,
            key=key(),
            dtype=jnp.float64,
        ),
        "Sobol": lambda: Sobol(
            d=dimension,
            scramble=scramble,
            key=key(),
            dtype=jnp.float64,
        ),
    }


def _benchmark_scipy(
    n_values: Sequence[int],
    dimension: int,
    scramble: bool,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, dict[int, Timing]]:
    factories = _scipy_factories(dimension, scramble, seed)
    results: dict[str, dict[int, Timing]] = {}

    for sequence, factory in factories.items():
        print(f"\n{sequence}: SciPy (CPU)")
        timings = {}
        for n in n_values:
            timing = _time_scipy_engine(factory, n, warmups, repeats)
            timings[n] = timing
            print(f"  n={n:>9,}: {_format_seconds(timing.median_seconds)}")
        results[sequence] = timings

    return results


def _benchmark_sqmc_backend(
    n_values: Sequence[int],
    dimension: int,
    scramble: bool,
    seed: int,
    device: jax.Device,
    warmups: int,
    repeats: int,
) -> dict[str, dict[int, Timing]]:
    factories = _sqmc_factories(dimension, scramble, seed, device)
    results: dict[str, dict[int, Timing]] = {}

    for sequence, factory in factories.items():
        print(f"\n{sequence}: QMC JAX ({device.platform.upper()}: {device})")
        timings = {}
        for n in n_values:
            timing = _time_sqmc_engine(factory, n, device, warmups, repeats)
            timings[n] = timing
            print(f"  n={n:>9,}: {_format_seconds(timing.median_seconds)}")
        results[sequence] = timings

    return results


def _rows_for_backend(
    backend: str,
    n_values: Sequence[int],
    dimension: int,
    scramble: bool,
    scipy_timings: dict[str, dict[int, Timing]],
    sqmc_timings: dict[str, dict[int, Timing]],
) -> list[BenchmarkRow]:
    rows = []
    for sequence in ("Halton", "Sobol"):
        rows.extend(
            BenchmarkRow(
                backend=backend,
                sequence=sequence,
                implementation="QMC JAX",
                number_of_samples=n,
                dimension=dimension,
                scramble=scramble,
                timing=sqmc_timings[sequence][n],
            )
            for n in n_values
        )
        rows.extend(
            BenchmarkRow(
                backend=backend,
                sequence=sequence,
                implementation="SciPy CPU",
                number_of_samples=n,
                dimension=dimension,
                scramble=scramble,
                timing=scipy_timings[sequence][n],
            )
            for n in n_values
        )
    return rows


def _plot_rows(
    rows: Sequence[BenchmarkRow],
    dimension: int,
    scramble: bool,
    backend: str,
    output_directory: Path,
) -> Path:
    """Create side-by-side Halton and Sobol runtime line charts."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    styles = {
        "QMC JAX": ("o", "-"),
        "SciPy CPU": ("s", "--"),
    }

    for axis, sequence in zip(axes, ("Halton", "Sobol")):
        for implementation, (marker, line_style) in styles.items():
            method_rows = sorted(
                (
                    row
                    for row in rows
                    if row.sequence == sequence
                    and row.implementation == implementation
                ),
                key=lambda row: row.number_of_samples,
            )
            counts = np.asarray(
                [row.number_of_samples for row in method_rows],
                dtype=np.int64,
            )
            medians = np.asarray(
                [row.timing.median_seconds for row in method_rows],
                dtype=np.float64,
            )
            lower = np.asarray(
                [row.timing.lower_quartile_seconds for row in method_rows],
                dtype=np.float64,
            )
            upper = np.asarray(
                [row.timing.upper_quartile_seconds for row in method_rows],
                dtype=np.float64,
            )
            log_counts = np.log10(counts)
            log_medians = np.log10(medians)
            log_lower = np.log10(lower)
            log_upper = np.log10(upper)
            (line,) = axis.plot(
                log_counts,
                log_medians,
                marker=marker,
                linestyle=line_style,
                linewidth=2.0,
                markersize=5.5,
                label=implementation,
            )
            axis.fill_between(
                log_counts,
                log_lower,
                log_upper,
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )

        axis.set_xlabel(r"$\log_{10}(n)$")
        axis.set_title(sequence)
        axis.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        axis.grid(False, which="minor")
        axis.legend(frameon=False)

    axes[0].set_ylabel(r"$\log_{10}(t / \mathrm{s})$")
    scramble_label = "scrambled" if scramble else "unscrambled"
    figure.suptitle(
        f"QMC sampling runtime, d={dimension} "
        f"({backend.upper()}, {scramble_label})\n"
        "Compilation, engine initialization, and transfer excluded; "
        "shaded region is the IQR"
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"qmc_benchmark_{backend}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _write_csv(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "backend",
                "sequence",
                "implementation",
                "dimension",
                "scramble",
                "number_of_samples",
                "median_seconds",
                "lower_quartile_seconds",
                "upper_quartile_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.backend,
                    row.sequence,
                    row.implementation,
                    row.dimension,
                    row.scramble,
                    row.number_of_samples,
                    row.timing.median_seconds,
                    row.timing.lower_quartile_seconds,
                    row.timing.upper_quartile_seconds,
                ]
            )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QMC JAX and SciPy Halton and Sobol sampling."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="QMC dimension (default: 5).",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=list(_DEFAULT_N_VALUES),
        help="Sample counts used on the x-axis.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Timed repetitions per point (default: 7).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed calls after compilation (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scramble",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_SCRAMBLE,
        help="Benchmark scrambled sequences (default: enabled).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.dimension <= 10_000:
        raise ValueError("dimension must be in [1, 10000].")
    if not arguments.n_values or any(n <= 0 for n in arguments.n_values):
        raise ValueError("all n-values must be positive.")
    if any(n > 2**_MAXBITS for n in arguments.n_values):
        raise ValueError(f"n-values must not exceed 2**{_MAXBITS}.")
    if len(set(arguments.n_values)) != len(arguments.n_values):
        raise ValueError("n-values must not contain duplicates.")
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if arguments.warmups < 0:
        raise ValueError("warmups must be non-negative.")


def main() -> None:
    arguments = _parse_arguments()
    _validate_arguments(arguments)
    n_values = tuple(sorted(arguments.n_values))

    cpu_device = jax.devices("cpu")[0]
    gpu_devices = _gpu_devices()

    print("QMC sampling benchmark")
    print(f"  dimension: {arguments.dimension}")
    print(f"  scramble: {arguments.scramble}")
    print(f"  n values: {list(n_values)}")
    print(f"  repeats: {arguments.repeats}")
    print(f"  JAX x64 enabled: {jax.config.jax_enable_x64}")
    print(f"  CPU device: {cpu_device}")
    print(f"  GPU devices: {gpu_devices if gpu_devices else 'none'}")

    scipy_timings = _benchmark_scipy(
        n_values,
        dimension=arguments.dimension,
        scramble=arguments.scramble,
        seed=arguments.seed,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )

    all_rows = []
    cpu_sqmc_timings = _benchmark_sqmc_backend(
        n_values,
        dimension=arguments.dimension,
        scramble=arguments.scramble,
        seed=arguments.seed,
        device=cpu_device,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    cpu_rows = _rows_for_backend(
        "cpu",
        n_values,
        dimension=arguments.dimension,
        scramble=arguments.scramble,
        scipy_timings=scipy_timings,
        sqmc_timings=cpu_sqmc_timings,
    )
    all_rows.extend(cpu_rows)
    cpu_chart = _plot_rows(
        cpu_rows,
        dimension=arguments.dimension,
        scramble=arguments.scramble,
        backend="cpu",
        output_directory=arguments.output_dir,
    )
    print(f"\nSaved CPU chart: {cpu_chart}")

    if gpu_devices:
        gpu_sqmc_timings = _benchmark_sqmc_backend(
            n_values,
            dimension=arguments.dimension,
            scramble=arguments.scramble,
            seed=arguments.seed,
            device=gpu_devices[0],
            warmups=arguments.warmups,
            repeats=arguments.repeats,
        )
        gpu_rows = _rows_for_backend(
            "gpu",
            n_values,
            dimension=arguments.dimension,
            scramble=arguments.scramble,
            scipy_timings=scipy_timings,
            sqmc_timings=gpu_sqmc_timings,
        )
        all_rows.extend(gpu_rows)
        gpu_chart = _plot_rows(
            gpu_rows,
            dimension=arguments.dimension,
            scramble=arguments.scramble,
            backend="gpu",
            output_directory=arguments.output_dir,
        )
        print(f"Saved GPU chart: {gpu_chart}")
    else:
        print("No JAX GPU detected; skipped the GPU benchmark and chart.")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output_dir / "qmc_benchmark.csv"
    _write_csv(all_rows, csv_path)
    print(f"Saved timings: {csv_path}")


if __name__ == "__main__":
    main()
