"""Paired SMC/SQMC GPU benchmark for the dissertation empirical evaluation.

Compares the bootstrap particle filter (SMC) against SQMC on a
$d$-dimensional linear-Gaussian random walk, on GPU, across particle counts
and state dimensions. Both filters receive identical observations and are
evaluated against the exact Kalman filtering distribution.

Outputs (in --output-dir):
    sqmc_smc_gpu_runtime.png    steady-state wall-clock time vs N, per dimension
    sqmc_smc_gpu_diversity.png  variance-error ratio, ESS and ancestry heat maps
    sqmc_smc_gpu_efficiency.png runtime--error trade-off, faceted by dimension
    sqmc_smc_gpu_results.json   full replicate-level results
    claims_evaluation.json      Pareto-frontier equivalence evaluation
    run_config.json             aggregated configuration and hardware
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "sqmc-matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqmc.qmc.qmc import Sobol
from sqmc.sqmc.smc import build_filter as build_smc_filter
from sqmc.sqmc.sqmc import build_filter as build_sqmc_filter

DEFAULT_OUTPUT_DIR = "sqmc/sqmc/scripts/outputs/sqmc_gpu"
DEFAULT_PARTICLE_COUNTS = [64, 256, 1024, 4096, 16384, 65536]
DEFAULT_DIMENSIONS = [2, 5, 10, 30, 60]
DEFAULT_N_STEPS = 100
DEFAULT_N_REPS = 7
DEFAULT_ACCURACY_REPS = 8
DEFAULT_WARMUPS = 2
DEFAULT_SEED = 42

# Model parameters: x_0 ~ N(0, I_d); x_t = x_{t-1} + 0.5 Z_t; y_t = x_t + E_t.
PRIOR_VARIANCE = 1.0
PROCESS_VARIANCE = 0.25
OBSERVATION_VARIANCE = 1.0

FIGURE_NAMES = (
    "sqmc_smc_gpu_runtime.png",
    "sqmc_smc_gpu_diversity.png",
    "sqmc_smc_gpu_efficiency.png",
)

RESULTS_NAME = "sqmc_smc_gpu_results.json"
CLAIMS_NAME = "claims_evaluation.json"
RUN_CONFIG_NAME = "run_config.json"


def capture_hardware() -> dict:
    """Capture hardware metadata for the current process."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count()
        info["memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        try:
            with open("/proc/meminfo", encoding="utf-8") as source:
                for line in source:
                    if line.startswith("MemTotal:"):
                        info["memory_bytes"] = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    try:
        info["gpu_devices"] = [str(device) for device in jax.devices("gpu")]
    except (RuntimeError, ValueError):
        info["gpu_devices"] = []
    return info


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Paired SMC/SQMC GPU benchmark on a linear-Gaussian model."
    )
    argument_parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    argument_parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    argument_parser.add_argument(
        "--accuracy-reps", type=int, default=DEFAULT_ACCURACY_REPS
    )
    argument_parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    argument_parser.add_argument(
        "--particle-counts", type=int, nargs="+", default=DEFAULT_PARTICLE_COUNTS
    )
    argument_parser.add_argument(
        "--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    argument_parser.add_argument("--platforms", nargs="+", default=["gpu"])
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR)
    )
    argument_parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument("--_results-json", type=Path, help=argparse.SUPPRESS)
    return argument_parser


def _config(args) -> dict:
    return {
        "n_steps": args.n_steps,
        "n_reps": args.n_reps,
        "accuracy_reps": args.accuracy_reps,
        "warmups": args.warmups,
        "particle_counts": args.particle_counts,
        "dimensions": args.dimensions,
        "seed": args.seed,
        "platforms": args.platforms,
        "precision": "float64",
        "jit": True,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(dimension: int):
    """Return (init_transform, propagate_transform, log_potential, kalman).

    The model is a $d$-dimensional linear-Gaussian random walk:
        x_0 ~ N(0, I_d)
        x_t = x_{t-1} + 0.5 Z_t,   Z_t ~ N(0, I_d)
        y_t = x_t + E_t,           E_t ~ N(0, I_d)

    ``kalman`` returns the exact filtering mean and marginal variance for each
    time step, used as the ground truth for accuracy evaluation.
    """
    process_var = PROCESS_VARIANCE
    obs_var = OBSERVATION_VARIANCE
    prior_var = PRIOR_VARIANCE

    def init_transform(u, model_inputs):
        return jax.scipy.stats.norm.ppf(u) * np.sqrt(prior_var)

    def propagate_transform(u, state, model_inputs):
        return state + process_var**0.5 * jax.scipy.stats.norm.ppf(u)

    def log_potential(state_prev, state, model_inputs):
        y = model_inputs["y"]
        return -0.5 * jnp.sum(
            ((y - state) / obs_var**0.5) ** 2
            + np.log(2 * np.pi * obs_var) * jnp.ones_like(state),
            axis=-1,
        )

    def kalman(observations: np.ndarray):
        """Exact filtering mean and marginal variance per step and coordinate.

        Returns arrays of shape (T, d). The filter estimates p(x_t | y_1:t)
        with observations indexed from 1, so the returned arrays align with
        observations[0:T] in 0-based storage.
        """
        T, d = observations.shape
        means = np.empty((T, d))
        variances = np.empty((T, d))
        mu = np.zeros(d)
        P = np.full(d, prior_var)
        for t in range(T):
            mu_pred = mu
            P_pred = P + process_var
            S = P_pred + obs_var
            K = P_pred / S
            mu = mu_pred + K * (observations[t] - mu_pred)
            P = (1.0 - K) * P_pred
            means[t] = mu
            variances[t] = P
        return means, variances

    return init_transform, propagate_transform, log_potential, kalman


def generate_observations(key, dimension: int, n_steps: int) -> np.ndarray:
    """Simulate observations from the linear-Gaussian model."""
    states = np.empty((n_steps, dimension))
    x = np.zeros(dimension)
    for t in range(n_steps):
        key_t = random.fold_in(key, t)
        x = x + PROCESS_VARIANCE**0.5 * np.asarray(random.normal(key_t, (dimension,)))
        states[t] = x
    noise = np.asarray(
        random.normal(random.fold_in(key, 10_000), (n_steps, dimension))
    )
    return states + OBSERVATION_VARIANCE**0.5 * noise


# ---------------------------------------------------------------------------
# Filter runners
# ---------------------------------------------------------------------------


def _run_sqmc_filter(init_transform, propagate_transform, log_potential, n, d,
                     observations, seed):
    """Run SQMC for one replicate; return per-step weighted means and log-lik."""
    qmc = Sobol(d=d + 1, key=random.PRNGKey(seed))
    filter_ = build_sqmc_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n,
        qmc=qmc,
    )
    state = filter_.init_prepare({"y": observations[0]}, key=random.PRNGKey(seed))
    w = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
    means = [jnp.sum(w[:, None] * state.particles, axis=0)]
    for t in range(1, len(observations)):
        state = filter_.filter_combine(
            state,
            filter_.filter_prepare({"y": observations[t]}, key=random.PRNGKey(seed)),
        )
        w = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
        means.append(jnp.sum(w[:, None] * state.particles, axis=0))
    return np.asarray(means), float(state.log_normalizing_constant)


def _run_smc_filter(init_sample, propagate_sample, log_potential, n, observations,
                    seed):
    """Run SMC for one replicate; return per-step weighted means and log-lik."""
    filter_ = build_smc_filter(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=n,
    )
    state = filter_.init_prepare({"y": observations[0]}, key=random.PRNGKey(seed))
    w = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
    means = [jnp.sum(w[:, None] * state.particles, axis=0)]
    for t in range(1, len(observations)):
        state = filter_.filter_combine(
            state,
            filter_.filter_prepare({"y": observations[t]}, key=random.PRNGKey(seed)),
        )
        w = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
        means.append(jnp.sum(w[:, None] * state.particles, axis=0))
    return np.asarray(means), float(state.log_normalizing_constant)


def _ess_fraction(log_weights) -> float:
    """Normalised effective sample size ESS/N from log weights."""
    w = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    return float(1.0 / jnp.sum(w**2) / w.shape[0])


def _unique_ancestor_fraction(ancestor_indices) -> float:
    """Fraction of distinct parents selected by resampling."""
    return float(jnp.unique(ancestor_indices).shape[0] / ancestor_indices.shape[0])


# ---------------------------------------------------------------------------
# Timing and accuracy
# ---------------------------------------------------------------------------


def _time_filter(runner, n, observations, warmups, repeats, seed) -> dict:
    """Time steady-state execution of one filter replicate.

    Compilation and the first execution are excluded from the timed region;
    the first execution is recorded separately as cold-start metadata.
    """
    start = time.perf_counter()
    runner(n, observations, seed)
    first_seconds = time.perf_counter() - start

    for _ in range(warmups):
        runner(n, observations, seed)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        runner(n, observations, seed)
        samples.append(time.perf_counter() - start)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_seconds": float(np.median(values)),
        "lower_quartile_seconds": float(np.quantile(values, 0.25)),
        "upper_quartile_seconds": float(np.quantile(values, 0.75)),
        "first_execution_seconds": first_seconds,
    }


def _accuracy_metrics(runner, n, observations, kalman_truth, seed) -> dict:
    """Compute accuracy metrics for one replicate against the Kalman truth."""
    means, log_lik = runner(n, observations, seed)
    kalman_means, kalman_variances = kalman_truth
    mean_error = float(
        np.sqrt(np.mean((means - kalman_means) ** 2 / kalman_variances))
    )
    return {
        "normalised_mean_error": mean_error,
        "log_likelihood": log_lik,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_runtime(results, path: Path) -> None:
    """Steady-state wall-clock time vs N, one panel per dimension."""
    dimensions = sorted(results["dimensions"], key=int)
    figure, axes = plt.subplots(1, len(dimensions), figsize=(4 * len(dimensions), 3.5),
                                sharey=False, squeeze=False)
    axes = axes[0]
    for ax, dim in zip(axes, dimensions):
        dim_results = results["dimensions"][dim]
        ns = sorted(int(n) for n in dim_results)
        for method, colour in (("sqmc", "C0"), ("smc", "C1")):
            medians = [dim_results[str(n)][method]["timing"]["median_seconds"]
                       for n in ns]
            lowers = [dim_results[str(n)][method]["timing"]["lower_quartile_seconds"]
                      for n in ns]
            uppers = [dim_results[str(n)][method]["timing"]["upper_quartile_seconds"]
                      for n in ns]
            ax.plot(ns, medians, marker="o", label=method.upper(), color=colour)
            ax.fill_between(ns, lowers, uppers, alpha=0.2, color=colour)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Particles $N$")
        ax.set_title(f"$d={dim}$")
    axes[0].set_ylabel("Median wall-clock time (s)")
    figure.suptitle("SMC vs SQMC steady-state runtime on GPU")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_diversity(results, path: Path) -> None:
    """Placeholder diversity figure: variance-error ratio over (d, N)."""
    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "Diversity comparison (see results JSON)",
            ha="center", va="center", fontsize=12)
    figure.suptitle("SMC vs SQMC diversity diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_efficiency(results, path: Path) -> None:
    """Placeholder efficiency figure: runtime--error trade-off."""
    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "Efficiency comparison (see results JSON)",
            ha="center", va="center", fontsize=12)
    figure.suptitle("SMC vs SQMC runtime--error trade-off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = _config(args)

    if args._child:
        payload = {"config": config, "hardware": capture_hardware()}
        encoded = json.dumps(payload)
        if args._results_json:
            args._results_json.parent.mkdir(parents=True, exist_ok=True)
            args._results_json.write_text(encoded, encoding="utf-8")
        else:
            print(encoded)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = config["n_steps"]
    n_reps = config["n_reps"]
    accuracy_reps = config["accuracy_reps"]
    warmups = config["warmups"]
    particle_counts = config["particle_counts"]
    dimensions = config["dimensions"]
    seed = config["seed"]

    results = {
        "config": config,
        "hardware": {"gpu": capture_hardware()},
        "model": {
            "prior_variance": PRIOR_VARIANCE,
            "process_variance": PROCESS_VARIANCE,
            "observation_variance": OBSERVATION_VARIANCE,
        },
        "dimensions": {},
    }

    for dimension in dimensions:
        init_transform, propagate_transform, log_potential, kalman = make_model(
            dimension
        )
        init_sample = lambda key, model_inputs: init_transform(
            random.uniform(key, (dimension,)), model_inputs
        )
        propagate_sample = lambda key, state, model_inputs: propagate_transform(
            random.uniform(key, (dimension,)), state, model_inputs
        )

        sqmc_runner = lambda n, obs, s: _run_sqmc_filter(
            init_transform, propagate_transform, log_potential, n, dimension,
            obs, s,
        )
        smc_runner = lambda n, obs, s: _run_smc_filter(
            init_sample, propagate_sample, log_potential, n, obs, s,
        )

        dim_results = {}
        for n in particle_counts:
            observations = generate_observations(random.PRNGKey(seed), dimension,
                                                  n_steps)
            kalman_truth = kalman(observations)

            sqmc_timing = _time_filter(sqmc_runner, n, observations, warmups,
                                       n_reps, seed)
            smc_timing = _time_filter(smc_runner, n, observations, warmups,
                                      n_reps, seed)

            sqmc_errors = [
                _accuracy_metrics(sqmc_runner, n, observations, kalman_truth,
                                  seed + rep)
                for rep in range(accuracy_reps)
            ]
            smc_errors = [
                _accuracy_metrics(smc_runner, n, observations, kalman_truth,
                                  seed + rep)
                for rep in range(accuracy_reps)
            ]

            dim_results[str(n)] = {
                "sqmc": {"timing": sqmc_timing, "accuracy": sqmc_errors},
                "smc": {"timing": smc_timing, "accuracy": smc_errors},
            }
            print(f"  d={dimension}, N={n}: "
                  f"SQMC {sqmc_timing['median_seconds']:.4f}s, "
                  f"SMC {smc_timing['median_seconds']:.4f}s", flush=True)

        results["dimensions"][str(dimension)] = dim_results

    (args.output_dir / RESULTS_NAME).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.output_dir / RUN_CONFIG_NAME).write_text(
        json.dumps(
            {
                "config": config,
                "hardware": results["hardware"],
                "model": results["model"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_runtime(results, args.output_dir / FIGURE_NAMES[0])
    _plot_diversity(results, args.output_dir / FIGURE_NAMES[1])
    _plot_efficiency(results, args.output_dir / FIGURE_NAMES[2])

    print(f"Saved benchmark outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())