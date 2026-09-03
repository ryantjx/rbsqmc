"""GPU statistical-efficiency benchmark for SMC and SQMC.

The timed path measures only the common filtering trajectory. Independent,
untimed diagnostic scans measure approximation error and particle diversity
against the exact Kalman filtering distribution for the same linear-Gaussian
model. CPU comparisons and Hilbert/QMC microbenchmarks intentionally live
outside this benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import t as student_t

from cuthbert.smc.particle_filter import build_filter as build_smc_filter
from cuthbertlib.resampling import systematic

from sqmc.sqmc.benchmark_sqmc import (
    DEFAULT_N_REPS,
    DEFAULT_N_STEPS,
    DEFAULT_SEED,
    SIGMA_0,
    SIGMA_X,
    SIGMA_Y,
    _unique_ancestor_fraction,
    _weighted_diagnostics,
    _weighted_mean,
    capture_hardware,
    generate_observations,
    make_sqmc_model,
    run_sqmc_diagnostics_jit,
    run_sqmc_jit,
    time_filter,
)


DEFAULT_ACCURACY_REPS = 8
DEFAULT_DIMENSIONS = [2, 5, 10, 30, 60]
DEFAULT_PARTICLE_COUNTS = [64, 256, 1024, 4096, 16384, 65536]
DEFAULT_WARMUPS = 2
DEFAULT_OUTPUT_DIR = "sqmc/sqmc/scripts/outputs/sqmc_gpu"
METHODS = ("smc", "sqmc")
METRICS = (
    "mean_nrmse",
    "variance_relative_rmse",
    "normalized_ess",
    "unique_ancestor_fraction",
)


def make_smc_model(dimension: int):
    """Return stochastic SMC transforms matching the SQMC model."""

    def init_sample(key, model_inputs):
        del model_inputs
        return SIGMA_0 * random.normal(key, (dimension,))

    def propagate_sample(key, state, model_inputs):
        del model_inputs
        return state + SIGMA_X * random.normal(key, (dimension,))

    def log_potential(state_prev, state, model_inputs):
        del state_prev
        y = model_inputs["y"]
        return (
            -0.5 * jnp.sum(((y - state) / SIGMA_Y) ** 2)
            - dimension * jnp.log(SIGMA_Y)
            - 0.5 * dimension * jnp.log(2.0 * jnp.pi)
        )

    return init_sample, propagate_sample, log_potential


def _initialize_smc_state(filter_, observations, n_particles, key, log_potential):
    """Initialize Cuthbert's state and apply the time-zero potential."""
    state = filter_.init_prepare({"y": observations[0]}, key=key)
    particles = state.particles
    initial_log_weights = jax.vmap(log_potential, (None, 0, None))(
        jnp.zeros_like(particles[0]), particles, {"y": observations[0]}
    )
    initial_lnc = jax.nn.logsumexp(initial_log_weights) - jnp.log(n_particles)
    return state._replace(
        log_weights=initial_log_weights.reshape(-1),
        log_normalizing_constant=initial_lnc,
    )


def _build_smc_jit(
    observations, n_particles, key, dimension, model, collect_diagnostics
):
    """Build the timed or diagnostic systematic-resampling SMC scan."""
    init_sample, propagate_sample, log_potential = model
    filter_ = build_smc_filter(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        resampling_fn=systematic.resampling,
    )
    init_key = random.fold_in(key, 0)
    init_state = _initialize_smc_state(
        filter_, observations, n_particles, init_key, log_potential
    )

    def summarize_state(state, ancestor_indices=None):
        if not collect_diagnostics:
            return _weighted_mean(
                state.particles, state.log_weights, n_particles
            )
        mean, variance, normalized_ess = _weighted_diagnostics(
            state.particles, state.log_weights, n_particles
        )
        ancestor_fraction = (
            jnp.asarray(jnp.nan, dtype=state.particles.dtype)
            if ancestor_indices is None
            else _unique_ancestor_fraction(ancestor_indices, n_particles)
        )
        return mean, variance, normalized_ess, ancestor_fraction

    init_summary = summarize_state(init_state)
    # Cuthbert consumes the key stored on the previous state for propagation,
    # then carries the prepared state's key forward.  Seed the initial state
    # with the first transition key and provide one look-ahead key per step so
    # that initialization and every transition use disjoint randomness.
    step_keys = random.split(random.fold_in(key, 17), observations.shape[0])
    init_state = init_state._replace(key=step_keys[0])
    scan_inputs = (
        jax.device_put(observations[1:]),
        jax.device_put(step_keys[1:]),
    )

    @jax.jit
    def scan_trajectory(state, inputs):
        def step(carry, step_inputs):
            y, step_key = step_inputs
            prepared = filter_.filter_prepare({"y": y}, key=step_key)
            new_state = filter_.filter_combine(carry, prepared)
            new_state = new_state._replace(log_weights=new_state.log_weights.reshape(-1))
            summary = summarize_state(new_state, new_state.ancestor_indices)
            return new_state, summary

        return jax.lax.scan(step, state, inputs)

    return scan_trajectory, init_state, scan_inputs, init_summary


def run_smc_jit(observations, n_particles, key, dimension, model):
    """Build the timed SMC scan, which emits only weighted means."""
    return _build_smc_jit(observations, n_particles, key, dimension, model, False)


def run_smc_diagnostics_jit(observations, n_particles, key, dimension, model):
    """Build the untimed SMC scan with posterior and diversity diagnostics."""
    return _build_smc_jit(observations, n_particles, key, dimension, model, True)


def kalman_filter_reference(observations, prior_variance=SIGMA_0**2):
    """Return exact filtering means and scalar marginal variances."""
    n_steps, dimension = observations.shape
    means = jnp.zeros((n_steps, dimension), dtype=observations.dtype)
    variances = jnp.zeros((n_steps,), dtype=observations.dtype)
    mean = jnp.zeros(dimension, dtype=observations.dtype)
    variance = jnp.asarray(prior_variance, dtype=observations.dtype)
    process_variance = jnp.asarray(SIGMA_X**2, dtype=observations.dtype)
    observation_variance = jnp.asarray(SIGMA_Y**2, dtype=observations.dtype)

    for step in range(n_steps):
        if step > 0:
            variance = variance + process_variance
        gain = variance / (variance + observation_variance)
        mean = mean + gain * (observations[step] - mean)
        variance = (1.0 - gain) * variance
        means = means.at[step].set(mean)
        variances = variances.at[step].set(variance)
    return means, variances


def _summary(values, *, nonnegative=False):
    """Return a mean, sample standard deviation, and Student-t 95% CI."""
    samples = np.asarray(values, dtype=float)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    critical = (
        float(student_t.ppf(0.975, samples.size - 1))
        if samples.size > 1
        else 0.0
    )
    half_width = critical * std / np.sqrt(samples.size)
    lower = mean - half_width
    if nonnegative:
        lower = max(0.0, lower)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(lower),
        "ci95_high": float(mean + half_width),
        "n": int(samples.size),
    }


def _replicate_keys(seed, dimension, replicate):
    """Return reproducible data, SMC, and SQMC keys for one paired replicate."""
    base = random.fold_in(random.PRNGKey(seed + replicate), dimension)
    return (
        random.fold_in(base, 101),
        random.fold_in(base, 202),
        random.fold_in(base, 303),
    )


def _diagnostic_once(
    run_fn, observations, n_particles, key, dimension, model, reference
):
    """Run an untimed diagnostic trajectory and compare it with Kalman."""
    runner, state, scan_inputs, initial = run_fn(
        observations, n_particles, key, dimension, model
    )
    final_state, trajectory = runner(state, scan_inputs)
    jax.block_until_ready((final_state, trajectory))
    means = jnp.concatenate((initial[0][None, :], trajectory[0]), axis=0)
    variances = jnp.concatenate((initial[1][None, :], trajectory[1]), axis=0)
    normalized_ess = jnp.concatenate((initial[2][None], trajectory[2]), axis=0)
    ancestor_fraction = trajectory[3]
    reference_mean, reference_variance = reference
    variance_scale = reference_variance[:, None]
    mean_nrmse = jnp.sqrt(jnp.mean((means - reference_mean) ** 2 / variance_scale))
    variance_relative_rmse = jnp.sqrt(
        jnp.mean(((variances - variance_scale) / variance_scale) ** 2)
    )
    return {
        "mean_nrmse": float(mean_nrmse),
        "variance_relative_rmse": float(variance_relative_rmse),
        "normalized_ess": float(jnp.mean(normalized_ess)),
        "unique_ancestor_fraction": float(jnp.mean(ancestor_fraction)),
    }


def _paired_diagnostics(n_particles, dimension, n_steps, seed, accuracy_reps):
    """Run paired SMC/SQMC replicates on identical simulated observations."""
    raw = {method: [] for method in METHODS}
    for replicate in range(accuracy_reps):
        data_key, smc_key, sqmc_key = _replicate_keys(seed, dimension, replicate)
        _, observations = generate_observations(data_key, n_steps, dimension)
        reference = kalman_filter_reference(observations)
        specifications = {
            "smc": (run_smc_diagnostics_jit, smc_key, make_smc_model),
            "sqmc": (run_sqmc_diagnostics_jit, sqmc_key, make_sqmc_model),
        }
        for method, (run_fn, method_key, model_factory) in specifications.items():
            metrics = _diagnostic_once(
                run_fn,
                observations,
                n_particles,
                method_key,
                dimension,
                model_factory(dimension),
                reference,
            )
            raw[method].append(
                {
                    "replicate": replicate,
                    "base_seed": seed,
                    "dimension": dimension,
                    "data_key": np.asarray(data_key).astype(int).tolist(),
                    "method_key": np.asarray(method_key).astype(int).tolist(),
                    **metrics,
                }
            )

    return {
        method: {
            "replicates": raw[method],
            "summary": {
                metric: _summary(
                    [entry[metric] for entry in raw[method]], nonnegative=True
                )
                for metric in METRICS
            },
        }
        for method in METHODS
    }


def _timing_entry(timing, method, dimension, n_particles, seed, n_reps):
    return {
        "method": method,
        "backend": "gpu",
        "label": f"{method.upper()}-GPU",
        "dimension": int(dimension),
        "n_particles": int(n_particles),
        "n_reps": int(n_reps),
        "seed": int(seed),
        "precision": "float64",
        "runtime": {
            key: timing[key]
            for key in (
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
                "median",
                "q25",
                "q75",
            )
        },
        "cold": {
            "end_to_end_time": timing["end_to_end_time"],
            "setup_time": timing["setup_time"],
            "compile_and_first_execution_time": timing["compile_time"],
        },
    }


def run_gpu(args):
    """Run every dimension on the already-selected GPU backend."""
    jax.config.update("jax_enable_x64", True)
    dimension_results = {}
    for dimension in args.dimensions:
        print(f"\nDimension d={dimension}", flush=True)
        data_key, _, _ = _replicate_keys(args.seed, dimension, 10_000)
        _, observations = generate_observations(data_key, args.n_steps, dimension)
        entries = {method: {} for method in METHODS}
        method_specs = {
            "smc": (run_smc_jit, make_smc_model),
            "sqmc": (run_sqmc_jit, make_sqmc_model),
        }
        for n_particles in args.particle_counts:
            for method, (run_fn, model_factory) in method_specs.items():
                method_key = random.fold_in(data_key, 401 if method == "smc" else 402)
                timing = time_filter(
                    run_fn,
                    observations,
                    n_particles,
                    method_key,
                    dimension,
                    model_factory(dimension),
                    n_reps=args.n_reps,
                    n_warmups=args.warmups,
                )
                entries[method][str(n_particles)] = _timing_entry(
                    timing,
                    method,
                    dimension,
                    n_particles,
                    args.seed,
                    args.n_reps,
                )
                print(
                    f"  {method.upper()} N={n_particles}: "
                    f"{timing['mean']:.5f}s "
                    f"[{timing['ci95_low']:.5f}, {timing['ci95_high']:.5f}]",
                    flush=True,
                )

            diagnostics = _paired_diagnostics(
                n_particles,
                dimension,
                args.n_steps,
                args.seed,
                args.accuracy_reps,
            )
            for method in METHODS:
                entries[method][str(n_particles)]["diagnostics"] = diagnostics[method]
        dimension_results[str(dimension)] = entries

    hardware = capture_hardware()
    hardware.update(
        {
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        }
    )
    return {"gpu": dimension_results}, hardware


def _runtime_interval_ratio(sqmc, smc):
    tiny = np.finfo(float).tiny
    return {
        "point": sqmc["mean"] / smc["mean"],
        "ci95_conservative_low": sqmc["ci95_low"] / max(smc["ci95_high"], tiny),
        "ci95_conservative_high": sqmc["ci95_high"] / max(smc["ci95_low"], tiny),
    }


def _comparison_status(summary, *, lower_is_better):
    if lower_is_better:
        if summary["ci95_high"] < 0.0:
            return "supported"
        if summary["ci95_low"] > 0.0:
            return "weakened"
    else:
        if summary["ci95_low"] > 0.0:
            return "supported"
        if summary["ci95_high"] < 0.0:
            return "weakened"
    return "inconclusive"


def _pareto_frontier(method_entries, metric):
    points = [
        {
            "n_particles": int(particle_count),
            "runtime": entry["runtime"]["mean"],
            "error": entry["diagnostics"]["summary"][metric]["mean"],
        }
        for particle_count, entry in method_entries.items()
    ]
    frontier = []
    for point in points:
        dominated = any(
            other["runtime"] <= point["runtime"]
            and other["error"] <= point["error"]
            and (
                other["runtime"] < point["runtime"]
                or other["error"] < point["error"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: point["runtime"])


def _combined_pareto_frontier(dimension_entries, metric):
    """Return the empirical frontier across both methods and its composition."""
    points = []
    for method in METHODS:
        for particle_count, entry in dimension_entries[method].items():
            points.append(
                {
                    "method": method,
                    "n_particles": int(particle_count),
                    "runtime": entry["runtime"]["mean"],
                    "error": entry["diagnostics"]["summary"][metric]["mean"],
                }
            )
    frontier = []
    for point in points:
        dominated = any(
            other["runtime"] <= point["runtime"]
            and other["error"] <= point["error"]
            and (
                other["runtime"] < point["runtime"]
                or other["error"] < point["error"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda point: point["runtime"])
    methods = {point["method"] for point in frontier}
    if methods == {"sqmc"}:
        status = "sqmc_only"
    elif methods == {"smc"}:
        status = "smc_only"
    else:
        status = "mixed"
    return {"points": frontier, "status": status}


def _matched_quality(dimension_entries):
    matches = {}
    smc_entries = dimension_entries["smc"]
    for sqmc_n, sqmc in dimension_entries["sqmc"].items():
        sqmc_mean = sqmc["diagnostics"]["summary"]["mean_nrmse"]
        sqmc_variance = sqmc["diagnostics"]["summary"]["variance_relative_rmse"]
        candidates = []
        for smc_n, smc in smc_entries.items():
            smc_mean = smc["diagnostics"]["summary"]["mean_nrmse"]
            smc_variance = smc["diagnostics"]["summary"]["variance_relative_rmse"]
            if (
                smc_mean["mean"] <= sqmc_mean["mean"]
                and smc_variance["mean"] <= sqmc_variance["mean"]
            ):
                candidates.append((smc_n, smc))
        if not candidates:
            matches[sqmc_n] = {"status": "inconclusive", "matched_smc_n": None}
            continue
        smc_n, smc = min(candidates, key=lambda item: item[1]["runtime"]["mean"])
        lower_particles = int(sqmc_n) < int(smc_n)
        faster_point = sqmc["runtime"]["mean"] < smc["runtime"]["mean"]
        faster_interval = (
            sqmc["runtime"]["ci95_high"] < smc["runtime"]["ci95_low"]
        )
        status = "inconclusive"
        if lower_particles and faster_point:
            status = "exploratory"
        elif not faster_point:
            status = "weakened"
        matches[sqmc_n] = {
            "status": status,
            "matched_smc_n": int(smc_n),
            "sqmc_n": int(sqmc_n),
            "particle_ratio_sqmc_over_smc": int(sqmc_n) / int(smc_n),
            "runtime_ratio_sqmc_over_smc": sqmc["runtime"]["mean"]
            / smc["runtime"]["mean"],
            "lower_particles": lower_particles,
            "faster_point_estimate": faster_point,
            "faster_with_timing_intervals": faster_interval,
        }
    return matches


def _strict_lower_particle_comparisons(dimension_entries):
    """Test every lower-N SQMC/higher-N SMC pair using paired error CIs."""
    comparisons = []
    for sqmc_n, sqmc in dimension_entries["sqmc"].items():
        for smc_n, smc in dimension_entries["smc"].items():
            if int(sqmc_n) >= int(smc_n):
                continue
            paired_errors = {}
            for metric in ("mean_nrmse", "variance_relative_rmse"):
                differences = [
                    sqmc_replicate[metric] - smc_replicate[metric]
                    for sqmc_replicate, smc_replicate in zip(
                        sqmc["diagnostics"]["replicates"],
                        smc["diagnostics"]["replicates"],
                    )
                ]
                paired_errors[metric] = _summary(differences)
                paired_errors[metric]["difference"] = "sqmc_minus_smc"
            faster_interval = (
                sqmc["runtime"]["ci95_high"] < smc["runtime"]["ci95_low"]
            )
            better_error_intervals = all(
                paired_errors[metric]["ci95_high"] < 0.0
                for metric in ("mean_nrmse", "variance_relative_rmse")
            )
            comparisons.append(
                {
                    "sqmc_n": int(sqmc_n),
                    "smc_n": int(smc_n),
                    "runtime_ratio_sqmc_over_smc": sqmc["runtime"]["mean"]
                    / smc["runtime"]["mean"],
                    "faster_with_timing_intervals": faster_interval,
                    "paired_error_differences": paired_errors,
                    "strictly_better_error_intervals": better_error_intervals,
                    "status": (
                        "supported"
                        if faster_interval and better_error_intervals
                        else "inconclusive"
                    ),
                }
            )
    return comparisons


def evaluate_claims(results, dimensions, particle_counts):
    """Evaluate same-N speed, paired diversity, and matched-quality efficiency."""
    gpu = results["gpu"]
    same_n_runtime = {}
    paired_diagnostics = {}
    pareto = {}
    matched_quality = {}
    strict_efficiency = {}
    statuses = {}
    for dimension in dimensions:
        dimension_key = str(dimension)
        entries = gpu[dimension_key]
        same_n_runtime[dimension_key] = {}
        paired_diagnostics[dimension_key] = {}
        for n_particles in particle_counts:
            particle_key = str(n_particles)
            smc = entries["smc"][particle_key]
            sqmc = entries["sqmc"][particle_key]
            ratio = _runtime_interval_ratio(sqmc["runtime"], smc["runtime"])
            if ratio["ci95_conservative_high"] < 1.0:
                speed_status = "sqmc_faster"
            elif ratio["ci95_conservative_low"] > 1.0:
                speed_status = "smc_faster"
            else:
                speed_status = "inconclusive"
            same_n_runtime[dimension_key][particle_key] = {
                **ratio,
                "status": speed_status,
            }

            smc_reps = smc["diagnostics"]["replicates"]
            sqmc_reps = sqmc["diagnostics"]["replicates"]
            comparisons = {}
            for metric in METRICS:
                differences = [
                    sqmc_rep[metric] - smc_rep[metric]
                    for smc_rep, sqmc_rep in zip(smc_reps, sqmc_reps)
                ]
                comparison = _summary(differences)
                comparison["difference"] = "sqmc_minus_smc"
                comparison["status"] = _comparison_status(
                    comparison,
                    lower_is_better=metric
                    in ("mean_nrmse", "variance_relative_rmse"),
                )
                comparisons[metric] = comparison
            coverage_supported = (
                comparisons["variance_relative_rmse"]["status"] == "supported"
            )
            ancestry_supported = any(
                comparisons[metric]["status"] == "supported"
                for metric in ("normalized_ess", "unique_ancestor_fraction")
            )
            comparisons["overall_diversity_status"] = (
                "supported"
                if coverage_supported and ancestry_supported
                else "inconclusive"
            )
            paired_diagnostics[dimension_key][particle_key] = comparisons

        pareto[dimension_key] = {
            "by_method": {
                method: {
                    metric: _pareto_frontier(entries[method], metric)
                    for metric in ("mean_nrmse", "variance_relative_rmse")
                }
                for method in METHODS
            },
            "combined": {
                metric: _combined_pareto_frontier(entries, metric)
                for metric in ("mean_nrmse", "variance_relative_rmse")
            },
        }
        matched_quality[dimension_key] = _matched_quality(entries)
        strict_efficiency[dimension_key] = _strict_lower_particle_comparisons(
            entries
        )
        match_statuses = {
            match["status"] for match in matched_quality[dimension_key].values()
        }
        if any(
            comparison["status"] == "supported"
            for comparison in strict_efficiency[dimension_key]
        ):
            statuses[dimension_key] = "supported"
        elif "exploratory" in match_statuses:
            statuses[dimension_key] = "exploratory"
        elif match_statuses == {"weakened"}:
            statuses[dimension_key] = "weakened"
        else:
            statuses[dimension_key] = "inconclusive"

    return {
        "same_n_runtime_ratio_sqmc_over_smc": same_n_runtime,
        "paired_diagnostic_differences": paired_diagnostics,
        "pareto_frontiers": pareto,
        "matched_quality": matched_quality,
        "strict_lower_particle_efficiency": strict_efficiency,
        "matched_quality_status_by_dimension": statuses,
        "uncertainty_notes": {
            "timing": "95% Student-t CIs over steady-state timing repetitions.",
            "diagnostics": "95% Student-t CIs over paired replicate differences.",
            "matched_quality": "Matched-quality mappings use point estimates and are exploratory.",
            "strict_efficiency": "Support requires lower N, non-overlapping runtime intervals, and paired mean- and variance-error CIs strictly below zero.",
        },
    }


def _format_n(n_particles):
    return f"{n_particles // 1024}k" if n_particles >= 1024 else str(n_particles)


def plot_runtime(results, dimensions, particle_counts, output_dir):
    """Plot SMC-GPU and SQMC-GPU runtime for every dimension."""
    figure, axes = plt.subplots(1, len(dimensions), figsize=(19, 4.3), squeeze=False)
    styles = {"smc": ("C0", "s"), "sqmc": ("C3", "o")}
    for column, dimension in enumerate(dimensions):
        axis = axes[0, column]
        entries = results["gpu"][str(dimension)]
        for method in METHODS:
            runtime = [entries[method][str(n)]["runtime"] for n in particle_counts]
            means = np.asarray([value["mean"] for value in runtime])
            low = np.asarray([value["ci95_low"] for value in runtime])
            high = np.asarray([value["ci95_high"] for value in runtime])
            color, marker = styles[method]
            axis.errorbar(
                particle_counts,
                means,
                yerr=np.vstack((means - low, high - means)),
                color=color,
                marker=marker,
                capsize=2,
                label=f"{method.upper()}-GPU",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"d={dimension}")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel("Particles N")
        if column == 0:
            axis.set_ylabel("Steady-state runtime (s)")
            axis.legend(frameon=False)
    figure.suptitle("SMC-GPU and SQMC-GPU runtime (mean ± 95% CI)")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sqmc_smc_gpu_runtime.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _centered_norm(values, center=0.0):
    limit = max(float(np.nanmax(np.abs(values - center))), 1e-12)
    return TwoSlopeNorm(
        vmin=min(center - limit, center - 1e-12),
        vcenter=center,
        vmax=max(center + limit, center + 1e-12),
    )


def plot_diversity(results, dimensions, particle_counts, output_dir):
    """Plot relative posterior-spread, ESS, and ancestry comparisons."""
    gpu = results["gpu"]
    variance_ratio = np.empty((len(dimensions), len(particle_counts)))
    ess_difference = np.empty_like(variance_ratio)
    ancestor_difference = np.empty_like(variance_ratio)
    for row, dimension in enumerate(dimensions):
        entries = gpu[str(dimension)]
        for column, n_particles in enumerate(particle_counts):
            smc = entries["smc"][str(n_particles)]["diagnostics"]["summary"]
            sqmc = entries["sqmc"][str(n_particles)]["diagnostics"]["summary"]
            variance_ratio[row, column] = (
                sqmc["variance_relative_rmse"]["mean"]
                / max(
                    smc["variance_relative_rmse"]["mean"],
                    np.finfo(float).tiny,
                )
            )
            ess_difference[row, column] = (
                sqmc["normalized_ess"]["mean"] - smc["normalized_ess"]["mean"]
            )
            ancestor_difference[row, column] = (
                sqmc["unique_ancestor_fraction"]["mean"]
                - smc["unique_ancestor_fraction"]["mean"]
            )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    panels = (
        (
            variance_ratio,
            "SQMC/SMC posterior-variance-error ratio",
            "coolwarm",
            1.0,
        ),
        (ess_difference, "SQMC − SMC normalized ESS", "PiYG", 0.0),
        (
            ancestor_difference,
            "SQMC − SMC unique-ancestor fraction",
            "PiYG",
            0.0,
        ),
    )
    for axis, (values, title, cmap, center) in zip(axes, panels):
        image = axis.imshow(
            values,
            cmap=cmap,
            norm=_centered_norm(values, center=center),
            aspect="auto",
        )
        axis.set_xticks(
            range(len(particle_counts)), [_format_n(n) for n in particle_counts]
        )
        axis.set_yticks(range(len(dimensions)), [str(d) for d in dimensions])
        axis.set_xlabel("Particles N")
        axis.set_ylabel("Dimension d")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle("SQMC-GPU versus SMC-GPU particle diversity")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sqmc_smc_gpu_diversity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_efficiency(results, dimensions, particle_counts, output_dir):
    """Plot runtime-versus-error frontiers with particle-count annotations."""
    figure, axes = plt.subplots(
        2, len(dimensions), figsize=(19, 8.2), squeeze=False
    )
    styles = {"smc": ("C0", "s"), "sqmc": ("C3", "o")}
    rows = (
        ("mean_nrmse", "Normalized filtering-mean RMSE"),
        ("variance_relative_rmse", "Relative marginal-variance RMSE"),
    )
    for column, dimension in enumerate(dimensions):
        entries = results["gpu"][str(dimension)]
        for row, (metric, ylabel) in enumerate(rows):
            axis = axes[row, column]
            for method in METHODS:
                method_entries = entries[method]
                runtime = np.asarray(
                    [method_entries[str(n)]["runtime"]["mean"] for n in particle_counts]
                )
                runtime_low = np.asarray(
                    [
                        method_entries[str(n)]["runtime"]["ci95_low"]
                        for n in particle_counts
                    ]
                )
                runtime_high = np.asarray(
                    [
                        method_entries[str(n)]["runtime"]["ci95_high"]
                        for n in particle_counts
                    ]
                )
                runtime_low = np.maximum(runtime_low, np.finfo(float).tiny)
                diagnostic_summaries = [
                    method_entries[str(n)]["diagnostics"]["summary"][metric]
                    for n in particle_counts
                ]
                error = np.asarray(
                    [diagnostic["mean"] for diagnostic in diagnostic_summaries]
                )
                error_low = np.asarray(
                    [diagnostic["ci95_low"] for diagnostic in diagnostic_summaries]
                )
                error_high = np.asarray(
                    [diagnostic["ci95_high"] for diagnostic in diagnostic_summaries]
                )
                error_low = np.maximum(error_low, np.finfo(float).tiny)
                color, marker = styles[method]
                axis.errorbar(
                    runtime,
                    error,
                    xerr=np.vstack((runtime - runtime_low, runtime_high - runtime)),
                    yerr=np.vstack((error - error_low, error_high - error)),
                    color=color,
                    marker=marker,
                    capsize=2,
                    label=method.upper(),
                )
                for x_value, y_value, n_particles in zip(
                    runtime, error, particle_counts
                ):
                    axis.annotate(
                        _format_n(n_particles),
                        (x_value, y_value),
                        fontsize=6,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            if row == 0:
                axis.set_title(f"d={dimension}")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 1:
                axis.set_xlabel("Steady-state runtime (s)")
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    figure.suptitle("GPU time-to-posterior-accuracy frontiers")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sqmc_smc_gpu_efficiency.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def parser():
    argument_parser = argparse.ArgumentParser(
        description="Compare SMC and SQMC statistical efficiency on GPU."
    )
    argument_parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    argument_parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    argument_parser.add_argument(
        "--accuracy-reps", type=int, default=DEFAULT_ACCURACY_REPS
    )
    argument_parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    argument_parser.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=DEFAULT_PARTICLE_COUNTS,
    )
    argument_parser.add_argument(
        "--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    argument_parser.add_argument(
        "--platforms", nargs="+", choices=["gpu"], default=["gpu"]
    )
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR)
    )
    argument_parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument(
        "--_results-json", type=Path, default=None, help=argparse.SUPPRESS
    )
    return argument_parser


def _validate(args):
    if args.n_steps <= 0 or args.n_reps <= 0 or args.accuracy_reps <= 0:
        raise ValueError("n-steps, n-reps, and accuracy-reps must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not args.dimensions or any(d <= 0 or d > 62 for d in args.dimensions):
        raise ValueError("dimensions must be in [1, 62]")
    if len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("dimensions must not contain duplicates")
    if not args.particle_counts or any(n <= 1 for n in args.particle_counts):
        raise ValueError("particle counts must be greater than one")
    if args.platforms != ["gpu"]:
        raise ValueError("this benchmark supports only the gpu platform")


def main(argv=None):
    args = parser().parse_args(argv)
    _validate(args)
    if args._child:
        results, hardware = run_gpu(args)
        payload = json.dumps({"results": results, "hardware": hardware})
        if args._results_json:
            args._results_json.parent.mkdir(parents=True, exist_ok=True)
            args._results_json.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    child_path = args.output_dir / "_sqmc_smc_gpu.json"
    child_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["JAX_PLATFORM_NAME"] = "gpu"
    command = [
        sys.executable,
        "-m",
        "sqmc.sqmc.benchmark_sqmc_smc",
        "--n-steps",
        str(args.n_steps),
        "--n-reps",
        str(args.n_reps),
        "--accuracy-reps",
        str(args.accuracy_reps),
        "--warmups",
        str(args.warmups),
        "--particle-counts",
        *(str(n) for n in args.particle_counts),
        "--dimensions",
        *(str(d) for d in args.dimensions),
        "--seed",
        str(args.seed),
        "--platforms",
        "gpu",
        "--_child",
        "--_results-json",
        str(child_path),
    ]
    print("Running paired SMC/SQMC benchmark on GPU...", flush=True)
    subprocess.run(command, env=environment, check=True)
    child_payload = json.loads(child_path.read_text(encoding="utf-8"))
    results = child_payload["results"]
    hardware = child_payload["hardware"]
    claims = evaluate_claims(results, args.dimensions, args.particle_counts)

    output_paths = [
        plot_runtime(results, args.dimensions, args.particle_counts, args.output_dir),
        plot_diversity(results, args.dimensions, args.particle_counts, args.output_dir),
        plot_efficiency(results, args.dimensions, args.particle_counts, args.output_dir),
    ]
    config = {
        "platforms": ["gpu"],
        "dimensions": args.dimensions,
        "particle_counts": args.particle_counts,
        "n_steps": args.n_steps,
        "n_reps": args.n_reps,
        "accuracy_reps": args.accuracy_reps,
        "warmups": args.warmups,
        "precision": "float64",
        "seed": args.seed,
    }
    model = {
        "prior_variance": SIGMA_0**2,
        "process_variance": SIGMA_X**2,
        "observation_variance": SIGMA_Y**2,
    }
    hilbert_bits = {
        str(dimension): 62 // dimension for dimension in args.dimensions
    }
    payload = {
        "gpu": results["gpu"],
        "config": config,
        "model": model,
        "hilbert_bits_per_dimension": hilbert_bits,
        "hardware": {"gpu": hardware},
        "claims_evaluation": claims,
    }
    results_path = args.output_dir / "sqmc_smc_gpu_results.json"
    claims_path = args.output_dir / "claims_evaluation.json"
    run_config_path = args.output_dir / "run_config.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(
            {
                "config": config,
                "model": model,
                "hilbert_bits_per_dimension": hilbert_bits,
                "hardware": payload["hardware"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    child_path.unlink(missing_ok=True)
    for path in output_paths:
        print(f"Figure saved to {path}")
    print(f"Results saved to {results_path}")
    print(f"Claims evaluation saved to {claims_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
