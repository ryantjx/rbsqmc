"""Four-method SQMC/SMC performance benchmark.

This module compares the same synthetic filtering problem using SMC and SQMC
on the CPU and GPU backends.  SMC is implemented locally with Cuthbert's
particle-filter API; ``sqmc/sqmc/sqmc.py`` is intentionally not modified.

The benchmark reports steady-state runtime scaling, cold end-to-end timing,
and filtering accuracy against the exact Kalman filtering mean.  The Hilbert
sort microbenchmark is kept separate because isolated kernel timings are not
additive to the fused filtering trajectory.
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
import numpy as np
from scipy.stats import t as student_t

from cuthbert.smc.particle_filter import build_filter as build_smc_filter
from cuthbertlib.resampling import systematic

from sqmc.sqmc.benchmark_sqmc import (
    DEFAULT_BASE_DIMENSION,
    DEFAULT_N_REPS,
    DEFAULT_N_STEPS,
    DEFAULT_PARTICLE_COUNTS,
    DEFAULT_SEED,
    SIGMA_X,
    SIGMA_Y,
    capture_hardware,
    generate_observations,
    make_sqmc_model,
    rmse,
    run_sqmc_jit,
    time_filter,
)


DEFAULT_ACCURACY_REPS = 8
PRIOR_VARIANCE = 1.0
DEFAULT_OUTPUT_DIR = "sqmc/sqmc/scripts/outputs/sqmc_gpu"
METHODS = ("smc", "sqmc")
PLATFORMS = ("gpu", "cpu")


def make_smc_model(dimension: int):
    """Return stochastic SMC transforms matching the SQMC model."""

    def init_sample(key, model_inputs):
        return random.normal(key, (dimension,))

    def propagate_sample(key, state, model_inputs):
        return state + SIGMA_X * random.normal(key, (dimension,))

    def log_potential(state_prev, state, model_inputs):
        y = model_inputs["y"]
        return (
            -0.5 * jnp.sum(((y - state) / SIGMA_Y) ** 2)
            - dimension * jnp.log(SIGMA_Y)
            - 0.5 * dimension * jnp.log(2.0 * jnp.pi)
        )

    return init_sample, propagate_sample, log_potential


def _weighted_mean(state, n_particles: int):
    """Compute a filtering expectation from normalized log weights."""
    particles = state.particles
    log_weights = state.log_weights.reshape(-1)
    weights = jax.nn.softmax(log_weights)
    return jnp.sum(particles * weights.reshape((n_particles,) + (1,) * (particles.ndim - 1)), axis=0)


def _initialize_smc_state(filter_, observations, n_particles: int, key, log_potential):
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


def run_smc_jit(observations, n_particles, key, dimension, model):
    """Build a fully jitted SMC trajectory using systematic resampling."""
    init_sample, propagate_sample, log_potential = model
    filter_ = build_smc_filter(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        resampling_fn=systematic.resampling,
    )
    init_state = _initialize_smc_state(
        filter_, observations, n_particles, key, log_potential
    )
    init_mean = _weighted_mean(init_state, n_particles)

    step_keys = random.split(random.fold_in(key, 17), max(observations.shape[0] - 1, 1))
    obs_device = jax.device_put(observations[1:])
    keys_device = jax.device_put(step_keys[: observations.shape[0] - 1])

    @jax.jit
    def scan_trajectory(state, scan_inputs):
        def step(carry, inputs):
            y, step_key = inputs
            prepared = filter_.filter_prepare({"y": y}, key=step_key)
            new_state = filter_.filter_combine(carry, prepared)
            new_state = new_state._replace(
                log_weights=new_state.log_weights.reshape(-1)
            )
            return new_state, _weighted_mean(new_state, n_particles)

        return jax.lax.scan(step, state, scan_inputs)

    return scan_trajectory, init_state, (obs_device, keys_device), init_mean


def kalman_filter_mean(observations, prior_variance: float = PRIOR_VARIANCE):
    """Exact filtering means for the independent linear-Gaussian model."""
    n_steps, dimension = observations.shape
    means = jnp.zeros((n_steps, dimension), dtype=observations.dtype)
    mean = jnp.zeros(dimension, dtype=observations.dtype)
    variance = jnp.asarray(prior_variance, dtype=observations.dtype)
    process_variance = jnp.asarray(SIGMA_X**2, dtype=observations.dtype)
    observation_variance = jnp.asarray(SIGMA_Y**2, dtype=observations.dtype)

    for t in range(n_steps):
        if t > 0:
            variance = variance + process_variance
        gain = variance / (variance + observation_variance)
        mean = mean + gain * (observations[t] - mean)
        variance = (1.0 - gain) * variance
        means = means.at[t].set(mean)
    return means


def _ci95(values):
    """Return mean, standard deviation, and a Student-t 95% CI."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    critical = float(student_t.ppf(0.975, values.size - 1)) if values.size > 1 else 0.0
    half = critical * std / np.sqrt(values.size)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": max(0.0, mean - half),
        "ci95_high": mean + half,
        "n": int(values.size),
    }


def _run_once(run_fn, observations, n_particles, key, dimension, model, reference):
    """Execute one method once and return its weighted estimates and RMSE."""
    runner, state, scan_inputs, init_mean = run_fn(
        observations, n_particles, key, dimension, model
    )
    final_state, trajectory_means = runner(state, scan_inputs)
    jax.block_until_ready((final_state, trajectory_means))
    means = jnp.concatenate([jnp.asarray([init_mean]), trajectory_means])
    return means, rmse(means, reference)


def _accuracy_replicates(
    run_fn,
    model_factory,
    n_particles,
    dimension,
    n_steps,
    seed,
    accuracy_reps,
):
    """Aggregate RMSE over independent data and method randomization seeds."""
    values = []
    for replicate in range(accuracy_reps):
        replicate_key = random.PRNGKey(seed + 1000 + replicate)
        _, observations = generate_observations(replicate_key, n_steps, dimension)
        reference = kalman_filter_mean(observations)
        method_key = random.fold_in(replicate_key, 7001)
        _, value = _run_once(
            run_fn,
            observations,
            n_particles,
            method_key,
            dimension,
            model_factory(dimension),
            reference,
        )
        values.append(value)
    return _ci95(values)


def _method_sweep(
    method,
    observations,
    dimension,
    particle_counts,
    n_reps,
    n_warmups,
    seed,
    n_steps,
    accuracy_reps,
    latent_reference=None,
):
    run_fn = run_smc_jit if method == "smc" else run_sqmc_jit
    model_factory = make_smc_model if method == "smc" else make_sqmc_model
    model = model_factory(dimension)
    reference = kalman_filter_mean(observations)
    sweep = {}
    for n in particle_counts:
        timing = time_filter(
            run_fn,
            observations,
            n,
            random.PRNGKey(seed + 11),
            dimension,
            model,
            n_reps=n_reps,
            n_warmups=n_warmups,
        )
        sweep[str(n)] = {
            "method": method,
            "backend": os.environ.get("JAX_PLATFORM_NAME", "unknown"),
            "label": f"{method.upper()}-{os.environ.get('JAX_PLATFORM_NAME', 'unknown').upper()}",
            "n_particles": int(n),
            "dimension": int(dimension),
            "n_steps": int(n_steps),
            "precision": "float64",
            "prior_variance": PRIOR_VARIANCE,
            "process_variance": SIGMA_X**2,
            "observation_variance": SIGMA_Y**2,
            "seed": int(seed),
            "accuracy_seeds": [int(seed + 1000 + r) for r in range(accuracy_reps)],
            "accuracy_reps": int(accuracy_reps),
            "median": timing["median"],
            "mean": timing["mean"],
            "std": timing["std"],
            "q25": timing["q25"],
            "q75": timing["q75"],
            "ci95_low": timing["ci95_low"],
            "ci95_high": timing["ci95_high"],
            "end_to_end_time": timing["end_to_end_time"],
            "cold_end_to_end_time": timing["end_to_end_time"],
            "setup_time": timing["setup_time"],
            "compile_time": timing["compile_time"],
            "compile_and_first_execution_time": timing["compile_time"],
            "rmse_single_seed": rmse(timing["means"], reference),
            "latent_rmse_single_seed": (
                rmse(timing["means"], latent_reference)
                if latent_reference is not None
                else None
            ),
            "accuracy": _accuracy_replicates(
                run_fn,
                model_factory,
                n,
                dimension,
                n_steps,
                seed,
                accuracy_reps,
            ),
        }
        entry = sweep[str(n)]
        print(
            f"  [{method.upper()}] d={dimension} N={n}: "
            f"{entry['mean']:.4f}s (95% CI "
            f"{entry['ci95_low']:.4f}-{entry['ci95_high']:.4f}), "
            f"cold {entry['end_to_end_time']:.4f}s, "
            f"RMSE {entry['accuracy']['mean']:.4f}",
            flush=True,
        )
    return sweep


def _plot_runtime(results, particle_counts, output_dir: Path):
    fig, axis = plt.subplots(figsize=(9, 6))
    styles = {
        "smc-cpu": ("C0", "s", "--"),
        "smc-gpu": ("C1", "s", "-"),
        "sqmc-cpu": ("C2", "o", "--"),
        "sqmc-gpu": ("C3", "o", "-"),
    }
    ns = [int(n) for n in particle_counts]
    for label, (method, backend) in {
        "SMC-CPU": ("smc", "cpu"),
        "SMC-GPU": ("smc", "gpu"),
        "SQMC-CPU": ("sqmc", "cpu"),
        "SQMC-GPU": ("sqmc", "gpu"),
    }.items():
        if backend not in results or method not in results[backend]:
            continue
        values = results[backend][method]
        mean = np.asarray([values[str(n)]["mean"] for n in ns])
        low = np.maximum(
            np.asarray([values[str(n)]["ci95_low"] for n in ns]),
            np.finfo(float).tiny,
        )
        high = np.asarray([values[str(n)]["ci95_high"] for n in ns])
        colour, marker, line = styles[label.lower()]
        axis.errorbar(
            ns,
            mean,
            yerr=np.vstack((np.maximum(mean - low, 0), np.maximum(high - mean, 0))),
            color=colour,
            marker=marker,
            linestyle=line,
            capsize=3,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of particles $N$")
    axis.set_ylabel("Steady-state wall-clock time (s)")
    axis.set_title("SMC and SQMC runtime scaling (mean ± 95% CI)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=False, ncol=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sqmc_smc_runtime.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_time_to_accuracy(results, particle_counts, output_dir: Path):
    fig, axis = plt.subplots(figsize=(9, 6))
    styles = {
        "smc-cpu": ("C0", "s", "--"),
        "smc-gpu": ("C1", "s", "-"),
        "sqmc-cpu": ("C2", "o", "--"),
        "sqmc-gpu": ("C3", "o", "-"),
    }
    ns = [int(n) for n in particle_counts]
    for label, (method, backend) in {
        "SMC-CPU": ("smc", "cpu"),
        "SMC-GPU": ("smc", "gpu"),
        "SQMC-CPU": ("sqmc", "cpu"),
        "SQMC-GPU": ("sqmc", "gpu"),
    }.items():
        if backend not in results or method not in results[backend]:
            continue
        values = results[backend][method]
        runtime = np.asarray([values[str(n)]["mean"] for n in ns])
        runtime_low = np.asarray([values[str(n)]["ci95_low"] for n in ns])
        runtime_high = np.asarray([values[str(n)]["ci95_high"] for n in ns])
        accuracy = np.asarray([values[str(n)]["accuracy"]["mean"] for n in ns])
        accuracy_low = np.maximum(
            np.asarray([values[str(n)]["accuracy"]["ci95_low"] for n in ns]),
            np.finfo(float).tiny,
        )
        accuracy_high = np.asarray([values[str(n)]["accuracy"]["ci95_high"] for n in ns])
        colour, marker, line = styles[label.lower()]
        axis.errorbar(
            runtime,
            accuracy,
            xerr=np.vstack((np.maximum(runtime - runtime_low, 0), np.maximum(runtime_high - runtime, 0))),
            yerr=np.vstack((np.maximum(accuracy - accuracy_low, 0), np.maximum(accuracy_high - accuracy, 0))),
            color=colour,
            marker=marker,
            linestyle=line,
            capsize=3,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Steady-state wall-clock time (s)")
    axis.set_ylabel("RMSE versus Kalman filtering mean")
    axis.set_title("Time-to-accuracy (independent accuracy replicates)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=False, ncol=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sqmc_smc_time_to_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _ratio_interval(cpu, gpu):
    """Conservative interval for CPU/GPU speedup from marginal CIs."""
    ratio_low = max(float(cpu["ci95_low"]), 0.0) / max(
        float(gpu["ci95_high"]), np.finfo(float).tiny
    )
    ratio_high = max(float(cpu["ci95_high"]), 0.0) / max(
        float(gpu["ci95_low"]), np.finfo(float).tiny
    )
    return ratio_low, ratio_high


def _time_to_threshold(values, threshold):
    """Return the fastest observed point whose accuracy meets ``threshold``."""
    candidates = [
        entry for entry in values.values() if entry["accuracy"]["mean"] <= threshold
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: entry["mean"])


def evaluate_claims(results, particle_counts):
    """Create a machine-readable, conservative claims assessment."""
    ns = [int(n) for n in particle_counts]
    speedups = {}
    overheads = {}
    slopes = {}
    crossover_points = {}
    for method in METHODS:
        if "cpu" in results and "gpu" in results:
            speedups[method] = {}
            for n in ns:
                cpu = results["cpu"][method][str(n)]
                gpu = results["gpu"][method][str(n)]
                low, high = _ratio_interval(cpu, gpu)
                speedups[method][str(n)] = {
                    "point": cpu["mean"] / gpu["mean"],
                    "ci95_conservative_low": low,
                    "ci95_conservative_high": high,
                }
            faster = [n for n in ns if speedups[method][str(n)]["point"] > 1.0]
            crossover_points[f"gpu_acceleration_{method}"] = {
                "first_observed_gpu_faster": faster[0] if faster else None,
                "speedup_threshold": 1.0,
            }
        for backend in PLATFORMS:
            if backend in results:
                values = results[backend][method]
                if len(ns) >= 2:
                    x = np.log(np.asarray(ns, dtype=float))
                    y = np.log(np.asarray([values[str(n)]["mean"] for n in ns]))
                    slopes[f"{method}-{backend}"] = float(np.polyfit(x, y, 1)[0])
        if method == "sqmc":
            for backend in PLATFORMS:
                if backend in results and "smc" in results[backend]:
                    overheads[f"sqmc-{backend}"] = {
                        str(n): results[backend]["sqmc"][str(n)]["mean"]
                        / results[backend]["smc"][str(n)]["mean"]
                        for n in ns
                    }
                    faster = [
                        n for n in ns if overheads[f"sqmc-{backend}"][str(n)] < 1.0
                    ]
                    crossover_points[f"sqmc_runtime_{backend}"] = {
                        "first_observed_sqmc_faster": faster[0] if faster else None,
                        "runtime_ratio_threshold": 1.0,
                    }

    time_to_accuracy = {}
    time_to_accuracy_status = {}
    for backend in PLATFORMS:
        if backend not in results or not all(m in results[backend] for m in METHODS):
            continue
        smc_values = results[backend]["smc"]
        sqmc_values = results[backend]["sqmc"]
        # Use the least stringent RMSE that both methods can attain, so neither
        # method is declared a winner merely because the other cannot reach an
        # arbitrarily demanding threshold.
        threshold = max(
            min(e["accuracy"]["mean"] for e in smc_values.values()),
            min(e["accuracy"]["mean"] for e in sqmc_values.values()),
        )
        smc_point = _time_to_threshold(smc_values, threshold)
        sqmc_point = _time_to_threshold(sqmc_values, threshold)
        record = {"target_rmse": float(threshold)}
        if smc_point is not None and sqmc_point is not None:
            record.update(
                {
                    "smc_time": float(smc_point["mean"]),
                    "sqmc_time": float(sqmc_point["mean"]),
                    "sqmc_over_smc": float(sqmc_point["mean"] / smc_point["mean"]),
                    "smc_particle_count": smc_point["n_particles"],
                    "sqmc_particle_count": sqmc_point["n_particles"],
                    "smc_time_ci95": [smc_point["ci95_low"], smc_point["ci95_high"]],
                    "sqmc_time_ci95": [sqmc_point["ci95_low"], sqmc_point["ci95_high"]],
                }
            )
            if sqmc_point["ci95_high"] < smc_point["ci95_low"]:
                time_to_accuracy_status[backend] = "supported"
            elif smc_point["ci95_high"] < sqmc_point["ci95_low"]:
                time_to_accuracy_status[backend] = "weakened"
            else:
                time_to_accuracy_status[backend] = "inconclusive"
        else:
            time_to_accuracy_status[backend] = "not_evaluated"
        time_to_accuracy[backend] = record

    statuses = {}
    for method in METHODS:
        values = speedups.get(method, {})
        largest = values.get(str(ns[-1]))
        if largest is None:
            statuses[f"gpu_acceleration_{method}"] = "not_evaluated"
        elif largest["ci95_conservative_low"] > 1.0:
            statuses[f"gpu_acceleration_{method}"] = "supported"
        elif largest["ci95_conservative_high"] < 1.0:
            statuses[f"gpu_acceleration_{method}"] = "weakened"
        else:
            statuses[f"gpu_acceleration_{method}"] = "inconclusive"
    statuses["hilbert_sort_bottleneck"] = "inconclusive"
    statuses["sqmc_time_to_accuracy"] = time_to_accuracy_status or "not_evaluated"

    return {
        "speedup_cpu_over_gpu": speedups,
        "sqmc_over_smc_runtime_ratio": overheads,
        "particle_count_crossover_points": crossover_points,
        "log_log_runtime_slopes": slopes,
        "time_to_accuracy": time_to_accuracy,
        "statuses": statuses,
        "delegated_claims": {
            "hilbert_sort_bottleneck": {
                "status": "inconclusive",
                "source": "sqmc/hilbert_sort/benchmark_hilbert_sort.py",
                "reason": "The fused trajectory benchmark does not isolate Hilbert-sort kernels.",
            }
        },
        "uncertainty_notes": {
            "timing": "95% Student-t CIs over steady-state timing repetitions; not estimator uncertainty.",
            "accuracy": "95% Student-t CIs over independent data/randomization seeds.",
        },
        "interpretation": (
            "GPU acceleration and SQMC time-to-accuracy claims are assessed from end-to-end measurements. "
            "Hilbert-sort dominance is not inferred from this benchmark."
        ),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark SMC and SQMC on CPU and GPU.")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    p.add_argument("--accuracy-reps", type=int, default=DEFAULT_ACCURACY_REPS)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--particle-counts", type=int, nargs="+", default=DEFAULT_PARTICLE_COUNTS)
    p.add_argument("--base-dimension", type=int, default=DEFAULT_BASE_DIMENSION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--platforms", nargs="+", choices=list(PLATFORMS), default=list(PLATFORMS))
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_results-json", type=Path, default=None, help=argparse.SUPPRESS)
    return p


def _validate(args):
    if args.n_steps <= 0 or args.n_reps <= 0 or args.accuracy_reps <= 0:
        raise ValueError("n-steps, n-reps, and accuracy-reps must be positive")
    if args.warmups < 0 or args.base_dimension <= 0:
        raise ValueError("warmups must be non-negative and dimension positive")
    if not args.particle_counts or any(n <= 0 for n in args.particle_counts):
        raise ValueError("particle counts must be positive")


def run_platform(platform_name, args):
    jax.config.update("jax_enable_x64", True)
    data_key = random.PRNGKey(args.seed)
    x_true, observations = generate_observations(
        data_key, args.n_steps, args.base_dimension
    )
    results = {}
    for method in METHODS:
        results[method] = _method_sweep(
            method,
            observations,
            args.base_dimension,
            args.particle_counts,
            args.n_reps,
            args.warmups,
            args.seed,
            args.n_steps,
            args.accuracy_reps,
            x_true,
        )
    hardware = capture_hardware()
    hardware.update(
        {
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        }
    )
    return {
        "platform": platform_name,
        "n_steps": args.n_steps,
        "base_dimension": args.base_dimension,
        "particle_counts": args.particle_counts,
        "n_reps": args.n_reps,
        "accuracy_reps": args.accuracy_reps,
        "warmups": args.warmups,
        "jit": True,
        "methods": results,
        "hardware": hardware,
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    if args._child:
        result = run_platform(args.platforms[0], args)
        payload = json.dumps(result)
        if args._results_json:
            args._results_json.parent.mkdir(parents=True, exist_ok=True)
            args._results_json.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    results = {}
    hardware = {}
    child_files = []
    for platform_name in args.platforms:
        child_path = args.output_dir / f"_sqmc_smc_{platform_name}.json"
        child_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["JAX_PLATFORM_NAME"] = platform_name
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
            "--base-dimension",
            str(args.base_dimension),
            "--seed",
            str(args.seed),
            "--platforms",
            platform_name,
            "--_child",
            "--_results-json",
            str(child_path),
        ]
        print(f"Running SMC/SQMC benchmark on {platform_name}...", flush=True)
        subprocess.run(command, env=env, check=True)
        child_payload = json.loads(child_path.read_text(encoding="utf-8"))
        results[platform_name] = child_payload["methods"]
        hardware[platform_name] = child_payload["hardware"]
        child_files.append(child_path)

    runtime_path = _plot_runtime(results, args.particle_counts, args.output_dir)
    accuracy_path = _plot_time_to_accuracy(results, args.particle_counts, args.output_dir)
    claims = evaluate_claims(results, args.particle_counts)
    merged = {
        "platforms": args.platforms,
        "n_steps": args.n_steps,
        "n_reps": args.n_reps,
        "accuracy_reps": args.accuracy_reps,
        "warmups": args.warmups,
        "particle_counts": args.particle_counts,
        "base_dimension": args.base_dimension,
        "precision": "float64",
        "model": {
            "prior_variance": PRIOR_VARIANCE,
            "process_variance": SIGMA_X**2,
            "observation_variance": SIGMA_Y**2,
        },
        "jit": True,
        "results": results,
        "hardware": hardware,
        "claims_evaluation": claims,
        "uncertainty_notes": {
            "timing": "95% Student-t CIs over steady-state timing repetitions; not estimator uncertainty.",
            "accuracy": "95% Student-t CIs over independent data/randomization seeds.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sqmc_smc_results.json"
    results_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    claims_path = args.output_dir / "claims_evaluation.json"
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    run_config = {
        "config": {
            "n_steps": args.n_steps,
            "n_reps": args.n_reps,
            "accuracy_reps": args.accuracy_reps,
            "warmups": args.warmups,
            "particle_counts": args.particle_counts,
            "base_dimension": args.base_dimension,
            "seed": args.seed,
            "platforms": args.platforms,
            "precision": "float64",
            "jit": True,
        },
        "hardware": {
            platform_name: json.loads(path.read_text(encoding="utf-8"))["hardware"]
            for platform_name, path in zip(args.platforms, child_files)
        },
    }
    run_config_path = args.output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    for path in child_files:
        path.unlink(missing_ok=True)
    print(f"Runtime chart saved to {runtime_path}")
    print(f"Time-to-accuracy chart saved to {accuracy_path}")
    print(f"Results saved to {results_path}")
    print(f"Claims evaluation saved to {claims_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
