"""Empirical evaluation: SQMC on GPU versus CPU.

The benchmark produces one runtime figure: mean steady-state wall-clock versus
particle count, with 95% confidence intervals for the GPU and CPU backends. It
also records cold end-to-end timings and machine metadata in JSON. QMC points
are generated inside the jitted trajectory from a pure time-indexed Sobol
kernel, so each filtering step receives a fresh scrambled block.

The GPU and CPU sweeps are run in separate subprocesses (one per platform)
because JAX caches its backend; the parent process merges the per-platform JSON
and renders the combined figure.

Usage:
    uv run -m sqmc.sqmc.benchmark_sqmc --platforms gpu cpu
    JAX_PLATFORM_NAME=cpu uv run -m sqmc.sqmc.benchmark_sqmc --platforms cpu \
        --_child --_results-json /tmp/sqmc_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
from jax import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from sqmc.sqmc.sqmc import resample_from_uniform
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort
from sqmc.qmc.qmc import Sobol, _MAXBITS, _sobol_sample_batched


# ---------------------------------------------------------------------------
# Model: d-dimensional random walk with Gaussian observation noise
#   x_t = x_{t-1} + sigma_x * Z_t,   Z_t ~ N(0, I_d)
#   y_t = x_t + sigma_y * E_t,       E_t ~ N(0, I_d)
# ---------------------------------------------------------------------------
SIGMA_X = 0.5
SIGMA_Y = 1.0
DEFAULT_N_STEPS = 100
DEFAULT_N_REPS = 7  # repetitions for steady-state timing
DEFAULT_N_WARMUPS = 1  # warmup runs (first captures JAX compile time)
DEFAULT_PARTICLE_COUNTS = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
]
DEFAULT_BASE_DIMENSION = 10
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Hardware spec capture
#
# Recorded per platform so the GPU-vs-CPU comparison is not confounded by a
# mismatch in machine (e.g. different RAM or CPU model) rather than by the
# algorithm being benchmarked.
# ---------------------------------------------------------------------------
def _memory_bytes():
    """Total physical RAM in bytes, via psutil or platform-specific fallback."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return int(out)
    except Exception:
        return None


def _gpu_devices() -> list:
    """List JAX GPU devices, tolerating builds without a GPU backend."""
    try:
        return [str(d) for d in jax.devices("gpu")]
    except Exception:
        return []


def _cpu_model() -> str | None:
    """Return the host CPU model when the platform exposes it."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _gpu_specs() -> list[dict]:
    """Return actual NVIDIA model/memory/driver metadata when available."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {"name": fields[0], "memory_mib": fields[1], "driver": fields[2]}
            )
    return devices


def capture_hardware() -> dict:
    """Return a dict describing the current host and its accelerators."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "node": platform.node(),
        "memory_bytes": _memory_bytes(),
        "gpu_devices": _gpu_devices(),
        "gpu_specs": _gpu_specs(),
    }


def make_sqmc_model(dimension):
    """Deterministic (inverse-CDF) model functions for SQMC."""

    def init_transform(u, model_inputs):
        return jax.scipy.stats.norm.ppf(u)

    def propagate_transform(u, state, model_inputs):
        return state + SIGMA_X * jax.scipy.stats.norm.ppf(u)

    def log_potential(state_prev, state, model_inputs):
        y = model_inputs["y"]
        return (
            -0.5 * jnp.sum(((y - state) / SIGMA_Y) ** 2)
            - dimension * jnp.log(SIGMA_Y)
            - 0.5 * dimension * jnp.log(2 * jnp.pi)
        )

    return init_transform, propagate_transform, log_potential


def generate_observations(key, n_steps, dimension):
    """Generate true states (T, d) and noisy observations (T, d)."""
    keys = random.split(key, n_steps)
    x_true = jnp.zeros((n_steps, dimension))
    x_prev = jnp.zeros(dimension)
    for t in range(n_steps):
        x_prev = x_prev + SIGMA_X * random.normal(keys[t], (dimension,))
        x_true = x_true.at[t].set(x_prev)
    obs_keys = random.split(random.fold_in(key, 999), n_steps)
    noise = jax.vmap(lambda k: SIGMA_Y * random.normal(k, (dimension,)))(obs_keys)
    y = x_true + noise
    return x_true, y


# ---------------------------------------------------------------------------
# Jitted SQMC trajectory runner
#
# The whole trajectory is compiled into a single jax.lax.scan so the filter runs
# on-device without per-step host<->device round-trips. The observations are
# pre-transferred to the device once; this transfer is included in the cold
# end-to-end measurement and excluded from steady-state scan timings.
# ---------------------------------------------------------------------------
def run_sqmc_jit(observations, n_particles, key, dimension, model):
    """Build a jitted SQMC scan with a fresh Sobol block at every time step.

    The QMC engine is initialized outside the timed scan, but point generation
    itself is expressed through the pure Sobol kernel. The scan receives time
    indices and computes the block start as ``1 + t * N``; no Python-side
    mutable sequence counter is touched while tracing the scan.
    """
    init_transform, propagate_transform, log_potential = model
    qmc = Sobol(d=dimension + 1, scramble=True, key=key)

    def sample_at(first_index):
        return _sobol_sample_batched(
            first_index=first_index,
            n=n_particles,
            direction_integers=qmc._direction_integers,
            digital_shift=qmc._digital_shift,
            num_bits=_MAXBITS,
            dtype=qmc.dtype,
        )

    init_points = sample_at(jnp.asarray(1, dtype=jnp.uint32))
    init_particles = jax.vmap(init_transform, (0, None))(
        init_points[:, :dimension], {"y": observations[0]}
    )
    initial_model_inputs = {"y": observations[0]}
    initial_log_weights = jax.vmap(log_potential, (None, 0, None))(
        jnp.zeros_like(init_particles[0]),
        init_particles,
        initial_model_inputs,
    )
    initial_lnc = jax.nn.logsumexp(initial_log_weights) - jnp.log(n_particles)
    init_state = (init_particles, initial_log_weights, initial_lnc)

    def weighted_particle_mean(state):
        # Filtering estimates are weighted expectations under the normalized
        # particle weights, rather than unweighted particle averages.
        particles, log_weights, _ = state
        weights = jax.nn.softmax(log_weights.reshape(-1))
        weight_shape = (n_particles,) + (1,) * (particles.ndim - 1)
        return jnp.sum(particles * weights.reshape(weight_shape), axis=0)

    init_mean = weighted_particle_mean(init_state)

    @jax.jit
    def scan_trajectory(init_state, scan_inputs):
        def step(carry, inputs):
            particles, log_weights, log_normalizing_constant = carry
            y, step_index = inputs
            first_index = (
                jnp.asarray(1, dtype=jnp.uint32)
                + step_index.astype(jnp.uint32) * jnp.uint32(n_particles)
            )
            points = sample_at(first_index)
            tau = jnp.argsort(points[:, 0], stable=True)
            h_order = hilbert_sort(particles)
            hilbert_log_weights = log_weights[h_order]
            idx, _ = resample_from_uniform(
                points[tau, 0], hilbert_log_weights
            )
            ancestor_indices = h_order[idx]
            ancestors = particles[ancestor_indices]
            next_particles = jax.vmap(
                propagate_transform, (0, 0, None)
            )(points[tau, 1:], ancestors, {"y": y})
            next_log_weights = jax.vmap(
                log_potential, (0, 0, None)
            )(ancestors, next_particles, {"y": y})
            next_lnc = (
                log_normalizing_constant
                + jax.nn.logsumexp(next_log_weights)
                - jnp.log(n_particles)
            )
            next_state = (next_particles, next_log_weights, next_lnc)
            return next_state, weighted_particle_mean(next_state)

        return jax.lax.scan(step, init_state, scan_inputs)

    step_indices = jnp.arange(1, observations.shape[0], dtype=jnp.uint32)
    obs_device = jax.device_put(observations[1:])
    step_indices_device = jax.device_put(step_indices)
    return scan_trajectory, init_state, (obs_device, step_indices_device), init_mean


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def time_filter(
    run_fn,
    observations,
    n_particles,
    key,
    dimension,
    model,
    n_reps=DEFAULT_N_REPS,
    n_warmups=DEFAULT_N_WARMUPS,
):
    """Record cold end-to-end and steady-state SQMC timings.

    The cold measurement includes runner construction, initial point generation,
    host-to-device transfer, JIT compilation, and the first complete trajectory.
    Steady-state measurements reuse the compiled runner and block on each result.
    """
    end_to_end_start = time.perf_counter()
    scan_trajectory, init_state, obs_device, init_mean = run_fn(
        observations, n_particles, key, dimension, model
    )
    # Include completion of initialization and host-to-device transfers in the
    # cold setup measurement; JAX dispatch is otherwise asynchronous.
    jax.block_until_ready((init_state, obs_device, init_mean))
    setup_complete = time.perf_counter()

    compile_start = time.perf_counter()
    state, means = scan_trajectory(init_state, obs_device)
    jax.block_until_ready((state, means))
    compile_time = time.perf_counter() - compile_start
    end_to_end_time = time.perf_counter() - end_to_end_start

    for _ in range(n_warmups):
        state, means = scan_trajectory(init_state, obs_device)
        jax.block_until_ready((state, means))

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        state, means = scan_trajectory(init_state, obs_device)
        jax.block_until_ready((state, means))
        times.append(time.perf_counter() - t0)
    means = jnp.concatenate([jnp.array([init_mean]), means])
    mean_time = float(np.mean(times))
    std_time = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    critical = (
        float(student_t.ppf(0.975, len(times) - 1)) if len(times) > 1 else 0.0
    )
    ci_half_width = critical * std_time / np.sqrt(len(times))
    return {
        "median": float(np.median(times)),
        "mean": mean_time,
        "std": std_time,
        "q25": float(np.percentile(times, 25)),
        "q75": float(np.percentile(times, 75)),
        "ci95_low": max(0.0, mean_time - ci_half_width),
        "ci95_high": mean_time + ci_half_width,
        "end_to_end_time": float(end_to_end_time),
        "setup_time": float(setup_complete - end_to_end_start),
        "compile_time": float(compile_time),
        "state": state,
        "means": means,
    }


def rmse(means, x_true):
    """Compute the filtering RMSE from the weighted trajectory estimates."""
    return float(jnp.sqrt(jnp.mean((means - x_true) ** 2)))


# ---------------------------------------------------------------------------
# Per-platform sweep (runs in a child subprocess with JAX_PLATFORM_NAME set)
# ---------------------------------------------------------------------------
def run_sweep(
    platform,
    n_steps,
    n_reps,
    n_warmups,
    particle_counts,
    base_dimension,
    seed,
):
    """Run the SQMC GPU/CPU runtime sweep on the current JAX backend."""
    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(seed)
    x_true, y = generate_observations(key, n_steps, base_dimension)
    model = make_sqmc_model(base_dimension)
    base_sweep = {}
    for n in particle_counts:
        result = time_filter(
            run_sqmc_jit, y, n, random.PRNGKey(0), base_dimension, model,
            n_reps, n_warmups,
        )
        base_sweep[n] = {
            "median": result["median"],
            "mean": result["mean"],
            "std": result["std"],
            "q25": result["q25"],
            "q75": result["q75"],
            "ci95_low": result["ci95_low"],
            "ci95_high": result["ci95_high"],
            "end_to_end_time": result["end_to_end_time"],
            "setup_time": result["setup_time"],
            "compile_time": result["compile_time"],
            "rmse": rmse(result["means"], x_true),
        }
        print(
            f"  [{platform}] d={base_dimension:>3} N={n:>6}: "
            f"mean {result['mean']:.4f}s (95% CI "
            f"{result['ci95_low']:.4f}-{result['ci95_high']:.4f}) "
            f"cold {result['end_to_end_time']:.4f}s "
            f"RMSE {base_sweep[n]['rmse']:.4f}",
            flush=True,
        )

    return {
        "platform": platform,
        "n_steps": n_steps,
        "base_dimension": base_dimension,
        "sweep": {str(n): base_sweep[n] for n in particle_counts},
        "devices": [str(d) for d in jax.devices()],
        "hardware": capture_hardware(),
    }


# ---------------------------------------------------------------------------
# Combined runtime figure
# ---------------------------------------------------------------------------
def plot_gpu_vs_cpu(results, particle_counts, output_dir):
    """Plot steady-state mean runtime with 95% confidence intervals."""
    platforms = [p for p in ("gpu", "cpu") if p in results]
    colours = {"gpu": "C0", "cpu": "C1"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = [int(n) for n in particle_counts]
    for p in platforms:
        sweep = results[p]["sweep"]
        mean = np.array([sweep[str(n)]["mean"] for n in ns])
        low = np.maximum(
            np.array([sweep[str(n)]["ci95_low"] for n in ns]),
            np.finfo(float).tiny,
        )
        high = np.array([sweep[str(n)]["ci95_high"] for n in ns])
        yerr = np.vstack((np.maximum(mean - low, 0.0), np.maximum(high - mean, 0.0)))
        ax.errorbar(
            ns, mean, yerr=yerr, fmt=colours[p] + "o-",
            label=p.upper(), linewidth=1.5, capsize=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Steady-state wall-clock time (s)")
    ax.set_title("SQMC GPU vs CPU runtime (mean ± 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "sqmc_gpu_vs_cpu.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SQMC GPU vs CPU benchmark.")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    p.add_argument("--warmups", type=int, default=DEFAULT_N_WARMUPS)
    p.add_argument("--particle-counts", type=int, nargs="+",
                   default=DEFAULT_PARTICLE_COUNTS)
    p.add_argument("--base-dimension", type=int, default=DEFAULT_BASE_DIMENSION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--platforms", type=str, nargs="+", default=["gpu", "cpu"],
                   choices=["gpu", "cpu"])
    p.add_argument("--output-dir", type=str, default="sqmc/sqmc/scripts/outputs/sqmc_gpu")
    # Internal: run a single platform in this process and write JSON.
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_results-json", type=str, default=None, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    # Child mode: run one platform on the already-selected backend and dump JSON.
    if args._child:
        platform = args.platforms[0]
        result = run_sweep(
            platform, args.n_steps, args.n_reps, args.warmups,
            args.particle_counts, args.base_dimension, args.seed,
        )
        payload = json.dumps(result)
        if args._results_json:
            with open(args._results_json, "w", encoding="utf-8") as fh:
                fh.write(payload)
        else:
            print(payload)
        return 0

    # Parent mode: spawn one subprocess per platform and merge results.
    results = {}
    child_files = {}
    for platform in args.platforms:
        results_json = os.path.join(args.output_dir, f"_sqmc_{platform}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        env = dict(os.environ)
        env["JAX_PLATFORM_NAME"] = platform
        cmd = [
            sys.executable, "-m", "sqmc.sqmc.benchmark_sqmc",
            "--n-steps", str(args.n_steps),
            "--n-reps", str(args.n_reps),
            "--warmups", str(args.warmups),
            "--particle-counts", *(str(n) for n in args.particle_counts),
            "--base-dimension", str(args.base_dimension),
            "--seed", str(args.seed),
            "--platforms", platform,
            "--_child", "--_results-json", results_json,
        ]
        print(f"Running SQMC sweep on {platform}...", flush=True)
        subprocess.run(cmd, env=env, check=True)
        with open(results_json, encoding="utf-8") as fh:
            results[platform] = json.load(fh)
        child_files[platform] = results_json

    out_path = plot_gpu_vs_cpu(results, args.particle_counts, args.output_dir)
    merged = {
        "platforms": args.platforms,
        "n_steps": args.n_steps,
        "n_reps": args.n_reps,
        "particle_counts": args.particle_counts,
        "base_dimension": args.base_dimension,
        "jit": True,
        "results": results,
    }
    json_path = os.path.join(args.output_dir, "sqmc_gpu_vs_cpu.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)

    # Run-config record: the parameters used plus the hardware specs captured on
    # each platform, so the GPU-vs-CPU comparison is auditable and not
    # confounded by a machine mismatch.
    run_config = {
        "config": {
            "n_steps": args.n_steps,
            "n_reps": args.n_reps,
            "warmups": args.warmups,
            "particle_counts": args.particle_counts,
            "base_dimension": args.base_dimension,
            "seed": args.seed,
            "platforms": args.platforms,
            "jit": True,
        },
        "hardware": {p: results[p]["hardware"] for p in results},
    }
    run_config_path = os.path.join(args.output_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    # Clean up per-platform intermediates.
    for f in child_files.values():
        try:
            os.remove(f)
        except OSError:
            pass

    print(f"\nFigure saved to {out_path}")
    print(f"Results saved to {json_path}")
    print(f"Run config saved to {run_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
