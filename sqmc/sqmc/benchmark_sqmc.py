"""Empirical evaluation: SQMC on GPU vs CPU.

Times only on-device execution (the host<->device sync is excluded by timing the
jitted ``jax.lax.scan`` whole-trajectory runner and blocking on the result inside
the timed region). Produces a single 4-panel figure for the dissertation section
"Empirical evaluation --- GPU vs CPU":

  1. Wall-clock vs N (log-log), GPU vs CPU, median + IQR (crossover).
  2. Speedup ratio time_CPU / time_GPU vs N, with a crossover line at 1.
  3. Per-step stacked breakdown (Sobol' / Hilbert index / argsort / propagate)
     on GPU vs CPU.
  4. Wall-clock vs filtering RMSE (log-log), GPU vs CPU, with a target-RMSE line.

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
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sqmc.sqmc.sqmc import build_filter as build_sqmc_filter
from sqmc.sqmc.sqmc import resample_from_uniform
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort, Hilbert_to_int
from sqmc.qmc.qmc import Sobol


# ---------------------------------------------------------------------------
# Model: d-dimensional random walk with Gaussian observation noise
#   x_t = x_{t-1} + sigma_x * Z_t,   Z_t ~ N(0, I_d)
#   y_t = x_t + sigma_y * E_t,       E_t ~ N(0, I_d)
# ---------------------------------------------------------------------------
SIGMA_X = 0.5
SIGMA_Y = 1.0
DEFAULT_N_STEPS = 100
DEFAULT_N_REPS = 5  # repetitions for timing / RMSE averaging
DEFAULT_N_WARMUPS = 1  # warmup runs (first captures JAX compile time)
DEFAULT_PARTICLE_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_BASE_DIMENSION = 10  # state dimension for the detailed 4-panel figure
DEFAULT_DIMENSIONS = [2, 5, 10, 20, 50, 60]  # state dimensions for the scaling sweep
DEFAULT_BREAKDOWN_N = 1024  # particle count for the per-step breakdown
DEFAULT_TARGET_RMSE = 0.5
DEFAULT_SEED = 42

STAGES = ("sobol", "hilbert_idx", "argsort", "propagate")


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


def capture_hardware() -> dict:
    """Return a dict describing the current host and its accelerators."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "node": platform.node(),
        "memory_bytes": _memory_bytes(),
        "gpu_devices": _gpu_devices(),
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
# pre-transferred to the device once (outside any timed region), so the
# host<->device transfer latency is excluded when the scan is timed.
# ---------------------------------------------------------------------------
def run_sqmc_jit(observations, n_particles, key, dimension, model):
    """Build the jitted SQMC trajectory scan and pre-transfer observations.

    Returns ``(scan_trajectory, init_state, obs_device, init_mean)``. The
    observations are moved to the device once here, so timing the returned
    ``scan_trajectory`` measures only the on-device filtering steps.
    """
    init_transform, propagate_transform, log_potential = model
    qmc = Sobol(d=dimension + 1)
    filter_ = build_sqmc_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )
    init_state = filter_.init_prepare({"y": observations[0]}, key=key)
    init_mean = jnp.mean(init_state.particles, axis=0)

    @jax.jit
    def scan_trajectory(init_state, obs):
        def step(carry, y):
            state = carry
            state2 = filter_.filter_prepare({"y": y}, key=key)
            new_state = filter_.filter_combine(state, state2)
            # filter_combine returns log_weights of shape (N, 1); reshape to
            # (N,) so the scan carry input/output types match.
            new_state = new_state._replace(
                log_weights=new_state.log_weights.reshape(-1)
            )
            return new_state, jnp.mean(new_state.particles, axis=0)

        final_state, means = jax.lax.scan(step, init_state, obs)
        return final_state, means

    # Move observations to the device once, outside the timed region.
    obs_device = jax.device_put(observations[1:])
    return scan_trajectory, init_state, obs_device, init_mean


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
    """Time only the SQMC filtering steps (the jitted scan).

    The observations are pre-transferred to the device by ``run_fn`` before any
    timing, so the host<->device transfer latency is excluded. Each timed call
    blocks on the result with ``jax.block_until_ready`` inside the timed region,
    so the recorded time is the on-device filtering time only. The first warmup
    call is reported separately as ``compile_time`` (XLA compilation), a one-off
    cost excluded from the steady-state statistics.

    Returns a dict with median/mean/std/q25/q75/compile_time and the final
    ``(state, means)``.
    """
    scan_trajectory, init_state, obs_device, init_mean = run_fn(
        observations, n_particles, key, dimension, model
    )
    compile_time = None
    state, means = None, None
    for w in range(n_warmups):
        t0 = time.perf_counter()
        state, means = scan_trajectory(init_state, obs_device)
        jax.block_until_ready((state, means))
        if w == 0:
            compile_time = time.perf_counter() - t0
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        state, means = scan_trajectory(init_state, obs_device)
        jax.block_until_ready((state, means))
        times.append(time.perf_counter() - t0)
    means = jnp.concatenate([jnp.array([init_mean]), means])
    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "q25": float(np.percentile(times, 25)),
        "q75": float(np.percentile(times, 75)),
        "compile_time": float(compile_time) if compile_time is not None else None,
        "state": state,
        "means": means,
    }


def rmse(means, x_true):
    """Overall RMSE across all state components and time steps."""
    return float(jnp.sqrt(jnp.mean((means - x_true) ** 2)))


# ---------------------------------------------------------------------------
# Per-step stage breakdown
#
# Each stage of ``filter_combine`` (sqmc/sqmc/sqmc.py) is timed as its own jitted
# callable over a fixed batch of ``breakdown_n`` particles, wrapped in
# ``jax.block_until_ready`` and averaged over ``n_reps`` after warmups. This
# avoids reintroducing a sync into the trajectory scan.
# ---------------------------------------------------------------------------
def _time_stage(fn, args, n_reps, n_warmups):
    for _ in range(n_warmups):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_breakdown(dimension, breakdown_n, key, n_reps, n_warmups):
    """Return ``{stage: median_seconds}`` for one SQMC step at ``breakdown_n``."""
    init_transform, propagate_transform, log_potential = make_sqmc_model(dimension)
    qmc = Sobol(d=dimension + 1)

    # A representative particle batch and matching log-weights / observation.
    particles = random.normal(key, (breakdown_n, dimension))
    log_weights = jnp.zeros(breakdown_n)
    model_inputs = {"y": jnp.zeros(dimension)}

    # Stage 1: Sobol' point generation + argsort of the resampling coordinate.
    # ``n`` is static because ``qmc.sample`` requires a Python integer.
    @partial(jax.jit, static_argnames=("n",))
    def stage_sobol(n):
        u = qmc.sample(n)
        tau = jnp.argsort(u[:, 0])
        return u, tau

    # Stage 2: Hilbert index computation (local, per-particle; no final argsort).
    @jax.jit
    def stage_hilbert_idx(p):
        work = p.astype(jnp.float64)
        means = jnp.mean(work, axis=0, keepdims=True)
        stds = jnp.std(work, axis=0, keepdims=True)
        safe = jnp.where(stds > 0.0, stds, jnp.ones_like(stds))
        unit = jax.nn.sigmoid((work - means) / safe)
        bits = 62 // dimension
        grid = 1 << bits
        ints = jnp.clip(jnp.floor(unit * grid), 0.0, grid - 1).astype(jnp.uint64)
        return jax.vmap(lambda c: Hilbert_to_int(c, grid))(ints)

    # Stage 3: the global argsort of the Hilbert indices.
    @jax.jit
    def stage_argsort(p):
        return hilbert_sort(p)

    # Stage 4: resample + propagate + reweight.
    u_full = qmc.sample(breakdown_n)
    tau_full = jnp.argsort(u_full[:, 0])

    @jax.jit
    def stage_propagate(p, lw, u, tau, mi):
        h_order = hilbert_sort(p)
        hilbert_log_weights = lw[h_order]
        idx, _ = resample_from_uniform(u[tau, 0], hilbert_log_weights)
        ancestor_indices = h_order[idx]
        ancestors = p[ancestor_indices]
        v = u[tau, 1:]
        next_particles = jax.vmap(propagate_transform, (0, 0, None))(v, ancestors, mi)
        log_potentials = jax.vmap(log_potential, (0, 0, None))(ancestors, next_particles, mi)
        return next_particles, log_potentials

    breakdown = {
        "sobol": _time_stage(stage_sobol, (breakdown_n,), n_reps, n_warmups),
        "hilbert_idx": _time_stage(stage_hilbert_idx, (particles,), n_reps, n_warmups),
        "argsort": _time_stage(stage_argsort, (particles,), n_reps, n_warmups),
        "propagate": _time_stage(
            stage_propagate, (particles, log_weights, u_full, tau_full, model_inputs),
            n_reps, n_warmups,
        ),
    }
    return breakdown


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
    dimensions,
    breakdown_n,
    target_rmse,
    target_margin,
    seed,
):
    """Run the full SQMC sweep on the current JAX backend. Returns a dict.

    Produces (i) a per-N sweep at ``base_dimension`` (for the detailed 4-panel
    figure) and (ii) per-dimension sweeps used for the time-to-target-error and
    speedup-vs-dimension figure.

    The absolute ``target_rmse`` is used only for the base-dimension panel. For
    the by-dimension figure the target is set per dimension to a small margin
    above that dimension's achievable RMSE floor (``min_rmse * (1 + margin)``),
    because the filtering-error floor grows with dimension and a single absolute
    target is unreachable at high d. The wall-clock reported for a dimension is
    the median at the smallest particle count reaching that per-dimension target.
    """
    jax.config.update("jax_enable_x64", True)

    def sweep_dimension(dimension):
        key = random.PRNGKey(seed)
        x_true, y = generate_observations(key, n_steps, dimension)
        model = make_sqmc_model(dimension)
        sweep = {}
        for n in particle_counts:
            result = time_filter(
                run_sqmc_jit, y, n, random.PRNGKey(0), dimension, model,
                n_reps, n_warmups,
            )
            sweep[n] = {
                "median": result["median"],
                "mean": result["mean"],
                "std": result["std"],
                "q25": result["q25"],
                "q75": result["q75"],
                "compile_time": result["compile_time"],
                "rmse": rmse(result["means"], x_true),
            }
            print(
                f"  [{platform}] d={dimension:>3} N={n:>5}: "
                f"med {result['median']:.4f}s (IQR {result['q25']:.4f}-"
                f"{result['q75']:.4f}) compile {result['compile_time']:.4f}s "
                f"RMSE {sweep[n]['rmse']:.4f}",
                flush=True,
            )
        return sweep

    base_sweep = sweep_dimension(base_dimension)

    dim_data = {}
    for d in dimensions:
        dim_sweep = sweep_dimension(d)
        floor = min(s["rmse"] for s in dim_sweep.values())
        dim_target = floor * (1.0 + target_margin)
        reached_n = min(
            (n for n, s in dim_sweep.items() if s["rmse"] <= dim_target),
            default=None,
        )
        if reached_n is None:
            dim_data[str(d)] = {
                "sweep": {str(n): dim_sweep[n] for n in particle_counts},
                "target": dim_target,
                "reached": False,
                "n_star": None,
                "time_to_target": None,
            }
        else:
            dim_data[str(d)] = {
                "sweep": {str(n): dim_sweep[n] for n in particle_counts},
                "target": dim_target,
                "reached": True,
                "n_star": reached_n,
                "time_to_target": dim_sweep[reached_n]["median"],
            }
        ttt = (f"{dim_data[str(d)]['time_to_target']:.4f}"
               if dim_data[str(d)]["reached"] else "-")
        print(
            f"  [{platform}] d={d:>3} reached={dim_data[str(d)]['reached']} "
            f"N*={dim_data[str(d)]['n_star'] or '-'} time_to_target={ttt}s",
            flush=True,
        )

    breakdown = time_breakdown(
        base_dimension, breakdown_n, random.PRNGKey(1), n_reps, n_warmups
    )
    print(f"  [{platform}] breakdown @ d={base_dimension} N={breakdown_n}: "
          f"{breakdown}", flush=True)

    return {
        "platform": platform,
        "n_steps": n_steps,
        "base_dimension": base_dimension,
        "breakdown_n": breakdown_n,
        "target_rmse": target_rmse,
        "sweep": {str(n): base_sweep[n] for n in particle_counts},
        "dimensions": dim_data,
        "breakdown": breakdown,
        "devices": [str(d) for d in jax.devices()],
        "hardware": capture_hardware(),
    }


# ---------------------------------------------------------------------------
# Combined 4-panel figure
# ---------------------------------------------------------------------------
def plot_gpu_vs_cpu(results, particle_counts, target_rmse, output_dir):
    platforms = [p for p in ("gpu", "cpu") if p in results]
    colours = {"gpu": "C0", "cpu": "C1"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SQMC: GPU vs CPU", fontsize=16)

    ns = [int(n) for n in particle_counts]

    # Panel 1: wall-clock vs N (log-log), mean + IQR error bars at each point.
    ax = axes[0, 0]
    for p in platforms:
        sweep = results[p]["sweep"]
        mean = np.array([sweep[str(n)]["mean"] for n in ns])
        q25 = np.array([sweep[str(n)]["q25"] for n in ns])
        q75 = np.array([sweep[str(n)]["q75"] for n in ns])
        # Error-bar extents must be non-negative. With few repetitions the mean
        # can fall outside the IQR, so clamp the extent at the data point rather
        # than passing a negative value to matplotlib.
        yerr = np.maximum(mean - q25, 0.0), np.maximum(q75 - mean, 0.0)
        yerr = np.vstack(yerr)
        ax.errorbar(
            ns, mean, yerr=yerr, fmt=colours[p] + "o-",
            label=p.upper(), linewidth=1.5, capsize=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Throughput (mean + IQR error bars)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: speedup ratio time_CPU / time_GPU.
    ax = axes[0, 1]
    if "gpu" in results and "cpu" in results:
        speedup = [
            results["cpu"]["sweep"][str(n)]["median"]
            / results["gpu"]["sweep"][str(n)]["median"]
            for n in ns
        ]
        ax.plot(ns, speedup, "C2o-", linewidth=1.5)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Speedup (time CPU / time GPU)")
    ax.set_title("Speedup ratio")
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: per-step stacked breakdown, GPU vs CPU.
    ax = axes[1, 0]
    width = 0.35
    xpos = np.arange(len(platforms))
    bottoms = np.zeros(len(platforms))
    stage_colours = dict(zip(STAGES, ("C0", "C1", "C2", "C3")))
    for stage in STAGES:
        vals = np.array([results[p]["breakdown"][stage] for p in platforms])
        ax.bar(xpos, vals, width, bottom=bottoms, label=stage,
               color=stage_colours[stage])
        bottoms += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels([p.upper() for p in platforms])
    ax.set_ylabel("Time per step (s)")
    ax.set_title(f"Per-step breakdown (N={results[platforms[0]]['breakdown_n']})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: wall-clock vs RMSE (log-log), target line.
    ax = axes[1, 1]
    for p in platforms:
        sweep = results[p]["sweep"]
        rmses = [sweep[str(n)]["rmse"] for n in ns]
        med = [sweep[str(n)]["median"] for n in ns]
        ax.plot(med, rmses, colours[p] + "o-", label=p.upper(), linewidth=1.5)
    ax.axhline(target_rmse, color="k", linestyle=":", linewidth=1,
               label=f"target RMSE={target_rmse}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Filtering RMSE")
    ax.set_title("Time to target error")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "sqmc_gpu_vs_cpu.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_gpu_vs_cpu_by_dimension(results, dimensions, output_dir):
    """Time-to-target error and speedup across state dimensions, GPU vs CPU.

    For each dimension the wall-clock at the smallest particle count reaching
    RMSE <= target is the operating point, so the panels compare the two
    backends at parity of error.
    """
    platforms = [p for p in ("gpu", "cpu") if p in results]
    colours = {"gpu": "C0", "cpu": "C1"}
    ds = [int(d) for d in dimensions]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("SQMC GPU vs CPU across state dimensions", fontsize=15)

    # Panel 1: time-to-target error (wall-clock at per-dimension target) vs d.
    ax = axes[0]
    for p in platforms:
        dims_ok, times = [], []
        for d in ds:
            entry = results[p]["dimensions"][str(d)]
            if entry["reached"]:
                dims_ok.append(d)
                times.append(entry["time_to_target"])
        if dims_ok:
            ax.plot(dims_ok, times, colours[p] + "o-", label=p.upper(),
                    linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("State dimension $d$")
    ax.set_ylabel("Wall-clock to per-dim target (s)")
    ax.set_title("Time to target error")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: speedup ratio time_CPU / time_GPU at the target particle count.
    ax = axes[1]
    if "gpu" in results and "cpu" in results:
        dims_ok, speedup = [], []
        for d in ds:
            g = results["gpu"]["dimensions"][str(d)]
            c = results["cpu"]["dimensions"][str(d)]
            if g["reached"] and c["reached"]:
                dims_ok.append(d)
                speedup.append(c["time_to_target"] / g["time_to_target"])
        if dims_ok:
            ax.plot(dims_ok, speedup, "C2o-", linewidth=1.5)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("State dimension $d$")
    ax.set_ylabel("Speedup (time CPU / time GPU)")
    ax.set_title("Speedup ratio at target error")
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: particles needed N* to reach target RMSE vs dimension.
    ax = axes[2]
    for p in platforms:
        dims_ok, nstars = [], []
        for d in ds:
            entry = results[p]["dimensions"][str(d)]
            if entry["reached"]:
                dims_ok.append(d)
                nstars.append(entry["n_star"])
        if dims_ok:
            ax.plot(dims_ok, nstars, colours[p] + "o-", label=p.upper(),
                    linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("State dimension $d$")
    ax.set_ylabel("Particles to target error $N^*$")
    ax.set_title("Particles to target error")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "sqmc_gpu_vs_cpu_by_dimension.png")
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
    p.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS,
                   help="State dimensions for the scaling sweep (time-to-target, speedup).")
    p.add_argument("--breakdown-n", type=int, default=DEFAULT_BREAKDOWN_N)
    p.add_argument("--target-rmse", type=float, default=DEFAULT_TARGET_RMSE)
    p.add_argument("--target-margin", type=float, default=0.05,
                   help="Fraction above each dimension's RMSE floor used as the "
                        "per-dimension target for the by-dimension figure.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--platforms", type=str, nargs="+", default=["gpu", "cpu"],
                   choices=["gpu", "cpu"])
    p.add_argument("--output-dir", type=str, default="sqmc/scripts/outputs/sqmc_gpu")
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
            args.particle_counts, args.base_dimension, args.dimensions,
            args.breakdown_n, args.target_rmse, args.target_margin, args.seed,
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
            "--dimensions", *(str(d) for d in args.dimensions),
            "--breakdown-n", str(args.breakdown_n),
            "--target-rmse", str(args.target_rmse),
            "--target-margin", str(args.target_margin),
            "--seed", str(args.seed),
            "--platforms", platform,
            "--_child", "--_results-json", results_json,
        ]
        print(f"Running SQMC sweep on {platform}...", flush=True)
        subprocess.run(cmd, env=env, check=True)
        with open(results_json, encoding="utf-8") as fh:
            results[platform] = json.load(fh)
        child_files[platform] = results_json

    out_path = plot_gpu_vs_cpu(results, args.particle_counts, args.target_rmse,
                               args.output_dir)
    by_dim_path = plot_gpu_vs_cpu_by_dimension(
        results, args.dimensions, args.output_dir
    )
    merged = {
        "platforms": args.platforms,
        "target_rmse": args.target_rmse,
        "dimensions": args.dimensions,
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
            "dimensions": args.dimensions,
            "breakdown_n": args.breakdown_n,
            "target_rmse": args.target_rmse,
            "seed": args.seed,
            "platforms": args.platforms,
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
    print(f"By-dimension figure saved to {by_dim_path}")
    print(f"Results saved to {json_path}")
    print(f"Run config saved to {run_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
