"""Empirical evaluation: SQMC vs SMC.

Produces the four plots for the dissertation section "Empirical evaluation ---
SQMC vs SMC":

  1. Throughput comparison (log N vs log time)
  2. Speedup ratio (log N vs time_SMC / time_SQMC)
  3. Per-step breakdown (SQMC step vs time per step / % of total)
  4. Time to target error (log run time vs log filtering RMSE)

Usage:
    uv run -m sqmc.sqmc.benchmark_sqmc_smc
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import numpy as np

from sqmc.sqmc.sqmc import build_filter as build_sqmc_filter
from sqmc.qmc.qmc import Sobol

from cuthbert.smc.particle_filter import build_filter as build_smc_filter
from cuthbertlib.resampling import systematic


# ---------------------------------------------------------------------------
# Model: 1D random walk with Gaussian observation noise
#   x_t = x_{t-1} + sigma_x * Z_t,   Z_t ~ N(0, 1)
#   y_t = x_t + sigma_y * E_t,       E_t ~ N(0, 1)
# ---------------------------------------------------------------------------
SIGMA_X = 0.5
SIGMA_Y = 1.0
DEFAULT_N_STEPS = 100
DEFAULT_N_REPS = 5  # repetitions for timing / RMSE averaging
DEFAULT_PARTICLE_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_BREAKDOWN_N = 1024
DEFAULT_SEED = 42


def init_transform(u, model_inputs):
    return jax.scipy.stats.norm.ppf(u)


def propagate_transform(u, state, model_inputs):
    return state + SIGMA_X * jax.scipy.stats.norm.ppf(u)


def init_sample(key, model_inputs):
    return random.normal(key, ())


def propagate_sample(key, state, model_inputs):
    return state + SIGMA_X * random.normal(key, ())


def log_potential(state_prev, state, model_inputs):
    y = model_inputs["y"]
    return (
        -0.5 * ((y - state) / SIGMA_Y) ** 2
        - jnp.log(SIGMA_Y)
        - 0.5 * jnp.log(2 * jnp.pi)
    )


def generate_observations(key, n_steps):
    keys = random.split(key, n_steps)
    x_true = jnp.zeros(n_steps)
    x_prev = 0.0
    for t in range(n_steps):
        x_prev = x_prev + SIGMA_X * random.normal(keys[t], ())
        x_true = x_true.at[t].set(x_prev)
    obs_keys = random.split(random.fold_in(key, 999), n_steps)
    noise = jax.vmap(lambda k: SIGMA_Y * random.normal(k, ()))(obs_keys)
    y = x_true + noise
    return x_true, y


# ---------------------------------------------------------------------------
# Filter runners
# ---------------------------------------------------------------------------
def run_sqmc(observations, n_particles, key):
    qmc = Sobol(d=2)
    filter_ = build_sqmc_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )
    state = filter_.init_prepare({"y": observations[0]}, key=key)
    means = [jnp.mean(state.particles)]
    for t in range(1, len(observations)):
        state = filter_.filter_combine(
            state,
            filter_.filter_prepare({"y": observations[t]}, key=key),
        )
        means.append(jnp.mean(state.particles))
    return state, jnp.array(means)


def run_smc(observations, n_particles, key):
    resampling_fn = systematic.resampling
    filter_ = build_smc_filter(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        resampling_fn=resampling_fn,
    )
    state = filter_.init_prepare({"y": observations[0]}, key=key)
    means = [jnp.mean(state.particles)]
    for t in range(1, len(observations)):
        state = filter_.filter_combine(
            state,
            filter_.filter_prepare({"y": observations[t]}, key=key),
        )
        means.append(jnp.mean(state.particles))
    return state, jnp.array(means)


# ---------------------------------------------------------------------------
# Jitted trajectory runners
#
# The per-step Python loop above forces a host<->device sync at every time
# step, which dominates wall-clock time on a GPU. These variants compile the
# whole trajectory into a single jax.lax.scan so the filter runs on-device
# without per-step round-trips. The filter is built once (Python-level) and
# the scan is jitted over (init_state, observations).
# ---------------------------------------------------------------------------
def run_sqmc_jit(observations, n_particles, key):
    qmc = Sobol(d=2)
    filter_ = build_sqmc_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )
    init_state = filter_.init_prepare({"y": observations[0]}, key=key)
    init_mean = jnp.mean(init_state.particles)

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
            return new_state, jnp.mean(new_state.particles)

        final_state, means = jax.lax.scan(step, init_state, obs)
        return final_state, means

    final_state, means = scan_trajectory(init_state, observations[1:])
    means = jnp.concatenate([jnp.array([init_mean]), means])
    return final_state, means


def run_smc_jit(observations, n_particles, key):
    resampling_fn = systematic.resampling
    filter_ = build_smc_filter(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        resampling_fn=resampling_fn,
    )
    init_state = filter_.init_prepare({"y": observations[0]}, key=key)
    init_mean = jnp.mean(init_state.particles)

    @jax.jit
    def scan_trajectory(init_state, obs):
        def step(carry, y):
            state = carry
            state2 = filter_.filter_prepare({"y": y}, key=key)
            new_state = filter_.filter_combine(state, state2)
            # Normalize log_weights to (N,) for scan carry type consistency.
            new_state = new_state._replace(
                log_weights=new_state.log_weights.reshape(-1)
            )
            return new_state, jnp.mean(new_state.particles)

        final_state, means = jax.lax.scan(step, init_state, obs)
        return final_state, means

    final_state, means = scan_trajectory(init_state, observations[1:])
    means = jnp.concatenate([jnp.array([init_mean]), means])
    return final_state, means


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def time_filter(run_fn, observations, n_particles, key, n_reps=DEFAULT_N_REPS):
    """Return (mean_time, std_time, final_state, means)."""
    times = []
    state, means = None, None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        state, means = run_fn(observations, n_particles, key)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)), state, means


def rmse(means, x_true):
    return float(jnp.sqrt(jnp.mean((means - x_true) ** 2)))


# ---------------------------------------------------------------------------
# Per-step timing breakdown for SQMC
# ---------------------------------------------------------------------------
def sqmc_step_breakdown(observations, n_particles, key):
    """Time each SQMC step individually and return per-step times."""
    qmc = Sobol(d=2)
    filter_ = build_sqmc_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )
    state = filter_.init_prepare({"y": observations[0]}, key=key)
    step_times = []
    for t in range(1, len(observations)):
        t0 = time.perf_counter()
        state = filter_.filter_combine(
            state,
            filter_.filter_prepare({"y": observations[t]}, key=key),
        )
        step_times.append(time.perf_counter() - t0)
    return np.array(step_times)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run the SQMC vs SMC empirical benchmark."
    )
    argument_parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="Number of time steps.",
    )
    argument_parser.add_argument(
        "--n-reps",
        type=int,
        default=DEFAULT_N_REPS,
        help="Number of timing repetitions per particle count.",
    )
    argument_parser.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=DEFAULT_PARTICLE_COUNTS,
        help="Particle counts to sweep over.",
    )
    argument_parser.add_argument(
        "--breakdown-n",
        type=int,
        default=DEFAULT_BREAKDOWN_N,
        help="Particle count for the per-step breakdown.",
    )
    argument_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for observation generation.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=str,
        default="sqmc/sqmc/outputs",
        help="Directory in which to save the plot.",
    )
    argument_parser.add_argument(
        "--jit",
        action="store_true",
        help="Compile the whole trajectory into a single jax.lax.scan "
        "instead of running a per-step Python loop.",
    )
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    n_steps = arguments.n_steps
    n_reps = arguments.n_reps
    particle_counts = arguments.particle_counts
    breakdown_n = arguments.breakdown_n
    seed = arguments.seed
    output_dir = arguments.output_dir
    use_jit = arguments.jit

    run_sqmc_fn = run_sqmc_jit if use_jit else run_sqmc
    run_smc_fn = run_smc_jit if use_jit else run_smc
    mode_label = "jitted (scan)" if use_jit else "per-step loop"

    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(seed)
    x_true, y = generate_observations(key, n_steps)

    # ------------------------------------------------------------------
    # 1. Throughput comparison: log N vs log time
    # ------------------------------------------------------------------
    sqmc_times, smc_times = [], []
    sqmc_times_std, smc_times_std = [], []
    sqmc_states, smc_states = {}, {}
    sqmc_means_all, smc_means_all = {}, {}

    print(f"Running throughput benchmark ({mode_label})...")
    for n in particle_counts:
        t_q, t_q_std, s_q, m_q = time_filter(run_sqmc_fn, y, n, random.PRNGKey(0), n_reps)
        t_s, t_s_std, s_s, m_s = time_filter(run_smc_fn, y, n, random.PRNGKey(0), n_reps)
        sqmc_times.append(t_q)
        smc_times.append(t_s)
        sqmc_times_std.append(t_q_std)
        smc_times_std.append(t_s_std)
        sqmc_states[n] = s_q
        smc_states[n] = s_s
        sqmc_means_all[n] = m_q
        smc_means_all[n] = m_s
        print(f"  N={n:>5}: SQMC {t_q:.4f}s ± {t_q_std:.4f} | SMC {t_s:.4f}s ± {t_s_std:.4f}")

    # ------------------------------------------------------------------
    # 2. Speedup ratio: log N vs time_SMC / time_SQMC
    # ------------------------------------------------------------------
    speedup = np.array(smc_times) / np.array(sqmc_times)

    # ------------------------------------------------------------------
    # 3. Per-step breakdown for SQMC at a representative N
    # ------------------------------------------------------------------
    N_breakdown = breakdown_n
    print(f"\nRunning per-step breakdown at N={N_breakdown}...")
    step_times = sqmc_step_breakdown(y, N_breakdown, random.PRNGKey(0))
    step_pct = 100.0 * step_times / step_times.sum()

    # ------------------------------------------------------------------
    # 4. Time to target error: log run time vs log filtering RMSE
    # ------------------------------------------------------------------
    sqmc_rmses = [rmse(sqmc_means_all[n], x_true) for n in particle_counts]
    smc_rmses = [rmse(smc_means_all[n], x_true) for n in particle_counts]

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  SQMC vs SMC benchmark  (T={n_steps}, reps={n_reps}, {mode_label})")
    print("=" * 72)
    print(f"{'N':>6} {'SQMC time (s)':>14} {'SMC time (s)':>14} {'Speedup':>9} {'SQMC RMSE':>11} {'SMC RMSE':>11}")
    print("-" * 72)
    for i, n in enumerate(particle_counts):
        print(
            f"{n:>6} {sqmc_times[i]:>14.4f} {smc_times[i]:>14.4f} "
            f"{speedup[i]:>9.3f} {sqmc_rmses[i]:>11.4f} {smc_rmses[i]:>11.4f}"
        )
    print("=" * 72)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Throughput: log N vs log time
    ax = axes[0, 0]
    ax.errorbar(particle_counts, sqmc_times, yerr=sqmc_times_std, fmt="C0o-", label="SQMC", capsize=3)
    ax.errorbar(particle_counts, smc_times, yerr=smc_times_std, fmt="C1s--", label="SMC", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Throughput comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # 2. Speedup ratio
    ax = axes[0, 1]
    ax.plot(particle_counts, speedup, "C2^-", linewidth=1.5)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Speedup (time SMC / time SQMC)")
    ax.set_title("Speedup ratio")
    ax.grid(True, alpha=0.3, which="both")

    # 3. Per-step breakdown
    ax = axes[1, 0]
    ax.plot(range(1, n_steps), step_times, "C0-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("SQMC step")
    ax.set_ylabel("Time per step (s)")
    ax.set_title(f"Per-step breakdown (N={N_breakdown})")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_steps), step_pct, "C3-", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("% of total time", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    # 4. Time to target error
    ax = axes[1, 1]
    ax.plot(sqmc_times, sqmc_rmses, "C0o-", label="SQMC", linewidth=1.5)
    ax.plot(smc_times, smc_rmses, "C1s--", label="SMC", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Filtering RMSE")
    ax.set_title("Time to target error")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "sqmc_vs_smc_benchmark.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())