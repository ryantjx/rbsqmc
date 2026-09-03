"""Compare SQMC (this package) vs SMC (cuthbert) on a 1D random-walk model.

Runs both filters on the same observations and reports:
  - log-likelihood estimates
  - filtered mean estimates
  - RMSE of the filtered mean vs the true state
  - wall-clock time

Produces a two-panel plot:
  1. Filtered mean vs true state over time
  2. Log-likelihood estimate vs particle count

Usage:
    uv run -m sqmc.scripts.compare_sqmc
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt

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
N_STEPS = 500


def init_transform(u, model_inputs):
    """Deterministic initial transform: x_0 ~ N(0, 1) via inverse-CDF."""
    return jax.scipy.stats.norm.ppf(u)


def propagate_transform(u, state, model_inputs):
    """Deterministic propagation: x_t = x_{t-1} + sigma_x * Phi^{-1}(u)."""
    return state + SIGMA_X * jax.scipy.stats.norm.ppf(u)


def init_sample(key, model_inputs):
    """Stochastic initial sampler for the standard SMC filter."""
    return random.normal(key, ())


def propagate_sample(key, state, model_inputs):
    """Stochastic propagation for the standard SMC filter."""
    return state + SIGMA_X * random.normal(key, ())


def log_potential(state_prev, state, model_inputs):
    """Log potential: log N(y_t | x_t, sigma_y^2)."""
    y = model_inputs["y"]
    return (
        -0.5 * ((y - state) / SIGMA_Y) ** 2
        - jnp.log(SIGMA_Y)
        - 0.5 * jnp.log(2 * jnp.pi)
    )


def generate_observations(key, n_steps):
    """Generate true states and noisy observations from the model."""
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


def run_sqmc(observations, n_particles, key):
    """Run the SQMC filter and return the final state plus per-step means."""
    qmc = Sobol(d=2)  # du=1 state + 1 resampling coordinate
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
    """Run the standard SMC filter and return the final state plus per-step means."""
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


def main():
    jax.config.update("jax_enable_x64", True)

    # Generate data
    key = random.PRNGKey(42)
    x_true, y = generate_observations(key, N_STEPS)

    # --- Single-run comparison at fixed N ---
    N = 1024
    key_sqmc = random.PRNGKey(0)
    key_smc = random.PRNGKey(0)

    t0 = time.perf_counter()
    sqmc_state, sqmc_means = run_sqmc(y, N, key_sqmc)
    t_sqmc = time.perf_counter() - t0

    t0 = time.perf_counter()
    smc_state, smc_means = run_smc(y, N, key_smc)
    t_smc = time.perf_counter() - t0

    sqmc_rmse = jnp.sqrt(jnp.mean((sqmc_means - x_true) ** 2))
    smc_rmse = jnp.sqrt(jnp.mean((smc_means - x_true) ** 2))

    print("=" * 60)
    print(f"  SQMC vs SMC comparison  (N={N}, T={N_STEPS})")
    print("=" * 60)
    print(f"{'Metric':<30} {'SQMC':>12} {'SMC':>12}")
    print("-" * 60)
    print(f"{'Log-likelihood':<30} {sqmc_state.log_normalizing_constant:>12.4f} {smc_state.log_normalizing_constant:>12.4f}")
    print(f"{'Final mean estimate':<30} {jnp.mean(sqmc_state.particles):>12.4f} {jnp.mean(smc_state.particles):>12.4f}")
    print(f"{'True final state':<30} {x_true[-1]:>12.4f} {x_true[-1]:>12.4f}")
    print(f"{'RMSE (filtered mean)':<30} {sqmc_rmse:>12.4f} {smc_rmse:>12.4f}")
    print(f"{'Wall-clock (s)':<30} {t_sqmc:>12.4f} {t_smc:>12.4f}")
    print("=" * 60)

    # --- Particle-count sweep: log-likelihood vs N ---
    particle_counts = [64, 128, 256, 512, 1024, 2048]
    sqmc_lls = []
    smc_lls = []
    for n in particle_counts:
        s, _ = run_sqmc(y, n, random.PRNGKey(0))
        sqmc_lls.append(float(s.log_normalizing_constant))
        s, _ = run_smc(y, n, random.PRNGKey(0))
        smc_lls.append(float(s.log_normalizing_constant))

    print(f"\n{'N':>6} {'SQMC log-lik':>14} {'SMC log-lik':>14} {'Diff':>10}")
    print("-" * 48)
    for n, ll_q, ll_s in zip(particle_counts, sqmc_lls, smc_lls):
        print(f"{n:>6} {ll_q:>14.4f} {ll_s:>14.4f} {ll_q - ll_s:>10.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Filtered mean vs true state
    ax1.plot(range(N_STEPS), x_true, "k-", linewidth=1.5, label="True state")
    ax1.plot(range(N_STEPS), sqmc_means, "C0--", linewidth=1.5, label="SQMC")
    ax1.plot(range(N_STEPS), smc_means, "C1:", linewidth=1.5, label="SMC")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("State")
    ax1.set_title(f"Filtered mean estimates (N={N})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Log-likelihood vs particle count
    ax2.plot(particle_counts, sqmc_lls, "C0o-", linewidth=1.5, label="SQMC")
    ax2.plot(particle_counts, smc_lls, "C1s--", linewidth=1.5, label="SMC")
    ax2.set_xlabel("Number of particles")
    ax2.set_ylabel("Log-likelihood")
    ax2.set_title("Log-likelihood vs particle count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "sqmc/sqmc/outputs/sqmc_vs_smc.png"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()