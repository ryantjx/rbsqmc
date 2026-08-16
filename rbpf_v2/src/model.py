from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .bivariate_poisson import daily_loglik
from .data import validate_results
from .kron import symmetrize
from .utils import (
    EMParams, FilterStates, FootballResults, ParticleMeans, RBPFFootballResults,
)


def compute_gamma_trajectory(
    model_inputs: FootballResults, gamma_0, kappa, num_teams: int | None = None
):
    """Compute the deterministic covariance recursion on the exact D-transition timeline."""
    M = gamma_0.shape[0] if num_teams is None else num_teams
    import time as _ctime
    _cgstart = _ctime.perf_counter()
    print(f"[filter {_ctime.strftime('%H:%M:%S')}] compute_gamma_trajectory start (D={model_inputs.timestamp.shape[0]})", flush=True)

    def day_step(previous, day):
        timestamp, timestamp_prev, home, away, mask = day
        phi = jnp.exp(-kappa * (timestamp - timestamp_prev))
        predicted = symmetrize(phi**2 * previous + (1.0 - phi**2) * gamma_0)

        def match_step(current, match):
            h, a, valid = match

            def condition(g):
                ids = jnp.array([h, a])
                observed = g[jnp.ix_(ids, ids)]
                cross = g[:, ids]
                # right solve cross @ inv(observed), using its Cholesky factor.
                chol = jnp.linalg.cholesky(observed)
                gain = jax.scipy.linalg.cho_solve((chol, True), cross.T).T
                updated = symmetrize(g - gain @ g[ids, :])
                keep = jnp.ones(M, dtype=g.dtype).at[ids].set(0.0)
                updated = updated * jnp.outer(keep, keep)
                return updated, (observed, gain)

            def padding(g):
                return g, (jnp.zeros((2, 2), g.dtype), jnp.zeros((M, 2), g.dtype))

            return jax.lax.cond(valid, condition, padding, current)

        posterior, (observed, gains) = jax.lax.scan(
            match_step, predicted, (home, away, mask)
        )
        return posterior, (posterior, predicted, observed, gains)

    _, values = jax.lax.scan(
        day_step, gamma_0,
        (model_inputs.timestamp, model_inputs.timestamp_prev,
         model_inputs.matches.home_id, model_inputs.matches.away_id,
         model_inputs.match_mask),
    )
    print(f"[filter {_ctime.strftime('%H:%M:%S')}] compute_gamma_trajectory done in {_ctime.perf_counter() - _cgstart:.1f}s", flush=True)
    return values


def augment_results(model_inputs: FootballResults, covariance_path) -> RBPFFootballResults:
    gamma, gamma_pred, gamma_observed, kalman_gain = covariance_path
    return RBPFFootballResults(
        model_inputs.date, model_inputs.timestamp, model_inputs.timestamp_prev,
        model_inputs.matches, model_inputs.match_mask, gamma, gamma_pred,
        gamma_observed, kalman_gain,
    )


def systematic_resample(key, log_weights, n_samples: int | None = None):
    n = log_weights.shape[0] if n_samples is None else n_samples
    probabilities = jax.nn.softmax(log_weights)
    cumulative = jnp.cumsum(probabilities).at[-1].set(1.0)
    u0 = jax.random.uniform(key, (), maxval=1.0 / n)
    positions = u0 + jnp.arange(n) / n
    return jnp.searchsorted(cumulative, positions, side="right")


def _propagate_component(key, parent_mean, day, params: EMParams):
    dt = day.timestamp - day.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    predicted = params.mean_0 + phi * (parent_mean - params.mean_0)

    def match_step(carry, match):
        current, rng = carry
        gain, observed, h, a, valid = match

        def sample(value):
            state, parent_key = value
            parent_key, draw_key = jax.random.split(parent_key)
            ids = jnp.array([h, a])
            mean_o = state[ids]
            covariance = jnp.kron(observed, params.B)
            sampled = jax.random.multivariate_normal(
                draw_key, mean_o.reshape(-1), covariance, method="svd"
            ).reshape(2, 2)
            updated = state + gain @ (sampled - mean_o)
            return updated.at[ids].set(sampled), parent_key

        return jax.lax.cond(valid, sample, lambda value: value, (current, rng)), None

    (state, _), _ = jax.lax.scan(
        match_step, (predicted, key),
        (day.kalman_gain, day.gamma_observed, day.matches.home_id,
         day.matches.away_id, day.match_mask),
    )
    return state


def run_filter(key, model_inputs: FootballResults, params: EMParams,
               n_particles: int, max_goals: int = 8):
    """Run a daily-resampled RBPF whose particles are component means."""
    if not isinstance(model_inputs.timestamp, jax.core.Tracer):
        validate_results(model_inputs, params.mean_0.shape[0])
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    covariance_path = compute_gamma_trajectory(
        model_inputs, params.gamma_0, params.kappa, params.mean_0.shape[0]
    )
    import time as _time
    def _flog(msg):
        print(f"[filter {_time.strftime('%H:%M:%S')}] {msg}", flush=True)
    augmented = augment_results(model_inputs, covariance_path)
    D = model_inputs.timestamp.shape[0]
    _flog(f"run_filter start: D={D} n_particles={n_particles}")
    means = [jnp.broadcast_to(params.mean_0, (n_particles,) + params.mean_0.shape)]
    weights = [jnp.full((n_particles,), -math.log(n_particles))]
    ancestors = []
    logz = [jnp.asarray(0.0)]
    rng = key
    _filter_start = _time.perf_counter()
    for t in range(D):
        if t % 1000 == 0 or t == D - 1:
            _flog(f"filter day {t}/{D} ({100 * t // max(D, 1)}%) in "
                  f"{_time.perf_counter() - _filter_start:.1f}s")
        rng, parent_key, propagation_key = jax.random.split(rng, 3)
        indices = systematic_resample(parent_key, weights[-1], n_particles)
        parents = means[-1][indices]
        keys = jax.random.split(propagation_key, n_particles)
        day = jax.tree.map(lambda x: x[t], augmented)
        next_means = jax.vmap(lambda k, m: _propagate_component(k, m, day, params))(
            keys, parents
        )
        increments = jax.vmap(
            lambda state: daily_loglik(state, day, params.alpha, params.beta, max_goals)
        )(next_means)
        normalizer = logsumexp(increments) - math.log(n_particles)
        next_weights = increments - logsumexp(increments)
        means.append(next_means)
        weights.append(next_weights)
        ancestors.append(indices)
        logz.append(logz[-1] + normalizer)
    _flog(f"run_filter done in {_time.perf_counter() - _filter_start:.1f}s")
    states = FilterStates(
        ParticleMeans(jnp.stack(means)), jnp.stack(weights),
        jnp.stack(ancestors) if ancestors else jnp.empty((0, n_particles), dtype=int),
        jnp.stack(logz),
    )
    return states, augmented


def mixture_moments(means, log_weights, gamma, B):
    """Dense mean/covariance of an RB Gaussian mixture at one state."""
    weights = jax.nn.softmax(log_weights)
    flat = means.reshape(means.shape[0], -1)
    mean = jnp.sum(weights[:, None] * flat, axis=0)
    centered = flat - mean
    covariance = jnp.kron(gamma, B) + (centered * weights[:, None]).T @ centered
    return mean.reshape(means.shape[1:]), covariance
