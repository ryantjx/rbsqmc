"""Standalone Cuthbert-integrated Rao--Blackwellized FFBS and MCEM backend."""

from __future__ import annotations

import math
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular
import optax
import cuthbert
import cuthbertlib
from cuthbert.smc.backward_sampler import build_smoother as cuthbert_build_smoother
from cuthbert.smc.particle_filter import ParticleFilterState

from rbpf_v3.src.bivariate_poisson import loglik
from rbpf_v3.src.helpers import (
    decode_EM_params,
    encode_EM_params,
    log_inverse_wishart_kernel,
)
from rbpf_v3.src.progress import progress
from rbpf_v3.src.utils import EMParams


# Extra degrees of freedom for the inverse-Wishart prior on gamma_0 (Sigma_0).
# The prior is InvWishart(nu, S) with nu = dimension + PRIOR_DOF_EXTRA. A larger
# value shrinks the fitted covariance toward the initial gamma_0, which helps
# stabilize EM when the M x M covariance is far higher-dimensional than the
# data can identify. The minimum valid nu is dimension + 1.
#
# Prior mean ratio E[Sigma]/Sigma0_init = (nu+M+1)/(nu-M-1):
#   extra=50  -> 3.0x (moderate)
#   extra=300 -> 1.3x (aggressive)
#   extra=500 -> 1.2x (very aggressive, near the practical ceiling)
PRIOR_DOF_EXTRA = 300.0


class BackwardDiagnostics(NamedTuple):
    ess_by_time: jax.Array
    entropy_by_time: jax.Array
    max_probability_by_time: jax.Array
    unique_indices_by_time: jax.Array
    probabilities: jax.Array | None


class SmoothedStates(NamedTuple):
    x: jax.Array
    component_indices: jax.Array
    diagnostics: BackwardDiagnostics


class MCEMConfig(NamedTuple):
    n_filter_particles: int = 32
    n_smoother_particles: int = 32
    n_epochs: int = 1
    n_gradient_steps: int = 2
    learning_rate: float = 1e-2
    max_goals: int = 8
    path_batch_size: int = 32
    log_every_gradient_steps: int = 5
    return_backward_probabilities: bool = False
    profile: bool = False
    acceptance_tolerance: float = 1e-6


class RBSmootherParticle(NamedTuple):
    x: jax.Array
    time_index: jax.Array


def symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + matrix.T)


def kron_logdet(gamma: jax.Array, B: jax.Array) -> jax.Array:
    gamma_chol = jnp.linalg.cholesky(gamma)
    b_chol = jnp.linalg.cholesky(B)
    return (
        B.shape[0] * 2.0 * jnp.sum(jnp.log(jnp.diag(gamma_chol)))
        + gamma.shape[0] * 2.0 * jnp.sum(jnp.log(jnp.diag(b_chol)))
    )


def kron_quad_batched(
    gamma_chol: jax.Array,
    B_chol: jax.Array,
    residuals: jax.Array,
) -> jax.Array:
    """Return Kronecker quadratics for arbitrary leading batch dimensions."""
    leading = residuals.shape[:-2]
    m, k = residuals.shape[-2:]
    flat = residuals.reshape((-1, m, k))
    team_rhs = flat.transpose(1, 0, 2).reshape((m, -1))
    team_white = solve_triangular(gamma_chol, team_rhs, lower=True)
    team_white = team_white.reshape((m, flat.shape[0], k)).transpose(1, 0, 2)
    trait_rhs = team_white.transpose(2, 0, 1).reshape((k, -1))
    white = solve_triangular(B_chol, trait_rhs, lower=True)
    white = white.reshape((k, flat.shape[0], m)).transpose(1, 2, 0)
    return jnp.sum(white * white, axis=(-2, -1)).reshape(leading)


def psd_sqrt(matrix: jax.Array, tolerance: float = 1e-7) -> jax.Array:
    values, vectors = jnp.linalg.eigh(symmetrize(matrix))
    if not isinstance(values, jax.core.Tracer):
        import numpy as np

        scale = max(1.0, float(np.max(np.abs(np.asarray(values)))))
        if float(np.min(np.asarray(values))) < -tolerance * scale:
            raise ValueError("matrix is not positive semidefinite")
    return (vectors * jnp.sqrt(jnp.maximum(values, 0.0))[None, :]) @ vectors.T


def sample_kron_psd_batched(
    key: jax.Array,
    means: jax.Array,
    gamma_sqrt: jax.Array,
    B_sqrt: jax.Array,
) -> jax.Array:
    noise = jax.random.normal(key, means.shape, dtype=means.dtype)
    transformed = jnp.einsum("ij,...jk,lk->...il", gamma_sqrt, noise, B_sqrt)
    return means + transformed


def backward_shared_terms(
    gamma_t: jax.Array,
    gamma_pred_next: jax.Array,
    phi_t: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    pred_chol = jnp.linalg.cholesky(gamma_pred_next)
    gain = phi_t * cho_solve((pred_chol, True), gamma_t.T).T
    conditional = symmetrize(gamma_t - gain @ gamma_pred_next @ gain.T)
    return gain, conditional


def _gaussian_kron_logpdf_from_cholesky(
    residuals: jax.Array,
    gamma_chol: jax.Array,
    B_chol: jax.Array,
) -> jax.Array:
    m, k = residuals.shape[-2:]
    logdet = (
        k * 2.0 * jnp.sum(jnp.log(jnp.diag(gamma_chol)))
        + m * 2.0 * jnp.sum(jnp.log(jnp.diag(B_chol)))
    )
    return -0.5 * (
        m * k * jnp.log(2.0 * jnp.pi)
        + logdet
        + kron_quad_batched(gamma_chol, B_chol, residuals)
    )


def gaussian_kron_logpdf(
    residuals: jax.Array, gamma: jax.Array, B: jax.Array
) -> jax.Array:
    return _gaussian_kron_logpdf_from_cholesky(
        residuals, jnp.linalg.cholesky(gamma), jnp.linalg.cholesky(B)
    )


def batched_backward_step(
    key: jax.Array,
    means_t: jax.Array,
    log_weights_t: jax.Array,
    x_next: jax.Array,
    mean_0: jax.Array,
    gamma_t: jax.Array,
    gamma_pred_next: jax.Array,
    phi_t: jax.Array,
    B: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select all S components and draw all S previous states in one operation."""
    selection_key, state_key = jax.random.split(key)
    predicted = mean_0 + phi_t * (means_t - mean_0)
    residuals = x_next[:, None, :, :] - predicted[None, :, :, :]
    pred_chol = jnp.linalg.cholesky(gamma_pred_next)
    b_chol = jnp.linalg.cholesky(B)
    logits = log_weights_t[None, :] + _gaussian_kron_logpdf_from_cholesky(
        residuals, pred_chol, b_chol
    )
    probabilities = jax.nn.softmax(logits, axis=-1)
    selection_keys = jax.random.split(selection_key, x_next.shape[0])
    chosen = jax.vmap(jax.random.categorical)(selection_keys, logits)
    gain, conditional_gamma = backward_shared_terms(
        gamma_t, gamma_pred_next, phi_t
    )
    conditional_means = means_t[chosen] + jnp.einsum(
        "ij,sjk->sik", gain, x_next - predicted[chosen]
    )
    previous = sample_kron_psd_batched(
        state_key,
        conditional_means,
        psd_sqrt(conditional_gamma),
        b_chol,
    )
    return previous, chosen, probabilities


def _unique_count(indices: jax.Array, n_components: int) -> jax.Array:
    return jnp.sum(jnp.bincount(indices, length=n_components) > 0)


def validate_inputs(filter_states, rb_inputs, params: EMParams) -> None:
    import numpy as np

    means = filter_states.particles.x
    weights = filter_states.log_weights
    d = means.shape[0] - 1
    if means.ndim != 4 or means.shape[-1] != 2:
        raise ValueError("filter means must have shape (D+1,N,M,2)")
    if weights.shape != means.shape[:2]:
        raise ValueError("filter log weights do not align with means")
    if rb_inputs.gamma.shape != (d, means.shape[-2], means.shape[-2]):
        raise ValueError("filtered covariance timeline is misaligned")
    if rb_inputs.gamma_pred.shape != rb_inputs.gamma.shape:
        raise ValueError("predicted covariance timeline is misaligned")
    dt = np.asarray(rb_inputs.timestamp - rb_inputs.timestamp_prev)
    if np.any(dt <= 0):
        raise ValueError("every elapsed time must be positive")
    if not np.all(np.isfinite(np.asarray(means))):
        raise ValueError("filter means contain non-finite values")
    if np.any(np.linalg.eigvalsh(np.asarray(rb_inputs.gamma_pred)) <= 0):
        raise ValueError("predicted covariance must be positive definite")
    if params.mean_0.shape != means.shape[-2:]:
        raise ValueError("parameter team dimensions do not match filter states")


def _daily_loglik(state: jax.Array, day, alpha, beta, max_goals: int) -> jax.Array:
    def match_body(total, match):
        home, away, home_score, away_score, valid = match
        value = loglik(
            jnp.asarray([home_score, away_score]),
            state[home],
            state[away],
            alpha,
            beta,
            max_goals,
        )
        return total + jnp.where(valid, value, 0.0), None

    total, _ = jax.lax.scan(
        match_body,
        jnp.asarray(0.0, dtype=state.dtype),
        (
            day.matches.home_id,
            day.matches.away_id,
            day.matches.home_score,
            day.matches.away_score,
            day.match_mask,
        ),
    )
    return total


def log_initial_density(params: EMParams, x_0: jax.Array) -> jax.Array:
    return gaussian_kron_logpdf(x_0 - params.mean_0, params.gamma_0, params.B)


def log_transition_density(
    params: EMParams, x_prev: jax.Array, x_next: jax.Array, dt: jax.Array
) -> jax.Array:
    phi = jnp.exp(-params.kappa * dt)
    residual = x_next - params.mean_0 - phi * (x_prev - params.mean_0)
    return gaussian_kron_logpdf(
        residual, (1.0 - phi**2) * params.gamma_0, params.B
    )


@partial(jax.jit, static_argnames=("max_goals",))
def complete_data_terms(
    params: EMParams,
    paths: jax.Array,
    model_inputs,
    max_goals: int,
) -> dict[str, jax.Array]:
    if paths.shape[0] != model_inputs.timestamp.shape[0] + 1:
        paths = paths.transpose(1, 0, 2, 3)
    initial = jnp.mean(jax.vmap(lambda x: log_initial_density(params, x))(paths[0]))

    def time_body(carry, inputs):
        x_prev, x_next, day = inputs
        dt = day.timestamp - day.timestamp_prev
        transition = jnp.mean(
            jax.vmap(lambda a, b: log_transition_density(params, a, b, dt))(
                x_prev, x_next
            )
        )
        observation = jnp.mean(
            jax.vmap(
                lambda state: _daily_loglik(
                    state, day, params.alpha, params.beta, max_goals
                )
            )(x_next)
        )
        return carry + jnp.asarray([transition, observation]), None

    totals, _ = jax.lax.scan(
        time_body,
        jnp.zeros((2,), dtype=paths.dtype),
        (paths[:-1], paths[1:], model_inputs),
    )
    return {
        "initial": initial,
        "transition": totals[0],
        "observation": totals[1],
    }


def mcem_objective(
    raw,
    mean_0,
    paths,
    model_inputs,
    max_goals: int,
    prior_scale,
    prior_dof,
):
    params = decode_EM_params(raw, mean_0)
    terms = complete_data_terms(params, paths, model_inputs, max_goals)
    prior = log_inverse_wishart_kernel(params.gamma_0, prior_scale, prior_dof)
    return terms["initial"] + terms["transition"] + terms["observation"] + prior


def _negative_objective(
    raw,
    mean_0,
    paths,
    model_inputs,
    max_goals: int,
    prior_scale,
    prior_dof,
):
    return -mcem_objective(
        raw, mean_0, paths, model_inputs, max_goals, prior_scale, prior_dof
    )


_objective_value_jit = jax.jit(mcem_objective, static_argnames=("max_goals",))
_negative_value_and_grad_jit = jax.jit(
    jax.value_and_grad(_negative_objective), static_argnames=("max_goals",)
)


def objective_diagnostics(
    params: EMParams,
    paths: jax.Array,
    model_inputs,
    max_goals: int,
    prior_scale,
    prior_dof,
) -> dict[str, jax.Array]:
    terms = complete_data_terms(params, paths, model_inputs, max_goals)
    prior = log_inverse_wishart_kernel(params.gamma_0, prior_scale, prior_dof)
    dimension = params.mean_0.size
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    scale = 1.0 - jnp.exp(-2.0 * params.kappa * dt)
    normalization = jnp.sum(
        jax.vmap(
            lambda value: -0.5 * dimension * jnp.log(2.0 * jnp.pi)
            - 0.5 * kron_logdet(value * params.gamma_0, params.B)
        )(scale)
    )
    return {
        **terms,
        "prior": prior,
        "total": terms["initial"] + terms["transition"] + terms["observation"] + prior,
        "transition_normalization": normalization,
        "transition_quadratic_penalty": terms["transition"] - normalization,
    }


def smoothed_path_diagnostics(
    params: EMParams, smoothed: SmoothedStates, model_inputs
) -> dict[str, jax.Array]:
    paths = smoothed.x
    dimension = params.mean_0.size
    initial = jax.vmap(
        lambda residual: kron_quad_batched(
            jnp.linalg.cholesky(params.gamma_0),
            jnp.linalg.cholesky(params.B),
            residual,
        )
    )(paths[0] - params.mean_0)

    def body(_, inputs):
        x_prev, x_next, dt = inputs
        phi = jnp.exp(-params.kappa * dt)
        residual = x_next - params.mean_0 - phi * (x_prev - params.mean_0)
        q_gamma = (1.0 - phi**2) * params.gamma_0
        quad = kron_quad_batched(
            jnp.linalg.cholesky(q_gamma), jnp.linalg.cholesky(params.B), residual
        )
        return None, quad

    _, transition = jax.lax.scan(
        body,
        None,
        (
            paths[:-1],
            paths[1:],
            model_inputs.timestamp - model_inputs.timestamp_prev,
        ),
    )
    return {
        "timeline_aligned": jnp.asarray(paths.shape[0] == model_inputs.timestamp.size + 1),
        "initial_mahalanobis_ratio": jnp.mean(initial) / dimension,
        "transition_mahalanobis_ratio": jnp.mean(transition) / dimension,
        "transition_mahalanobis_median": jnp.median(transition),
        "transition_mahalanobis_p05": jnp.percentile(transition, 5),
        "transition_mahalanobis_p95": jnp.percentile(transition, 95),
        "smoothed_mean": jnp.mean(paths, axis=1),
        "smoothed_variance": jnp.var(paths, axis=1),
        "lag_one_moment": jnp.mean(paths[:-1] * paths[1:], axis=1),
    }


def make_rb_smoother_filter_states(filter_states, rb_inputs, params: EMParams):
    """Adapt the filter using only means plus an O(D*N) scalar time label."""
    means = jax.lax.stop_gradient(filter_states.particles.x)
    n_times, n_components = means.shape[:2]
    if n_times != rb_inputs.timestamp.size + 1:
        raise ValueError("expected D+1 filter states for D transitions")
    particles = RBSmootherParticle(
        x=means,
        time_index=jnp.broadcast_to(
            jnp.arange(n_times)[:, None], (n_times, n_components)
        ),
    )
    component_ids = jnp.broadcast_to(
        jnp.arange(n_components), (n_times, n_components)
    )
    return ParticleFilterState(
        key=jax.random.split(jax.random.key(0), n_times),
        particles=particles,
        log_weights=filter_states.log_weights,
        ancestor_indices=component_ids,
        model_inputs=jnp.arange(n_times),
        log_normalizing_constant=filter_states.log_normalizing_constant,
    )


def rb_terminal_resampling(
    key,
    logits,
    positions,
    n,
    *,
    gamma_terminal,
    B_sqrt,
):
    """Use Cuthbert's exact systematic resampler, then draw terminal states."""
    index_key, draw_key = jax.random.split(key)
    indices, output_logits, selected = (
        cuthbertlib.resampling.systematic.resampling(
            index_key, logits, positions, n
        )
    )
    terminal = sample_kron_psd_batched(
        draw_key, selected.x, psd_sqrt(gamma_terminal), B_sqrt
    )
    return indices, output_logits, selected._replace(x=terminal)


def rb_backward_sampling_fn(
    key,
    x0_all,
    x1_all,
    log_weight_x0_all,
    log_density,
    x1_ancestor_indices,
    *,
    mean_0,
    B,
    gamma_filtered,
    gamma_pred,
    phi,
):
    del log_density, x1_ancestor_indices
    t = x0_all.time_index[0]
    gamma_t = jax.lax.dynamic_index_in_dim(gamma_filtered, t, keepdims=False)
    gamma_pred_next = jax.lax.dynamic_index_in_dim(gamma_pred, t, keepdims=False)
    phi_t = jax.lax.dynamic_index_in_dim(phi, t, keepdims=False)
    previous, components, _ = batched_backward_step(
        key,
        x0_all.x,
        log_weight_x0_all,
        x1_all.x,
        mean_0,
        gamma_t,
        gamma_pred_next,
        phi_t,
        B,
    )
    selected = jax.tree.map(lambda values: values[components], x0_all)
    return selected._replace(x=previous), components


def complete_state_joint_log_potential(
    previous,
    current,
    day,
    *,
    params,
    max_goals,
):
    dt = day.timestamp - day.timestamp_prev
    return log_transition_density(params, previous.x, current.x, dt) + _daily_loglik(
        current.x, day, params.alpha, params.beta, max_goals
    )


def build_smoother(
    params: EMParams,
    rb_inputs,
    n_smoother_particles: int,
    max_goals: int = 8,
):
    gamma_filtered = jnp.concatenate([params.gamma_0[None], rb_inputs.gamma])
    dt = rb_inputs.timestamp - rb_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    return cuthbert_build_smoother(
        log_potential=partial(
            complete_state_joint_log_potential,
            params=params,
            max_goals=max_goals,
        ),
        backward_sampling_fn=partial(
            rb_backward_sampling_fn,
            mean_0=params.mean_0,
            B=params.B,
            gamma_filtered=gamma_filtered,
            gamma_pred=rb_inputs.gamma_pred,
            phi=phi,
        ),
        resampling_fn=partial(
            rb_terminal_resampling,
            gamma_terminal=gamma_filtered[-1],
            B_sqrt=jnp.linalg.cholesky(params.B),
        ),
        n_smoother_particles=n_smoother_particles,
    )


@partial(jax.jit, static_argnames=("return_backward_probabilities",))
def _cuthbert_diagnostics_jit(
    paths,
    component_indices,
    means,
    log_weights,
    gamma_pred,
    phi,
    mean_0,
    B,
    *,
    return_backward_probabilities,
):
    n_components = means.shape[1]

    def body(_, inputs):
        means_t, weights_t, pred_t, phi_t, x_next, indices = inputs
        predicted = mean_0 + phi_t * (means_t - mean_0)
        residuals = x_next[:, None] - predicted[None]
        probabilities = jax.nn.softmax(
            weights_t[None]
            + _gaussian_kron_logpdf_from_cholesky(
                residuals, jnp.linalg.cholesky(pred_t), jnp.linalg.cholesky(B)
            ),
            axis=-1,
        )
        ess = jnp.mean(1.0 / jnp.sum(probabilities**2, axis=-1))
        entropy = jnp.mean(
            -jnp.sum(
                jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0),
                axis=-1,
            )
        )
        return None, (
            ess,
            entropy,
            jnp.max(probabilities),
            _unique_count(indices, n_components),
            probabilities,
        )

    _, outputs = jax.lax.scan(
        body,
        None,
        (
            means[:-1],
            log_weights[:-1],
            gamma_pred,
            phi,
            paths[1:],
            component_indices[:-1],
        ),
    )
    ess, entropy, maximum, unique, probabilities = outputs
    terminal_prob = jax.nn.softmax(log_weights[-1])
    terminal_probabilities = jnp.broadcast_to(
        terminal_prob, (paths.shape[1], n_components)
    )
    return BackwardDiagnostics(
        jnp.concatenate([ess, (1.0 / jnp.sum(terminal_prob**2))[None]]),
        jnp.concatenate(
            [
                entropy,
                (-jnp.sum(jnp.where(terminal_prob > 0, terminal_prob * jnp.log(terminal_prob), 0)))[None],
            ]
        ),
        jnp.concatenate([maximum, jnp.max(terminal_prob)[None]]),
        jnp.concatenate(
            [
                unique,
                _unique_count(component_indices[-1], n_components)[None],
            ]
        ),
        (
            jnp.concatenate([probabilities, terminal_probabilities[None]], axis=0)
            if return_backward_probabilities
            else None
        ),
    )


def run_cuthbert_smoother(
    key,
    filter_states,
    rb_inputs,
    params,
    n_smoother_particles,
    max_goals=8,
    return_backward_probabilities=False,
):
    validate_inputs(filter_states, rb_inputs, params)
    adapted = make_rb_smoother_filter_states(filter_states, rb_inputs, params)
    raw = cuthbert.smoother(
        build_smoother(params, rb_inputs, n_smoother_particles, max_goals),
        adapted,
        rb_inputs,
        parallel=False,
        key=key,
    )
    paths = raw.particles.x
    indices = raw.ancestor_indices
    dt = rb_inputs.timestamp - rb_inputs.timestamp_prev
    diagnostics = _cuthbert_diagnostics_jit(
        paths,
        indices,
        filter_states.particles.x,
        filter_states.log_weights,
        rb_inputs.gamma_pred,
        jnp.exp(-params.kappa * dt),
        params.mean_0,
        params.B,
        return_backward_probabilities=return_backward_probabilities,
    )
    return SmoothedStates(paths, indices, diagnostics)


def rb_backward_simulation(
    key,
    filter_states,
    rb_inputs,
    params,
    n_smoother_particles,
    return_backward_probabilities=False,
):
    return run_cuthbert_smoother(
        key,
        filter_states,
        rb_inputs,
        params,
        n_smoother_particles,
        return_backward_probabilities=return_backward_probabilities,
    )


def E_step(
    params,
    model_inputs,
    n_particles,
    n_smoother_particles,
    max_goals,
    key,
    *,
    return_backward_probabilities=False,
):
    from rbpf_v3.src.model import run_filter
    filter_key, smoother_key = jax.random.split(key)
    filter_start = time.perf_counter()
    filtered, augmented = run_filter(
        filter_key, model_inputs, params, n_particles, max_goals
    )
    jax.block_until_ready(filtered.log_normalizing_constant)
    filter_seconds = time.perf_counter() - filter_start
    backward_start = time.perf_counter()
    smoothed = run_cuthbert_smoother(
        smoother_key,
        filtered,
        augmented,
        params,
        n_smoother_particles,
        max_goals,
        return_backward_probabilities,
    )
    jax.block_until_ready(smoothed.x)
    backward_seconds = time.perf_counter() - backward_start
    return smoothed, filtered, augmented, {
        "filter_seconds": filter_seconds,
        "backward_seconds": backward_seconds,
    }


def run_mcem(
    key,
    model_inputs,
    initial_params,
    config: MCEMConfig = MCEMConfig(),
    *,
    e_step_fn=E_step,
):
    """Run MCEM with a fixed zero stationary mean and safe rollback."""
    fixed_mean = jnp.zeros_like(initial_params.mean_0)
    initial_params = initial_params._replace(mean_0=fixed_mean)
    raw = encode_EM_params(initial_params)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(raw)
    dimension = initial_params.gamma_0.shape[0] # M x M covariance matrix dimension
    prior_dof = float(dimension + PRIOR_DOF_EXTRA)
    prior_scale = (prior_dof + dimension + 1.0) * initial_params.gamma_0
    params_history = [initial_params]
    mstep_history = []
    log_marginal_history = []
    diagnostics_history = []
    timing_history = []
    rng = key
    last = None
    progress("backend=cuthbert EM loop starting")

    for epoch in range(config.n_epochs):
        epoch_start = time.perf_counter()
        rng, e_key = jax.random.split(rng)
        params = decode_EM_params(raw, fixed_mean)
        progress(
            f"epoch {epoch}/{config.n_epochs}: running E-step "
            "(filter + backward simulation)..."
        )
        smoothed, filtered, augmented, e_timing = e_step_fn(
            params,
            model_inputs,
            config.n_filter_particles,
            config.n_smoother_particles,
            config.max_goals,
            e_key,
            return_backward_probabilities=config.return_backward_probabilities,
        )
        paths = jax.lax.stop_gradient(smoothed.x)
        progress(
            f"epoch {epoch}/{config.n_epochs}: E-step complete in "
            f"{e_timing['filter_seconds'] + e_timing['backward_seconds']:.1f}s; "
            f"logZ={float(filtered.log_normalizing_constant[-1]):.3f}"
        )
        start_raw, start_opt_state = raw, opt_state
        start_value = _objective_value_jit(
            raw,
            fixed_mean,
            paths,
            model_inputs,
            config.max_goals,
            prior_scale,
            prior_dof,
        )
        jax.block_until_ready(start_value)
        mstep_start = time.perf_counter()
        progress(
            f"epoch {epoch}/{config.n_epochs}: running M-step "
            f"({config.n_gradient_steps} updates)..."
        )
        for step in range(config.n_gradient_steps):
            loss, gradient = _negative_value_and_grad_jit(
                raw,
                fixed_mean,
                paths,
                model_inputs,
                config.max_goals,
                prior_scale,
                prior_dof,
            )
            updates, opt_state = optimizer.update(gradient, opt_state, raw)
            raw = optax.apply_updates(raw, updates)
            if step % max(config.log_every_gradient_steps, 1) == 0:
                jax.block_until_ready(loss)
                progress(
                    f"epoch {epoch}/{config.n_epochs}: M-step "
                    f"{step}/{config.n_gradient_steps}; loss={float(loss):.3f}"
                )
        candidate = _objective_value_jit(
            raw,
            fixed_mean,
            paths,
            model_inputs,
            config.max_goals,
            prior_scale,
            prior_dof,
        )
        jax.block_until_ready(candidate)
        accepted = bool(
            jnp.isfinite(candidate)
            & (candidate + config.acceptance_tolerance >= start_value)
        )
        if not accepted:
            raw, opt_state, candidate = start_raw, start_opt_state, start_value
        mstep_seconds = time.perf_counter() - mstep_start
        final_epoch_params = decode_EM_params(raw, fixed_mean)
        record = objective_diagnostics(
            final_epoch_params,
            paths,
            model_inputs,
            config.max_goals,
            prior_scale,
            prior_dof,
        )
        record.update(
            {
                "epoch": epoch,
                "start_objective": start_value,
                "candidate_objective": candidate,
                "accepted": accepted,
            }
        )
        params_history.append(final_epoch_params)
        mstep_history.append(record)
        log_marginal_history.append(filtered.log_normalizing_constant[-1])
        diagnostics_history.append(
            {
                **smoothed_path_diagnostics(final_epoch_params, smoothed, augmented),
                "backward_ess": smoothed.diagnostics.ess_by_time,
                "backward_entropy": smoothed.diagnostics.entropy_by_time,
                "backward_max_probability": smoothed.diagnostics.max_probability_by_time,
                "unique_indices_by_time": smoothed.diagnostics.unique_indices_by_time,
            }
        )
        timing_history.append(
            {
                **e_timing,
                "mstep_seconds": mstep_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        progress(
            f"epoch {epoch}/{config.n_epochs}: M-step complete in "
            f"{mstep_seconds:.1f}s; accepted={accepted}"
        )
        last = (smoothed, filtered, augmented)

    final_params = decode_EM_params(raw, fixed_mean)
    rng, final_key = jax.random.split(rng)
    final_smoothed, final_filtered, final_augmented, final_timing = e_step_fn(
        final_params,
        model_inputs,
        config.n_filter_particles,
        config.n_smoother_particles,
        config.max_goals,
        final_key,
        return_backward_probabilities=config.return_backward_probabilities,
    )
    progress("backend=cuthbert MCEM complete")
    return {
        "backend": "cuthbert",
        "final_params": final_params,
        "params_history": params_history,
        "mstep_history": mstep_history,
        "log_marginal_history": log_marginal_history,
        "diagnostics_history": diagnostics_history,
        "timing_history": timing_history,
        "final_timing": final_timing,
        "final_log_marginal_likelihood": final_filtered.log_normalizing_constant[-1],
        "final_filter_states": final_filtered,
        "final_augmented_data": final_augmented,
        "final_smoothed_states": final_smoothed,
        "config": config._asdict(),
    }


run_EM = run_mcem


__all__ = [
    "BackwardDiagnostics",
    "SmoothedStates",
    "MCEMConfig",
    "RBSmootherParticle",
    "E_step",
    "backward_shared_terms",
    "batched_backward_step",
    "complete_data_terms",
    "build_smoother",
    "make_rb_smoother_filter_states",
    "rb_backward_sampling_fn",
    "rb_terminal_resampling",
    "run_cuthbert_smoother",
    "gaussian_kron_logpdf",
    "kron_logdet",
    "kron_quad_batched",
    "psd_sqrt",
    "rb_backward_simulation",
    "run_mcem",
    "sample_kron_psd_batched",
    "validate_inputs",
]
