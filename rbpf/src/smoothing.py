from rbpf.src.model import run_filter
from rbpf.src.utils import EMParams, RBPFState, RBPFFootballResults, FootballResults, RawEMParams
from rbpf.src.helpers import default_init_params
from rbpf.src.bivariate_poisson import loglik
from rbpf.src.data import ACTIVE_TEAMS, get_results, WORLDCUP_2026_TEAMS
from rbpf.src.helpers import (
    decode_EM_params,
    encode_EM_params,
    log_inverse_wishart_kernel,
    parameter_diagnostics,
    save_em_results,
    timeline_diagnostics,
)
from rbpf.src.graphic import plot_em_results

import jax
import cuthbert
import cuthbertlib
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import optax
import os
import json
from cuthbert.smc.backward_sampler import build_smoother
from cuthbertlib.smc.smoothing.exact_sampling import simulate as exact_sampling_simulate

"""
optimizer should be outside of M-step -> so that it continues from the previous step rather than re-initializing the optimizer each time. This is important for optimizers like Adam that maintain state across steps.
"""

def _kron_quad(A, B, V):
    """v_i^T (A (x) B)^{-1} v_i for each row v_i of V.

    V: (N, M*K) where each row is vec_C(S_i) of an (M, K) matrix S_i.
    A: (M, M), B: (K, K). Returns (N,).
    """
    K = B.shape[0]
    N = V.shape[0]
    S = V.reshape(N, -1, K)  # (N, M, K)
    B_inv = jnp.linalg.inv(B)  # (K, K), K is tiny
    # Treat particles/paths and traits as multiple right-hand sides. This
    # factorizes A once instead of broadcasting A and repeating the same solve
    # N times.
    M = S.shape[1]
    rhs = S.transpose(1, 0, 2).reshape(M, N * K)
    A_inv_rhs = jnp.linalg.solve(A, rhs)
    A_inv_S = A_inv_rhs.reshape(M, N, K).transpose(1, 0, 2)
    St_Ainv_S = jnp.matmul(S.transpose(0, 2, 1), A_inv_S)  # (N, K, K)
    # tr(S^T A^{-1} S B^{-T}) = sum((S^T A^{-1} S) * B^{-1})
    return jnp.sum(St_Ainv_S * B_inv[None], axis=(-2, -1))


def _kron_logdet(A, B):
    """logdet(A (x) B) = K logdet(A) + M logdet(B). A: (M, M), B: (K, K)."""
    M = A.shape[0]
    K = B.shape[0]
    _, logdet_A = jnp.linalg.slogdet(A)
    _, logdet_B = jnp.linalg.slogdet(B)
    return K * logdet_A + M * logdet_B

def _psd_sqrt(A):
    values, vectors = jnp.linalg.eigh(0.5 * (A + A.T))
    # Only roundoff-sized negative values should reach this point.
    values = jnp.maximum(values, 0.0)
    return (vectors * jnp.sqrt(values)[None, :]) @ vectors.T

######################## E-step: RBPF filter + smoother ########################

# def materialize_rb_filter(
#     key: jax.Array,
#     filtered_states: RBPFState,
#     gamma: jax.Array,
#     gamma_0: jax.Array,
#     B: jax.Array,
# ):
#     """
#     Convert RB Gaussian-component means into a fixed full-state particle cloud.

#     filtered_states.particles.x: (D+1, N, num_teams, 2)
#     gamma:                      (D, num_teams, num_teams)
#     """
#     means = filtered_states.particles.x
#     # State 0 is the prior. State d+1 is the posterior after day d.
#     component_gammas = jnp.concatenate([gamma_0[None], gamma],axis=0,)

#     if component_gammas.shape[0] != means.shape[0]:
#         raise ValueError(
#             "Covariance and filter timelines are inconsistent: "
#             f"{component_gammas.shape[0]} versus {means.shape[0]}"
#         )

#     L_gamma = jax.vmap(_psd_sqrt)(component_gammas)
#     L_B = _psd_sqrt(B)
#     # One independent key for every time/component pair.
#     keys = jax.random.split(key, means.shape[:2])
#     def materialize_time(keys_t, means_t, L_gamma_t):
#         def materialize_component(component_key, component_mean):
#             # sample from N(0, I) and transform to N(mean, gamma_t (x) B)
#             noise = jax.random.normal(component_key,component_mean.shape)
#             return (component_mean + L_gamma_t @ noise @ L_B.T)
#         return jax.vmap(materialize_component)(keys_t,means_t)
#     # generate full states based on the means and covariances from the RBPF filter
#     full_states = jax.vmap(materialize_time)(keys,means,L_gamma,)
#     # Only replace particles.x. Keep weights, ancestors and inputs unchanged.
#     materialized_particles = filtered_states.particles._replace(x=full_states)
#     return filtered_states._replace(particles=materialized_particles)

def joint_log_potential(
    x_prev: jax.Array,
    x: jax.Array,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    max_goals: int,
) -> jax.Array:
    """
    transition + observation log-potential for a single time step.
    
    this is log p(x_t | x_{t-1}) + sum_m log p(y_t^m | x_t^m) for a single time step t.
    """
    # --- Transition term: log p(x_t | x_{t-1}) ---
    # OU: x_t = mu_0 + phi (x_{t-1} - mu_0) + eps,  eps ~ N(0, Q)

    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)

    pred_mean = params.mean_0 + phi * (x_prev - params.mean_0)  # (M, 2)
    delta = (x - pred_mean).reshape(-1)  # (2M,)
    dim = delta.size

    # now there wont be issue of phi = 1.0 when dt = 0 because the teams are grouped into a single observation
    var_scale = 1.0 - phi**2
    # (x_t  - pred_mean)^T (var_scale * gamma_0 (x) B)^{-1} (x_t - pred_mean)
    quad = _kron_quad(var_scale * params.gamma_0, params.B, delta[None])[0]
    # determinant term: logdet(var_scale * gamma_0 (x) B)
    log_det = _kron_logdet(var_scale * params.gamma_0, params.B)
    log_transition = (
        -0.5 * dim * jnp.log(2 * jnp.pi)
        - 0.5 * log_det
        - 0.5 * quad
    )
    # --- Observation term: sum_m log p(y_t^m | x_t^m) ---
    home_id = model_inputs.matches.home_id        # (M_max,)
    away_id = model_inputs.matches.away_id        # (M_max,)
    home_score = model_inputs.matches.home_score  # (M_max,)
    away_score = model_inputs.matches.away_score  # (M_max,)
    valid = model_inputs.match_mask               # (M_max,) bool

    def log_match(_, match):
        h, a, yh, ya, v = match
        y = jnp.array([yh, ya])                  # observed goals (2,)
        x_i = x[h]                               # (2,) home attack/defence
        x_j = x[a]                               # (2,) away attack/defence
        ll = loglik(y, x_i, x_j, alpha=params.alpha, beta=params.beta,
                    max_goals=max_goals, scale=1.0)
        return _, jnp.where(v, ll, 0.0)          # padded matches contribute 0

    _, logliks = jax.lax.scan(
        log_match, None, (home_id, away_id, home_score, away_score, valid)
    )
    log_observation = jnp.sum(logliks)

    return log_transition + log_observation

def E_step(
    params: EMParams,
    model_inputs: FootballResults,
    n_particles: int,
    n_smoother_particles: int,
    max_goals: int,
    key: jax.Array,
):
    key, filter_key, materialize_key, smoother_key = jax.random.split(key, 4)
    # 1. run the RBPF filter
    filtered_states, model_inputs_rbpf = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals
    )

    # # 2. materialize a fixed full state cloud 
    # materialized_states = materialize_rb_filter(
    #     materialize_key, filtered_states, model_inputs_rbpf.gamma,
    #     params.gamma_0, params.B
    # )

    # 3. build an run the smoother
    # smoother_obj = build_smoother(
    #     log_potential=partial(
    #         lambda sp, s, mi, p: joint_log_potential(sp.x, s.x, mi, p, max_goals),
    #         p=params,
    #     ),
    #     backward_sampling_fn=exact_sampling_simulate,
    #     resampling_fn=cuthbertlib.resampling.systematic.resampling,
    #     n_smoother_particles=n_smoother_particles,
    # )
    smoothed_states = cuthbert.smoother(
        smoother_obj,
        filtered_states,
        model_inputs_rbpf,
        parallel=False,
        key=smoother_key
    )
    smoothed_states = cuthbert.smoother(
        smoother_obj, 
        materialized_states, 
        model_inputs_rbpf, 
        False, 
        smoother_key
    )
    return smoothed_states, filtered_states, materialized_states, model_inputs_rbpf

######### M-step: gradient-based optimization of complete-data log-likelihood ####################

# log initial density: log p(x_0 | theta)
def log_initial_density(params: EMParams, x_0: jax.Array):
    delta = (x_0 - params.mean_0).reshape(-1)  # (2M,)
    dim = delta.size

    quad = _kron_quad(params.gamma_0, params.B, delta[None])[0]
    log_det = _kron_logdet(params.gamma_0, params.B)
    return (
        -0.5 * dim * jnp.log(2 * jnp.pi)
        - 0.5 * log_det
        - 0.5 * quad
    )

# log transition density: log p(x_t | x_{t-1}, theta)
def log_transition_density(
    params: EMParams,
    x_prev: jax.Array,
    x_next: jax.Array,
    day_inputs: RBPFFootballResults,
) -> jax.Array:
    dt = day_inputs.timestamp - day_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    variance_scale = 1.0 - phi**2

    predicted_mean = (params.mean_0 + phi * (x_prev - params.mean_0))

    residual = (x_next - predicted_mean).reshape(-1)
    dimension = residual.size

    team_covariance = variance_scale * params.gamma_0

    quad = _kron_quad(team_covariance, params.B, residual[None], )[0]

    log_det = _kron_logdet(team_covariance, params.B)

    return (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * log_det
        -0.5 * quad
    )

def materialized_cloud_diagnostics(
    params: EMParams,
    materialized_states,
) -> dict:
    """Validate the fixed initial full-state cloud before backward selection."""
    initial = jnp.asarray(materialized_states.particles.x[0])
    n_particles = initial.shape[0]
    dimension = params.mean_0.size
    residual = (initial - params.mean_0).reshape(n_particles, dimension)
    mahalanobis = _kron_quad(params.gamma_0, params.B, residual)

    empirical_mean = jnp.mean(residual, axis=0)
    centered = residual - empirical_mean
    denominator = jnp.maximum(n_particles - 1, 1)
    empirical_covariance = centered.T @ centered / denominator
    expected_covariance = jnp.kron(params.gamma_0, params.B)
    relative_covariance_error = (
        jnp.linalg.norm(empirical_covariance - expected_covariance)
        / jnp.linalg.norm(expected_covariance)
    )
    ratio = jnp.mean(mahalanobis) / dimension

    return {
        "n_particles": n_particles,
        "latent_dimension": dimension,
        "initial_mahalanobis_mean": jnp.mean(mahalanobis),
        "initial_mahalanobis_median": jnp.median(mahalanobis),
        "initial_mahalanobis_ratio": ratio,
        "initial_mean_error_l2": jnp.linalg.norm(empirical_mean),
        "initial_covariance_relative_frobenius_error": relative_covariance_error,
        "rb_means_as_states_suspected": ratio < 0.1,
    }


def smoothed_path_diagnostics(
    params: EMParams,
    smoothed_states,
    model_inputs: RBPFFootballResults,
) -> dict:
    """Check whether smoother paths behave like complete OU state draws.

    The latent dimension is an unconditional OU reference for these
    Mahalanobis statistics. Backward selection conditions on all observations,
    so a smoothed-path ratio need not equal one exactly.
    """
    paths = jax.lax.stop_gradient(
        smoothed_states.particles.x.transpose(1, 0, 2, 3)
    )
    n_paths, n_states, num_teams, n_traits = paths.shape
    dimension = num_teams * n_traits

    initial_residual = (paths[:, 0] - params.mean_0).reshape(n_paths, -1)
    initial_quad = _kron_quad(
        params.gamma_0,
        params.B,
        initial_residual,
    )
    initial_logdet = _kron_logdet(params.gamma_0, params.B)
    initial_normalization = (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * initial_logdet
    )

    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    variance_scale = 1.0 - phi**2
    previous_by_time = paths[:, :-1].transpose(1, 0, 2, 3)
    next_by_time = paths[:, 1:].transpose(1, 0, 2, 3)

    def transition_quad_at_time(values):
        phi_t, scale, previous, current = values
        predicted = (
            params.mean_0[None]
            + phi_t * (previous - params.mean_0[None])
        )
        residual = current - predicted
        return _kron_quad(
            scale * params.gamma_0,
            params.B,
            residual.reshape(n_paths, -1),
        )

    # lax.map limits peak memory compared with materializing all residuals and
    # broadcasting a solve across every time/path pair simultaneously.
    transition_quad_by_time_path = jax.lax.map(
        transition_quad_at_time,
        (phi, variance_scale, previous_by_time, next_by_time),
    )

    transition_logdet = jax.vmap(
        lambda scale: _kron_logdet(
            scale * params.gamma_0,
            params.B,
        )
    )(variance_scale)
    transition_normalization = (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * transition_logdet
    )

    flat_transition_quad = transition_quad_by_time_path.reshape(-1)
    initial_ratio = jnp.mean(initial_quad) / dimension
    transition_ratio = jnp.mean(flat_transition_quad) / dimension

    ancestor_indices = jnp.asarray(smoothed_states.ancestor_indices)

    def unique_count(indices):
        sorted_indices = jnp.sort(indices)
        return 1 + jnp.sum(sorted_indices[1:] != sorted_indices[:-1])

    unique_particles = jax.vmap(unique_count)(ancestor_indices)

    return {
        "n_paths": n_paths,
        "n_states": n_states,
        "latent_dimension": dimension,
        "timeline_aligned": n_states == dt.size + 1,
        "initial_mahalanobis_mean": jnp.mean(initial_quad),
        "initial_mahalanobis_median": jnp.median(initial_quad),
        "initial_mahalanobis_min": jnp.min(initial_quad),
        "initial_mahalanobis_max": jnp.max(initial_quad),
        "initial_mahalanobis_ratio": initial_ratio,
        "initial_normalization": initial_normalization,
        "initial_log_density_mean": (
            initial_normalization - 0.5 * jnp.mean(initial_quad)
        ),
        "transition_mahalanobis_mean": jnp.mean(flat_transition_quad),
        "transition_mahalanobis_median": jnp.median(flat_transition_quad),
        "transition_mahalanobis_p05": jnp.percentile(flat_transition_quad, 5),
        "transition_mahalanobis_p95": jnp.percentile(flat_transition_quad, 95),
        "transition_mahalanobis_ratio": transition_ratio,
        "transition_normalization_sum": jnp.sum(transition_normalization),
        "transition_quadratic_penalty_mean_path": (
            -0.5 * jnp.mean(jnp.sum(transition_quad_by_time_path, axis=0))
        ),
        "transition_log_density_mean_path": (
            jnp.sum(transition_normalization)
            - 0.5 * jnp.mean(jnp.sum(transition_quad_by_time_path, axis=0))
        ),
        "unique_smoother_particles_min": jnp.min(unique_particles),
        "unique_smoother_particles_mean": jnp.mean(unique_particles),
        "unique_smoother_particles_final": unique_particles[-1],
    }


def density_decomposition_diagnostics(
    params: EMParams,
    model_inputs: RBPFFootballResults,
    initial_log_density: jax.Array,
    transition_log_density: jax.Array,
) -> dict:
    """Recover normalization and Mahalanobis terms from saved log densities."""
    dimension = params.mean_0.size
    initial_normalization = (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * _kron_logdet(params.gamma_0, params.B)
    )

    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    variance_scale = 1.0 - jnp.exp(-2.0 * params.kappa * dt)
    transition_normalization = jnp.sum(
        jax.vmap(
            lambda scale: (
                -0.5 * dimension * jnp.log(2.0 * jnp.pi)
                -0.5 * _kron_logdet(scale * params.gamma_0, params.B)
            )
        )(variance_scale)
    )
    initial_quad = 2.0 * (initial_normalization - initial_log_density)
    transition_quad = 2.0 * (
        transition_normalization - transition_log_density
    )
    n_transitions = dt.size

    return {
        "initial_normalization": initial_normalization,
        "initial_mahalanobis_mean": initial_quad,
        "initial_mahalanobis_ratio": initial_quad / dimension,
        "transition_normalization_sum": transition_normalization,
        "transition_quadratic_penalty": -0.5 * transition_quad,
        "transition_mahalanobis_sum": transition_quad,
        "transition_mahalanobis_mean": transition_quad / n_transitions,
        "transition_mahalanobis_ratio": (
            transition_quad / (n_transitions * dimension)
        ),
    }


# log observation density: sum_m log p(y_t^m | x_t^m, theta)
def log_observation_density(
    params: EMParams,
    x_day: jax.Array,
    day_inputs: RBPFFootballResults,
    max_goals: int,
) -> jax.Array:
    matches = day_inputs.matches

    def one_match(_, match):
        home, away, home_score, away_score, valid = match
        value = jax.lax.cond(valid,
            lambda _: loglik(
                jnp.array([home_score, away_score]),
                x_day[home],
                x_day[away],
                alpha=params.alpha,
                beta=params.beta,
                max_goals=max_goals,
                scale=1.0,
            ),
            lambda _: jnp.array(
                0.0,
                dtype=x_day.dtype,
            ),
            operand=None,
        )

        return None, value

    _, values = jax.lax.scan(
        one_match,
        None,
        (
            matches.home_id,
            matches.away_id,
            matches.home_score,
            matches.away_score,
            day_inputs.match_mask,
        ),
    )

    return jnp.sum(values)

def log_joint_density_terms(
    params: EMParams,
    paths: jax.Array,
    model_inputs: RBPFFootballResults,
    max_goals: int,
):
    """
    path[0]   is the timestamp-zero prior state.
    path[d+1] is the state for observed day d.
    """
    initial = log_initial_density(params, paths[0])

    transition, observation = jax.vmap(
        lambda x_prev, x_day, day_inputs: (
            log_transition_density(
                params,
                x_prev,
                x_day,
                day_inputs,
            ),
            log_observation_density(
                params,
                x_day,
                day_inputs,
                max_goals,
            ),
        )
    )(paths[:-1], paths[1:],model_inputs)

    return (initial, jnp.sum(transition), jnp.sum(observation))

def log_joint_density(
    params: EMParams,
    path: jax.Array,
    model_inputs: RBPFFootballResults,
    max_goals: int,
) -> jax.Array:
    initial, transition, observation = log_joint_density_terms(
        params,
        path,
        model_inputs,
        max_goals,
    )

    return initial + transition + observation

def M_step(
    smoothed_states: jax.Array,
    model_inputs: RBPFFootballResults,
    raw_params: RawEMParams,
    fixed_mean_0: jax.Array,
    prior_scale: jax.Array,
    prior_dof: float,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    max_goals: int,
    n_gradient_steps: int = 501,
):
    # smoothed_states.particles.x: (T+1, M, num_teams, 2) -> (M, T+1, num_teams, 2)
    paths = jax.lax.stop_gradient(
        smoothed_states.particles.x.transpose(1, 0, 2, 3)
    )

    def loss_fn(param_raw: RawEMParams):
        candidate_param = decode_EM_params(param_raw, fixed_mean_0=fixed_mean_0)
        initial, transition, observation = jax.vmap(
            lambda path: log_joint_density_terms(
                candidate_param,
                path,
                model_inputs,
                max_goals=max_goals
            )
        )(paths)
        log_joint = (jnp.mean(initial) + jnp.mean(transition) + jnp.mean(observation))

        # Recalculated for every candidate Gamma_0.
        log_prior = log_inverse_wishart_kernel(
            candidate_param.gamma_0,
            scale=prior_scale,
            dof=prior_dof,
        )

        log_posterior_objective = log_joint + log_prior

        loss = -log_posterior_objective
        # If we want to add a inverse-wishard prior to constrain the covariance matrices, we can add it here. For now, we just return the negative log-likelihood as the loss.
        diagnostics = (
            jnp.mean(initial),
            jnp.mean(transition),
            jnp.mean(observation),
            log_prior
        )
        return loss, diagnostics
    # jax.value_and_grad(loss_fn, has_aux=True) returns a function that computes both the value of loss_fn and its gradient with respect to param_raw. The has_aux=True argument indicates that loss_fn returns auxiliary data (diagnostics) in addition to the main loss value.
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    starting_raw = raw_params
    starting_opt_state = opt_state

    initial_loss, initial_terms = loss_fn(raw_params)

    for _ in range(n_gradient_steps):
        (loss, terms), grads = value_and_grad_fn(raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)

    candidate_loss, candidate_terms = loss_fn(raw_params)

    accepted = bool(
        jnp.isfinite(candidate_loss)
        & (candidate_loss <= initial_loss + 1e-6)
    )
    if accepted:
        final_loss = candidate_loss
        final_terms = candidate_terms
    else:
        raw_params = starting_raw
        opt_state = starting_opt_state
        final_loss = initial_loss
        final_terms = initial_terms

    final_params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)
    final_initial_term, final_transition_term, final_observation_term, final_prior_term = final_terms
    decomposition = density_decomposition_diagnostics(
        final_params,
        model_inputs,
        initial_log_density=final_initial_term,
        transition_log_density=final_transition_term,
    )

    diagnostics = {
        "accepted": accepted,
        "initial_objective": initial_loss,
        "candidate_objective": candidate_loss,
        "final_objective": final_loss,
        "initial_log_density": final_initial_term,
        "transition_log_density": final_transition_term,
        "observation_log_density": final_observation_term,
        "prior_log_density": final_prior_term,
        "density_decomposition": decomposition,
    }
    final_raw_params = raw_params
    return final_raw_params, final_params, opt_state, diagnostics


############## EM Algorithm: E-step + M-step ##############

def run_EM(
    model_inputs: FootballResults,
    init_params: EMParams,
    n_particles: int,
    n_smoother_particles: int,
    n_epochs: int,
    max_goals: int,
    learning_rate: float = 1e-3,
    n_gradient_steps: int = 50,
    key: jax.Array = jax.random.PRNGKey(0)
):
    run_key = key
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    invalid_dt = jnp.where(dt <= 0)[0]
    if invalid_dt.size > 0:
        raise ValueError(
            "Every observed day must occur strictly after its previous state. "
            f"Found dt <= 0 at day indices "
            f"{jax.device_get(invalid_dt).tolist()}."
        )

    
    # mean_0 is fixed for all runs
    fixed_mean_0 = jax.lax.stop_gradient(init_params.mean_0)
    gamma_0_prior = jax.lax.stop_gradient(init_params.gamma_0)
    raw_params = encode_EM_params(init_params)
    params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)

    gamma_0_prior = jax.lax.stop_gradient(init_params.gamma_0)
    dim = gamma_0_prior.shape[0]
    prior_dof = float(dim + 10)
    prior_scale = jax.lax.stop_gradient(
        (prior_dof + dim + 1.0) * gamma_0_prior
    )

    # optimizer initialization
    optimizer = optax.adam(
        learning_rate=learning_rate
    )
    
    opt_state = optimizer.init(raw_params)

    params_history = [params]
    log_marginal_history = []
    mstep_history = []
    diagnostics_history = []
    timeline_summary = timeline_diagnostics(model_inputs)

    for epoch in range(n_epochs):
        key, e_step_key = jax.random.split(key)
        print(f"\n[EM] Epoch {epoch + 1}/{n_epochs}")

        # ------------
        # E-step for theta^k
        # -----------
        (
            smoothed_states,
            filtered_states,
            materialized_states,
            model_inputs_rbpf,
        ) = E_step(
            params=params,
            model_inputs=model_inputs,
            n_particles=n_particles,
            n_smoother_particles=n_smoother_particles,
            max_goals=max_goals,
            key=e_step_key,
        )
        log_marginal_history.append(filtered_states.log_normalizing_constant[-1])
        print(
            f"  [E-step] log marginal likelihood: "
            f"{filtered_states.log_normalizing_constant[-1]:.4f}"
        )
        e_step_diagnostics = smoothed_path_diagnostics(
            params,
            smoothed_states,
            model_inputs_rbpf,
        )
        cloud_diagnostics = materialized_cloud_diagnostics(
            params,
            materialized_states,
        )
        if bool(cloud_diagnostics["rb_means_as_states_suspected"]):
            raise RuntimeError(
                "The initial materialized cloud has near-zero Mahalanobis "
                "variation. RB component means appear to be used as full "
                "latent states."
            )
        if not bool(e_step_diagnostics["timeline_aligned"]):
            raise RuntimeError(
                "Smoothed-state and transition timelines are not aligned."
            )
        e_step_parameter_diagnostics = parameter_diagnostics(params)
        # -----------
        # M-step on fixed paths sampled under theta^k
        # ----------
        updated_raw_params, updated_params, updated_opt_state, mstep_diagnostics = M_step(
            smoothed_states=smoothed_states,
            raw_params=raw_params,
            fixed_mean_0=fixed_mean_0,
            prior_scale=prior_scale,
            prior_dof=prior_dof,
            model_inputs=model_inputs_rbpf,
            opt_state=opt_state,
            optimizer=optimizer,
            max_goals=max_goals,
            n_gradient_steps=n_gradient_steps,
        )
        print(
            f"  [M-step] objective: "
            f"{mstep_diagnostics['initial_objective']:.4f} -> "
            f"{mstep_diagnostics['final_objective']:.4f} "
            f"({'accepted' if mstep_diagnostics['accepted'] else 'rejected'})"
        )

        # update for new epoch
        raw_params = updated_raw_params
        params = updated_params
        opt_state = updated_opt_state

        # add to history
        params_history.append(params)
        mstep_history.append({
            "epoch": epoch,
            **mstep_diagnostics
        })
        diagnostics_history.append({
            "epoch": epoch,
            "e_step_parameter_index": epoch,
            "updated_parameter_index": epoch + 1,
            "materialized_cloud": cloud_diagnostics,
            "smoother_paths": e_step_diagnostics,
            "e_step_parameters": e_step_parameter_diagnostics,
            "updated_parameters": parameter_diagnostics(params),
            "m_step_density": mstep_diagnostics["density_decomposition"],
        })

    # Run the final E-step to get the final log-marginal likelihood
    print("\n[EM] Running final E-step...")
    key, final_filter_key = jax.random.split(key)
    final_filtered_state, model_inputs_rbpf = run_filter(
        key=final_filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals
    )
    final_log_marginal_likelihood = final_filtered_state.log_normalizing_constant[-1]
    log_marginal_history.append(final_log_marginal_likelihood)

    print(f"[EM] Final log-marginal likelihood: {final_log_marginal_likelihood:.4f}")
    print("[EM] EM run complete.")

    return {
        "final_params": params,
        "opt_state": opt_state,
        "params_history": params_history,
        "log_marginal_history": log_marginal_history,
        "mstep_history": mstep_history,
        "diagnostics_history": diagnostics_history,
        "final_log_marginal_likelihood": final_log_marginal_likelihood,
        "run_metadata": {
            "n_filter_particles": n_particles,
            "n_smoother_particles": n_smoother_particles,
            "n_epochs": n_epochs,
            "n_gradient_steps": n_gradient_steps,
            "learning_rate": learning_rate,
            "max_goals": max_goals,
            "n_teams": params.mean_0.shape[0],
            "latent_dimension": params.mean_0.size,
            "initial_prng_key": run_key,
            "materialized_full_state_cloud": True,
            "timeline": timeline_summary,
        },
        "final_filtered_state": final_filtered_state,
        "model_inputs_rbpf": model_inputs_rbpf,
    }
#################################################

def main():
    print("============= Running EM Algorithm ===============")
    start_date = "2000-01-01"
    end_date = "2026-01-01"
    teams = WORLDCUP_2026_TEAMS
    N = 100
    MAX_GOALS = 8
    
    df, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=teams,
    )
    num_teams = len(team_id_to_name)
    key = jax.random.PRNGKey(42)

    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)
    
    print(f"Data loaded: {len(df)} matches from {start_date} to {end_date} for {num_teams} teams. Timestamps: {len(model_inputs.timestamp)}. Max goals: {MAX_GOALS}.")

    key, subkey = jax.random.split(key)
    results = run_EM(
        model_inputs=model_inputs,
        init_params=params,
        n_particles=N,
        n_smoother_particles=N,
        n_epochs=5,
        max_goals=MAX_GOALS,
        learning_rate=1e-3,
        n_gradient_steps=20,
        key=subkey
    )

    from rbpf.src.graphic import plot_all
    plot_all(
        filtered_states=results["final_filtered_state"],
        augmented_results=results["model_inputs_rbpf"],
        team_id_to_name=team_id_to_name,
        top_n = 10,
        save_path = './rbpf/outputs/smoothing',
        timestamps=df["date"].to_numpy(),
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "smoothing"
    )
    save_em_results(results, output_dir)
    plot_em_results(results, output_dir)

if __name__ == "__main__":
    main()

# def _complete_log_likelihood_terms(params, X, model_inputs):


# def _complete_log_likelihood(params, X, model_inputs):
#     init_ll, trans_ll, obs_ll = _complete_log_likelihood_terms()
#     # total loss
#     return init_ll + trans_ll + obs_ll

# def M_step(
#     smoothed_states: jax.Array,
#     model_inputs: RBPFFootballResults,
#     prev_params: EMParams,
#     learning_rate: float = 1e-3,
#     n_gradient_steps: int = 1,
# ):
    
#     return final_params, loss_trace

# @partial(jax.jit, static_argnames=("n_smoother_particles",))
# def _eval_loss_components(
#     smoothed_states: jax.Array,
#     model_inputs: RBPFFootballResults,
#     params: EMParams,
#     n_smoother_particles: int,
# ):
#     """Mean complete-data log-likelihood component scores over smoothed paths.

#     ``smoothed_states.particles.x`` has shape ``(T+1, M, num_teams, 2)`` and is
#     rearranged to ``(M, T+1, num_teams, 2)`` so each row is one smoothed
#     trajectory. Returns the mean (over M smoother particles) of each term:
#     ``(init_mean, trans_mean, obs_mean)``.
#     """
#     trajectories = smoothed_states.particles.x.transpose(1, 0, 2, 3)
#     init_ll, trans_ll, obs_ll = jax.vmap(
#         lambda X: _complete_log_likelihood_terms(params, X, model_inputs)
#     )(jax.lax.stop_gradient(trajectories))
#     return jnp.mean(init_ll), jnp.mean(trans_ll), jnp.mean(obs_ll)

# def run_EM(
#     n_epochs: int,
#     init_params: EMParams,
#     model_inputs: FootballResults,
#     n_particles: int,
#     n_smoother_particles: int,
#     num_teams: int,
#     learning_rate: float = 1e-3,
#     n_gradient_steps: int = 1,
#     key: jax.Array = jax.random.PRNGKey(0),
# ):


# def main():
#     # --- Load data ---
#     from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
#     data, model_inputs, team_id_to_name = get_results(
#         start_date="2000-01-01",
#         end_date="2026-01-01",
#         max_goals=MAX_GOALS,
#         include_friendly=False,
#         teams_only=WORLDCUP_2026_TEAMS,
#     )
#     print("DataFrame head:")
#     print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(3))
#     print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(3))
#     num_teams = len(team_id_to_name)
#     key = jax.random.PRNGKey(42)

#     params = default_init_params(num_teams=num_teams, team_id_to_name=None)

#     # best_params, params_track, log_marginal_likelihoods, loss_components, loss_traces = run_EM(
#     #     n_epochs=5,
#     #     init_params=params,
#     #     model_inputs=model_inputs,
#     #     n_particles=100,
#     #     n_smoother_particles=100,
#     #     num_teams=num_teams,
#     #     learning_rate=1e-3,
#     #     n_gradient_steps=10,
#     #     key=key,
#     # )

#     # --- Save results and plot diagnostics ---
#     import os
#     output_dir = os.path.join(
#         os.path.dirname(__file__), "..", "outputs", "smoothing"
#     )
#     save_em_results(
#         best_params=best_params,
#         log_marginal_likelihoods=log_marginal_likelihoods,
#         output_dir=output_dir,
#         extra={
#             "n_epochs": 5,
#             "n_particles": 100,
#             "n_smoother_particles": 100,
#             "num_teams": num_teams,
#             "learning_rate": 1e-3,
#             "n_gradient_steps": 10,
#             # per-epoch complete-data loss contribution of each term
#             "loss_components": {
#                 k: jnp.asarray(v).tolist() for k, v in loss_components.items()
#             },
#             # per-gradient-step loss trace per epoch: (n_epochs, n_gradient_steps, 4)
#             # columns = [total, init, transition, observation]
#             "loss_traces": jnp.asarray(loss_traces).tolist(),
#         },
#     )
#     plot_log_marginal_likelihood_curve(
#         log_marginal_likelihoods,
#         save_path=os.path.join(output_dir, "em_log_marginal_likelihood_curve.png"),
#     )
#     plot_loss_components(
#         loss_traces,
#         save_path=os.path.join(output_dir, "em_loss_components_curve.png"),
#     )
#     print("Done.")
# if __name__ == "__main__":
#     main()
