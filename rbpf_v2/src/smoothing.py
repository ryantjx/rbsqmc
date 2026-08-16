from __future__ import annotations

import math
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

from .bivariate_poisson import daily_loglik
from .helpers import decode_EM_params, encode_EM_params, log_inverse_wishart_kernel
from .kron import kron_logdet, kron_quad, rts_kron_terms, sample_kron_psd, symmetrize
from .model import run_filter, systematic_resample
from .utils import EMParams, ParticleMeans, SmoothedStates


def _log(message: str) -> None:
    """Timestamped, unbuffered progress log (visible with ``python -u``)."""
    print(f"[smoothing {time.strftime('%H:%M:%S')}] {message}", flush=True)


class MCEMConfig(NamedTuple):
    n_filter_particles: int = 32
    n_smoother_particles: int = 32
    n_epochs: int = 1
    n_gradient_steps: int = 2
    learning_rate: float = 1e-2
    max_goals: int = 8
    acceptance_tolerance: float = 1e-6


def gaussian_kron_logpdf(residual, gamma, B):
    dimension = gamma.shape[0] * B.shape[0]
    return -0.5 * (
        dimension * jnp.log(2.0 * jnp.pi)
        + kron_logdet(gamma, B)
        + kron_quad(gamma, B, residual)
    )


def backward_statistics(means_t, log_weights_t, x_next, mean_0,
                        gamma_filtered, gamma_pred_next, B, phi):
    """RB backward logits plus the shared conditional team covariance and gain."""
    predicted = mean_0 + phi * (means_t - mean_0)
    residuals = x_next[None] - predicted
    compatibility = gaussian_kron_logpdf(residuals, gamma_pred_next, B)
    logits = log_weights_t + compatibility
    J_gamma, gamma_cond = rts_kron_terms(gamma_filtered, gamma_pred_next, phi)
    return logits, predicted, J_gamma, gamma_cond


def terminal_resample(key, means, log_weights, gamma_filtered, B, n_paths):
    """Select mixture components and independently materialize full terminal states."""
    key, index_key, draw_key = jax.random.split(key, 3)
    indices = systematic_resample(index_key, log_weights, n_paths)
    keys = jax.random.split(draw_key, n_paths)
    draws = jax.vmap(lambda k, m: sample_kron_psd(k, m, gamma_filtered, B))(
        keys, means[indices]
    )
    return draws, indices


def rb_backward_simulation(key, filter_states, augmented_data, params: EMParams,
                           n_smoother_particles: int):
    """Draw temporally coherent complete states from the RB filtering mixture."""
    means = filter_states.particles.x
    log_weights = filter_states.log_weights
    D, N = means.shape[0] - 1, means.shape[1]
    if augmented_data.gamma.shape[0] != D or augmented_data.gamma_pred.shape[0] != D:
        raise ValueError("covariance timeline must contain D entries for D+1 states")
    gamma_filtered = jnp.concatenate([params.gamma_0[None], augmented_data.gamma])
    rng, terminal_key = jax.random.split(key)
    terminal, terminal_indices = terminal_resample(
        terminal_key, means[D], log_weights[D], gamma_filtered[D], params.B,
        n_smoother_particles,
    )
    states = [None] * (D + 1)
    indices = [None] * (D + 1)
    probabilities = [None] * (D + 1)
    states[D], indices[D] = terminal, terminal_indices
    probabilities[D] = jnp.broadcast_to(jax.nn.softmax(log_weights[D]),
                                         (n_smoother_particles, N))

    for t in range(D - 1, -1, -1):
        if t % 1000 == 0 or t == 0 or t == D - 1:
            _log(f"rb_backward_simulation: t={t}/{D} ({100 * (D - 1 - t) // max(D - 1, 1)}%)")
        rng, selection_key, draw_key = jax.random.split(rng, 3)
        phi = jnp.exp(-params.kappa * (
            augmented_data.timestamp[t] - augmented_data.timestamp_prev[t]
        ))
        logits, predicted, J_gamma, gamma_cond = jax.vmap(
            lambda future: backward_statistics(
                means[t], log_weights[t], future, params.mean_0,
                gamma_filtered[t], augmented_data.gamma_pred[t], params.B, phi
            ), out_axes=(0, None, None, None)
        )(states[t + 1])
        probs = jax.nn.softmax(logits, axis=-1)
        selection_keys = jax.random.split(selection_key, n_smoother_particles)
        chosen = jax.vmap(jax.random.categorical)(selection_keys, logits)
        selected_means = means[t][chosen]
        selected_predicted = predicted[chosen]
        conditional_means = selected_means + jnp.einsum(
            "ij,sjk->sik", J_gamma, states[t + 1] - selected_predicted
        )
        draw_keys = jax.random.split(draw_key, n_smoother_particles)
        draws = jax.vmap(
            lambda k, m: sample_kron_psd(k, m, gamma_cond, params.B)
        )(draw_keys, conditional_means)
        states[t], indices[t], probabilities[t] = draws, chosen, probs

    return SmoothedStates(
        ParticleMeans(jnp.stack(states)), jnp.stack(indices),
        jnp.stack(probabilities),
    )


def E_step(params, model_inputs, n_particles, n_smoother_particles,
           max_goals, key):
    key, filter_key, smoother_key = jax.random.split(key, 3)
    filtered, augmented = run_filter(filter_key, model_inputs, params,
                                     n_particles, max_goals)
    smoothed = rb_backward_simulation(
        smoother_key, filtered, augmented, params, n_smoother_particles
    )
    return smoothed, filtered, augmented


def log_initial_density(params: EMParams, x_0):
    return gaussian_kron_logpdf(x_0 - params.mean_0, params.gamma_0, params.B)


def log_transition_density(params: EMParams, x_prev, x_next, day):
    dt = day.timestamp - day.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    predicted = params.mean_0 + phi * (x_prev - params.mean_0)
    return gaussian_kron_logpdf(
        x_next - predicted, (1.0 - phi**2) * params.gamma_0, params.B
    )


def complete_data_terms(params: EMParams, paths, model_inputs, max_goals: int):
    """Mean initial, summed transition, and summed observation terms."""
    # Accept state-major (D+1,S,M,2) and path-major (S,D+1,M,2).
    if paths.shape[0] == model_inputs.timestamp.shape[0] + 1:
        paths = paths.transpose(1, 0, 2, 3)
    initial = jax.vmap(lambda path: log_initial_density(params, path[0]))(paths)

    def path_terms(path):
        transition = jnp.asarray(0.0)
        observation = jnp.asarray(0.0)
        for t in range(model_inputs.timestamp.shape[0]):
            day = jax.tree.map(lambda x: x[t], model_inputs)
            transition = transition + log_transition_density(params, path[t], path[t + 1], day)
            observation = observation + daily_loglik(path[t + 1], day, params.alpha,
                                                      params.beta, max_goals)
        return transition, observation

    transitions, observations = jax.vmap(path_terms)(paths)
    return {"initial": jnp.mean(initial), "transition": jnp.mean(transitions),
            "observation": jnp.mean(observations)}


def mcem_objective(raw, mean_0, paths, model_inputs, max_goals, nu, S_gamma):
    params = decode_EM_params(raw, mean_0)
    terms = complete_data_terms(params, paths, model_inputs, max_goals)
    prior = log_inverse_wishart_kernel(params.gamma_0, nu, S_gamma)
    return terms["initial"] + terms["transition"] + terms["observation"] + prior


def _terms_to_record(params, paths, model_inputs, max_goals, nu, S_gamma):
    terms = complete_data_terms(params, paths, model_inputs, max_goals)
    prior = log_inverse_wishart_kernel(params.gamma_0, nu, S_gamma)
    dimension = params.mean_0.size
    matches = jnp.sum(model_inputs.match_mask)
    dimension = params.mean_0.size
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    scale = 1.0 - jnp.exp(-2.0 * params.kappa * dt)
    transition_normalization = jnp.sum(jax.vmap(
        lambda s: -0.5 * dimension * jnp.log(2 * jnp.pi)
        - 0.5 * kron_logdet(s * params.gamma_0, params.B)
    )(scale))
    transition_quadratic = terms["transition"] - transition_normalization
    return {
        **terms, "prior": prior,
        "total": terms["initial"] + terms["transition"] + terms["observation"] + prior,
        "transition_normalization": transition_normalization,
        "transition_quadratic_penalty": transition_quadratic,
        "initial_per_dimension": terms["initial"] / dimension,
        "transition_per_dimension_time": terms["transition"] / (dimension * model_inputs.timestamp.size),
        "observation_per_match": terms["observation"] / jnp.maximum(matches, 1),
    }


RB_BACKWARD_SIMULATION = rb_backward_simulation


def run_mcem(key, model_inputs, initial_params: EMParams,
             config: MCEMConfig = MCEMConfig(), *, e_step_fn=None):
    """Run MCEM, rejecting any non-finite or worsening fixed-path M-step."""
    if e_step_fn is None:
        e_step_fn = E_step
    _log(f"run_mcem start: n_epochs={config.n_epochs} "
         f"n_filter_particles={config.n_filter_particles} "
         f"n_smoother_particles={config.n_smoother_particles} "
         f"n_gradient_steps={config.n_gradient_steps} "
         f"learning_rate={config.learning_rate} "
         f"timeline_days={model_inputs.timestamp.shape[0]}")
    fixed_mean = jax.lax.stop_gradient(initial_params.mean_0)
    raw = encode_EM_params(initial_params)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(raw)
    nu = initial_params.gamma_0.shape[0] + 10
    S_gamma = (nu + initial_params.gamma_0.shape[0] + 1) * initial_params.gamma_0
    params_history = [initial_params]
    mstep_history, log_marginal_history, diagnostics_history = [], [], []
    last_smoothed = last_filtered = last_augmented = None
    rng = key
    _log("EM loop starting")

    for epoch in range(config.n_epochs):
        epoch_start = time.perf_counter()
        rng, e_key = jax.random.split(rng)
        params = decode_EM_params(raw, fixed_mean)
        _log(f"epoch {epoch}/{config.n_epochs}: running E-step (filter + backward simulation)...")
        smoothed, filtered, augmented = e_step_fn(
            params, model_inputs, config.n_filter_particles,
            config.n_smoother_particles, config.max_goals, e_key,
        )
        _log(f"epoch {epoch}: E-step done ({time.perf_counter() - epoch_start:.1f}s)")
        paths = jax.lax.stop_gradient(smoothed.particles.x)
        start_raw, start_opt = raw, opt_state
        objective = lambda value: mcem_objective(
            value, fixed_mean, paths, model_inputs, config.max_goals, nu, S_gamma
        )
        start_value = objective(raw)
        _log(f"epoch {epoch}: evaluating objective at start (start_value={float(start_value):.3f})")
        grad_step_start = time.perf_counter()
        for step in range(config.n_gradient_steps):
            if step % 5 == 0 or step == config.n_gradient_steps - 1:
                _log(f"epoch {epoch}: gradient step {step}/{config.n_gradient_steps} "
                     f"({time.perf_counter() - grad_step_start:.1f}s elapsed)")
            value, gradient = jax.value_and_grad(lambda value: -objective(value))(raw)
            updates, opt_state = optimizer.update(gradient, opt_state, raw)
            raw = optax.apply_updates(raw, updates)
        candidate_value = objective(raw)
        _log(f"epoch {epoch}: M-step done in "
             f"{time.perf_counter() - grad_step_start:.1f}s; "
             f"start={float(start_value):.3f} candidate={float(candidate_value):.3f}")
        accepted = bool(jnp.isfinite(candidate_value) &
                        (candidate_value + config.acceptance_tolerance >= start_value))
        if not accepted:
            raw, opt_state, candidate_value = start_raw, start_opt, start_value
            _log(f"epoch {epoch}: step REJECTED; reverting to start params")
        final_epoch_params = decode_EM_params(raw, fixed_mean)
        record = _terms_to_record(final_epoch_params, paths, model_inputs,
                                  config.max_goals, nu, S_gamma)
        record.update({"epoch": epoch, "start_objective": start_value,
                       "candidate_objective": candidate_value, "accepted": accepted})
        mstep_history.append(record)
        params_history.append(final_epoch_params)
        log_marginal_history.append(filtered.log_normalizing_constant[-1])
        diagnostics_history.append(smoothed_path_diagnostics(final_epoch_params,
                                                              smoothed, augmented))
        last_smoothed, last_filtered, last_augmented = smoothed, filtered, augmented
        _log(f"epoch {epoch}/{config.n_epochs} complete in "
             f"{time.perf_counter() - epoch_start:.1f}s "
             f"(accepted={accepted}, objective={float(candidate_value):.3f})")

    _log("EM loop complete; running final filter")
    final_params = decode_EM_params(raw, fixed_mean)
    rng, final_key = jax.random.split(rng)
    final_filtered, final_augmented = run_filter(
        final_key, model_inputs, final_params, config.n_filter_particles, config.max_goals
    )
    if last_smoothed is None:
        rng, smooth_key = jax.random.split(rng)
        last_smoothed, _, _ = e_step_fn(
            final_params, model_inputs, config.n_filter_particles,
            config.n_smoother_particles, config.max_goals, smooth_key,
        )
    _log("run_mcem complete")
    return {
        "final_params": final_params, "params_history": params_history,
        "mstep_history": mstep_history, "log_marginal_history": log_marginal_history,
        "diagnostics_history": diagnostics_history,
        "final_log_marginal_likelihood": final_filtered.log_normalizing_constant[-1],
        "final_filter_states": final_filtered, "final_augmented_data": final_augmented,
        "final_smoothed_paths": last_smoothed.particles.x,
        "backward_probabilities": last_smoothed.backward_probabilities,
        "backward_component_indices": last_smoothed.component_indices,
        "config": config._asdict(),
    }


run_EM = run_mcem


def smoothed_path_diagnostics(params, smoothed, model_inputs):
    paths = smoothed.particles.x.transpose(1, 0, 2, 3)
    dimension = params.mean_0.size
    initial = kron_quad(params.gamma_0, params.B, paths[:, 0] - params.mean_0)
    quads = []
    for t in range(model_inputs.timestamp.shape[0]):
        dt = model_inputs.timestamp[t] - model_inputs.timestamp_prev[t]
        phi = jnp.exp(-params.kappa * dt)
        residual = paths[:, t + 1] - params.mean_0 - phi * (paths[:, t] - params.mean_0)
        quads.append(kron_quad((1 - phi**2) * params.gamma_0, params.B, residual))
    q = jnp.stack(quads)
    probs = smoothed.backward_probabilities
    ess = 1.0 / jnp.sum(probs**2, axis=-1)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0), axis=-1)
    unique = jnp.asarray([jnp.unique(smoothed.component_indices[t], size=smoothed.component_indices.shape[1],
                                    fill_value=-1).shape[0]
                          for t in range(smoothed.component_indices.shape[0])])
    # The exact unique count is kept eager because diagnostics are not jitted.
    import numpy as np
    unique = jnp.asarray([len(np.unique(np.asarray(v))) for v in smoothed.component_indices])
    return {
        "timeline_aligned": paths.shape[1] == model_inputs.timestamp.size + 1,
        "initial_mahalanobis_ratio": jnp.mean(initial) / dimension,
        "transition_mahalanobis_ratio": jnp.mean(q) / dimension,
        "transition_mahalanobis_median": jnp.median(q),
        "transition_mahalanobis_p05": jnp.percentile(q, 5),
        "transition_mahalanobis_p95": jnp.percentile(q, 95),
        "backward_ess_min": jnp.min(ess), "backward_ess_mean": jnp.mean(ess),
        "backward_entropy_mean": jnp.mean(entropy),
        "backward_max_probability": jnp.max(probs),
        "unique_component_indices_by_time": unique,
        "smoothed_mean": jnp.mean(paths, axis=0),
        "smoothed_variance": jnp.var(paths, axis=0),
        "lag_one_moment": jnp.mean(paths[:, :-1, :, :, None, None] *
                                    paths[:, 1:, None, None, :, :], axis=0),
    }
