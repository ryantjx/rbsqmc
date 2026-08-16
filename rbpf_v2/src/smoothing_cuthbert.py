"""Cuthbert integration for the RB-aware football backward simulator.

The model-specific Gaussian calculations live in :mod:`rbpf_v2.src.smoothing`.
This module only adapts those calculations to Cuthbert's particle-tree,
terminal-resampling, and backward-callback protocols.
"""

from __future__ import annotations

from functools import partial

import cuthbert
import cuthbertlib
from cuthbert.smc.backward_sampler import build_smoother as cuthbert_build_smoother
from cuthbert.smc.particle_filter import ParticleFilterState
import jax
import jax.numpy as jnp

from .bivariate_poisson import daily_loglik
from .kron import rts_kron_terms, sample_kron_psd
from .model import run_filter
from .smoothing import (
    MCEMConfig,
    backward_statistics,
    gaussian_kron_logpdf,
    log_transition_density,
    run_mcem as _run_mcem,
)
from .utils import EMParams, ParticleMeans, RBSmootherParticle, SmoothedStates


def make_rb_smoother_filter_states(filtered_states, rb_inputs, params: EMParams):
    """Adapt v2 filter output to Cuthbert's time-stacked particle-tree contract."""
    means = jax.lax.stop_gradient(filtered_states.particles.x)
    n_states, n_components = means.shape[:2]
    D = n_states - 1
    filtered_gamma = jnp.concatenate([params.gamma_0[None], rb_inputs.gamma])
    dt = rb_inputs.timestamp - rb_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    if n_states != dt.size + 1:
        raise ValueError("Expected D+1 filter states for D transitions")
    if filtered_gamma.shape[0] != n_states:
        raise ValueError("Filtered covariance timeline is misaligned")
    if rb_inputs.gamma_pred.shape[0] != D:
        raise ValueError("Predicted covariance timeline is misaligned")
    if not isinstance(rb_inputs.gamma_pred, jax.core.Tracer):
        import numpy as np

        if np.any(np.asarray(dt) <= 0):
            raise ValueError("Every elapsed time dt must be strictly positive")
        if np.any(np.linalg.eigvalsh(np.asarray(rb_inputs.gamma_pred)) <= 0):
            raise ValueError("Every predicted covariance must be positive definite")

    pred_terminal = jnp.concatenate(
        [rb_inputs.gamma_pred, rb_inputs.gamma_pred[-1:]], axis=0
    )
    phi_terminal = jnp.concatenate([phi, phi[-1:]], axis=0)

    def broadcast(values):
        return jnp.broadcast_to(
            values[:, None], (n_states, n_components) + values.shape[1:]
        )

    particles = RBSmootherParticle(
        x=means,
        gamma_filtered=broadcast(filtered_gamma),
        gamma_pred_next=broadcast(pred_terminal),
        phi_next=broadcast(phi_terminal),
    )
    # Cuthbert propagates these labels after the custom selection. Giving each
    # component its own ID makes the returned labels the backward component IDs,
    # not the forward resampling genealogy.
    component_ids = jnp.broadcast_to(
        jnp.arange(n_components), (n_states, n_components)
    )
    return ParticleFilterState(
        key=jax.random.split(jax.random.key(0), n_states),
        particles=particles,
        log_weights=filtered_states.log_weights,
        ancestor_indices=component_ids,
        model_inputs=jnp.arange(n_states),
        log_normalizing_constant=filtered_states.log_normalizing_constant,
    )


def rb_terminal_resampling(key, logits, positions, n, *, B):
    """Select terminal components and materialize independent complete states."""
    _, index_key, state_key = jax.random.split(key, 3)
    indices, output_logits, selected = (
        cuthbertlib.resampling.systematic.resampling(
            index_key, logits, positions, n
        )
    )
    state_keys = jax.random.split(state_key, n)
    terminal_states = jax.vmap(
        lambda draw_key, mean, gamma: sample_kron_psd(
            draw_key, mean, gamma, B
        )
    )(state_keys, selected.x, selected.gamma_filtered)
    return indices, output_logits, selected._replace(x=terminal_states)


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
):
    """Cuthbert callback implementing marginalized RB selection and conditioning."""
    del log_density, x1_ancestor_indices
    component_means = x0_all.x
    gamma_t = x0_all.gamma_filtered[0]
    gamma_pred_next = x0_all.gamma_pred_next[0]
    phi_t = x0_all.phi_next[0]
    predicted = mean_0[None] + phi_t * (component_means - mean_0[None])
    J_gamma, gamma_cond = rts_kron_terms(gamma_t, gamma_pred_next, phi_t)

    def sample_one(trajectory_key, x_next):
        component_key, state_key = jax.random.split(trajectory_key)
        logits = jnp.asarray(log_weight_x0_all) + gaussian_kron_logpdf(
            x_next[None] - predicted, gamma_pred_next, B
        )
        component = jax.random.categorical(component_key, logits)
        conditional_mean = component_means[component] + J_gamma @ (
            x_next - predicted[component]
        )
        previous = sample_kron_psd(
            state_key, conditional_mean, gamma_cond, B
        )
        return previous, component

    keys = jax.random.split(key, x1_all.x.shape[0])
    previous, component_indices = jax.vmap(sample_one)(keys, x1_all.x)
    selected = jax.tree.map(lambda values: values[component_indices], x0_all)
    return selected._replace(x=previous), component_indices


def _joint_log_potential(previous, current, day, params, max_goals):
    """Complete-state potential required by Cuthbert's smoother interface."""
    return (
        log_transition_density(params, previous.x, current.x, day)
        + daily_loglik(current.x, day, params.alpha, params.beta, max_goals)
    )


def build_smoother(params: EMParams, n_smoother_particles: int,
                   max_goals: int = 8):
    """Build a Cuthbert smoother with RB-aware terminal and backward operations."""
    return cuthbert_build_smoother(
        log_potential=partial(
            _joint_log_potential, params=params, max_goals=max_goals
        ),
        backward_sampling_fn=partial(
            rb_backward_sampling_fn, mean_0=params.mean_0, B=params.B
        ),
        resampling_fn=partial(rb_terminal_resampling, B=params.B),
        n_smoother_particles=n_smoother_particles,
    )


def run_cuthbert_smoother(key, filter_states, augmented_data, params: EMParams,
                          n_smoother_particles: int, max_goals: int = 8):
    """Run Cuthbert and convert its output to the common v2 smoother result."""
    adapted = make_rb_smoother_filter_states(filter_states, augmented_data, params)
    raw = cuthbert.smoother(
        build_smoother(params, n_smoother_particles, max_goals),
        adapted,
        augmented_data,
        parallel=False,
        key=key,
    )
    paths = raw.particles.x
    D, N = paths.shape[0] - 1, filter_states.particles.x.shape[1]
    probabilities = [None] * (D + 1)
    probabilities[D] = jnp.broadcast_to(
        jax.nn.softmax(filter_states.log_weights[D]),
        (n_smoother_particles, N),
    )
    for t in range(D - 1, -1, -1):
        logits = jax.vmap(lambda future: backward_statistics(
            filter_states.particles.x[t], filter_states.log_weights[t], future,
            params.mean_0, raw.particles.gamma_filtered[t, 0],
            raw.particles.gamma_pred_next[t, 0], params.B,
            raw.particles.phi_next[t, 0],
        )[0])(paths[t + 1])
        probabilities[t] = jax.nn.softmax(logits, axis=-1)
    return SmoothedStates(
        ParticleMeans(paths), raw.ancestor_indices, jnp.stack(probabilities)
    )


def E_step(params, model_inputs, n_particles, n_smoother_particles,
           max_goals, key):
    """Cuthbert-backed RBPF/FFBS expectation step."""
    _, filter_key, smoother_key = jax.random.split(key, 3)
    filtered, augmented = run_filter(
        filter_key, model_inputs, params, n_particles, max_goals
    )
    smoothed = run_cuthbert_smoother(
        smoother_key, filtered, augmented, params,
        n_smoother_particles, max_goals,
    )
    return smoothed, filtered, augmented


def run_mcem(key, model_inputs, initial_params: EMParams,
             config: MCEMConfig = MCEMConfig()):
    """Run the shared MCEM loop with the Cuthbert-backed E-step."""
    return _run_mcem(
        key, model_inputs, initial_params, config, e_step_fn=E_step
    )


run_EM = run_mcem

__all__ = [
    "E_step",
    "build_smoother",
    "make_rb_smoother_filter_states",
    "rb_backward_sampling_fn",
    "rb_terminal_resampling",
    "run_cuthbert_smoother",
    "run_mcem",
    "run_EM",
]
