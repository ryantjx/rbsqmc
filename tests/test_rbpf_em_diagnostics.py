import json
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rbpf.src.helpers import (
    decode_EM_params,
    encode_EM_params,
    parameter_diagnostics,
    save_em_results,
    timeline_diagnostics,
)
from rbpf.src.smoothing import (
    materialize_rb_filter,
    materialized_cloud_diagnostics,
    smoothed_path_diagnostics,
)
from rbpf.src.utils import EMParams, RBPFState


class DummyStateHistory(NamedTuple):
    particles: RBPFState
    ancestor_indices: jax.Array


def _params(num_teams=3):
    base = jnp.arange(1, num_teams + 1, dtype=jnp.float32)
    gamma = 0.12 * jnp.eye(num_teams) + 0.01 * jnp.outer(base, base)
    return EMParams(
        mean_0=jnp.zeros((num_teams, 2)),
        gamma_0=gamma,
        B=jnp.diag(jnp.array([1.25, 0.8])),
        kappa=jnp.array(0.03),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )


def test_encode_decode_preserves_identified_kronecker_covariance():
    params = _params()
    raw = encode_EM_params(params)
    decoded = decode_EM_params(raw, fixed_mean_0=params.mean_0)

    expected = np.kron(np.asarray(params.gamma_0), np.asarray(params.B))
    actual = np.kron(np.asarray(decoded.gamma_0), np.asarray(decoded.B))

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.linalg.det(np.asarray(decoded.B)), 1.0, rtol=1e-6)


def test_materialized_initial_cloud_has_full_state_covariance():
    params = _params(num_teams=2)
    n_particles = 4000
    means = jnp.broadcast_to(
        params.mean_0,
        (1, n_particles) + params.mean_0.shape,
    )
    filtered = DummyStateHistory(
        particles=RBPFState(x=means),
        ancestor_indices=jnp.broadcast_to(jnp.arange(n_particles), (1, n_particles)),
    )

    materialized = materialize_rb_filter(
        jax.random.PRNGKey(1),
        filtered,
        gamma=jnp.empty((0, 2, 2)),
        gamma_0=params.gamma_0,
        B=params.B,
    )
    samples = np.asarray(materialized.particles.x[0]).reshape(n_particles, -1)
    diagnostics = materialized_cloud_diagnostics(params, materialized)

    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.025)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False),
        np.kron(np.asarray(params.gamma_0), np.asarray(params.B)),
        rtol=0.12,
        atol=0.015,
    )
    assert not bool(diagnostics["rb_means_as_states_suspected"])
    assert float(diagnostics["initial_mahalanobis_ratio"]) == pytest.approx(
        1.0, abs=0.05
    )


def test_materialization_preserves_zero_residual_covariance_rows():
    params = _params(num_teams=3)
    n_particles = 64
    means = jax.random.normal(
        jax.random.PRNGKey(2),
        (2, n_particles, 3, 2),
    )
    posterior_gamma = params.gamma_0.at[0, :].set(0.0).at[:, 0].set(0.0)
    filtered = DummyStateHistory(
        particles=RBPFState(x=means),
        ancestor_indices=jnp.broadcast_to(jnp.arange(n_particles), (2, n_particles)),
    )

    materialized = materialize_rb_filter(
        jax.random.PRNGKey(3),
        filtered,
        gamma=posterior_gamma[None],
        gamma_0=params.gamma_0,
        B=params.B,
    )

    np.testing.assert_allclose(
        np.asarray(materialized.particles.x[1, :, 0]),
        np.asarray(means[1, :, 0]),
        atol=1e-6,
    )


def _simulate_ou_paths(params, n_paths=256, n_transitions=20):
    keys = jax.random.split(jax.random.PRNGKey(4), n_transitions + 1)
    L_gamma = jnp.linalg.cholesky(params.gamma_0)
    L_B = jnp.linalg.cholesky(params.B)

    initial_noise = jax.random.normal(
        keys[0],
        (n_paths,) + params.mean_0.shape,
    )
    initial = params.mean_0 + jax.vmap(
        lambda noise: L_gamma @ noise @ L_B.T
    )(initial_noise)

    dt = jnp.arange(1, n_transitions + 1) % 4 + 1

    def step(previous, item):
        key, delta = item
        phi = jnp.exp(-params.kappa * delta)
        scale = jnp.sqrt(1.0 - phi**2)
        noise = jax.random.normal(key, previous.shape)
        innovation = jax.vmap(
            lambda z: scale * L_gamma @ z @ L_B.T
        )(noise)
        current = params.mean_0 + phi * (previous - params.mean_0) + innovation
        return current, current

    _, states = jax.lax.scan(step, initial, (keys[1:], dt))
    paths = jnp.concatenate([initial[None], states], axis=0)
    ancestors = jnp.broadcast_to(jnp.arange(n_paths), (n_transitions + 1, n_paths))
    smoothed = DummyStateHistory(RBPFState(paths), ancestors)
    model_inputs = SimpleNamespace(
        timestamp=jnp.cumsum(dt),
        timestamp_prev=jnp.concatenate([jnp.array([0]), jnp.cumsum(dt)[:-1]]),
    )
    return smoothed, model_inputs


def test_full_ou_paths_have_dimension_scaled_mahalanobis_statistics():
    params = _params()
    smoothed, model_inputs = _simulate_ou_paths(params)
    diagnostics = smoothed_path_diagnostics(params, smoothed, model_inputs)

    assert bool(diagnostics["timeline_aligned"])
    assert float(diagnostics["initial_mahalanobis_ratio"]) == pytest.approx(
        1.0, abs=0.12
    )
    assert float(diagnostics["transition_mahalanobis_ratio"]) == pytest.approx(
        1.0, abs=0.06
    )


def test_conditional_means_trigger_materialized_cloud_warning():
    params = _params()
    n_paths = 16
    n_transitions = 3
    paths = jnp.zeros((n_transitions + 1, n_paths, 3, 2))
    ancestors = jnp.broadcast_to(jnp.arange(n_paths), (n_transitions + 1, n_paths))
    smoothed = DummyStateHistory(RBPFState(paths), ancestors)
    diagnostics = materialized_cloud_diagnostics(params, smoothed)
    assert float(diagnostics["initial_mahalanobis_ratio"]) == 0.0
    assert bool(diagnostics["rb_means_as_states_suspected"])


def test_timeline_and_parameter_diagnostics():
    params = _params()
    inputs = SimpleNamespace(
        timestamp=jnp.array([2, 4, 11]),
        timestamp_prev=jnp.array([0, 2, 4]),
    )
    timeline = timeline_diagnostics(inputs)
    parameter = parameter_diagnostics(params)

    assert bool(timeline["all_dt_positive"])
    assert int(timeline["n_transitions"]) == 3
    assert float(timeline["dt_min"]) == 2.0
    assert float(parameter["gamma_min_eigenvalue"]) > 0
    assert float(parameter["gamma_condition_number"]) >= 1
    assert set(parameter["Q_eigenvalues_by_dt"]) == {"1", "2", "7", "30"}


def test_save_em_results_records_final_likelihood_and_diagnostics(tmp_path):
    params = _params()
    results = {
        "final_params": params,
        "params_history": [params],
        "log_marginal_history": [jnp.array(-12.0)],
        "mstep_history": [],
        "diagnostics_history": [{"check": jnp.array(True)}],
        "run_metadata": {"seed": 7},
        "final_log_marginal_likelihood": jnp.array(-12.0),
    }

    save_em_results(results, str(tmp_path))

    metadata = json.loads((tmp_path / "em_run_metadata.json").read_text())
    diagnostics = json.loads(
        (tmp_path / "em_diagnostics_history.json").read_text()
    )
    assert metadata["seed"] == 7
    assert metadata["final_log_marginal_likelihood"] == -12.0
    assert diagnostics == [{"check": True}]
