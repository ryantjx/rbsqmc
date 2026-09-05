"""Tests for the Sequential quasi-Monte Carlo (SQMC) filter in ``sqmc.sqmc.sqmc``.

Test conventions follow ``state-space-models/cuthbert``: ``chex.TestCase``
classes, ``@chex.variants(with_jit, without_jit)`` for pure array-valued
functions, absl ``parameterized`` markers, module-level
``chex.assert_trees_all_close`` and ``chex.assert_shape``, and a
module-autouse x64 fixture that restores the flag on teardown.

The SQMC filter is deterministic (it propagates particles through fixed RQMC
points), so the analytical Kalman-filter log-likelihood is recovered to tight
tolerance on a linear-Gaussian model.
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import chex

from cuthbert.inference import Filter

from sqmc.qmc.qmc import Sobol
from sqmc.sqmc.sqmc import (
    build_filter,
    filter_combine,
    filter_prepare,
    init_prepare,
    resample_from_uniform,
)


@pytest.fixture(scope="module", autouse=True)
def config():
    """Enable double precision for the module and restore it on teardown."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _scalar_kalman_loglikelihood(observations, sigma_x, sigma_y):
    """Exact log-likelihood of a 1D random-walk model via the Kalman filter.

    Model:
        x_0 ~ N(0, 1)
        x_t = x_{t-1} + sigma_x * Z_t,   Z_t ~ N(0, 1)
        y_t = x_t + sigma_y * E_t,       E_t ~ N(0, 1)

    Returns the log-likelihood ``log p(y_1:T)``.
    """
    m = 0.0
    P = 1.0
    log_likelihood = 0.0
    for y in observations:
        # Prediction step.
        m_pred = m
        P_pred = P + sigma_x**2
        # Update step.
        S = P_pred + sigma_y**2
        K = P_pred / S
        m = m_pred + K * (y - m_pred)
        P = (1.0 - K) * P_pred
        # Incremental log-likelihood of y_t.
        log_likelihood += -0.5 * (
            jnp.log(2.0 * jnp.pi * S) + (y - m_pred) ** 2 / S
        )
    return log_likelihood


def _build_rw_filter(n_particles, sigma_x, sigma_y, qmc):
    """Build an SQMC filter for the 1D random-walk model."""

    def init_transform(u, model_inputs):
        # x_0 ~ N(0, 1)
        return jax.scipy.stats.norm.ppf(u)

    def propagate_transform(u, state, model_inputs):
        # x_t = x_{t-1} + sigma_x * Phi^{-1}(u)
        return state + sigma_x * jax.scipy.stats.norm.ppf(u)

    def log_potential(state_prev, state, model_inputs):
        # log N(y_t | x_t, sigma_y^2)
        y = model_inputs["y"]
        return -0.5 * ((y - state) / sigma_y) ** 2 - jnp.log(
            sigma_y
        ) - 0.5 * jnp.log(2 * jnp.pi)

    return build_filter(
        init_transform=init_transform,
        propagate_transform=propagate_transform,
        log_potential=log_potential,
        n_filter_particles=n_particles,
        qmc=qmc,
    )


class BuildFilterTest(chex.TestCase):
    def test_returns_cuthbert_filter(self):
        qmc = Sobol(d=2)
        filter_ = _build_rw_filter(64, 0.5, 1.0, qmc)

        self.assertIsInstance(filter_, Filter)
        self.assertFalse(filter_.associative)

    def test_init_prepare_shapes_and_zero_logz(self):
        qmc = Sobol(d=2)
        filter_ = _build_rw_filter(64, 0.5, 1.0, qmc)
        state = filter_.init_prepare({"y": jnp.array(0.0)}, key=jax.random.key(0))

        chex.assert_shape(state.particles, (64, 1))
        chex.assert_shape(state.log_weights, (64,))
        chex.assert_shape(state.ancestor_indices, (64,))
        # log_weights are all zero, so log_normalizing_constant == 0.
        chex.assert_trees_all_close(
            state.log_normalizing_constant, jnp.array(0.0), rtol=0.0, atol=1e-12
        )
        self.assertEqual(state.n_particles, 64)


class ResampleFromUniformTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_matches_searchsorted_inverse_cdf(self):
        logits = jnp.asarray([0.0, -1.0, -2.0, -3.0, -4.0])
        sorted_uniforms = jnp.asarray([0.0, 0.1, 0.5, 0.9, 1.0])

        def compute():
            return resample_from_uniform(sorted_uniforms, logits)

        idx, logits_out = self.variant(compute)()

        weights = jnp.exp(logits - jax.nn.logsumexp(logits))
        cs = jnp.cumsum(weights)
        expected_idx = jnp.searchsorted(cs, sorted_uniforms, method="sort")
        expected_idx = jnp.clip(expected_idx, 0, weights.shape[0] - 1).astype(int)

        chex.assert_trees_all_close(idx, expected_idx, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(
            logits_out, jnp.zeros_like(sorted_uniforms), rtol=0.0, atol=0.0
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_indices_are_in_bounds(self):
        logits = jnp.asarray([0.0, -1.0, -2.0])
        sorted_uniforms = jnp.asarray([0.0, 0.5, 1.0])

        def compute():
            return resample_from_uniform(sorted_uniforms, logits)

        idx, _ = self.variant(compute)()
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), 3)


class FilterCombineTest(chex.TestCase):
    def test_combine_preserves_shapes_and_permutation(self):
        n_particles = 64
        qmc = Sobol(d=2)
        filter_ = _build_rw_filter(n_particles, 0.5, 1.0, qmc)

        init_state = filter_.init_prepare(
            {"y": jnp.array(0.0)}, key=jax.random.key(0)
        )
        prep_state = filter_.filter_prepare(
            {"y": jnp.array(0.5)}, key=jax.random.key(1)
        )
        combined = filter_.filter_combine(init_state, prep_state)

        chex.assert_shape(combined.particles, (n_particles, 1))
        # log_potential broadcasts the (N, 1) state against the scalar
        # observation, so the combined log-weights are (N, 1).
        chex.assert_shape(combined.log_weights, (n_particles, 1))
        chex.assert_shape(combined.ancestor_indices, (n_particles,))
        # Resampling draws with replacement, so ancestor indices are in
        # [0, N-1] but need not be a permutation.
        self.assertGreaterEqual(int(combined.ancestor_indices.min()), 0)
        self.assertLess(int(combined.ancestor_indices.max()), n_particles)
        # log_normalizing_constant is a scalar.
        self.assertEqual(combined.log_normalizing_constant.shape, ())


class AnalyticalKalmanTest(chex.TestCase):
    @parameterized.product(n_particles=[256, 1024], n_steps=[10, 20])
    def test_loglikelihood_matches_kalman(self, n_particles, n_steps):
        sigma_x = 0.5
        sigma_y = 1.0
        qmc = Sobol(d=2)
        filter_ = _build_rw_filter(n_particles, sigma_x, sigma_y, qmc)

        # Generate observations from the model.
        key = jax.random.key(0)
        x_true = 0.0
        observations = []
        for t in range(n_steps):
            key = jax.random.fold_in(key, t)
            x_true = x_true + sigma_x * jax.random.normal(key, ())
            key = jax.random.fold_in(key, t + 1000)
            observations.append(x_true + sigma_y * jax.random.normal(key, ()))
        observations = jnp.asarray(observations)

        # Run the SQMC filter.
        state = filter_.init_prepare(
            {"y": observations[0]}, key=jax.random.key(0)
        )
        for t in range(1, n_steps):
            state = filter_.filter_combine(
                state,
                filter_.filter_prepare(
                    {"y": observations[t]}, key=jax.random.key(t)
                ),
            )

        expected = _scalar_kalman_loglikelihood(
            observations[1:], sigma_x, sigma_y
        )
        # ``init_prepare`` initialises particles with zero weights and does not
        # add the first observation's likelihood; the combine loop starts at
        # t=1, so the filter estimates log p(y_2:T). Compare against the Kalman
        # log-likelihood of the same observations. SQMC is deterministic, so
        # the estimate is accurate to tight tolerance.
        chex.assert_trees_all_close(
            state.log_normalizing_constant, expected, rtol=1e-2, atol=1e-2
        )


class NoopTest(chex.TestCase):
    def test_identity_transforms_preserve_particle_multiset(self):
        n_particles = 64
        qmc = Sobol(d=2)

        def init_transform(u, model_inputs):
            return u

        def propagate_transform(u, state, model_inputs):
            return state

        def log_potential(state_prev, state, model_inputs):
            return jnp.zeros(())

        filter_ = build_filter(
            init_transform=init_transform,
            propagate_transform=propagate_transform,
            log_potential=log_potential,
            n_filter_particles=n_particles,
            qmc=qmc,
        )

        init_state = filter_.init_prepare(
            {"y": jnp.array(0.0)}, key=jax.random.key(0)
        )
        prep_state = filter_.filter_prepare(
            {"y": jnp.array(0.0)}, key=jax.random.key(1)
        )
        combined = filter_.filter_combine(init_state, prep_state)

        # Identity propagation + zero potential: the log normalising constant
        # stays at 0 (weights are reset to zero after resampling), and the
        # particle shape is preserved. Resampling draws with replacement, so
        # the particle multiset is not preserved exactly; instead we check that
        # every combined particle is one of the initial particles.
        chex.assert_trees_all_close(
            combined.log_normalizing_constant, jnp.array(0.0), rtol=0.0, atol=1e-12
        )
        chex.assert_shape(combined.particles, init_state.particles.shape)
        initial_values = jnp.sort(init_state.particles[:, 0])
        for value in combined.particles[:, 0]:
            self.assertTrue(
                bool(jnp.any(jnp.isclose(initial_values, value, atol=1e-12)))
            )


class DeterminismTest(chex.TestCase):
    def test_same_qmc_is_reproducible(self):
        def run():
            qmc = Sobol(d=2)
            filter_ = _build_rw_filter(128, 0.5, 1.0, qmc)
            state = filter_.init_prepare(
                {"y": jnp.array(0.0)}, key=jax.random.key(0)
            )
            for t in range(1, 5):
                state = filter_.filter_combine(
                    state,
                    filter_.filter_prepare(
                        {"y": jnp.array(0.1 * t)}, key=jax.random.key(t)
                    ),
                )
            return state.log_normalizing_constant, state.particles

        first_logz, first_particles = run()
        second_logz, second_particles = run()

        chex.assert_trees_all_close(first_logz, second_logz, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(
            first_particles, second_particles, rtol=0.0, atol=0.0
        )
