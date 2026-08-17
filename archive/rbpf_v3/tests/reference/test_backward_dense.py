import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from rbpf_v3.src import smoothing, smoothing_noncuthbert


@pytest.mark.parametrize("module", [smoothing, smoothing_noncuthbert])
def test_kronecker_log_density_and_conditionals_match_dense(module):
    gamma = jnp.asarray([[1.3, 0.2], [0.2, 0.8]])
    gamma_pred = jnp.asarray([[1.7, 0.1], [0.1, 1.1]])
    B = jnp.asarray([[1.2, 0.15], [0.15, 0.9]])
    residuals = jnp.asarray(
        [[[[0.2, -0.1], [0.4, 0.3]], [[-0.2, 0.7], [0.1, -0.4]]]]
    )
    actual = module.gaussian_kron_logpdf(residuals, gamma_pred, B)
    covariance = np.kron(np.asarray(gamma_pred), np.asarray(B))
    expected = np.asarray(
        [
            [multivariate_normal.logpdf(np.asarray(item).reshape(-1), cov=covariance) for item in row]
            for row in np.asarray(residuals)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    phi = jnp.asarray(0.73)
    gain, conditional = module.backward_shared_terms(gamma, gamma_pred, phi)
    dense_gain = phi * np.kron(np.asarray(gamma), np.asarray(B)) @ np.linalg.inv(covariance)
    expected_gain = np.kron(np.asarray(gain), np.eye(2))
    np.testing.assert_allclose(expected_gain, dense_gain, rtol=1e-5, atol=1e-6)
    dense_conditional = np.kron(np.asarray(gamma), np.asarray(B)) - dense_gain @ covariance @ dense_gain.T
    np.testing.assert_allclose(
        np.kron(np.asarray(conditional), np.asarray(B)),
        dense_conditional,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("module", [smoothing, smoothing_noncuthbert])
def test_backward_logits_use_predicted_covariance_not_complete_state_q(module):
    means = jnp.asarray([[[0.0, 0.0]], [[1.2, -0.4]]])
    weights = jnp.log(jnp.asarray([0.4, 0.6]))
    future = jnp.asarray([[[0.8, -0.2]]])
    mean_0 = jnp.zeros((1, 2))
    gamma_t = jnp.asarray([[0.2]])
    predicted_r = jnp.asarray([[1.4]])
    complete_q = jnp.asarray([[0.05]])
    B = jnp.eye(2)
    phi = jnp.asarray(0.8)

    predicted = mean_0 + phi * (means - mean_0)
    correct = weights + module.gaussian_kron_logpdf(
        future[:, None] - predicted[None], predicted_r, B
    )[0]
    wrong = weights + module.gaussian_kron_logpdf(
        future[:, None] - predicted[None], complete_q, B
    )[0]
    assert np.max(np.abs(np.asarray(jax.nn.softmax(correct) - jax.nn.softmax(wrong)))) > 0.1
