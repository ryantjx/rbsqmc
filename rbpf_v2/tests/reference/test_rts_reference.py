import jax.numpy as jnp

from rbpf_v2.src.kron import rts_kron_terms


def test_rts_gain_covariance_dense_reference():
    filtered = jnp.array([[.7, .2], [.2, .6]])
    predicted = jnp.array([[.9, .2], [.2, .8]])
    B = jnp.array([[1., .1], [.1, 1.1]])
    J, C = rts_kron_terms(filtered, predicted, .75)
    P, R = jnp.kron(filtered, B), jnp.kron(predicted, B)
    dense_J = .75 * P @ jnp.linalg.inv(R)
    dense_C = P - dense_J @ R @ dense_J.T
    assert jnp.allclose(jnp.kron(J, jnp.eye(2)), dense_J, atol=1e-5)
    assert jnp.allclose(jnp.kron(C, B), dense_C, atol=1e-5)
