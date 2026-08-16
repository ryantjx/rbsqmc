import jax.numpy as jnp

from rbpf_v2.src.kron import rts_kron_terms
from rbpf_v2.src.smoothing import backward_statistics


def test_backward_kernel_matches_dense_and_uses_R(small_model):
    _, _, params = small_model
    gamma_t = jnp.array([[.8, .1], [.1, .5]])
    gamma_pred = jnp.array([[.9, .12], [.12, .7]])
    B = jnp.diag(jnp.array([1.4, 1/1.4]))
    phi = .7
    means = jnp.stack([jnp.zeros((2, 2)), jnp.ones((2, 2)) * .2])
    future = jnp.ones((2, 2)) * .1
    weights = jnp.log(jnp.array([.4, .6]))
    logits, predicted, Jg, Cg = backward_statistics(
        means, weights, future, jnp.zeros((2, 2)), gamma_t, gamma_pred, B, phi)
    P, R = jnp.kron(gamma_t, B), jnp.kron(gamma_pred, B)
    dense_J = phi * P @ jnp.linalg.inv(R)
    dense_C = P - dense_J @ R @ dense_J.T
    expected = []
    for i in range(2):
        r = (future - predicted[i]).reshape(-1)
        expected.append(weights[i] - .5 * (4*jnp.log(2*jnp.pi) +
                         jnp.linalg.slogdet(R)[1] + r @ jnp.linalg.solve(R, r)))
    assert jnp.allclose(logits, jnp.asarray(expected), atol=1e-5)
    assert jnp.allclose(jnp.kron(Jg, jnp.eye(2)), dense_J, atol=1e-5)
    assert jnp.allclose(jnp.kron(Cg, B), dense_C, atol=1e-5)


def test_conditional_mean_matches_dense():
    gamma, pred, phi = jnp.array([[.7, .1], [.1, .6]]), jnp.array([[.9,.1],[.1,.8]]), .8
    J, _ = rts_kron_terms(gamma, pred, phi)
    mean, forecast, future = jnp.zeros((2,2)), jnp.ones((2,2))*.1, jnp.ones((2,2))*.3
    got = mean + J @ (future - forecast)
    dense = mean.reshape(-1) + jnp.kron(J, jnp.eye(2)) @ (future-forecast).reshape(-1)
    assert jnp.allclose(got.reshape(-1), dense)
