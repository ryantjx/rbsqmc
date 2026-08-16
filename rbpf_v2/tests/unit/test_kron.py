import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rbpf_v2.src.kron import kron_logdet, kron_quad, psd_sqrt, sample_kron_psd


def test_kron_primitives_match_dense(key):
    A = jnp.array([[2., .3], [.3, 1.]])
    B = jnp.array([[1.2, .1], [.1, .8]])
    residual = jax.random.normal(key, (5, 2, 2))
    dense = jnp.kron(A, B)
    expected = jax.vmap(lambda r: r.reshape(-1) @ jnp.linalg.solve(dense, r.reshape(-1)))(residual)
    assert jnp.allclose(kron_quad(A, B, residual), expected, atol=1e-5)
    assert jnp.allclose(kron_logdet(A, B), jnp.linalg.slogdet(dense)[1])


def test_psd_sqrt_and_negative_rejection():
    A = jnp.array([[1., 1.], [1., 1.]])
    root = psd_sqrt(A)
    assert jnp.allclose(root @ root.T, A, atol=1e-5)
    with pytest.raises(ValueError):
        psd_sqrt(jnp.diag(jnp.array([1., -0.1])))


@pytest.mark.slow
def test_kron_sampling_moments(key):
    A, B = jnp.array([[1., .2], [.2, .7]]), jnp.diag(jnp.array([1.3, .8]))
    keys = jax.random.split(key, 6000)
    draws = jax.vmap(lambda k: sample_kron_psd(k, jnp.zeros((2, 2)), A, B))(keys)
    assert np.allclose(np.cov(np.asarray(draws).reshape(6000, -1), rowvar=False),
                       np.asarray(jnp.kron(A, B)), atol=.06)
