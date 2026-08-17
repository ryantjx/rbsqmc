import jax
import jax.numpy as jnp
import numpy as np

from rbpf_v2.src.kron import sample_kron_psd
from rbpf_v2.src.smoothing import terminal_resample


def test_terminal_draw_has_residual_uncertainty(key):
    means = jnp.zeros((4, 2, 2))
    gamma, B = jnp.array([[1., .2], [.2, .8]]), jnp.diag(jnp.array([1.2, 1/1.2]))
    draws, indices = terminal_resample(key, means, jnp.zeros(4), gamma, B, 512)
    assert draws.shape == (512, 2, 2)
    assert float(jnp.var(draws)) > .3
    assert indices.min() >= 0 and indices.max() < 4


def test_conditional_draw_moments(key):
    mean = jnp.arange(4.).reshape(2,2) / 10
    gamma, B = jnp.array([[.4, .1], [.1, .3]]), jnp.diag(jnp.array([1.5, 1/1.5]))
    keys = jax.random.split(key, 5000)
    draws = jax.vmap(lambda k: sample_kron_psd(k, mean, gamma, B))(keys)
    assert np.allclose(np.asarray(draws.mean(0)), np.asarray(mean), atol=.04)
    assert np.allclose(np.cov(np.asarray(draws).reshape(5000,-1), rowvar=False),
                       np.asarray(jnp.kron(gamma, B)), atol=.04)
