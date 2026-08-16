import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rbpf_v3.src import smoothing, smoothing_noncuthbert


@pytest.mark.slow
@pytest.mark.parametrize("module", [smoothing, smoothing_noncuthbert])
def test_batched_kron_draw_moments(module):
    gamma = jnp.asarray([[1.0, 0.25], [0.25, 0.7]])
    B = jnp.asarray([[1.3, -0.1], [-0.1, 0.8]])
    means = jnp.zeros((20000, 2, 2))
    draws = module.sample_kron_psd_batched(
        jax.random.key(1), means, module.psd_sqrt(gamma), module.psd_sqrt(B)
    )
    empirical = np.cov(np.asarray(draws).reshape((draws.shape[0], -1)), rowvar=False)
    np.testing.assert_allclose(empirical, np.kron(gamma, B), rtol=0.06, atol=0.04)
