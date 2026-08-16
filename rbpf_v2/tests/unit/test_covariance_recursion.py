import jax.numpy as jnp

from rbpf_v2.src.model import compute_gamma_trajectory


def test_covariance_shapes_timeline_and_conditioning(small_model):
    _, data, params = small_model
    gamma, predicted, observed, gain = compute_gamma_trajectory(data, params.gamma_0,
                                                                 params.kappa)
    D, L = data.match_mask.shape
    M = params.mean_0.shape[0]
    assert gamma.shape == predicted.shape == (D, M, M)
    assert observed.shape == (D, L, 2, 2) and gain.shape == (D, L, M, 2)
    assert jnp.linalg.eigvalsh(predicted).min() > 0
    assert jnp.allclose(gamma, gamma.transpose(0, 2, 1))
    # All four teams play in the fixture, hence no within-component covariance remains.
    assert jnp.allclose(gamma, 0, atol=1e-6)
