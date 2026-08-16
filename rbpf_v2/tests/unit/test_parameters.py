import jax
import jax.numpy as jnp

from rbpf_v2.src.helpers import decode_EM_params, encode_EM_params, log_inverse_wishart_kernel


def test_parameter_roundtrip_and_constraints(small_model):
    _, _, params = small_model
    decoded = decode_EM_params(encode_EM_params(params), params.mean_0)
    assert jnp.allclose(decoded.gamma_0, params.gamma_0, atol=2e-5)
    assert decoded.kappa > 0
    assert jnp.allclose(jnp.linalg.det(decoded.B), 1.)
    assert jnp.allclose(decoded.B, jnp.diag(jnp.diag(decoded.B)))
    assert jnp.linalg.eigvalsh(decoded.gamma_0).min() > 0


def test_prior_and_gradients_finite(small_model):
    _, _, params = small_model
    value = log_inverse_wishart_kernel(params.gamma_0, 14, 19 * params.gamma_0)
    grad = jax.grad(lambda g: log_inverse_wishart_kernel(g, 14, 19 * params.gamma_0))(params.gamma_0)
    assert jnp.isfinite(value) and jnp.isfinite(grad).all()
