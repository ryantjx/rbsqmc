import jax
import jax.numpy as jnp

from rbpf_v2.src.smoothing import E_step


def test_e_step_contract(small_model):
    _, data, params = small_model
    smooth, filtered, augmented = E_step(params, data, 8, 8, 8, jax.random.key(1))
    assert smooth.particles.x.shape == (data.timestamp.size + 1, 8, 4, 2)
    assert filtered.particles.x.shape == (data.timestamp.size + 1, 8, 4, 2)
    assert augmented.gamma_pred.shape[0] == data.timestamp.size
    assert jnp.isfinite(smooth.particles.x).all()
