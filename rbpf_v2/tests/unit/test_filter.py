import jax
import jax.numpy as jnp

from rbpf_v2.src.model import run_filter


def test_filter_contract_reproducibility(small_model, key):
    _, data, params = small_model
    first, augmented = run_filter(key, data, params, 8, 8)
    second, _ = run_filter(key, data, params, 8, 8)
    assert first.particles.x.shape == (data.timestamp.size + 1, 8, 4, 2)
    assert first.log_weights.shape == (data.timestamp.size + 1, 8)
    assert jnp.allclose(jnp.exp(first.log_weights).sum(axis=1), 1)
    assert jnp.isfinite(first.log_normalizing_constant).all()
    assert jnp.array_equal(first.particles.x, second.particles.x)
    changed, augmented_changed = run_filter(jax.random.key(99), data, params, 8, 8)
    assert not jnp.array_equal(first.particles.x, changed.particles.x)
    assert jnp.array_equal(augmented.gamma, augmented_changed.gamma)


def test_filter_jit_agrees_with_eager(small_model, key):
    _, data, params = small_model
    eager, _ = run_filter(key, data, params, 4, 8)
    compiled, _ = jax.jit(run_filter, static_argnames=("n_particles", "max_goals"))(
        key, data, params, 4, 8
    )
    assert jnp.allclose(eager.particles.x, compiled.particles.x)
