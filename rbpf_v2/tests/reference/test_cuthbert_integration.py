import jax
import jax.numpy as jnp

from cuthbert.inference import Smoother

from rbpf_v2.src.model import run_filter
from rbpf_v2.src.smoothing_cuthbert import (
    build_smoother,
    make_rb_smoother_filter_states,
    rb_backward_sampling_fn,
    run_cuthbert_smoother,
)
from rbpf_v2.src.utils import RBSmootherParticle


def test_build_smoother_returns_cuthbert_object(small_model):
    _, _, params = small_model
    assert isinstance(build_smoother(params, 8, 8), Smoother)


def test_adapter_attaches_exact_timeline_metadata(small_model):
    _, data, params = small_model
    filtered, augmented = run_filter(jax.random.key(0), data, params, 8, 8)
    adapted = make_rb_smoother_filter_states(filtered, augmented, params)
    D, N = data.timestamp.size, 8
    assert adapted.particles.x.shape == (D + 1, N, 4, 2)
    assert adapted.particles.gamma_filtered.shape == (D + 1, N, 4, 4)
    assert jnp.allclose(adapted.particles.gamma_filtered[0, 0], params.gamma_0)
    assert jnp.allclose(adapted.particles.gamma_filtered[1:, 0], augmented.gamma)
    assert jnp.allclose(adapted.particles.gamma_pred_next[:-1, 0], augmented.gamma_pred)
    expected_phi = jnp.exp(-params.kappa * (data.timestamp - data.timestamp_prev))
    assert jnp.allclose(adapted.particles.phi_next[:-1, 0], expected_phi)
    assert jnp.array_equal(adapted.ancestor_indices[0], jnp.arange(N))


def test_callback_ignores_point_density_and_forward_ancestors(small_model):
    _, data, params = small_model
    filtered, augmented = run_filter(jax.random.key(1), data, params, 8, 8)
    adapted = make_rb_smoother_filter_states(filtered, augmented, params)
    x0 = jax.tree.map(lambda x: x[0], adapted.particles)
    future_x = jnp.broadcast_to(params.mean_0, (5,) + params.mean_0.shape)
    x1 = RBSmootherParticle(
        future_x,
        jnp.broadcast_to(params.gamma_0, (5,) + params.gamma_0.shape),
        jnp.broadcast_to(augmented.gamma_pred[0], (5,) + params.gamma_0.shape),
        jnp.ones(5),
    )

    def forbidden_density(*_):
        raise AssertionError("point-state log density must not be evaluated")

    kwargs = dict(
        key=jax.random.key(9), x0_all=x0, x1_all=x1,
        log_weight_x0_all=filtered.log_weights[0],
        log_density=forbidden_density, mean_0=params.mean_0, B=params.B,
    )
    first, first_indices = rb_backward_sampling_fn(
        **kwargs, x1_ancestor_indices=jnp.zeros(5, dtype=int)
    )
    second, second_indices = rb_backward_sampling_fn(
        **kwargs, x1_ancestor_indices=jnp.full(5, 7, dtype=int)
    )
    assert jnp.array_equal(first_indices, second_indices)
    assert jnp.array_equal(first.x, second.x)
    assert jnp.array_equal(first.gamma_filtered, x0.gamma_filtered[first_indices])
    assert jnp.array_equal(first.gamma_pred_next, x0.gamma_pred_next[first_indices])


def test_cuthbert_smoother_returns_full_paths_and_component_ids(small_model):
    _, data, params = small_model
    filtered, augmented = run_filter(jax.random.key(2), data, params, 16, 8)
    smoothed = run_cuthbert_smoother(
        jax.random.key(3), filtered, augmented, params, 32, 8
    )
    assert smoothed.particles.x.shape == (data.timestamp.size + 1, 32, 4, 2)
    assert smoothed.component_indices.shape == (data.timestamp.size + 1, 32)
    assert smoothed.backward_probabilities.shape == (data.timestamp.size + 1, 32, 16)
    assert smoothed.component_indices.min() >= 0
    assert smoothed.component_indices.max() < 16
    # State zero is a full Gaussian prior draw, not the repeated component mean.
    assert jnp.var(smoothed.particles.x[0] - params.mean_0) > 0
