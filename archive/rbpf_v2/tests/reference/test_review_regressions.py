import jax
import jax.numpy as jnp

from rbpf_v2.src.model import run_filter
from rbpf_v2.src.smoothing import rb_backward_simulation, smoothed_path_diagnostics


def test_rb_smoother_materializes_and_selects_fresh_components(small_model):
    _, data, params = small_model
    filtered, augmented = run_filter(jax.random.key(2), data, params, 32, 8)
    smoothed = rb_backward_simulation(jax.random.key(3), filtered, augmented, params, 64)
    assert smoothed.particles.x.shape == (data.timestamp.size + 1, 64, 4, 2)
    assert float(jnp.var(smoothed.particles.x[0] - params.mean_0)) > 0
    assert len(jnp.unique(smoothed.component_indices[0])) > 1
    diagnostics = smoothed_path_diagnostics(params, smoothed, augmented)
    assert jnp.isfinite(diagnostics["transition_mahalanobis_ratio"])
    assert diagnostics["transition_mahalanobis_ratio"] < 10
