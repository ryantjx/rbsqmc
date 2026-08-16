import jax.numpy as jnp

from rbpf_v2.src.evaluation import backward_metrics, brier_score, parameter_diagnostics, result_probabilities


def test_known_metrics(small_model):
    _, _, params = small_model
    p = jnp.array([[[.5, .5], [.25, .75]]])
    metrics = backward_metrics(p, jnp.array([[0, 1]]))
    assert jnp.allclose(metrics["ess_min"], 1 / (.25**2 + .75**2))
    assert parameter_diagnostics(params)["gamma_effective_rank"] == 4
    grid = jnp.array([[.2, .1], [.6, .1]])
    assert jnp.allclose(result_probabilities(grid), jnp.array([.6, .3, .1]))
    assert brier_score(jnp.array([1., 0., 0.]), 0) == 0
