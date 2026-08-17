import jax.numpy as jnp
import jax

from rbpf_v2.src.data import slice_results
from rbpf_v2.src.evaluation import predictive_score_grid, result_probabilities, rolling_origin_predictive_evaluation
from rbpf_v2.src.smoothing import MCEMConfig, run_mcem


def test_predictive_grid_is_valid(small_model):
    _, _, params = small_model
    states = jnp.stack([params.mean_0, params.mean_0 + .1])
    grid = predictive_score_grid(states, jnp.log(jnp.array([.5, .5])), 0, 1, params, 8)
    assert jnp.isfinite(grid).all() and grid.min() >= 0 and grid.sum() <= 1.00001
    assert jnp.allclose(result_probabilities(grid).sum(), grid.sum())


def test_chronological_holdout_is_scored(small_model):
    _, data, params = small_model
    train, holdout = slice_results(data, None, -1), slice_results(data, -1, None)
    result = run_mcem(jax.random.key(0), train, params,
                      MCEMConfig(8, 8, 1, 1, 1e-3, 8, 1e-6))
    metrics = rolling_origin_predictive_evaluation(result, holdout, seed=0)
    assert metrics["available"] and metrics["n_matches"] == 2
    assert jnp.isfinite(metrics["mean_negative_log_predictive_density"])
