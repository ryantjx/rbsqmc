import importlib.util
from pathlib import Path

import jax.numpy as jnp
from jax import random
import numpy as np
import pytest

from sqmc.sqmc.benchmark_sqmc import (
    _unique_ancestor_fraction,
    _weighted_diagnostics,
    generate_observations,
    make_sqmc_model,
    run_sqmc_diagnostics_jit,
    run_sqmc_jit,
)
from sqmc.sqmc.benchmark_sqmc_smc import (
    METRICS,
    _replicate_keys,
    evaluate_claims,
    kalman_filter_reference,
    make_smc_model,
    run_smc_diagnostics_jit,
    run_smc_jit,
)


def test_kalman_reference_matches_scalar_analytic_recursion():
    observations = jnp.zeros((3, 1), dtype=jnp.float64)

    means, variances = kalman_filter_reference(observations)

    np.testing.assert_allclose(means, 0.0)
    np.testing.assert_allclose(variances, [0.5, 3.0 / 7.0, 19.0 / 47.0])


def test_generated_initial_state_uses_the_filter_prior():
    key = random.PRNGKey(7)
    initial_key, _, observation_key = random.split(key, 3)

    states, observations = generate_observations(key, n_steps=1, dimension=3)

    expected_state = random.normal(initial_key, (3,))
    expected_noise_key = random.split(observation_key, 1)[0]
    expected_observation = expected_state + random.normal(expected_noise_key, (3,))
    np.testing.assert_allclose(states[0], expected_state)
    np.testing.assert_allclose(observations[0], expected_observation)


def test_normalized_ess_for_uniform_and_concentrated_weights():
    particles = jnp.arange(4.0)[:, None]
    mean, variance, uniform_ess = _weighted_diagnostics(
        particles, jnp.zeros(4), n_particles=4
    )
    _, _, concentrated_ess = _weighted_diagnostics(
        particles, jnp.array([0.0, -100.0, -100.0, -100.0]), n_particles=4
    )

    np.testing.assert_allclose(mean, [1.5])
    np.testing.assert_allclose(variance, [1.25])
    assert float(uniform_ess) == pytest.approx(1.0)
    assert float(concentrated_ess) == pytest.approx(0.25)


def test_unique_ancestor_fraction_extremes():
    assert float(_unique_ancestor_fraction(jnp.arange(4), 4)) == pytest.approx(1.0)
    assert float(_unique_ancestor_fraction(jnp.zeros(4, dtype=int), 4)) == pytest.approx(
        0.25
    )


def test_paired_replicates_share_data_but_use_independent_method_keys():
    first = _replicate_keys(42, dimension=30, replicate=3)
    second = _replicate_keys(42, dimension=30, replicate=3)
    other_replicate = _replicate_keys(42, dimension=30, replicate=4)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[1], first[2])
    assert not np.array_equal(first[0], other_replicate[0])
    _, observations_a = generate_observations(first[0], 4, 2)
    _, observations_b = generate_observations(second[0], 4, 2)
    np.testing.assert_array_equal(observations_a, observations_b)


@pytest.mark.parametrize(
    ("timed_runner", "diagnostic_runner", "model_factory"),
    [
        (run_smc_jit, run_smc_diagnostics_jit, make_smc_model),
        (run_sqmc_jit, run_sqmc_diagnostics_jit, make_sqmc_model),
    ],
)
def test_timed_scan_omits_diagnostic_outputs(
    timed_runner, diagnostic_runner, model_factory
):
    observations = jnp.zeros((3, 2))
    key = random.PRNGKey(5)
    timed = timed_runner(observations, 4, key, 2, model_factory(2))
    diagnostic = diagnostic_runner(observations, 4, key, 2, model_factory(2))

    assert timed[3].shape == (2,)
    assert isinstance(diagnostic[3], tuple)
    assert len(diagnostic[3]) == 4


def _benchmark_entry(method, error_shift):
    replicates = []
    summary = {}
    values_by_metric = {
        "mean_nrmse": 0.2 + error_shift,
        "variance_relative_rmse": 0.3 + error_shift,
        "normalized_ess": 0.6 - error_shift,
        "unique_ancestor_fraction": 0.5 - error_shift,
    }
    for replicate in range(8):
        replicates.append(
            {
                "replicate": replicate,
                **{metric: value for metric, value in values_by_metric.items()},
            }
        )
    for metric, value in values_by_metric.items():
        summary[metric] = {
            "mean": value,
            "std": 0.0,
            "ci95_low": value,
            "ci95_high": value,
            "n": 8,
        }
    runtime = 2.0 if method == "sqmc" else 1.0
    return {
        "runtime": {
            "mean": runtime,
            "std": 0.0,
            "ci95_low": runtime,
            "ci95_high": runtime,
        },
        "diagnostics": {"replicates": replicates, "summary": summary},
    }


def test_claims_use_gpu_dimension_method_particle_schema():
    results = {
        "gpu": {
            "2": {
                "smc": {"64": _benchmark_entry("smc", 0.05)},
                "sqmc": {"64": _benchmark_entry("sqmc", 0.0)},
            }
        }
    }

    claims = evaluate_claims(results, dimensions=[2], particle_counts=[64])

    assert claims["same_n_runtime_ratio_sqmc_over_smc"]["2"]["64"]["point"] == 2.0
    assert set(
        claims["paired_diagnostic_differences"]["2"]["64"]
    ).issuperset(METRICS)
    assert claims["pareto_frontiers"]["2"]["combined"]["mean_nrmse"]["status"]


def test_gpu_runner_validates_multidimensional_gpu_only_config():
    runner_path = (
        Path(__file__).parents[1]
        / "sqmc"
        / "scripts"
        / "run_sqmc_benchmarks_gpu.py"
    )
    spec = importlib.util.spec_from_file_location("sqmc_gpu_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    config = {
        "remote_output_dir": "outputs",
        "sqmc": {
            "n_steps": 100,
            "n_reps": 7,
            "accuracy_reps": 8,
            "warmups": 2,
            "particle_counts": [64, 256],
            "dimensions": [2, 60],
            "platforms": ["gpu"],
            "seed": 42,
            "precision": "float64",
            "jit": True,
        },
    }

    assert runner.validate_config(config) is config
    command = runner.benchmark_command(config, Path("outputs"))
    assert "--dimensions" in command
    assert "--base-dimension" not in command
    config["sqmc"]["platforms"] = ["cpu"]
    with pytest.raises(ValueError, match="exactly"):
        runner.validate_config(config)
