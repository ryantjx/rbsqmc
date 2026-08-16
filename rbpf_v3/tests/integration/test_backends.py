import jax
import numpy as np
import pytest

from rbpf_v3.src import smoothing, smoothing_noncuthbert


@pytest.mark.parametrize("module", [smoothing, smoothing_noncuthbert])
def test_backend_e_step_and_mcem_schema(module, small_problem):
    data, params, _, _ = small_problem
    config = module.MCEMConfig(
        n_filter_particles=6,
        n_smoother_particles=5,
        n_epochs=1,
        n_gradient_steps=1,
        return_backward_probabilities=True,
    )
    result = module.run_mcem(jax.random.key(4), data, params, config)
    smoothed = result["final_smoothed_states"]
    assert smoothed.x.shape == (data.timestamp.size + 1, 5, 2, 2)
    assert smoothed.component_indices.shape == (data.timestamp.size + 1, 5)
    assert smoothed.diagnostics.probabilities.shape == (
        data.timestamp.size + 1,
        5,
        6,
    )
    assert np.isfinite(float(result["final_log_marginal_likelihood"]))
    assert result["mstep_history"][0]["accepted"]


def test_backend_distributional_equivalence_on_saved_filter(small_problem):
    _, params, filtered, augmented = small_problem
    direct = smoothing_noncuthbert.rb_backward_simulation(
        jax.random.key(8), filtered, augmented, params, 512, False
    )
    cuthbert = smoothing.rb_backward_simulation(
        jax.random.key(8), filtered, augmented, params, 512, False
    )
    jax.block_until_ready((direct.x, cuthbert.x))
    np.testing.assert_allclose(
        np.asarray(direct.x).mean(axis=1),
        np.asarray(cuthbert.x).mean(axis=1),
        rtol=0.2,
        atol=0.15,
    )
    np.testing.assert_allclose(
        np.asarray(direct.x).var(axis=1),
        np.asarray(cuthbert.x).var(axis=1),
        rtol=0.3,
        atol=0.15,
    )
    np.testing.assert_allclose(
        direct.diagnostics.ess_by_time,
        cuthbert.diagnostics.ess_by_time,
        rtol=0.15,
        atol=0.15,
    )


def test_default_diagnostics_omit_full_probabilities(small_problem):
    _, params, filtered, augmented = small_problem
    for module in (smoothing, smoothing_noncuthbert):
        result = module.rb_backward_simulation(
            jax.random.key(2), filtered, augmented, params, 4
        )
        assert result.diagnostics.probabilities is None
