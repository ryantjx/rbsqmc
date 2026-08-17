import jax
import jax.numpy as jnp

from rbpf_v2.src.smoothing import MCEMConfig
from rbpf_v2.src.smoothing_cuthbert import run_mcem


def test_cuthbert_mcem_entrypoint(small_model):
    _, data, params = small_model
    result = run_mcem(
        jax.random.key(12), data, params,
        MCEMConfig(8, 8, 1, 1, 1e-3, 8, 1e-6),
    )
    assert result["final_smoothed_paths"].shape == (data.timestamp.size + 1, 8, 4, 2)
    assert jnp.isfinite(result["final_log_marginal_likelihood"])
    assert result["mstep_history"][0]["candidate_objective"] + 1e-6 >= (
        result["mstep_history"][0]["start_objective"]
    )
