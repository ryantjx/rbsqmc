import jax
import jax.numpy as jnp

from rbpf_v2.src.smoothing import MCEMConfig, run_mcem


def test_one_epoch_nonworsening_and_finite(small_model):
    _, data, params = small_model
    result = run_mcem(jax.random.key(4), data, params,
                      MCEMConfig(8, 8, 1, 1, 1e-3, 8, 1e-6))
    record = result["mstep_history"][0]
    assert record["candidate_objective"] + 1e-6 >= record["start_objective"]
    assert all(jnp.isfinite(record[name]) for name in ("initial", "transition", "observation", "prior", "total"))
    assert jnp.linalg.det(result["final_params"].B) == jnp.asarray(1., dtype=result["final_params"].B.dtype)
