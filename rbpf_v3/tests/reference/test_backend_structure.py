import inspect
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from rbpf_v3.src import smoothing, smoothing_noncuthbert


def test_cuthbert_adapter_is_compact_and_ignores_density_and_genealogy(small_problem):
    _, params, filtered, augmented = small_problem
    adapted = smoothing.make_rb_smoother_filter_states(filtered, augmented, params)
    assert adapted.particles._fields == ("x", "time_index")
    assert adapted.particles.time_index.shape == filtered.log_weights.shape
    assert adapted.particles.x.shape == filtered.particles.x.shape
    assert adapted.particles.time_index.ndim == 2
    x0 = jax.tree.map(lambda value: value[1], adapted.particles)
    x1 = jax.tree.map(lambda value: value[2, :4], adapted.particles)
    gamma_filtered = jnp.concatenate([params.gamma_0[None], augmented.gamma])
    phi = jnp.exp(-params.kappa * (augmented.timestamp - augmented.timestamp_prev))
    kwargs = dict(
        mean_0=params.mean_0,
        B=params.B,
        gamma_filtered=gamma_filtered,
        gamma_pred=augmented.gamma_pred,
        phi=phi,
    )
    first = smoothing.rb_backward_sampling_fn(
        jax.random.key(9), x0, x1, filtered.log_weights[1], object(), jnp.zeros(4), **kwargs
    )
    second = smoothing.rb_backward_sampling_fn(
        jax.random.key(9), x0, x1, filtered.log_weights[1], lambda *_: jnp.inf, jnp.arange(4), **kwargs
    )
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(left, right)


def test_direct_module_imports_with_cuthbert_blocked():
    code = (
        "import sys; sys.modules['cuthbert']=None; sys.modules['cuthbertlib']=None; "
        "import rbpf_v3.src.smoothing_noncuthbert"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_direct_core_is_reverse_scan_and_has_no_cross_backend_import():
    source = inspect.getsource(smoothing_noncuthbert._rb_backward_simulation_jit)
    assert "jax.lax.scan" in source
    assert "reverse=True" in source
    module_source = inspect.getsource(smoothing_noncuthbert)
    assert "import cuthbert" not in module_source
    assert "from rbpf_v3.src.smoothing import" not in module_source
