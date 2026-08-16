import time

import jax
import pytest

from rbpf_v3.src import smoothing, smoothing_noncuthbert


@pytest.mark.slow
@pytest.mark.parametrize("module", [smoothing, smoothing_noncuthbert])
def test_warmed_smoother_runs_at_d512_without_oom(module, small_problem):
    _, params, filtered, augmented = small_problem
    # Repeat the valid small timeline to exercise a long fixed-shape graph.
    repeats = 128
    means = jax.numpy.concatenate(
        [jax.numpy.tile(filtered.particles.x[:-1], (repeats, 1, 1, 1)), filtered.particles.x[-1:]],
        axis=0,
    )
    weights = jax.numpy.concatenate(
        [jax.numpy.tile(filtered.log_weights[:-1], (repeats, 1)), filtered.log_weights[-1:]],
        axis=0,
    )
    logz = jax.numpy.concatenate(
        [
            jax.numpy.tile(filtered.log_normalizing_constant[:-1], repeats),
            filtered.log_normalizing_constant[-1:],
        ],
        axis=0,
    )
    long_filter = filtered._replace(
        particles=filtered.particles._replace(x=means),
        log_weights=weights,
        log_normalizing_constant=logz,
    )
    long_augmented = jax.tree.map(lambda value: jax.numpy.tile(value, (repeats,) + (1,) * (value.ndim - 1)), augmented)
    result = module.rb_backward_simulation(
        jax.random.key(5), long_filter, long_augmented, params, 8
    )
    jax.block_until_ready(result.x)
    assert result.x.shape[0] == 513
