import math

import jax
import jax.numpy as jnp
import numpy as np

from rbpf_v2.src.bivariate_poisson import daily_loglik, loglik, loglik_grid


def test_grid_matches_direct_shared_poisson_sum():
    xh, xa = jnp.array([.2, -.1]), jnp.array([-.3, .1])
    alpha, beta = .1, -1.2
    grid = np.exp(np.asarray(loglik_grid(xh, xa, alpha, beta, 5)))
    l1, l2, l3 = np.exp(alpha + xh[0] - xa[1]), np.exp(alpha + xa[0] - xh[1]), np.exp(beta)
    yh, ya = 2, 1
    direct = sum(math.exp(-(l1+l2+l3)) * l1**(yh-k) / math.factorial(yh-k)
                 * l2**(ya-k) / math.factorial(ya-k) * l3**k / math.factorial(k)
                 for k in range(min(yh, ya)+1))
    assert np.isclose(grid[yh, ya], direct, rtol=1e-5)
    assert 0 < grid.sum() <= 1


def test_bounds_swap_and_gradients():
    xi, xj, y = jnp.array([.1, .2]), jnp.array([-.2, .3]), jnp.array([2, 1])
    lhs = loglik(y, xi, xj, .1, -1., 4)
    rhs = loglik(y[::-1], xj, xi, .1, -1., 4)
    assert jnp.allclose(lhs, rhs)
    assert jnp.isneginf(loglik(jnp.array([5, 5]), xi, xj, .1, -1., 4))
    grad = jax.grad(lambda a: loglik(y, xi, xj, a, -1., 4))(.1)
    assert jnp.isfinite(grad)


def test_padding_is_zero(small_model):
    _, data, params = small_model
    day = jax.tree.map(lambda x: x[0], data)
    value = daily_loglik(params.mean_0, day, params.alpha, params.beta, 8)
    assert jnp.isfinite(value)
