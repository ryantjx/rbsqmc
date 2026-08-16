"""Numerically stable bivariate-Poisson score probabilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp


def _log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


@partial(jax.jit, static_argnames=("max_goals",))
def loglik(y, x_i, x_j, alpha, beta, max_goals: int, scale=1.0):
    """Return log p(home_score, away_score), or ``-inf`` outside the bound."""
    y = jnp.asarray(y)
    yh, ya = y[0], y[1]
    l1 = alpha + (x_i[0] - x_j[1]) / scale
    l2 = alpha + (x_j[0] - x_i[1]) / scale
    l3 = beta
    base = -(jnp.exp(l1) + jnp.exp(l2) + jnp.exp(l3))
    base += yh * l1 - gammaln(yh + 1) + ya * l2 - gammaln(ya + 1)
    k = jnp.arange(max_goals + 1)
    terms = _log_binom(yh, k) + _log_binom(ya, k) + gammaln(k + 1)
    terms += k * (l3 - l1 - l2)
    limit = jnp.minimum(yh, ya)
    terms = jnp.where(k <= limit, terms, -jnp.inf)
    valid = (yh >= 0) & (ya >= 0) & (limit <= max_goals)
    return jnp.where(valid, base + logsumexp(terms), -jnp.inf)


@partial(jax.jit, static_argnames=("max_goals",))
def loglik_grid(x_i, x_j, alpha, beta, max_goals: int, scale=1.0):
    scores = jnp.arange(max_goals + 1)
    return jax.vmap(
        lambda yh: jax.vmap(
            lambda ya: loglik(
                jnp.array([yh, ya]), x_i, x_j, alpha, beta, max_goals, scale
            )
        )(scores)
    )(scores)


def daily_loglik(state, day, alpha, beta, max_goals: int):
    """Sum valid match likelihoods; padding contributes exactly zero."""
    def one(h, a, yh, ya, valid):
        value = loglik(jnp.array([yh, ya]), state[h], state[a], alpha, beta,
                       max_goals)
        return jnp.where(valid, value, 0.0)

    return jnp.sum(jax.vmap(one)(day.matches.home_id, day.matches.away_id,
                                  day.matches.home_score, day.matches.away_score,
                                  day.match_mask))
