import jax
import jax.numpy as jnp
from typing import NamedTuple


class FootballResults(NamedTuple):
    match_index_id: jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array


class RBPFState(NamedTuple):
    x: jax.Array


class RBPFFootballResults(NamedTuple):
    """Augmented model inputs carrying the deterministic team-covariance
    trajectory.

    The OU model uses a Kronecker-structured covariance ``Sigma = gamma (x) B``
    with a *shared* attack/defence factor ``B``. This lets us track only the
    ``M x M`` team covariance ``gamma_t`` (plus the fixed ``2 x 2`` ``B``)
    instead of the full ``2M x 2M`` matrix, giving a 4x memory reduction that
    scales to many teams.

    - ``gamma_t``:      (T, M, M) filtered posterior team covariance ``Gamma_{t|t}``
    - ``gamma_pred_t``: (T, M, M) prediction team covariance ``Gamma_{t|t-1}``
    - ``kalman_gain_t``:(T, M, 2)  Kalman gain in team space (all teams vs the
      two observed teams)
    """
    match_index_id: jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array
    gamma_t: jax.Array          # (T, M, M) filtered posterior team covariance
    gamma_pred_t: jax.Array     # (T, M, M) prediction team covariance
    kalman_gain_t: jax.Array    # (T, M, 2)  Kalman gain (all teams vs observed)


class EMParams(NamedTuple):
    """All parameters that the model optimizes.

    The covariance is Kronecker-structured ``Sigma = gamma (x) B`` with a
    *shared* attack/defence factor ``B``, so the initial/stationary covariance
    is ``Sigma_0 = gamma_0 (x) B``.

    The transition is a scalar-phi AR(1) (discrete OU) with mean-reversion rate
    ``kappa``:

        phi_t = exp(-kappa * dt)
        mu_{t|t-1} = mu + phi_t (mu_{t-1|t-1} - mu)
        gamma_{t|t-1} = phi_t^2 gamma_{t-1|t-1} + (1 - phi_t^2) gamma_0

    The convex-combination covariance is PD by construction (sum of two PD
    matrices with positive weights) and team-specific (heavily-observed teams
    have small posterior covariance, so their transition noise is small).

    - mean_0:  (M, 2) initial mean (fixed at zeros during training).
    - gamma_0: (M, M) team factor of the initial/stationary covariance ``Sigma_0 = gamma_0 (x) B``.
    - B:       (2, 2) shared attack/defence factor (used for both Sigma_0 and Q).
    - kappa:   scalar mean-reversion rate of the OU transition.
    - alpha:   scalar baseline scoring rate.
    - beta:    scalar shared-scoring / correlation parameter.

    ``scale`` is fixed at 1.0 (not a parameter): it is unidentifiable with
    ``gamma_0`` (scaling ``x -> a x``, ``gamma_0 -> a^2 gamma_0``, ``scale -> a scale``
    leaves the likelihood unchanged), so fixing it at 1 breaks the degeneracy and
    lets ``gamma_0`` carry the true state-variance scale.
    """
    mean_0: jax.Array      # (M, 2)
    gamma_0: jax.Array     # (M, M)
    B: jax.Array           # (2, 2)
    kappa: float
    alpha: float
    beta: float
