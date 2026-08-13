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

    The random-walk model uses a Kronecker-structured covariance
    ``Sigma = gamma (x) B`` with a *shared* attack/defence factor ``B``
    (``B_0 = B_Q = B``). This lets us track only the ``M x M`` team covariance
    ``gamma_t`` (plus the fixed ``2 x 2`` ``B``) instead of the full
    ``2M x 2M`` matrix, giving a 4x memory reduction that scales to many teams.

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
    *shared* attack/defence factor ``B`` (``B_0 = B_Q = B``), so the initial
    covariance is ``Sigma_0 = gamma_0 (x) B`` and the transition covariance is
    ``Q = gamma_Q (x) B``.

    The transition is a pure random walk (no mean reversion):

        X_t = X_{t-1} + eps_t,   eps_t ~ N(0, dt * Q),  Q = gamma_Q (x) B

    - mean_0:  (M, 2) initial mean (fixed at zeros during training).
    - gamma_0: (M, M) team factor of the initial covariance ``Sigma_0 = gamma_0 (x) B``.
    - gamma_Q: (M, M) team factor of the transition covariance ``Q = gamma_Q (x) B``.
    - B:       (2, 2) shared attack/defence factor (used for both Sigma_0 and Q).
    - alpha:   scalar baseline scoring rate.
    - beta:    scalar shared-scoring / correlation parameter.
    """
    mean_0: jax.Array
    gamma_0: jax.Array
    gamma_Q: jax.Array
    B: jax.Array
    alpha: float
    beta: float
