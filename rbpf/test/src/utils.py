import jax
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
    """Augmented model inputs carrying the deterministic covariance trajectory.

    Unlike the OU reference (which stored a Kronecker ``gamma_t`` of shape
    ``(M, M)``), the random-walk model in MODEL.md has a *general* ``2M x 2M``
    covariance trajectory. ``sigma_t`` is the filtered posterior covariance
    ``Sigma_{t|t}``, ``sigma_pred_t`` is the prediction covariance
    ``Sigma_{t|t-1}``, and ``kalman_gain_t`` is the Kalman gain ``K_t`` used to
    condition the unobserved (Rao-Blackwellized) teams on the sampled observed
    block.
    """
    match_index_id: jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array
    sigma_t: jax.Array          # (T, 2M, 2M) filtered posterior covariance
    sigma_pred_t: jax.Array     # (T, 2M, 2M) prediction covariance
    kalman_gain_t: jax.Array    # (T, 2M, 4)  Kalman gain (all teams vs observed)


class EMParams(NamedTuple):
    """All parameters that EM optimizes.

    - mean_0:  (M, 2) initial mean (fixed at zeros during EM).
    - sigma_0: (2M, 2M) general initial covariance (Cholesky-parameterized in
      the M-step; NOT a Kronecker product).
    - gamma_Q: (M, M) team-correlation factor of the transition covariance
      ``Q = gamma_Q (x) B_Q``.
    - B_Q:     (2, 2) attack/defence factor of the transition covariance.
    - alpha:   scalar baseline scoring rate.
    - beta:    scalar shared-scoring / correlation parameter.
    """
    mean_0: jax.Array      # (M, 2)
    sigma_0: jax.Array     # (2M, 2M)
    gamma_Q: jax.Array     # (M, M)
    B_Q: jax.Array         # (2, 2)
    alpha: float
    beta: float
