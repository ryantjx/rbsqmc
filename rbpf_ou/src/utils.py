import jax
import jax.numpy as jnp
from typing import NamedTuple


def kron_sample_psd(key, mean, A, B):
    """Sample from N(mean, A (x) B) without forming A (x) B.

    mean: (M*K,) flattened vec_C of an (M, K) matrix. A: (M, M), B: (K, K).
    Returns (M*K,). Eigenvalues of A and B are clipped to >= 0 so observed
    (zero-variance) teams stay exactly at their mean (PSD-aware).
    """
    M = A.shape[0]
    K = B.shape[0]
    mean_MK = mean.reshape(M, K)
    eigvals_A, eigvecs_A = jnp.linalg.eigh(A)
    eigvals_A = jnp.clip(eigvals_A, 0.0)
    eigvals_B, eigvecs_B = jnp.linalg.eigh(B)
    eigvals_B = jnp.clip(eigvals_B, 0.0)
    z = jax.random.normal(key, (M, K))
    Z_A = (eigvecs_A * jnp.sqrt(eigvals_A)[None, :]) @ z
    X = Z_A @ (eigvecs_B * jnp.sqrt(eigvals_B)[None, :]).T
    return (mean_MK + X).reshape(-1)


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

    The random-walk model in MODEL.md uses a Kronecker-structured covariance
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
    """All parameters that EM optimizes.

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

    - mean_0:  (M, 2) initial mean (fixed at zeros during EM).
    - gamma_0: (M, M) team factor of the initial/stationary covariance ``Sigma_0 = gamma_0 (x) B``.
    - B:       (2, 2) shared attack/defence factor (used for both Sigma_0 and Q).
    - kappa:   scalar mean-reversion rate of the OU transition.
    - alpha:   scalar baseline scoring rate.
    - beta:    scalar shared-scoring / correlation parameter.
    - scale:   scalar controlling the influence of team strength on the goal
      rates (``log_lambda = alpha + (x_att - x_def)/scale``). Free during EM.
    """
    mean_0: jax.Array      # (M, 2)
    gamma_0: jax.Array     # (M, M)
    B: jax.Array           # (2, 2)
    kappa: float
    alpha: float
    beta: float
    scale: float
