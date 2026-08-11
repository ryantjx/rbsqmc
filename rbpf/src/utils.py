import jax
from typing import NamedTuple

class FootballResults(NamedTuple):
    match_index_id : jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array

class RBPFState(NamedTuple):
    x: jax.Array

class RBPFFootballResults(NamedTuple):
    match_index_id : jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array
    gamma_t: jax.Array
    gamma_pred_t: jax.Array
    kalman_gain_t: jax.Array
    
class EMParams(NamedTuple):
    """All parameters that EM optimizes."""
    mean_0: jax.Array      # (M, 2)
    gamma_0: jax.Array     # (M, M)
    B: jax.Array         # (2, 2)
    kappa: float
    alpha: float
    beta: float