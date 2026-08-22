"""
Run prediction

take in and filter
sequentially generate and store prediction
compare predictions to actual results
update the filter with new match results
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from rbsqmc.src.data.bivariate_poisson import loglik_grid
from rbsqmc.src.data.data import concat_football_results
from rbsqmc.src.model.optimization import run_filter_unbiased
from rbsqmc.src.utils.type import EMParams, Matches, RBPFState, FootballResults

@jax.jit(static_argnames=("max_goals",))
def predict_match_score(
    particles_x: jax.Array,     # (N, num_teams, 2)
    log_weights: jax.Array,     # (N,)
    home_id: int,
    away_id: int,
    alpha: float,
    beta: float,
    max_goals: int,
):
    """Weighted posterior predictive score grid (G x G), normalized to sum to 1."""
    # (N, G, G) log-likelihood grids, one per particle
    log_grid = jax.vmap(
        lambda p: loglik_grid(
            p[home_id], p[away_id], alpha=alpha, beta=beta,
            max_goals=max_goals, scale=1.0)
    )(particles_x)

    # Weighted average of likelihoods: logsumexp over particles with the
    # unnormalized log-weights, normalized by sum(weights).
    w = jnp.exp(log_weights)                     # (N,)
    log_w = jnp.log(w + 1e-12)
    log_w = log_w - logsumexp(log_w)             # normalize log-weights
    logp = logsumexp(log_grid + log_w[:, None, None], axis=0)  # (G, G)

    grid = jnp.exp(logp)
    return grid / grid.sum()

# ---------------------------------------------------------------------------
# Sequential predict -> update -> compare driver
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def run_sequential_predict(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
):
    """One-step-ahead prediction over the prediction window.

    Returns per-day predicted grids and the predictive log-prob of the actual
    score, for every prediction day.
    """
    # ---- 1. Filter over the FULL concatenated sequence --------------------
    # concat_football_results fixes the timestamp_prev boundary and pads M.
    full_inputs = concat_football_results(observed_inputs, prediction_inputs)
    filtered_states, _ = run_filter_unbiased(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )
    # filtered_states.{particles.x, log_weights} have leading dim T_full + 1
    # (index 0 is the initial state). Day t (0-based, in full_inputs coords)
    # has its POSTERIOR at filtered_states index t+1, and the state used to
    # PREDICT day t is the day-(t-1) posterior at filtered_states index t.

    G = max_goals + 1
    # In full-sequence coords, prediction day p (0-based) is full day
    # t = n_obs_days + p. Its predictive state is at filtered index t.
    n_obs_days = observed_inputs.timestamp.shape[0]  # days in the observed sequence
    pred_start = n_obs_days  # full-sequence day index of first prediction day

    def scan_body(_, pred_day):
        t = pred_start + pred_day            # full-sequence day index
        # predictive state = posterior BEFORE day t's matches are absorbed
        x_prev = filtered_states.particles.x[t]      # (N, teams, 2)
        lw_prev = filtered_states.log_weights[t]     # (N,)

        home_id = prediction_inputs.matches.home_id[pred_day]   # (M,)
        away_id = prediction_inputs.matches.away_id[pred_day]
        valid = prediction_inputs.match_mask[pred_day]
        yh = prediction_inputs.matches.home_score[pred_day]
        ya = prediction_inputs.matches.away_score[pred_day]

        def one_match(h, a, v, sh, sa):
            grid = predict_match_score(x_prev, lw_prev, h, a,
                                       params.alpha, params.beta, max_goals)
            grid = jnp.where(v, grid, jnp.zeros((G, G)))
            # predictive log-prob of the ACTUAL score (compare pred vs actual)
            logp_actual = jnp.where(
                v, jnp.log(grid[sh, sa] + 1e-12), 0.0
            )
            return grid, logp_actual

        grids, logp_actual = jax.vmap(one_match)(home_id, away_id, valid, yh, ya)
        # grids: (M, G, G); logp_actual: (M,)
        day_logp = jnp.sum(logp_actual)            # predictive log-score for the day
        return None, (grids, logp_actual, day_logp)

    n_pred_days = prediction_inputs.timestamp.shape[0]
    _, (all_grids, all_logp_actual, day_logp) = jax.lax.scan(
        scan_body, None, jnp.arange(n_pred_days)
    )
    # all_grids:      (P, M, G, G)  predicted grid per match per day
    # all_logp_actual:(P, M)        predictive log-prob of actual score per match
    # day_logp:       (P,)          total predictive log-prob per day
    return all_grids, all_logp_actual, day_logp


def main():
    pass

if __name__ == "__main__":
    main()