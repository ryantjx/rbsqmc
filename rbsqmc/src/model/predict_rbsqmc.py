"""Sequential one-step-ahead prediction with the RB-SQMC filter."""

from functools import partial

import jax
import jax.numpy as jnp

from rbsqmc.src.data.data import concat_football_results, unpack_football_results
from rbsqmc.src.model.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.predict import predict_match_score
from rbsqmc.src.utils.type import EMParams, FootballResults


def run_sequential_predict_rbsqmc(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
):
    """Unpack to match-level steps, then run compiled SQMC prediction."""
    return _run_sequential_predict_rbsqmc_jitted(
        key=key,
        observed_inputs=unpack_football_results(observed_inputs),
        prediction_inputs=unpack_football_results(prediction_inputs),
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def _run_sequential_predict_rbsqmc_jitted(
    key: jax.Array,
    observed_inputs: FootballResults,
    prediction_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
):
    """Run SQMC and return one-step-ahead score forecasts.

    The full observed-plus-prediction sequence is filtered once. For prediction
    timestep ``t``, the forecast uses the particles at history index ``t+1``:
    their positions have been propagated for the current match, while their
    score-dependent weights are replaced by uniform pre-likelihood weights.
    Prediction inputs should contain one match per leading timestep so
    same-date matches are forecast and updated sequentially.
    """
    full_inputs = concat_football_results(observed_inputs, prediction_inputs)
    result, _ = run_filter_sqmc(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )

    particles_x = result["particles_x"]
    log_weights = result["log_weights"]
    grid_size = max_goals + 1
    prediction_start = observed_inputs.timestamp.shape[0]

    def scan_body(_, prediction_step):
        history_index = prediction_start + prediction_step + 1
        particles_predictive = particles_x[history_index]
        weights_predictive = jnp.zeros_like(log_weights[history_index])

        home_ids = prediction_inputs.matches.home_id[prediction_step]
        away_ids = prediction_inputs.matches.away_id[prediction_step]
        valid = prediction_inputs.match_mask[prediction_step]
        home_scores = prediction_inputs.matches.home_score[prediction_step]
        away_scores = prediction_inputs.matches.away_score[prediction_step]

        def predict_one(home_id, away_id, is_valid, home_score, away_score):
            grid = predict_match_score(
                particles_predictive,
                weights_predictive,
                home_id,
                away_id,
                params.alpha,
                params.beta,
                max_goals,
            )
            grid = jnp.where(
                is_valid,
                grid,
                jnp.zeros((grid_size, grid_size)),
            )
            score_is_known = (
                is_valid
                & (home_score >= 0)
                & (away_score >= 0)
                & (home_score <= max_goals)
                & (away_score <= max_goals)
            )
            safe_home_score = jnp.clip(home_score, 0, max_goals)
            safe_away_score = jnp.clip(away_score, 0, max_goals)
            log_probability = jnp.where(
                score_is_known,
                jnp.log(grid[safe_home_score, safe_away_score] + 1e-12),
                0.0,
            )
            return grid, log_probability

        grids, log_probabilities = jax.vmap(predict_one)(
            home_ids,
            away_ids,
            valid,
            home_scores,
            away_scores,
        )
        return None, (
            grids,
            log_probabilities,
            jnp.sum(log_probabilities),
        )

    n_prediction_steps = prediction_inputs.timestamp.shape[0]
    _, outputs = jax.lax.scan(
        scan_body,
        None,
        jnp.arange(n_prediction_steps),
    )
    return outputs


__all__ = ["run_sequential_predict_rbsqmc"]
