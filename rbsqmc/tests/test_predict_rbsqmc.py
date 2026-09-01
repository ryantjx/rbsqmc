"""Tests for SQMC sequential prediction and forecast evaluation."""

import jax
import jax.numpy as jnp
import numpy as np

from rbsqmc.src.model.predict import (
    evaluate_match_predictions,
    run_sequential_predict,
)
from rbsqmc.src.model.predict_rbsqmc import run_sequential_predict_rbsqmc
from rbsqmc.src.utils.type import EMParams, FootballResults, Matches


def _inputs(timestamp, timestamp_prev, home, away, home_score, away_score):
    return FootballResults(
        date=jnp.asarray(timestamp) + 10,
        timestamp=jnp.asarray(timestamp),
        timestamp_prev=jnp.asarray(timestamp_prev),
        matches=Matches(
            home_id=jnp.asarray(home)[:, None],
            away_id=jnp.asarray(away)[:, None],
            home_score=jnp.asarray(home_score)[:, None],
            away_score=jnp.asarray(away_score)[:, None],
        ),
        match_mask=jnp.ones((len(timestamp), 1), dtype=bool),
    )


def _params():
    return EMParams(
        mean_0=jnp.zeros((4, 2)),
        gamma_0=jnp.eye(4),
        B=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
        kappa=jnp.array(0.001),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )


def test_sqmc_prediction_matches_smc_output_contract():
    observed = _inputs([2], [0], [0], [1], [1], [0])
    prediction = _inputs([3, 3], [2, 3], [2, 0], [3, 2], [2, -1], [1, -1])
    arguments = {
        "key": jax.random.PRNGKey(0),
        "observed_inputs": observed,
        "prediction_inputs": prediction,
        "params": _params(),
        "n_particles": 8,
        "max_goals": 3,
    }

    smc = run_sequential_predict(**arguments)
    sqmc = run_sequential_predict_rbsqmc(**arguments)
    jax.block_until_ready(sqmc[0])

    assert smc[0].shape == sqmc[0].shape == (2, 1, 4, 4)
    assert smc[1].shape == sqmc[1].shape == (2, 1)
    assert smc[2].shape == sqmc[2].shape == (2,)
    np.testing.assert_allclose(np.asarray(smc[0]).sum(axis=(-2, -1)), 1.0)
    np.testing.assert_allclose(np.asarray(sqmc[0]).sum(axis=(-2, -1)), 1.0)
    assert np.isfinite(np.asarray(smc[0])).all()
    assert np.isfinite(np.asarray(sqmc[0])).all()
    # The second fixture has no known score and must not receive a bogus log score.
    assert float(smc[1][1, 0]) == 0.0
    assert float(sqmc[1][1, 0]) == 0.0


def test_prediction_evaluation_includes_multiclass_brier_reference():
    predictions = [
        {
            "actual_home_score": 1,
            "actual_away_score": 0,
            "predicted_home_score": 1,
            "predicted_away_score": 0,
            "prob_home_win": 1.0,
            "prob_draw": 0.0,
            "prob_away_win": 0.0,
            "log_likelihood": 0.0,
        },
        {
            "actual_home_score": 0,
            "actual_away_score": 1,
            "predicted_home_score": 1,
            "predicted_away_score": 0,
            "prob_home_win": 1.0,
            "prob_draw": 0.0,
            "prob_away_win": 0.0,
            "log_likelihood": -2.0,
        },
        {
            "actual_home_score": -1,
            "actual_away_score": -1,
            "predicted_home_score": 0,
            "predicted_away_score": 0,
            "prob_home_win": 0.3,
            "prob_draw": 0.4,
            "prob_away_win": 0.3,
            "log_likelihood": 0.0,
        },
    ]

    evaluation = evaluate_match_predictions(predictions)

    assert evaluation["n_predictions"] == 3
    assert evaluation["n_scored"] == 2
    assert evaluation["mean_brier_score"] == 1.0
    assert evaluation["uniform_reference_brier_score"] == 2.0 / 3.0
    assert evaluation["brier_skill_score_vs_uniform"] == -0.5
    assert evaluation["outcome_accuracy"] == 0.5
    assert predictions[0]["brier_score"] == 0.0
    assert predictions[1]["brier_score"] == 2.0
    assert predictions[2]["brier_score"] is None
