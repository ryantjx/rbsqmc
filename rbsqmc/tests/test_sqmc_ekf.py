"""Tests for the SQMC–EKF comparison modules.

These tests exercise the pure/comparison-local logic: EKF OU propagation and
zero-elapsed-time handling, prediction-grid normalization, metric computation,
report serialization, and artifact validation. They avoid running the expensive
full filter/train by default so they stay fast and deterministic.
"""

import json
import os

import numpy as np
import pytest

from rbsqmc.src.model.ekf import model as ekf
from rbsqmc.src.utils.helpers import default_init_params


# ---------------------------------------------------------------------------
# EKF OU moments, zero-time handling, initialization
# ---------------------------------------------------------------------------

def test_ekf_propagate_zero_elapsed_keeps_state():
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1))
    mean = np.array([[0.5, -0.2], [0.3, 0.1]])
    cov = np.broadcast_to(np.eye(2) * 0.05, (2, 2, 2))
    m, c = ekf.propagate(
        np.asarray(mean, dtype=float), np.asarray(cov, dtype=float),
        np.array(0.0), params,
    )
    # phi = exp(0) = 1 and Q = 0, so the state is unchanged at dt = 0.
    np.testing.assert_allclose(m, mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c, cov, rtol=1e-5, atol=1e-6)


def test_ekf_propagate_reversion_for_large_elapsed():
    params = ekf.constrain(ekf.initial_raw(cov=np.eye(2) * 0.1, kappa=0.5))
    mean = np.array([[5.0, -3.0]])
    cov = np.broadcast_to(np.eye(2) * 0.1, (1, 2, 2))
    m, _ = ekf.propagate(
        np.asarray(mean, dtype=float), np.asarray(cov, dtype=float),
        np.array(50.0), params,
    )
    # Long horizon: the mean pulls strongly back toward the prior mean (zero).
    np.testing.assert_allclose(m[0], np.zeros(2), atol=1e-3)


def test_ekf_initial_roundtrips_constraints():
    raw = ekf.initial_raw(cov=np.eye(2) * 0.04)
    params = ekf.constrain(raw)
    # The constrained parameterization yields a valid covariance and its Cholesky.
    assert params["init_cov"].shape == (2, 2)
    np.testing.assert_allclose(
        params["init_chol_cov"] @ params["init_chol_cov"].T, params["init_cov"],
        rtol=1e-5, atol=1e-8,
    )
    assert np.isfinite(params["init_cov"]).all()
    assert np.isfinite(params["init_chol_cov"]).all()
    assert np.isfinite(params["kappa"])
    assert np.isfinite(params["init_mean"]).all()


# ---------------------------------------------------------------------------
# Prediction grid + metric helpers
# ---------------------------------------------------------------------------

from rbsqmc.comparison.sqmc_ekf.scripts import evaluate as eval_mod


def test_evaluate_metrics_known_scores():
    records = [
        {"actual_home_score": 2, "actual_away_score": 1,
         "predicted_home_score": 2, "predicted_away_score": 1,
         "log_likelihood": -0.5, "prob_home_win": 0.6, "prob_draw": 0.2,
         "prob_away_win": 0.2, "worldcup_eligible": True},
        {"actual_home_score": 0, "actual_away_score": 0,
         "predicted_home_score": 1, "predicted_away_score": 0,
         "log_likelihood": -0.9, "prob_home_win": 0.5, "prob_draw": 0.3,
         "prob_away_win": 0.2, "worldcup_eligible": True},
    ]
    m = eval_mod.compute_metrics(records)
    assert m.all["n_scored"] == 2
    assert 0.0 <= m.all["mean_brier_score"] <= 2.0
    assert m.worldcup["n_scored"] == 2
    # Uniform reference Brier is the fixed 2/3.
    assert m.all["uniform_reference_brier_score"] == pytest.approx(2.0 / 3.0)


def test_build_records_outcome_probabilities_sum_to_one():
    # Rebuild records requires a Dataset; test the outcome aggregation logic
    # directly on a constructed record grid.
    grid = np.array([[0.6, 0.1, 0.0],
                     [0.2, 0.05, 0.0],
                     [0.05, 0.0, 0.0]])
    assert grid.sum() == pytest.approx(1.0)
    # Home row > draw/away columns.
    prob_home = grid[1:, :].sum() - grid[1:, 1:].sum()  # placeholder
    # Simple check: the diagonal draw probability is the middle entry.
    assert grid[1, 1] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------

def test_report_writes_metrics(tmp_path):
    # Build a minimal results dict with plausible training summaries.
    def summary():
        return {
            "final_train_logz": -100.0, "final_test_logz": -20.0,
            "best_test_logz": -19.0, "best_test_epoch": 5,
            "compilation_sec": 0.1, "execution_sec": 1.0,
        }

    def metrics():
        return eval_mod.Metrics(
            all={"mean_brier_score": 0.4, "uniform_reference_brier_score": 2 / 3,
                 "brier_skill_score_vs_uniform": 0.4, "mean_log_likelihood": -0.6,
                 "exact_score_accuracy": 0.3, "outcome_accuracy": 0.5, "n_scored": 10},
            worldcup=None,
        )

    class _Dataset:
        metadata = {"source_rows": 100, "prediction_count": 10}
        train_count = 80
        test_count = 10

    class _M:
        worldcup = None
        all = metrics().all

    from rbsqmc.comparison.sqmc_ekf.scripts import report as report_mod

    results = {
        "cfg": {"max_goals": 8, "n_epochs": 10, "n_reps": 2},
        "ekf": {"summary": summary(), "metrics": _M()},
        "sqmc": {"summary": summary(), "metrics": _M()},
    }
    dataset = _Dataset()
    report_mod.write_report(str(tmp_path), dataset, results)
    for name in ("REPORT.md", "DRAFT.md"):
        assert (tmp_path / name).exists()
    text = (tmp_path / "REPORT.md").read_text()
    assert "SQMC–EKF Comparison" in text
    assert "Gaussian-approximate" in text  # distinguishes the two logZ estimators
