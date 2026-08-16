import pandas as pd
import pytest

from rbpf_v2.src.data import chronological_split, prepare_results


def test_sorted_grouped_stable_ids_and_split(small_model):
    frame, data, _ = small_model
    assert data.timestamp.size == frame.date.nunique()
    assert (data.timestamp - data.timestamp_prev > 0).all()
    train, holdout = chronological_split(frame, "2024-01-05")
    assert train.date.max() < holdout.date.min()


def test_duplicate_team_rejected():
    frame = pd.DataFrame({"date": ["2024-01-01"] * 2,
                          "home_team": ["A", "A"], "away_team": ["B", "C"],
                          "home_score": [1, 0], "away_score": [0, 0]})
    with pytest.raises(ValueError, match="twice"):
        prepare_results(frame)


def test_scores_above_bound_are_excluded(small_model):
    frame, _, _ = small_model
    frame.loc[0, "home_score"] = 99
    filtered, _, _ = prepare_results(frame, max_goals=8)
    assert 99 not in filtered.home_score.to_numpy()
