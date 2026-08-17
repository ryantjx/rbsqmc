from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .utils import FootballResults, Matches


REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score"}


def validate_results(results: FootballResults, num_teams: int | None = None) -> None:
    """Validate all non-JIT structural invariants required by the RBPF."""
    dt = np.asarray(results.timestamp - results.timestamp_prev)
    if dt.ndim != 1 or np.any(dt <= 0):
        raise ValueError("Every elapsed time dt must be strictly positive")
    mask = np.asarray(results.match_mask, dtype=bool)
    arrays = results.matches
    if mask.shape != np.asarray(arrays.home_id).shape:
        raise ValueError("match_mask and padded match arrays must have equal shape")
    for t in range(mask.shape[0]):
        ids = np.concatenate([
            np.asarray(arrays.home_id[t])[mask[t]],
            np.asarray(arrays.away_id[t])[mask[t]],
        ])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"A team appears more than once on day index {t}")
        if np.any(ids < 0) or (num_teams is not None and np.any(ids >= num_teams)):
            raise ValueError("valid team IDs are outside [0, M)")


def prepare_results(
    matches: pd.DataFrame,
    *,
    max_goals: int = 8,
    origin: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, FootballResults, dict[int, str]]:
    """Sort, filter and group a match DataFrame into one transition per date."""
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"missing match columns: {sorted(missing)}")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.dropna(subset=["home_score", "away_score"])
    frame[["home_score", "away_score"]] = frame[["home_score", "away_score"]].astype(int)
    frame = frame[
        frame.home_score.between(0, max_goals)
        & frame.away_score.between(0, max_goals)
    ].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no matches remain after score/date filtering")

    names = sorted(set(frame.home_team) | set(frame.away_team))
    name_to_id = {name: i for i, name in enumerate(names)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    frame["home_id"] = frame.home_team.map(name_to_id).astype(int)
    frame["away_id"] = frame.away_team.map(name_to_id).astype(int)

    appearances = pd.concat([
        frame[["date", "home_team"]].rename(columns={"home_team": "team"}),
        frame[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    duplicate = appearances.duplicated(["date", "team"], keep=False)
    if duplicate.any():
        row = appearances.loc[duplicate].sort_values(["date", "team"]).iloc[0]
        raise ValueError(f"A team cannot play twice on one day: {row.team} on {row.date.date()}")

    groups = list(frame.groupby("date", sort=True))
    D, L = len(groups), max(len(g) for _, g in groups)
    home = np.zeros((D, L), np.int32)
    away = np.zeros((D, L), np.int32)
    hs = np.zeros((D, L), np.int32)
    aws = np.zeros((D, L), np.int32)
    mask = np.zeros((D, L), bool)
    dates = np.asarray([np.datetime64(date, "D").astype(np.int64) for date, _ in groups])
    if origin is None:
        origin_day = dates[0] - 1
    else:
        origin_day = np.datetime64(pd.Timestamp(origin).normalize(), "D").astype(np.int64)
    timestamps = dates - origin_day
    previous = np.concatenate([[0], timestamps[:-1]])
    for t, (_, group) in enumerate(groups):
        n = len(group)
        home[t, :n] = group.home_id
        away[t, :n] = group.away_id
        hs[t, :n] = group.home_score
        aws[t, :n] = group.away_score
        mask[t, :n] = True
    results = FootballResults(
        date=jnp.asarray(dates), timestamp=jnp.asarray(timestamps, dtype=jnp.float32),
        timestamp_prev=jnp.asarray(previous, dtype=jnp.float32),
        matches=Matches(*(jnp.asarray(x) for x in (home, away, hs, aws))),
        match_mask=jnp.asarray(mask),
    )
    validate_results(results, len(names))
    return frame, results, id_to_name


def chronological_split(matches: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff).normalize()
    dates = pd.to_datetime(matches.date).dt.normalize()
    return matches.loc[dates <= cutoff].copy(), matches.loc[dates > cutoff].copy()


def slice_results(results: FootballResults, start=None, stop=None) -> FootballResults:
    """Chronologically slice days while retaining their original elapsed-time contract."""
    import jax
    return jax.tree.map(lambda value: value[slice(start, stop)], results)


def get_results(path: str | Path, **kwargs):
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return prepare_results(frame, **kwargs)


def synthetic_results() -> tuple[pd.DataFrame, FootballResults, dict[int, str]]:
    """Deterministic four-team fixture used by examples and smoke tests."""
    return prepare_results(pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-05", "2024-01-05", "2024-01-09", "2024-01-09"],
        "home_team": ["A", "C", "A", "B", "A", "B"],
        "away_team": ["B", "D", "C", "D", "D", "C"],
        "home_score": [1, 0, 2, 1, 1, 0],
        "away_score": [0, 1, 1, 1, 2, 0],
    }))
