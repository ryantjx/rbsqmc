import os
import json
import pandas as pd
from rbsqmc.src.utils.type import Matches, FootballResults
import numpy as np
import jax.numpy as jnp

RAW_URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")  # rbpf/data/
PARQUET_PATH = os.path.join(_DATA_DIR, "results.parquet")
WORLDCUP_2026_PATH = os.path.join(_DATA_DIR, "worldcup2026.json")
ACITVE_TEAMS_PATH = os.path.join(_DATA_DIR, "active_teams.json")
TEAMS_SMALL_PATH = os.path.join(_DATA_DIR, "teams_small.json")

with open(WORLDCUP_2026_PATH) as f:
    WORLDCUP_2026_TEAMS: set[str] = set(json.load(f))

with open(ACITVE_TEAMS_PATH) as f:
    ACTIVE_TEAMS: set[str] = set(json.load(f)['teams'])

with open(TEAMS_SMALL_PATH) as f:
    TEAMS_SMALL: set[str] = set(json.load(f)['teams'])

def _drop_duplicate_teams_per_day(data: pd.DataFrame) -> pd.DataFrame:
    """Assume that teams do not play more than once per day, otherwise it causes an issue with the propagation."""
    home_appearances = data[["date", "home_team"]].rename(columns={"home_team": "team"})
    away_appearances = data[["date", "away_team"]].rename(columns={"away_team": "team"})
    team_appearances = pd.concat([home_appearances, away_appearances], ignore_index=True)
    duplicated = team_appearances[team_appearances.duplicated(subset=["date", "team"], keep=False)]
    if not duplicated.empty:
        print("Warning: Dropping duplicate team appearances on the same day:")
        print(duplicated.sort_values(["date", "team"]).to_string(index=False))
    return data[~data.index.isin(duplicated.index)].reset_index(drop=True)

def filter_teams(
    data: pd.DataFrame,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp | None,
    max_goals: int,
    include_friendly: bool,
    teams_only: set[str] | None,
) -> pd.DataFrame:
    # filter data by date range
    data = data[
        (data["date"] >= start_datetime)
        & (data["date"] <= end_datetime if end_datetime is not None else True)
    ]
    # filter data by max_goals
    data = data[
        (data["home_score"] <= max_goals) & (data["away_score"] <= max_goals)
    ]
    # filter out friendly matches if include_friendly is False
    data["friendly"] = data["tournament"].str.contains(
        "Friendly", case=False, na=False
    )
    # filter by teams_only if provided
    if teams_only is not None:
        data = data[
            data["home_team"].isin(teams_only) & data["away_team"].isin(teams_only)
        ]
    if not include_friendly:
        data = data[~data["friendly"]]
    data = data.reset_index(drop=True)
    # Fix dates
    if data["date"].min() >= pd.Timestamp("2026-01-18"):
        data.loc[
            (data["home_team"] == "Morocco")
            & (data["away_team"] == "Senegal")
            & (data["date"] == "2026-01-18"),
            ["home_score", "away_score"],
        ] = [0, 1]

    # future games have nan scores - fill with -1
    data[["home_score", "away_score"]] = (
        data[["home_score", "away_score"]].fillna(-1).astype(int)
    )
    data = _drop_duplicate_teams_per_day(data)
    return data

def generate_team_id_mapping(data: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    # Create mapping of team names to integer IDs
    all_teams = pd.unique(data[["home_team", "away_team"]].values.ravel())
    # Create a stable mapping: team_name -> team_id
    team_name_to_id = {name: i for i, name in enumerate(sorted(all_teams))}
    team_id_to_name = {i: name for i, name in enumerate(sorted(all_teams))}

    # Assign integer IDs back to the dataframe
    data["home_id"] = data["home_team"].map(team_name_to_id)
    data["away_id"] = data["away_team"].map(team_name_to_id)
    return team_name_to_id, team_id_to_name

def generate_results_jax(matches: pd.DataFrame) -> tuple[pd.DataFrame, FootballResults]:
    # group by date
    date_grouped = matches.groupby("date")
    # get number of unique dates and max number of matches per day
    T = date_grouped.ngroups
    M = date_grouped.size().max()

    print(f"Generating JAX arrays for {T} unique dates and up to {M} matches per day.")

    # Use NumPy arrays for accumulation (mutable); converted to JAX at the end.
    # Pad with -1 as a sentinel: never a valid score or team id (team ids are >= 0).
    home_score = np.full((T, M), 0, dtype=np.int32)
    away_score = np.full((T, M), 0, dtype=np.int32)
    home_id = np.full((T, M), 0, dtype=np.int32)
    away_id = np.full((T, M), 0, dtype=np.int32)
    match_mask = np.zeros((T, M), dtype=bool)

    dates = []
    timestamps = []
    timestamps_prev = []

    prev_timestamp = 0
    # loop over each date group and fill in the arrays
    for t, (date, group) in enumerate(date_grouped):
        n = len(group)

        home_score[t, :n] = group["home_score"].to_numpy()
        away_score[t, :n] = group["away_score"].to_numpy()
        home_id[t, :n] = group["home_id"].to_numpy()
        away_id[t, :n] = group["away_id"].to_numpy()

        match_mask[t, :n] = True

        timestamp = group["timestamp"].iloc[0]

        dates.append(date)
        timestamps.append(timestamp)
        timestamps_prev.append(prev_timestamp)
        # update prev_timestamp for the next iteration
        prev_timestamp = timestamp

    matches_jax = Matches(
        home_score=jnp.asarray(home_score),
        away_score=jnp.asarray(away_score),
        home_id=jnp.asarray(home_id),
        away_id=jnp.asarray(away_id),
    )

    # dates are pandas Timestamps -> convert to numeric (days since start_date)
    # via np.datetime64 for JAX compatibility. DataFrame keeps the readable form.
    date_values = jnp.asarray([np.datetime64(d).astype("datetime64[D]").astype("int64")
                               for d in dates])

    results = FootballResults(
        date=date_values, # (T, ). used for reference in the future.
        timestamp=jnp.asarray(timestamps), # (T, )
        timestamp_prev=jnp.asarray(timestamps_prev), # (T, )
        matches=matches_jax, # (T, M)
        match_mask=jnp.asarray(match_mask), # (T, M) boolean mask
    )
    # per-date matches as readable dictionaries (one entry per date).
    # Distinct from `results.matches` (the JAX Matches tensor).
    matches_per_date = [
        [
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "tournament": row["tournament"],
            }
            for _, row in group.iterrows()
        ]
        for _, group in date_grouped
    ]

    df = pd.DataFrame(
        {
            "date": dates,
            "timestamp": timestamps,
            "timestamp_prev": timestamps_prev,
            "matches": matches_per_date,   # list of T lists of match dicts
        }
    )
    return df, results

def get_results(
    start_date: str = "1872-11-30",  # date of first game
    end_date: str | None = None,
    max_goals: int = 8,
    include_friendly: bool = False,
    teams_only: set[str] | None = None,
    download: bool = False,
):
    """
    Fetch and process international football results.

    If `download` is True, pulls the latest data from the CSV source over the
    network; otherwise reads from the local parquet cache (no internet needed).
    Both paths apply identical filtering/processing.

    Outputs:
        - results_df: pd.DataFrame with columns ['match_index_id', 'timestamp',
          'home_team_id', 'away_team_id', 'home_score', 'away_score']
        - results: FootballResults named tuple with the same data as jax arrays
        - team_id_to_name: dict mapping team_id to team_name
    """
    if download:
        data = pd.read_csv(RAW_URL, date_format="%Y-%m-%d")
    else:
        data = pd.read_parquet(PARQUET_PATH)

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) if end_date else None
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date", inplace=True)

    data = filter_teams(
        data=data,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        max_goals=max_goals,
        include_friendly=include_friendly,
        teams_only=teams_only,
    )
    team_name_to_id, team_id_to_name = generate_team_id_mapping(data)

    # timestamp is days since start_date (start_date is the reference point)
    data["timestamp"] = (data["date"] - start_datetime).dt.days

    # group by date and convert each matches to jax array
    # then add in timestamp and timestamp_prev columns
    df, football_results = generate_results_jax(data)

    return df, football_results, team_id_to_name

def get_training_data(
    train_start_date: str,  # date of first game
    test_start_date: str,
    prediction_start_date: str,
    max_goals: int = 8,
    include_friendly: bool = False,
    teams_only: set[str] | None = None,
    download: bool = False,
):
    """
    Train, Test, Prediction data split.
    """
    if download:
        data = pd.read_csv(RAW_URL, date_format="%Y-%m-%d")
    else:
        data = pd.read_parquet(PARQUET_PATH)
    train_start_date, test_start_date, prediction_start_date = map(
        pd.to_datetime, [train_start_date, test_start_date, prediction_start_date]
    )
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date", inplace=True)


def main():
    df, results_jax, team_id_to_name = get_results(
        start_date="2020-01-01", end_date="2026-12-31", max_goals=8, download=True
    )
    print("DataFrame head:")
    print(df[["date", "timestamp", "timestamp_prev", "matches"]].head(5))
    last = df.iloc[-1]["matches"][-1]
    print("Last match details:")
    print("   Date:       ", df.iloc[-1]["date"])
    print("   Timestamp:  ", df.iloc[-1]["timestamp"])
    print("   Timestamp Prev: ", df.iloc[-1]["timestamp_prev"])
    print("   Home Team:  ", last["home_team"])
    print("   Away Team:  ", last["away_team"])
    print("   Home Score: ", last["home_score"])
    print("   Away Score: ", last["away_score"])
    print("   Tournament: ", last["tournament"])
    # print(df[['date', 'timestamp', 'timestamp_prev', 'matches']].tail(5))
    # # print("\nFootballResults named tuple:")
    # # print(results_jax)
    # print("\nTeam ID to Name mapping:")
    # print(f"ID 1 : {team_id_to_name[1]}")
    # print(f"ID 2 : {team_id_to_name[2]}")

if __name__ == "__main__":
    main()
