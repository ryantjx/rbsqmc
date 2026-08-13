import os
import json
import pandas as pd
import numpy as np
from typing import NamedTuple
import jax
from rbpf_rw_v2.src.utils import FootballResults
from rbpf_rw_v2.src.helpers import to_jax_data

RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # rbpf_rw_v2/data/
PARQUET_PATH = os.path.join(_DATA_DIR, "results.parquet")
WORLDCUP_2026_PATH = os.path.join(_DATA_DIR, "worldcup2026.json")
ACITVE_TEAMS_PATH = os.path.join(_DATA_DIR, "active_teams.json")

with open(WORLDCUP_2026_PATH) as f:
    WORLDCUP_2026_TEAMS: set[str] = set(json.load(f))

with open(ACITVE_TEAMS_PATH) as f:
    ACTIVE_TEAMS: set[str] = set(json.load(f)['teams'])


def get_results(
    start_date: str = "1872-11-30",  # date of first game
    end_date: str | None = None,
    max_goals: int = 8,
    include_friendly: bool = False,
    teams_only: set[str] | None = None,
    download: bool = False,
) -> tuple[pd.DataFrame, FootballResults, dict[int, str]]:
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
    # Ensure 'date' is parsed as datetime (parquet may store it as a string)
    data["date"] = pd.to_datetime(data["date"])

    # Fix dates
    if data["date"].min() >= pd.Timestamp("2026-01-18"):
        data.loc[
            (data["home_team"] == "Morocco")
            & (data["away_team"] == "Senegal")
            & (data["date"] == "2026-01-18"),
            ["home_score", "away_score"],
        ] = [0, 1]

    # Process time data into days since origin date
    data = data[
        (data["date"] >= start_date)
        & (data["date"] <= end_date if end_date else True)
    ]
    data.sort_values(by="date", inplace=True)
    data["timestamp"] = (data["date"] - data["date"].min()).dt.days
    data["friendly"] = data["tournament"].str.contains(
        "Friendly", case=False, na=False
    )
    if not include_friendly:
        data = data[~data["friendly"]]
    data[["home_score", "away_score"]] = (
        data[["home_score", "away_score"]].fillna(-1).astype(int)
    )
    data = data[
        (data["home_score"] <= max_goals) & (data["away_score"] <= max_goals)
    ]

    if teams_only is not None:
        data = data[
            data["home_team"].isin(teams_only) & data["away_team"].isin(teams_only)
        ]
    data["timestamp_prev"] = data["timestamp"].shift(1).fillna(0).astype(int)
    data = data.reset_index(drop=True)

    all_teams = pd.unique(data[["home_team", "away_team"]].values.ravel())
    # Create a stable mapping: team_name -> team_id
    team_name_to_id = {name: i for i, name in enumerate(sorted(all_teams))}
    team_id_to_name = {i: name for i, name in enumerate(sorted(all_teams))}

    # Assign integer IDs back to the dataframe
    data["home_team_id"] = data["home_team"].map(team_name_to_id)
    data["away_team_id"] = data["away_team"].map(team_name_to_id)
    return data, to_jax_data(data), team_id_to_name


def main():
    data, results, team_id_to_name = get_results(
        start_date="2020-01-01", max_goals=8, download=True
    )
    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'timestamp', 'timestamp_prev']].head())
    print("\nFootballResults named tuple:")
    print(results)
    print("\nTeam ID to Name mapping:")
    print(f"ID 1 : {team_id_to_name[1]}")
    print(f"ID 2 : {team_id_to_name[2]}")


if __name__ == "__main__":
    main()
