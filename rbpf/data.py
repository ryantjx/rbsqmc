import pandas as pd
import numpy as np
from typing import NamedTuple
import jax

jax.config.update("jax_platforms", "cpu")
RAW_URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

class FootballResults(NamedTuple):
    match_index_id : jax.Array
    timestamp: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array


def download_results(
        start_date: str = "1872-11-30", # date of first game
        end_date: str | None = None,
        max_goals: int = 8
) -> tuple[pd.DataFrame, FootballResults, dict[int, str]]:
    """
    
    Outputs:
        - results_df: pd.DataFrame with columns ['match_index_id', 'timestamp', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
        - results: FootballResults named tuple with the same data as jax arrays
        - team_id_to_name: dict mapping team_id to team_name
    """

    data = pd.read_csv(RAW_URL, date_format="%Y-%m-%d")

    # Fix dates
    if data['date'].min() >= "2026-01-18":
        data.loc[
            (data["home_team"] == "Morocco")
            & (data["away_team"] == "Senegal")
            & (data["date"] == "2026-01-18"),
            ["home_score", "away_score"],
        ] = [0, 1]

    # Process time data into days since origin date
    data["date"] = pd.to_datetime(data["date"])
    data = data[(data["date"] >= start_date) & (data["date"] <= end_date if end_date else True)]
    data["timestamp"] = (data["date"] - data["date"].min()).dt.days
    data["friendly"] = data["tournament"].str.contains(
        "Friendly", case=False, na=False
    )
    data[["home_score", "away_score"]] = data[["home_score", "away_score"]].fillna(-1).astype(int)
    data = data[(data['home_score'] <= max_goals) & (data['away_score'] <= max_goals)]

    all_teams = pd.unique(data[["home_team", "away_team"]].values.ravel())
    # Create a stable mapping: team_name -> team_id
    team_name_to_id = {name: i + 1 for i, name in enumerate(sorted(all_teams))}
    team_id_to_name = {i + 1: name for i, name in enumerate(sorted(all_teams))}

    # Assign integer IDs back to the dataframe
    data["home_team_id"] = data["home_team"].map(team_name_to_id)
    data["away_team_id"] = data["away_team"].map(team_name_to_id)
    # print(data.head())
    return data, to_jax_data(data), team_id_to_name

def to_jax_data(df: pd.DataFrame) -> FootballResults:
    """
    Convert a pandas DataFrame to a FootballResults named tuple with jax arrays.
    """
    return FootballResults(
        match_index_id=jax.numpy.array(df.index.values),
        timestamp=jax.numpy.array(df["timestamp"].values),
        home_team_id=jax.numpy.array(df["home_team_id"].values),
        away_team_id=jax.numpy.array(df["away_team_id"].values),
        home_score=jax.numpy.array(df["home_score"].values),
        away_score=jax.numpy.array(df["away_score"].values),
    )

def main():
    data, results, team_id_to_name = download_results(start_date="2020-01-01", max_goals=8)
    print("DataFrame head:")
    print(data.head())
    print("\nFootballResults named tuple:")
    print(results)
    print("\nTeam ID to Name mapping:")
    print(f"ID 1 : {team_id_to_name[1]}")
    print(f"ID 2 : {team_id_to_name[2]}")

if __name__ == "__main__":
    main()