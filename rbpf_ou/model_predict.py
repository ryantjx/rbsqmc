"""Predict 2026 FIFA World Cup group-stage matches from trained EM parameters.

Runs the forward filter over the training data with the trained ``EMParams``
to obtain each team's final latent strength (attack, defence), then for every
fixture in ``data/worldcup2026_fixtures.json`` computes the bivariate-Poisson
score distribution and reports win / draw / loss probabilities plus the most
likely scoreline.

This script is intended to run AFTER ``model_trained.py`` (which produces the
trained filter states and graphics). It reuses the same trained params and the
same data range / team set so the predictions are consistent with the training.

Usage:
    python model_predict.py [--params-path PATH] [--output-dir DIR]

Defaults:
    --params-path  rbpf_ou/outputs_gpu/em_params_final.json
    --output-dir   rbpf_ou/outputs/predictions
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Ensure the repo root is importable even when this file is run directly as a
# script (in which case sys.path[0] is the script's own directory, not the root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rbpf_ou.src.data import get_results, WORLDCUP_2026_TEAMS, ACTIVE_TEAMS
from rbpf_ou.src.helpers import load_params, generate_augmented_data
from rbpf_ou.src.model import run_filter, compute_gamma_trajectory
from rbpf_ou.src.bivariate_poisson import loglik_grid

jax.config.update("jax_platforms", "cpu")

MAX_GOALS = 8

# Read the training config so the filter uses the SAME data range and particle
# count as the EM run (otherwise the predictions are generated on a different
# time horizon, which changes the team strengths).
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoothing_gpu_config.json")
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH) as _f:
        _CONFIG = json.load(_f)
else:
    _CONFIG = {}
N = int(_CONFIG.get("N", 200))
START_DATE = str(_CONFIG.get("start_date", "1950-01-01"))
END_DATE = str(_CONFIG.get("end_date", "2026-01-01"))
_TEAM_SETS = {
    "ACTIVE_TEAMS": ACTIVE_TEAMS,
    "WORLDCUP_2026_TEAMS": WORLDCUP_2026_TEAMS,
}
TEAMS = _TEAM_SETS.get(_CONFIG.get("teams", "WORLDCUP_2026_TEAMS"), WORLDCUP_2026_TEAMS)

_FIXTURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "worldcup2026_fixtures.json")


def _load_fixtures() -> list[dict]:
    """Load the World Cup group-stage fixtures from JSON."""
    with open(_FIXTURES_PATH) as f:
        return json.load(f)


def _score_distribution(
    x_home: jnp.ndarray,
    x_away: jnp.ndarray,
    alpha: float,
    beta: float,
    scale: float,
    max_goals: int = MAX_GOALS,
) -> jnp.ndarray:
    """Return the (max_goals+1, max_goals+1) matrix of P(home_goals, away_goals).

    Uses the bivariate-Poisson log-likelihood grid, exponentiated and
    normalized to a proper probability distribution.
    """
    log_grid = loglik_grid(
        x_i=x_home, x_j=x_away, alpha=alpha, beta=beta,
        max_goals=max_goals, scale=scale,
    )
    log_grid = log_grid - jax.scipy.special.logsumexp(log_grid)
    return jnp.exp(log_grid)


def main():
    parser = argparse.ArgumentParser(description="Predict World Cup matches from trained params.")
    parser.add_argument(
        "--params-path",
        type=str,
        default="rbpf_ou/outputs_gpu/em_params_final.json",
        help="Path to trained EMParams JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rbpf_ou/outputs/predictions",
        help="Directory to write prediction outputs.",
    )
    args = parser.parse_args()

    # --- 1. Load data and run the forward filter to get final team strengths ---
    data, model_inputs, team_id_to_name = get_results(
        start_date=START_DATE,
        end_date=END_DATE,
        max_goals=MAX_GOALS,
        teams_only=TEAMS,
    )
    NUM_TEAMS = len(team_id_to_name)
    print(f"Loaded {len(data)} matches from {data['date'].min()} to {data['date'].max()}")

    params = load_params(args.params_path)
    print(f"Loaded params: kappa={params.kappa}, alpha={params.alpha}, beta={params.beta}, scale={params.scale}")

    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        gamma_Q=params.gamma_Q,
        kappa=params.kappa,
        num_teams=NUM_TEAMS,
    )
    augmented_results = generate_augmented_data(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain,
    )

    key = jax.random.PRNGKey(42)
    key, filter_key = jax.random.split(key)
    filtered_states, _ = run_filter(
        key=filter_key,
        model_inputs=augmented_results,
        params=params,
        num_teams=NUM_TEAMS,
        n_particles=N,
    )
    print(f"Filtered {len(filtered_states.particles.x)} states for {len(data)} matches")

    # Final filtered posterior mean per team: (M, 2) [attack, defence].
    x_final = np.asarray(filtered_states.particles.x[-1])  # (N, M, 2)
    final_mean = x_final.mean(axis=0)  # (M, 2)
    team_name_to_id = {name: i for i, name in team_id_to_name.items()}

    # --- 2. Predict each fixture ---
    fixtures = _load_fixtures()
    print(f"Loaded {len(fixtures)} fixtures")

    predictions = []
    total_loglik = 0.0
    n_scored = 0
    for fx in fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        if home_name not in team_name_to_id or away_name not in team_name_to_id:
            print(f"  WARNING: fixture {home_name} vs {away_name} has a team not in the trained set; skipping")
            continue
        x_home = jnp.asarray(final_mean[team_name_to_id[home_name]])
        x_away = jnp.asarray(final_mean[team_name_to_id[away_name]])

        grid = _score_distribution(
            x_home, x_away, params.alpha, params.beta, params.scale
        )  # (G+1, G+1)

        # Win / draw / loss probabilities.
        p_home_win = float(jnp.sum(jnp.triu(grid, 1)))
        p_draw = float(jnp.sum(jnp.diag(grid)))
        p_away_win = float(jnp.sum(jnp.tril(grid, -1)))

        # Most likely scoreline.
        flat_idx = int(jnp.argmax(grid))
        home_goals = flat_idx // (MAX_GOALS + 1)
        away_goals = flat_idx % (MAX_GOALS + 1)
        most_likely = (home_goals, away_goals)
        p_most_likely = float(grid[home_goals, away_goals])

        # Expected goals.
        goals = jnp.arange(MAX_GOALS + 1)
        exp_home = float(jnp.sum(goals * jnp.sum(grid, axis=1)))
        exp_away = float(jnp.sum(goals * jnp.sum(grid, axis=0)))

        # Actual score (if present in the fixture) and its log-likelihood.
        actual_home = fx.get("home_score")
        actual_away = fx.get("away_score")
        loglik = None
        if actual_home is not None and actual_away is not None:
            loglik = float(jnp.log(grid[actual_home, actual_away]))
            total_loglik += loglik
            n_scored += 1

        predictions.append({
            "date": fx["date"],
            "home": home_name,
            "away": away_name,
            "p_home_win": round(p_home_win, 4),
            "p_draw": round(p_draw, 4),
            "p_away_win": round(p_away_win, 4),
            "most_likely_score": f"{home_goals}-{away_goals}",
            "p_most_likely": round(p_most_likely, 4),
            "expected_goals_home": round(exp_home, 3),
            "expected_goals_away": round(exp_away, 3),
            "actual_score": (f"{actual_home}-{actual_away}"
                             if actual_home is not None and actual_away is not None else ""),
            "log_likelihood": (round(loglik, 4) if loglik is not None else None),
        })

    # --- 3. Write outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate evaluation metrics (only over fixtures with actual scores).
    eval_metrics = {}
    if n_scored > 0:
        eval_metrics = {
            "n_matches_scored": n_scored,
            "total_log_likelihood": round(total_loglik, 4),
            "mean_log_likelihood": round(total_loglik / n_scored, 4),
        }
        print(f"\n=== EVALUATION (over {n_scored} scored matches) ===")
        print(f"  total log-likelihood = {total_loglik:.4f}")
        print(f"  mean log-likelihood   = {total_loglik / n_scored:.4f}")

    json_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_path, "w") as f:
        json.dump({"evaluation": eval_metrics, "predictions": predictions}, f, indent=2)
    print(f"Saved predictions to {json_path}")

    # CSV summary.
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("date,home,away,p_home_win,p_draw,p_away_win,most_likely_score,p_most_likely,expected_goals_home,expected_goals_away,actual_score,log_likelihood\n")
        for p in predictions:
            f.write(
                f"{p['date']},{p['home']},{p['away']},{p['p_home_win']},{p['p_draw']},"
                f"{p['p_away_win']},{p['most_likely_score']},{p['p_most_likely']},"
                f"{p['expected_goals_home']},{p['expected_goals_away']},"
                f"{p['actual_score']},{p['log_likelihood']}\n"
            )
    print(f"Saved predictions CSV to {csv_path}")

    # Console summary.
    print("\n=== PREDICTIONS ===")
    for p in predictions:
        actual = f"actual {p['actual_score']}" if p["actual_score"] else "no actual"
        ll = f"ll={p['log_likelihood']:.2f}" if p["log_likelihood"] is not None else ""
        print(
            f"{p['date']}  {p['home']:>20} vs {p['away']:<20}  "
            f"W{p['p_home_win']:.2f} D{p['p_draw']:.2f} L{p['p_away_win']:.2f}  "
            f"most likely {p['most_likely_score']} ({p['p_most_likely']:.2f})  "
            f"xG {p['expected_goals_home']:.2f}-{p['expected_goals_away']:.2f}  "
            f"{actual} {ll}"
        )


if __name__ == "__main__":
    main()
