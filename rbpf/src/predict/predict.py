"""Sequential match predictions using a fitted RBPF model.

Given a fitted parameter file (EMParams JSON, e.g. ``optimized_params.json``)
and a fixtures list (see ``rbpf/data/fixtures.json``), this script:

  1. Loads the trained parameters.
  2. Groups fixtures by date (matches on the same day are processed together,
     matching the filter's per-day conditioning step).
  3. For each match day, uses the *current* filtered team-strength posterior
     to score every upcoming match via the bivariate-Poisson predictive
     distribution, then conditions on the observed scores (via the forward
     filter) to update the latent state for the next day.

This makes predictions sequentially, exactly mirroring how the model is fit.

Teams in the fixtures that are not present in the trained parameter set are
dropped (with a warning). Matches involving a dropped team are skipped.

Usage:
    python -m rbpf.src.predict --params rbpf/outputs/smoothing/optimized_params.json \
        --fixtures rbpf/data/fixtures.json --out rbpf/outputs/predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rbpf.src.utils.helpers import load_params
from rbpf.src.data.bivariate_poisson import loglik_grid
from rbpf.src.utils.type import EMParams, FootballResults


def _load_fixtures(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        fixtures = json.load(f)
    # Normalise field names (accept both "home"/"away" and "home_team"/"away_team").
    out = []
    for fx in fixtures:
        out.append({
            "date": fx["date"],
            "home": fx.get("home", fx.get("home_team")),
            "away": fx.get("away", fx.get("away_team")),
            "home_score": int(fx.get("home_score", -1)),
            "away_score": int(fx.get("away_score", -1)),
        })
    return out


def _filter_known_teams(
    fixtures: list[dict[str, Any]],
    team_id_to_name: dict[int, str],
) -> tuple[list[dict[str, Any]], int]:
    """Keep only fixtures whose both teams are present in the trained params."""
    known = set(team_id_to_name.values())
    dropped_teams: set[str] = set()
    dropped_fixtures = 0
    kept = []
    for fx in fixtures:
        if fx["home"] in known and fx["away"] in known:
            kept.append(fx)
        else:
            dropped_fixtures += 1
            dropped_teams.add(fx["home"])
            dropped_teams.add(fx["away"])
    if dropped_teams:
        dropped_sorted = sorted(d for d in dropped_teams if d)
        print(
            f"Warning: dropping {dropped_fixtures} fixture(s) involving teams "
            f"not in the trained model: {dropped_sorted}",
            flush=True,
        )
    return kept, dropped_fixtures


def _group_by_date(
    fixtures: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fx in fixtures:
        by_date[fx["date"]].append(fx)
    # Process chronologically.
    return sorted(by_date.items())


def _predict_day(
    states: jnp.ndarray,        # (N_particles, num_teams, 2) current posterior
    log_weights: jnp.ndarray,   # (N_particles,)
    params: EMParams,
    day_matches: list[dict[str, Any]],
    team_id_to_name: dict[int, str],
    max_goals: int,
) -> list[dict[str, Any]]:
    """Score each match in a day from the *pre-observation* state."""
    name_to_id = {name: i for i, name in team_id_to_name.items()}

    # Particle weights (normalised, stable).
    log_w = np.asarray(log_weights)
    log_w = log_w - np.max(log_w)
    weights = np.exp(log_w)
    weights = weights / (weights.sum() + 1e-12)

    # Posterior predictive mean of team strengths: weighted average over particles.
    mean_state = np.sum(np.asarray(states) * weights[:, None, None], axis=0)  # (T,2)

    results = []
    for match in day_matches:
        h = name_to_id[match["home"]]
        a = name_to_id[match["away"]]
        x_h = jnp.asarray(mean_state[h])   # (2,) attack/defence of home
        x_a = jnp.asarray(mean_state[a])   # (2,)

        # Predictive score grid: log p(Y_home = i, Y_away = j).
        grid = loglik_grid(
            x_h, x_a,
            alpha=params.alpha,
            beta=params.beta,
            max_goals=max_goals,
            scale=1.0,
        )  # (G, G)

        # Normalise to a predictive distribution over scores 0..max_goals.
        grid = grid - jnp.max(grid)
        probs = jnp.exp(grid)
        probs = probs / jnp.sum(probs)

        y_h, y_a = match["home_score"], match["away_score"]

        # Log-likelihood of the observed score under the predictive grid.
        if y_h >= 0 and y_a >= 0 and y_h <= max_goals and y_a <= max_goals:
            log_prob_obs = float(grid[y_h, y_a])
            prob_obs = float(probs[y_h, y_a])
        else:
            log_prob_obs = None
            prob_obs = None

        # Most likely score / margin.
        flat = np.asarray(probs)
        argmax = int(np.argmax(flat))
        pred_h, pred_a = argmax // (max_goals + 1), argmax % (max_goals + 1)

        # Most likely score / margin.
        flat = np.asarray(probs)
        argmax = int(np.argmax(flat))
        pred_h, pred_a = argmax // (max_goals + 1), argmax % (max_goals + 1)

        # Marginal home-win / draw / away-win probabilities.
        scores = np.arange(max_goals + 1)
        pHome, pDraw, pAway = 0.0, 0.0, 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > j:
                    pHome += float(probs[i, j])
                elif i == j:
                    pDraw += float(probs[i, j])
                else:
                    pAway += float(probs[i, j])

        # Full bivariate-Poisson probability breakdown over all scorelines
        # (home 0..max_goals x away 0..max_goals). Useful for plotting and for
        # a granular view of the predictive distribution.
        score_probabilities = [
            {
                "home": int(i),
                "away": int(j),
                "probability": float(probs[i, j]),
            }
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
        ]

        results.append({
            "date": match["date"],
            "home": match["home"],
            "away": match["away"],
            "actual_home_score": y_h,
            "actual_away_score": y_a,
            "predicted_home_score": int(pred_h),
            "predicted_away_score": int(pred_a),
            "log_likelihood": None if log_prob_obs is None else float(log_prob_obs),
            "probability": None if prob_obs is None else float(prob_obs),
            "prob_home_win": float(pHome),
            "prob_draw": float(pDraw),
            "prob_away_win": float(pAway),
            "score_probabilities": score_probabilities,
        })
    return results


def _score_prediction(pred: dict[str, Any]) -> bool:
    """1 if the predicted score matched exactly, else 0."""
    if pred.get("actual_home_score", -1) < 0 or pred.get("actual_away_score", -1) < 0:
        return False
    return (
        pred["predicted_home_score"] == pred["actual_home_score"]
        and pred["predicted_away_score"] == pred["actual_away_score"]
    )


def _build_timeline(
    fixtures: list[dict[str, Any]],
    team_id_to_name: dict[int, str],
) -> FootballResults:
    """Build a single FootballResults over the whole fixtures timeline.

    Matches are grouped by date, ordered chronologically, with monotonically
    increasing day timestamps so the OU transition uses the correct dt. This
    mirrors how the model was trained (one conditioning step per day).
    """
    import pandas as pd
    from rbpf.src.data.data import generate_results_jax

    name_to_id = {name: i for i, name in team_id_to_name.items()}
    rows = []
    for m in fixtures:
        rows.append({
            "date": m["date"],
            "timestamp": int(pd.Timestamp(m["date"]).toordinal()),
            "home_team": m["home"],
            "away_team": m["away"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "tournament": "FIFA World Cup",
            "home_id": name_to_id[m["home"]],
            "away_id": name_to_id[m["away"]],
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    _, results = generate_results_jax(df)
    return results


def run_predictions(
    params: EMParams,
    team_id_to_name: dict[int, str],
    fixtures: list[dict[str, Any]],
    max_goals: int = 8,
    n_particles: int = 2500,
    seed: int = 0,
) -> dict[str, Any]:
    """Run sequential predictions and return a results dict.

    A single forward filter is run over the entire fixtures timeline. For each
    match day t, the filtered posterior state from day t-1 (i.e. the predictive
    state before conditioning on day t's scores) is used to score day t's
    matches. This gives a genuine sequential / out-of-sample prediction.
    """
    from rbpf.src.model.model import run_filter

    # Drop fixtures involving teams not in the model.
    fixtures, n_dropped = _filter_known_teams(fixtures, team_id_to_name)
    grouped = _group_by_date(fixtures)

    key = jax.random.PRNGKey(seed)
    timeline = _build_timeline(fixtures, team_id_to_name)
    filtered, _ = run_filter(
        key=key,
        model_inputs=timeline,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals,
    )

    # filtered.particles.x: (T+1, N, M, 2) with the prepended prior at index 0.
    # filtered.log_weights: (T+1, N).
    x_all = np.asarray(filtered.particles.x)
    log_w_all = np.asarray(filtered.log_weights)

    # For day t (0-indexed into grouped), the pre-observation predictive state
    # is the filter output at time t (prior at t=0, posterior after t-1 days).
    predictions: list[dict[str, Any]] = []
    for day_idx, (date, day_matches) in enumerate(grouped):
        states = x_all[day_idx]       # (N, M, 2)
        log_weights = log_w_all[day_idx]  # (N,)
        day_preds = _predict_day(
            states,
            log_weights,
            params,
            day_matches,
            team_id_to_name,
            max_goals,
        )
        predictions.extend(day_preds)

    # --- Aggregate metrics. ---
    scored = [p for p in predictions if p["log_likelihood"] is not None]
    exact = [p for p in scored if _score_prediction(p)]
    accuracy = len(exact) / len(scored) if scored else None

    total_log_lik = float(sum(p["log_likelihood"] for p in scored)) if scored else None

    # Outcome accuracy (home/draw/away vs actual).
    outcome_correct = 0
    for p in scored:
        actual_h, actual_a = p["actual_home_score"], p["actual_away_score"]
        actual_outcome = "home" if actual_h > actual_a else ("draw" if actual_h == actual_a else "away")
        pred_outcome = (
            "home"
            if p["prob_home_win"] >= p["prob_draw"] and p["prob_home_win"] >= p["prob_away_win"]
            else ("draw" if p["prob_draw"] >= p["prob_away_win"] else "away")
        )
        p["actual_outcome"] = actual_outcome
        p["predicted_outcome"] = pred_outcome
        if actual_outcome == pred_outcome:
            outcome_correct += 1
    outcome_accuracy = outcome_correct / len(scored) if scored else None

    return {
        "params_path": None,
        "n_fixtures_total": n_dropped + len(fixtures),
        "n_fixtures_used": len(fixtures),
        "n_fixtures_dropped": n_dropped,
        "max_goals": max_goals,
        "n_particles": n_particles,
        "predictions": predictions,
        "summary": {
            "n_predictions": len(predictions),
            "n_scored": len(scored),
            "n_exact_score_matches": len(exact),
            "score_accuracy": accuracy,
            "outcome_accuracy": outcome_accuracy,
            "total_log_likelihood": total_log_lik,
        },
    }


def run_predictions_from_config(
    *,
    cfg: dict,
    params: EMParams,
    team_id_to_name: dict[int, str],
    save_path: str,
    max_goals: int,
) -> None:
    """Run the sequential prediction pipeline and save results to ``prediction/``.

    Loads the fixtures from ``cfg["fixtures"]`` (defaulting to
    ``rbpf/data/fixtures.json``), runs the predictive filter with the trained
    params, and writes a ``predictions.json`` plus a human-readable summary
    into ``<save_path>/prediction/``.
    """
    fixtures_path = cfg.get("fixtures", "rbpf/data/fixtures.json")
    pred_path = save_path + "prediction"
    os.makedirs(pred_path, exist_ok=True)

    if not os.path.isfile(fixtures_path):
        print(f"[predict] fixtures file not found: {fixtures_path}; skipping predictions", flush=True)
        return

    fixtures = _load_fixtures(fixtures_path)
    print(f"[predict] Loaded {len(fixtures)} fixtures from {fixtures_path}", flush=True)

    result = run_predictions(
        params=params,
        team_id_to_name=team_id_to_name,
        fixtures=fixtures,
        max_goals=max_goals,
        n_particles=cfg.get("prediction_n_particles", 1000),
        seed=cfg.get("prediction_seed", 0),
    )
    result["params_path"] = None

    with open(pred_path + "/predictions.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[predict] Wrote predictions to {os.path.abspath(pred_path + '/predictions.json')}", flush=True)

    # Generate prediction plots (score heatmaps + outcome probabilities).
    from rbpf.src.utils.graphic import plot_all_predictions
    plot_all_predictions(
        result,
        max_goals=max_goals,
        save_path=pred_path,
    )

    s = result["summary"]
    print(
        f"[predict] Predictions: {s['n_predictions']} (scored {s['n_scored']}), "
        f"exact-score {s['n_exact_score_matches']}, "
        f"score-accuracy {s['score_accuracy']}, "
        f"outcome-accuracy {s['outcome_accuracy']}, "
        f"total-loglik {s['total_log_likelihood']}",
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sequential RBPF match predictions")
    parser.add_argument("--params", required=True,
                        help="Path to fitted EMParams JSON (e.g. optimized_params.json)")
    parser.add_argument("--fixtures", required=True,
                        help="Path to fixtures JSON (e.g. rbpf/data/fixtures.json)")
    parser.add_argument("--out", default="rbpf/outputs/predictions.json",
                        help="Where to write predictions.json")
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--n-particles", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config",
                        help="Optional smoothing config JSON. If given, uses its "
                             "'teams', 'max_goals', 'prediction_n_particles' and "
                             "'prediction_seed' to match how the model was trained.")
    parser.add_argument("--plot", default=None, nargs="?", const="",
                        help="Where to save prediction plots (default: <out dir>/plots). "
                             "Omit the value to use the default, or pass --no-plots to skip.")
    args = parser.parse_args(argv)

    params = load_params(args.params)

    # Resolve teams / hyperparameters. If --config is given, use its team set so
    # the team-index -> name mapping is faithful to training (fixes cases where
    # the model was trained on worldcup2026 but predict.py assumed teams_small).
    cfg = {}
    if args.config:
        import json as _json
        with open(args.config) as f:
            cfg = _json.load(f)
        print(f"Loaded config from {args.config}")

    from rbpf.src.data.data import get_results
    from rbpf.src.utils.helpers import resolve_teams

    teams_only = resolve_teams(cfg) if cfg else None
    max_goals = int(cfg.get("max_goals", args.max_goals)) if cfg else args.max_goals
    n_particles = int(cfg.get("prediction_n_particles", args.n_particles)) if cfg else args.n_particles
    seed = int(cfg.get("prediction_seed", args.seed)) if cfg else args.seed

    num_teams = params.mean_0.shape[0]
    print(f"Loaded params with {num_teams} teams.")

    fixtures = _load_fixtures(args.fixtures)

    if teams_only is None:
        # No teams filter: infer the team set from the model dimension. This is
        # only reliable if the fixtures cover exactly the trained teams.
        teams_only = {fx["home"] for fx in fixtures} | {fx["away"] for fx in fixtures}

    # Reconstruct the team-id -> name mapping using the same team ordering used
    # at training time.
    df_dummy, _, team_id_to_name = get_results(
        start_date="1900-01-01", end_date="2099-12-31",
        teams_only=teams_only, include_friendly=False,
    )
    if len(team_id_to_name) != num_teams:
        print(
            f"Warning: trained model has {num_teams} teams but the resolved team "
            f"set has {len(team_id_to_name)}. The name mapping may be incorrect; "
            f"verify your training teams config.",
            flush=True,
        )

    result = run_predictions(
        params=params,
        team_id_to_name=team_id_to_name,
        fixtures=fixtures,
        max_goals=max_goals,
        n_particles=n_particles,
        seed=seed,
    )
    result["params_path"] = args.params
    result["teams"] = sorted(teams_only)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote predictions to {os.path.abspath(args.out)}")

    # Generate prediction plots (heatmaps + outcome probabilities).
    if args.plot is not None:
        if not args.plot:
            # Default to the directory containing the output file.
            args.plot = os.path.dirname(args.out) or "."
        from rbpf.src.utils.graphic import plot_all_predictions
        plot_all_predictions(
            result,
            max_goals=max_goals,
            save_path=args.plot,
        )

    s = result["summary"]
    print(
        f"Predictions: {s['n_predictions']} (scored {s['n_scored']}), "
        f"exact-score {s['n_exact_score_matches']}, "
        f"score-accuracy {s['score_accuracy']}, "
        f"outcome-accuracy {s['outcome_accuracy']}, "
        f"total-loglik {s['total_log_likelihood']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
