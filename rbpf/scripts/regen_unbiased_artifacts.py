"""Regenerate filter + prediction artifacts from fitted params.

Given a fitted ``params_unbiased.json`` (and the run config), this reruns the
final filter and the prediction step so the artifact PNGs/JSONs that were not
downloaded from Colab (top strengths, correlation plots, prediction plots, ...)
are reproduced locally. ``filter_states.npz`` is also rewritten so everything
is consistent.

Usage:
    python rbpf/scripts/regen_unbiased_artifacts.py \
        --out rbpf/outputs/filter_unbiased_gpu \
        --config rbpf/scripts/config/model_unbiased_gpu_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Ensure the repo root is importable (scripts can be run from anywhere).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rbpf/outputs/filter_unbiased_gpu",
                        help="Output dir containing params_unbiased.json")
    parser.add_argument("--config", default=None,
                        help="Optional run config JSON (for teams/seed/start/end dates)")
    args = parser.parse_args()

    out = args.out.rstrip("/")
    # Load fitted params.
    from rbpf.src.utils.helpers import load_params
    params = load_params(os.path.join(out, "params_unbiased.json"))

    # Load config (teams set, dates, seed, particles).
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    start_date = cfg.get("start_date", "1950-01-01")
    end_date = cfg.get("end_date", "2026-01-01")
    teams_cfg = cfg.get("teams", "worldcup2026")
    n_particles = cfg.get("n_particles", 500)
    max_goals = cfg.get("max_goals", 8)
    seed = cfg.get("seed", 0)

    from rbpf.src.data.data import get_results
    from rbpf.src.utils.helpers import resolve_teams
    from rbpf.src.model_unbiased import run_filter_unbiased
    from rbpf.src.utils.graphic import plot_all
    from rbpf.src.predict.predict import run_predictions, _load_fixtures
    from rbpf.src.utils.graphic import plot_all_predictions

    import jax

    teams_only = resolve_teams({"teams": teams_cfg})
    data, model_inputs, team_id_to_name = get_results(
        start_date=start_date, end_date=end_date,
        max_goals=max_goals, include_friendly=False,
        teams_only=teams_only,
    )
    print(f"Data: {len(team_id_to_name)} teams, {len(data)} dates.")

    # 1. Run the final filter with the fitted params.
    key = jax.random.PRNGKey(seed)
    key, filter_key = jax.random.split(key)
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=filter_key, model_inputs=model_inputs, params=params,
        n_particles=n_particles, max_goals=max_goals,
    )
    print(f"Final filter logZ = {float(filtered_states.log_normalizing_constant[-1]):.4f}")

    # 2. Write filter states + plots (matches main()).
    from rbpf.src.utils.graphic import save_filter_states, plot_all
    save_filter_states(filtered_states, model_inputs_rbpf,
                       save_path=os.path.join(out, "filter_states.npz"))
    plot_all(filtered_states=filtered_states,
             augmented_results=model_inputs_rbpf,
             team_id_to_name=team_id_to_name,
             top_n=10, save_path=out,
             timestamps=data["date"].to_numpy(),
             params=params)
    print(f"Wrote filter plots + filter_states.npz to {out}")

    # 3. Prediction step.
    fixtures_path = "rbpf/data/fixtures.json"
    if not os.path.isfile(fixtures_path):
        print(f"fixtures not found: {fixtures_path}")
        return 0
    fixtures = _load_fixtures(fixtures_path)
    pred_result = run_predictions(
        params=params, team_id_to_name=team_id_to_name,
        fixtures=fixtures, max_goals=max_goals,
        n_particles=n_particles, seed=seed,
    )
    pred_result["params_path"] = os.path.join(out, "params_unbiased.json")
    pred_result["teams"] = sorted(team_id_to_name.values())
    with open(os.path.join(out, "predictions.json"), "w") as f:
        json.dump(pred_result, f, indent=2)
    s = pred_result["summary"]
    print(f"Predictions: {s['n_predictions']} scored, exact {s['score_accuracy']}, "
          f"outcome {s['outcome_accuracy']}")
    plot_all_predictions(pred_result, max_goals=max_goals,
                         save_path=os.path.join(out, "prediction_plots"))
    print(f"Wrote predictions.json + prediction_plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
