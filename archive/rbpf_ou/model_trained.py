
"""Run the filter with trained EM parameters and produce diagnostic graphics.

Loads a trained ``EMParams`` JSON (by default the GPU EM output), runs the
forward filter over the full dataset, and writes the plots defined in
``rbpf/src/graphic.py``.

Usage:
    python model_trained.py [--params-path PATH] [--output-dir DIR]

Defaults:
    --params-path  rbpf/outputs_gpu/em_params_final.json
    --output-dir   rbpf/outputs/smoothed
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp

# Ensure the repo root is importable even when this file is run directly as a
# script (in which case sys.path[0] is the script's own directory, not the root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
from rbpf.src.helpers import load_params, generate_augmented_data
from rbpf.src.model import run_filter, compute_gamma_trajectory
from rbpf.src.graphic import (
    plot_top_strengths,
    plot_top_filter_states,
    plot_correlation_matrix,
    plot_correlation_extremes,
    plot_log_normalizing_constant,
)

jax.config.update("jax_platforms", "cpu")

N = 1000
MAX_GOALS = 8


def main():
    parser = argparse.ArgumentParser(description="Run filter with trained params.")
    parser.add_argument(
        "--params-path",
        type=str,
        default="rbpf/outputs_gpu/em_params_final.json",
        help="Path to trained EMParams JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rbpf/outputs/smoothed",
        help="Directory to write graphics.",
    )
    args = parser.parse_args()

    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01",
        end_date="2026-01-01",
        max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    NUM_TEAMS = len(team_id_to_name)
    print(f"Loaded {len(data)} matches from {data['date'].min()} to {data['date'].max()}")

    params = load_params(args.params_path)
    print(f"Loaded params: kappa={params.kappa}, alpha={params.alpha}, beta={params.beta}")

    # Augment model inputs with the deterministic gamma trajectory.
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
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

    os.makedirs(args.output_dir, exist_ok=True)

    plot_top_strengths(
        filtered_states, team_id_to_name, top_n=10,
        save_path=os.path.join(args.output_dir, "top_strengths.png"),
    )
    plot_top_filter_states(
        filtered_states, team_id_to_name, top_n=5,
        save_path=os.path.join(args.output_dir, "top_filter_states.png"),
    )
    plot_correlation_matrix(
        augmented_results, team_id_to_name,
        save_path=os.path.join(args.output_dir, "correlation_matrix.png"),
    )
    plot_correlation_extremes(
        augmented_results, team_id_to_name, top_n=5,
        save_path=os.path.join(args.output_dir, "correlation_extremes.png"),
    )
    plot_log_normalizing_constant(
        filtered_states,
        save_path=os.path.join(args.output_dir, "log_normalizing_constant.png"),
    )

    print(f"Graphics saved under {args.output_dir}/")


if __name__ == "__main__":
    main()