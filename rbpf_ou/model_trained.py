"""Run the filter with trained EM parameters and produce diagnostic graphics.

Loads a trained ``EMParams`` JSON (by default the GPU EM output), runs the
forward filter over the full dataset, and writes the plots defined in
``rbpf_ou/src/graphic.py``.

Usage:
    python model_trained.py [--params-path PATH] [--output-dir DIR]

Defaults:
    --params-path  rbpf_ou/outputs_gpu/em_params_final.json
    --output-dir   rbpf_ou/outputs/smoothed
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

from rbpf_ou.src.data import get_results, WORLDCUP_2026_TEAMS
from rbpf_ou.src.helpers import load_params, generate_augmented_data
from rbpf_ou.src.model import run_filter, compute_gamma_trajectory
from rbpf_ou.src.graphic import (
    plot_top_strengths,
    plot_top_filter_states,
    plot_correlation_matrix,
    plot_correlation_extremes,
    plot_log_normalizing_constant,
)

jax.config.update("jax_platforms", "cpu")

MAX_GOALS = 8

# Read the training config so the filter uses the SAME data range and particle
# count as the EM run (otherwise the graphics are generated on a different
# time horizon, which changes the rankings).
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoothing_gpu_config.json")
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH) as _f:
        _CONFIG = json.load(_f)
else:
    _CONFIG = {}
N = int(_CONFIG.get("N", 200))
START_DATE = str(_CONFIG.get("start_date", "1950-01-01"))
END_DATE = str(_CONFIG.get("end_date", "2026-01-01"))


def main():
    parser = argparse.ArgumentParser(description="Run filter with trained params.")
    parser.add_argument(
        "--params-path",
        type=str,
        default="rbpf_ou/outputs_gpu/em_params_final.json",
        help="Path to trained EMParams JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rbpf_ou/outputs/smoothed",
        help="Directory to write graphics.",
    )
    args = parser.parse_args()

    data, model_inputs, team_id_to_name = get_results(
        start_date=START_DATE,
        end_date=END_DATE,
        max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    NUM_TEAMS = len(team_id_to_name)
    print(f"Loaded {len(data)} matches from {data['date'].min()} to {data['date'].max()}")

    params = load_params(args.params_path)
    print(f"Loaded params: kappa={params.kappa}, alpha={params.alpha}, beta={params.beta}")

    # Augment model inputs with the deterministic covariance trajectory.
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
        augmented_results, team_id_to_name, num_teams=NUM_TEAMS,
        save_path=os.path.join(args.output_dir, "correlation_matrix.png"),
    )
    plot_correlation_extremes(
        augmented_results, team_id_to_name, top_n=5, num_teams=NUM_TEAMS,
        save_path=os.path.join(args.output_dir, "correlation_extremes.png"),
    )
    plot_log_normalizing_constant(
        filtered_states,
        save_path=os.path.join(args.output_dir, "log_normalizing_constant.png"),
    )

    # --- Save final filter states and full correlation matrix as .npy ---
    # Final filtered posterior mean per team: (M, 2) [attack, defence].
    x_final = np.asarray(filtered_states.particles.x[-1])  # (N, M, 2)
    final_mean = x_final.mean(axis=0)  # (M, 2)
    np.save(os.path.join(args.output_dir, "final_filter_states.npy"), final_mean)

    # Full between-team correlation matrix from the final team covariance
    # gamma_t[-1] (M, M), normalized to a correlation matrix.
    gamma_final = np.asarray(augmented_results.gamma_t[-1])  # (M, M)
    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)
    np.save(os.path.join(args.output_dir, "correlation_matrix.npy"), corr)

    # Also save the team names for reference.
    team_names = [team_id_to_name[i] for i in range(NUM_TEAMS)]
    np.save(os.path.join(args.output_dir, "team_names.npy"), np.array(team_names))

    # --- Export final filter states to CSV (team_name, attack, defence) ---
    csv_path = os.path.join(args.output_dir, "final_filter_states.csv")
    with open(csv_path, "w") as f:
        f.write("team_name,attack,defence\n")
        for name, a, d in zip(team_names, final_mean[:, 0], final_mean[:, 1]):
            f.write(f"{name},{a:.6f},{d:.6f}\n")

    print(f"Saved final filter states and correlation matrix under {args.output_dir}/")
    print(f"Saved CSV: {csv_path}")
    print(f"Graphics saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
