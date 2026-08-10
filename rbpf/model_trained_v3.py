"""Run the RBPF filter using trained (EM-estimated) parameters from smoothing_v3.py.

Loads the smoothed parameters produced by `smoothing_v3.py` (V3: fixed μ₀=0,
fixed Γ₀ with small variance, fixed B=I₂, estimated κ, α, β), runs the forward
filter on the full dataset, and generates diagnostic plots and saved results.

Usage:
    python model_trained_v3.py [--params-path ./outputs_gpu_v3/em_params_final.json]
"""

import argparse
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

from data import download_results, read_results, WORLDCUP_2026_TEAMS
from graphic import generate_all_plots, save_results, weighted_mean
from model import run_filter, compute_gamma_trajectory, MAX_GOALS
from smoothing_v3 import load_params, EMParams

MAX_GOALS = 8
N = 1000

jax.config.update("jax_platforms", "cuda")


def find_latest_params(output_dir: str) -> str:
    """Find the best available EM parameter file.

    Prefers `em_params_final.json`; falls back to the highest-numbered epoch.
    """
    final_path = os.path.join(output_dir, "em_params_final.json")
    if os.path.exists(final_path):
        return final_path

    epoch_files = sorted(
        glob.glob(os.path.join(output_dir, "em_params_epoch_*.json")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    if not epoch_files:
        raise FileNotFoundError(
            f"No EM parameter files found in {output_dir}/. "
            "Run smoothing_v3.py first to produce trained parameters."
        )
    return epoch_files[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Run RBPF filter with trained EM parameters (V3: fixed μ₀=0, small Γ₀, fixed B=I₂)."
    )
    parser.add_argument(
        "--params-path", type=str, default=None,
        help="Path to EM params JSON. If omitted, auto-detects the latest in ./outputs_gpu_v3.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs_gpu_v3/trained",
        help="Directory to save filter results and plots. Default: ./outputs_gpu_v3/trained",
    )
    parser.add_argument(
        "--start-date", type=str, default="2000-01-01",
        help="Start date for the dataset.",
    )
    parser.add_argument(
        "--end-date", type=str, default="2025-12-31",
        help="End date for the dataset.",
    )
    args = parser.parse_args()

    # --- 1. Load data ---
    print("Loading match results...")
    data, results, team_id_to_name = read_results(
        start_date=args.start_date, end_date=args.end_date, max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    NUM_TEAMS = len(team_id_to_name)
    print(f"  {len(results.timestamp)} matches, {NUM_TEAMS} teams")

    # --- 2. Load trained parameters ---
    params_path = args.params_path or find_latest_params("./outputs_gpu_v3")
    print(f"\nLoading trained parameters from: {params_path}")
    params: EMParams = load_params(params_path)

    print(f"  κ (kappa)        = {params.init_kappa:.6f} [ESTIMATED]")
    print(f"  α (alpha)        = {params.init_alpha:.6f} [ESTIMATED]")
    print(f"  β (beta)         = {params.init_beta:.6f} [ESTIMATED]")
    print(f"  friendly_scale   = {params.init_friendly_scale:.6f}")
    print(f"  μ_0 (init_mean)  shape = {params.init_mean.shape} [FIXED at 0]")
    print(f"  Γ_0 (init_gamma) shape = {params.init_gamma.shape} [FIXED, small variance]")
    print(f"  B   (init_B)     shape = {params.init_B.shape} [FIXED]")

    # --- 3. Run the filter with trained parameters ---
    print("\nRunning filter with trained parameters...")
    key = jax.random.PRNGKey(42)
    key, filter_key = jax.random.split(key)
    filtered_states = run_filter(
        key=filter_key,
        results=results,
        init_gamma=params.init_gamma,
        init_kappa=params.init_kappa,
        num_teams=NUM_TEAMS,
        n=N,
        init_mean=params.init_mean,
        init_B=params.init_B,
        init_alpha=params.init_alpha,
        init_beta=params.init_beta,
        init_friendly_scale=params.init_friendly_scale,
    )
    gamma_trajectory = compute_gamma_trajectory(
        results, params.init_gamma, params.init_kappa, NUM_TEAMS,
    )

    # --- 4. Report results ---
    print(f"\n{'='*60}")
    print("FILTER COMPLETE (V3 trained parameters)")
    print(f"{'='*60}")
    print(f"  particles.x:           {filtered_states.particles.x.shape}")
    print(f"  gamma_trajectory:      {gamma_trajectory.shape}")
    print(f"  log_weights:           {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_const: {filtered_states.log_normalizing_constant.shape}")

    final_log_w = filtered_states.log_weights[-1]
    final_x_mean = weighted_mean(filtered_states.particles.x[-1], final_log_w)
    final_gamma = gamma_trajectory[-1]

    print(f"\nFinal state (weighted mean) — x shape: {final_x_mean.shape}")
    print(f"Final gamma — shape: {final_gamma.shape}")

    # Top 10 teams by attack strength
    final_att = np.array(final_x_mean[:, 0])
    top_idx = np.argsort(final_att)[::-1][:10]
    print("\nTop 10 teams by attack strength:")
    for tid in top_idx:
        name = team_id_to_name.get(int(tid), f"Team {tid}")
        print(f"  {name:30s}  attack={final_att[tid]:+.4f}  defense={final_x_mean[tid, 1]:+.4f}")

    # Print initial vs final correlation summaries
    from graphic import plot_correlation_matrix
    print("\n--- Generating correlation matrices ---")
    for feat, feat_name in [(0, "Attack"), (1, "Defense")]:
        for t_idx, t_label in [(0, "initial"), (-1, "final")]:
            # Compute mean absolute off-diagonal correlation
            all_x = filtered_states.particles.x
            x_t = np.array(all_x[t_idx])  # (N, M, 2)
            vals = x_t[:, :, feat]  # (N, M)
            corr = np.corrcoef(vals, rowvar=False)
            # Mask diagonal
            mask = ~np.eye(corr.shape[0], dtype=bool)
            mean_abs_corr = np.mean(np.abs(corr[mask]))
            print(f"  Mean |corr(off-diag)| {feat_name} ({t_label}): {mean_abs_corr:.4f}")

    # --- 5. Generate plots and save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating plots in {args.output_dir}/...")

    # Load EM log marginal history if available
    history_path = os.path.join(os.path.dirname(args.params_path), "em_log_marginal_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            log_marginals = json.load(f)
        print(f"Loaded EM history ({len(log_marginals)} epochs) from {history_path}")
        plot_em_convergence(
            log_marginals,
            save_path=f"{args.output_dir}/em_convergence.png",
        )
    else:
        print(f"No EM history found at {history_path}, skipping em_convergence.png")

    generate_all_plots(
        filtered_states,
        gamma_trajectory,
        team_id_to_name,
        output_dir=args.output_dir,
        max_teams=10,
    )

    print("Saving results to parquet...")
    save_results(
        filtered_states,
        gamma_trajectory,
        team_id_to_name,
        output_dir=args.output_dir,
    )

    print(f"\nDone. All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
