"""
Optimized RBPF: Load optimal params from smoothed_params.json, run filter
with optimized initialization, and compare against original filter.
Generates rbpf_results_optimized.png.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import archive.rbpf_1.highdim_rbpf_filter as filt
from archive.rbpf_1.highdim_rbpf_filter import run_rbpf, generate_observations, D, P, H
from archive.rbpf_1.generate_observations import Observation


def load_optimal_params(filepath: str = "./rbpf/outputs/smoothed_params_em.json") -> dict:
    """Load optimal model parameters from EM JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    op = data['optimal_params']
    return {
        'kappa': op['kappa'],
        'phi': op['phi'],
        'mu_0': jnp.array(op['mu_0']),          # (D, P)
        'Gamma_0': jnp.array(op['Gamma_0']),    # (D, D)
        'B': jnp.array(op['B']),                # (P, P)
        'R': jnp.array(op['R']),                # (H, H)
        'Sigma_0': jnp.array(op['Sigma_0']),    # (D*P, D*P)
    }


def apply_optimal_params(params: dict):
    """Override the filter module's global INIT_* constants with optimal params."""
    filt.INIT_KAPPA = params['kappa']
    filt.INIT_MU = params['mu_0']
    filt.INIT_GAMMA = params['Gamma_0']
    filt.INIT_B = params['B']
    filt.INIT_SIGMA = params['Sigma_0']
    filt.INIT_R = params['R']


def compute_filter_mean(states_history) -> jax.Array:
    """Extract weighted mean from filter states at each timestep."""
    T_filter = states_history.particles.x.shape[0]
    filter_means = []
    for i in range(T_filter):
        log_weights = states_history.log_weights[i]
        weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
        x_mean = jnp.sum(states_history.particles.x[i] * weights[:, None, None], axis=0)
        filter_means.append(x_mean)
    return jnp.stack(filter_means)  # (T_filter, D, P)


def plot_optimized_results(
    states_history_orig,
    states_history_opt,
    true_states: jax.Array,
    observations: Observation,
):
    """Plot True, Original Filter, and Optimized Filter trajectories.
    
    Saves to ./rbpf/outputs/rbpf_results_optimized.png
    """
    T = states_history_orig.particles.x.shape[0] + 1
    n_features = D
    
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 28), sharex=True)
    
    # Track observed timesteps per feature
    observed_mask = {d: [] for d in range(n_features)}
    n_obs = observations.x1_index.shape[0]
    for i in range(n_obs):
        t = i + 1
        x1_idx = int(observations.x1_index[i].item())
        x2_idx = int(observations.x2_index[i].item())
        observed_mask[x1_idx].append(t)
        observed_mask[x2_idx].append(t)
    
    # Compute filter means
    filter_mean_orig = compute_filter_mean(states_history_orig)  # (T-1, D, P)
    filter_mean_opt = compute_filter_mean(states_history_opt)    # (T-1, D, P)
    
    for feat_idx in range(n_features):
        # True trajectory
        att_true = true_states[:, feat_idx, 0]
        def_true = true_states[:, feat_idx, 1]
        
        # Original filter (prepend true state at t=0)
        att_orig = [float(true_states[0, feat_idx, 0])]
        def_orig = [float(true_states[0, feat_idx, 1])]
        for i in range(T - 1):
            att_orig.append(float(filter_mean_orig[i, feat_idx, 0]))
            def_orig.append(float(filter_mean_orig[i, feat_idx, 1]))
        att_orig = jnp.array(att_orig)
        def_orig = jnp.array(def_orig)
        
        # Optimized filter (prepend true state at t=0)
        att_opt = [float(true_states[0, feat_idx, 0])]
        def_opt = [float(true_states[0, feat_idx, 1])]
        for i in range(T - 1):
            att_opt.append(float(filter_mean_opt[i, feat_idx, 0]))
            def_opt.append(float(filter_mean_opt[i, feat_idx, 1]))
        att_opt = jnp.array(att_opt)
        def_opt = jnp.array(def_opt)
        
        # --- Attack (column 0) ---
        ax = axes[feat_idx, 0]
        ax.plot(range(T), att_true, 'b-', alpha=0.5, linewidth=1.5, label='True')
        ax.plot(range(T), att_orig, 'r-', alpha=0.5, linewidth=1.5, label='Filter (orig)')
        ax.plot(range(T), att_opt, 'g-', alpha=0.8, linewidth=1.5, label='Filter (opt)')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax.scatter(obs_times, [float(att_true[t]) for t in obs_times],
                       c='blue', marker='x', s=50, zorder=5)
            # Predicted points (filter estimates) at observed timesteps
            ax.scatter(obs_times, [float(att_orig[t]) for t in obs_times],
                       c='red', marker='x', s=50, zorder=5)
            ax.scatter(obs_times, [float(att_opt[t]) for t in obs_times],
                       c='green', marker='x', s=50, zorder=5)
        ax.set_ylabel(f'F{feat_idx}')
        if feat_idx == 0:
            ax.set_title('Attack')
        if feat_idx == n_features - 1:
            ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # --- Defense (column 1) ---
        ax = axes[feat_idx, 1]
        ax.plot(range(T), def_true, 'b-', alpha=0.5, linewidth=1.5, label='True')
        ax.plot(range(T), def_orig, 'r-', alpha=0.5, linewidth=1.5, label='Filter (orig)')
        ax.plot(range(T), def_opt, 'g-', alpha=0.8, linewidth=1.5, label='Filter (opt)')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax.scatter(obs_times, [float(def_true[t]) for t in obs_times],
                       c='blue', marker='x', s=50, zorder=5)
            # Predicted points (filter estimates) at observed timesteps
            ax.scatter(obs_times, [float(def_orig[t]) for t in obs_times],
                       c='red', marker='x', s=50, zorder=5)
            ax.scatter(obs_times, [float(def_opt[t]) for t in obs_times],
                       c='green', marker='x', s=50, zorder=5)
        if feat_idx == 0:
            ax.set_title('Defense')
        if feat_idx == n_features - 1:
            ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        'RBPF Optimized (Blue=True, Red=Filter-orig, Green=Filter-opt, X=Observed/Predicted)',
        y=1.0
    )
    plt.tight_layout()
    plt.savefig('./rbpf/outputs/rbpf_results_optimized.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ./rbpf/outputs/rbpf_results_optimized.png")


def main():
    print("=== RBPF Optimized: Filter with Optimal Params ===\n")
    
    # 1. Generate observations and true states (same seed as smoothing)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    observations, true_states = generate_observations(subkey, T=200)
    print(f"Generated observations: {observations.t.shape[0]} timesteps")
    print(f"True states shape: {true_states.shape}")
    
    # 2. Run ORIGINAL filter (with initial guess params)
    print("\n--- Running original filter ---")
    key, subkey = jax.random.split(key)
    states_history_orig = run_rbpf(subkey, observations, n_particles=100)
    print(f"Original filter completed: {states_history_orig.particles.x.shape[0]} timesteps")
    
    # 3. Load optimal params from smoothed_params.json
    print("\n--- Loading optimal params from smoothed_params.json ---")
    optimal_params = load_optimal_params("./rbpf/outputs/smoothed_params_em.json")
    print(f"  kappa:   {optimal_params['kappa']:.6f}")
    print(f"  phi:     {optimal_params['phi']:.6f}")
    print(f"  mu_0:    shape={optimal_params['mu_0'].shape}")
    print(f"  Gamma_0: shape={optimal_params['Gamma_0'].shape}")
    print(f"  B:       {optimal_params['B']}")
    print(f"  R:       {optimal_params['R']}")
    
    # 4. Apply optimal params to filter module
    apply_optimal_params(optimal_params)
    print("\n  -> Overrode INIT_KAPPA, INIT_MU, INIT_GAMMA, INIT_B, INIT_SIGMA, INIT_R")
    
    # 5. Run OPTIMIZED filter (with optimal params)
    print("\n--- Running optimized filter ---")
    key, subkey = jax.random.split(key)
    states_history_opt = run_rbpf(subkey, observations, n_particles=100)
    print(f"Optimized filter completed: {states_history_opt.particles.x.shape[0]} timesteps")
    
    # 6. Compute RMSE for both filters
    T_filter = states_history_orig.particles.x.shape[0]
    true_aligned = true_states[1:T_filter + 1]  # Align with filter output (t=1 to T-1)
    
    filter_mean_orig = compute_filter_mean(states_history_orig)
    filter_mean_opt = compute_filter_mean(states_history_opt)
    
    rmse_orig = jnp.sqrt(jnp.mean((filter_mean_orig - true_aligned) ** 2))
    rmse_opt = jnp.sqrt(jnp.mean((filter_mean_opt - true_aligned) ** 2))
    
    print(f"\n=== RMSE Comparison ===")
    print(f"  Filter (original): {rmse_orig:.4f}")
    print(f"  Filter (optimized): {rmse_opt:.4f}")
    print(f"  Improvement:        {rmse_orig - rmse_opt:.4f}")
    
    # 7. Per-feature RMSE breakdown
    print(f"\n=== Per-Feature RMSE ===")
    print(f"  {'Feature':>8}  {'Original':>10}  {'Optimized':>10}  {'Improvement':>12}")
    for d in range(D):
        rmse_orig_d = jnp.sqrt(jnp.mean((filter_mean_orig[:, d, :] - true_aligned[:, d, :]) ** 2))
        rmse_opt_d = jnp.sqrt(jnp.mean((filter_mean_opt[:, d, :] - true_aligned[:, d, :]) ** 2))
        print(f"  F{d:>5}  {float(rmse_orig_d):>10.4f}  {float(rmse_opt_d):>10.4f}  {float(rmse_orig_d - rmse_opt_d):>12.4f}")
    
    # 8. Plot results
    plot_optimized_results(
        states_history_orig, states_history_opt, true_states, observations
    )
    
    print(f"\n=== Summary ===")
    print(f"  Original filter RMSE:  {rmse_orig:.4f}")
    print(f"  Optimized filter RMSE: {rmse_opt:.4f}")
    print(f"  Improvement:           {rmse_orig - rmse_opt:.4f} ({(rmse_orig - rmse_opt) / rmse_orig * 100:.1f}%)")


if __name__ == "__main__":
    main()
