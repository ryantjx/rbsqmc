"""
EM Algorithm for RB-PF with Kronecker Structure.

Iterates E-step (filter + smoother) and M-step (parameter estimation)
until convergence. Saves optimal params to smoothed_params_em.json.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import copy
import jax
import jax.numpy as jnp

import highdim_rbpf_filter as filt
import highdim_rbpf_smoothing as sm
from highdim_rbpf_filter import (
    run_rbpf, generate_observations,
    D, P, H,
)
from highdim_rbpf_smoothing import (
    run_rbpf_smoother, m_step, validate_smoother,
)
from generate_observations import Observation


def apply_params(filt_module, sm_module, params: dict):
    """Override both filter and smoothing module's global INIT_* constants."""
    filt_module.INIT_KAPPA = params['kappa']
    filt_module.INIT_MU = params['mu_0']
    filt_module.INIT_GAMMA = params['Gamma_0']
    filt_module.INIT_B = params['B']
    filt_module.INIT_SIGMA = params['Sigma_0']
    filt_module.INIT_R = params['R']
    # Also update smoothing module since it imports by value
    sm_module.INIT_KAPPA = params['kappa']
    sm_module.INIT_MU = params['mu_0']
    sm_module.INIT_GAMMA = params['Gamma_0']
    sm_module.INIT_B = params['B']
    sm_module.INIT_SIGMA = params['Sigma_0']
    # INIT_R is not imported by smoothing module, but set it anyway if present
    if hasattr(sm_module, 'INIT_R'):
        sm_module.INIT_R = params['R']


def get_params(filt_module) -> dict:
    """Read current INIT_* constants from the filter module."""
    return {
        'kappa': float(filt_module.INIT_KAPPA),
        'mu_0': filt_module.INIT_MU,
        'Gamma_0': filt_module.INIT_GAMMA,
        'B': filt_module.INIT_B,
        'Sigma_0': filt_module.INIT_SIGMA,
        'R': filt_module.INIT_R,
    }


def params_distance(p1: dict, p2: dict) -> float:
    """Compute relative distance between two param dicts for convergence check."""
    diff = 0.0
    diff += abs(p1['kappa'] - p2['kappa']) / (abs(p2['kappa']) + 1e-10)
    diff += jnp.sum(jnp.abs(p1['mu_0'] - p2['mu_0'])) / (jnp.sum(jnp.abs(p2['mu_0'])) + 1e-10)
    diff += jnp.sum(jnp.abs(p1['Gamma_0'] - p2['Gamma_0'])) / (jnp.sum(jnp.abs(p2['Gamma_0'])) + 1e-10)
    diff += jnp.sum(jnp.abs(p1['B'] - p2['B'])) / (jnp.sum(jnp.abs(p2['B'])) + 1e-10)
    diff += jnp.sum(jnp.abs(p1['R'] - p2['R'])) / (jnp.sum(jnp.abs(p2['R'])) + 1e-10)
    return float(diff)


def run_em(
    key: jax.Array,
    observations: Observation,
    true_states: jax.Array,
    n_particles: int = 100,
    n_smooth_trajectories: int = 10,
    max_iter: int = 5,
    tol: float = 1e-3,
    step_size: float = 0.3,
    verbose: bool = True,
) -> dict:
    """
    Run the EM algorithm: iterate E-step and M-step until convergence.
    
    E-step: Run filter + smoother with current params
    M-step: Estimate optimal params from smoothed trajectories
    
    Uses step_size damping to prevent divergence: new_param = (1-alpha)*old + alpha*new
    
    Returns:
        dict with final optimal params and convergence history
    """
    # Save original params
    original_params = get_params(filt)
    current_params = copy.deepcopy(original_params)
    best_params = copy.deepcopy(original_params)
    best_rmse = float('inf')
    
    history = []
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n{'='*60}")
            print(f"EM Iteration {iteration + 1}/{max_iter}")
            print(f"{'='*60}")
        
        # === E-step: Filter + Smoother ===
        key, subkey = jax.random.split(key)
        states_history = run_rbpf(subkey, observations, n_particles=n_particles)
        
        key, smoother_key = jax.random.split(key)
        smoothed_trajectories, smoothed_params = run_rbpf_smoother(
            key=smoother_key,
            states_historical=states_history,
            observations=observations,
            n_smooth_trajectories=n_smooth_trajectories,
        )
        
        # Compute smoothed RMSE
        smoothed_mean = smoothed_params['mean']
        true_aligned = true_states[:smoothed_mean.shape[0]]
        smoothed_rmse = float(jnp.sqrt(jnp.mean((smoothed_mean - true_aligned) ** 2)))
        
        # Check for NaN in filter output
        has_nan = bool(jnp.any(jnp.isnan(states_history.particles.x)))
        if has_nan:
            if verbose:
                print(f"  WARNING: NaN detected in filter output, reverting to previous params")
            # Don't update params, just record and continue
            history.append({
                'iteration': iteration + 1,
                'smoothed_rmse': float('nan'),
                'param_distance': float('nan'),
                'kappa': current_params['kappa'],
                'phi': float('nan'),
            })
            continue
        
        # === M-step: Estimate optimal params ===
        m_step_params = m_step(smoothed_trajectories, observations, true_states)
        
        # Apply damping: new = (1 - alpha) * old + alpha * m_step
        alpha = step_size
        new_params = {
            'kappa': (1 - alpha) * current_params['kappa'] + alpha * m_step_params['kappa'],
            'mu_0': (1 - alpha) * current_params['mu_0'] + alpha * m_step_params['mu_0'],
            'Gamma_0': (1 - alpha) * current_params['Gamma_0'] + alpha * m_step_params['Gamma_0'],
            'B': (1 - alpha) * current_params['B'] + alpha * m_step_params['B'],
            'R': (1 - alpha) * current_params['R'] + alpha * m_step_params['R'],
            'phi': m_step_params['phi'],
        }
        
        # Clamp R to prevent divergence: cap diagonal at 10x original, zero off-diag growth
        R_orig_diag = jnp.diag(original_params['R'])
        R_new_diag = jnp.minimum(jnp.diag(new_params['R']), 10.0 * R_orig_diag)
        R_new_diag = jnp.maximum(R_new_diag, 0.01 * R_orig_diag)  # Also floor
        new_params['R'] = jnp.diag(R_new_diag)
        
        # Clamp B diagonal to prevent blowup
        B_orig_diag = jnp.diag(original_params['B'])
        B_new_diag = jnp.minimum(jnp.diag(new_params['B']), 5.0 * B_orig_diag)
        B_new_diag = jnp.maximum(B_new_diag, 0.1 * B_orig_diag)
        # Keep off-diagonal small
        B_off = new_params['B'] - jnp.diag(jnp.diag(new_params['B']))
        B_off_max = 0.3 * jnp.sqrt(B_new_diag[0] * B_new_diag[1])
        B_off = jnp.clip(B_off, -B_off_max, B_off_max)
        new_params['B'] = jnp.diag(B_new_diag) + B_off
        
        # Clamp kappa to reasonable range
        new_params['kappa'] = float(jnp.clip(new_params['kappa'], 1e-4, 1.0))
        
        # Clamp Gamma_0 to prevent blowup: cap at 5x original
        Gamma_0_orig = original_params['Gamma_0']
        Gamma_0_max = 5.0 * Gamma_0_orig
        new_params['Gamma_0'] = jnp.minimum(new_params['Gamma_0'], Gamma_0_max)
        # Also floor at 0.1x original to prevent collapse
        Gamma_0_min = 0.1 * Gamma_0_orig
        new_params['Gamma_0'] = jnp.maximum(new_params['Gamma_0'], Gamma_0_min)
        
        # Ensure positive definite via eigenvalue clipping
        def ensure_pd(M, min_eig=1e-4):
            """Clip eigenvalues to be at least min_eig."""
            w, V = jnp.linalg.eigh(0.5 * (M + M.T))  # symmetrize first
            w = jnp.maximum(w, min_eig)
            return V @ jnp.diag(w) @ V.T
        
        new_params['Gamma_0'] = ensure_pd(new_params['Gamma_0'])
        new_params['B'] = ensure_pd(new_params['B'])
        new_params['R'] = ensure_pd(new_params['R'])
        
        new_params['Sigma_0'] = jnp.kron(new_params['Gamma_0'], new_params['B'])
        
        if verbose:
            print(f"  kappa:   {new_params['kappa']:.6f} (M-step: {m_step_params['kappa']:.6f})")
            print(f"  phi:     {new_params['phi']:.6f}")
            print(f"  B diag:  {jnp.diag(new_params['B'])}")
            print(f"  R diag:  {jnp.diag(new_params['R'])}")
            print(f"  Smoothed RMSE: {smoothed_rmse:.4f}")
        
        # Check convergence
        if iteration > 0:
            dist = params_distance(new_params, current_params)
            if verbose:
                print(f"  Param distance: {dist:.6f}")
            if dist < tol:
                if verbose:
                    print(f"\n  Converged at iteration {iteration + 1} (distance < {tol})")
                history.append({
                    'iteration': iteration + 1,
                    'smoothed_rmse': smoothed_rmse,
                    'param_distance': dist,
                    'kappa': new_params['kappa'],
                    'phi': new_params['phi'],
                })
                current_params = new_params
                break
        
        history.append({
            'iteration': iteration + 1,
            'smoothed_rmse': smoothed_rmse,
            'param_distance': float('inf') if iteration == 0 else dist,
            'kappa': new_params['kappa'],
            'phi': new_params['phi'],
        })
        
        # Apply new params for next iteration
        apply_params(filt, sm, new_params)
        current_params = new_params
        
        # Track best params (lowest smoothed RMSE)
        if smoothed_rmse < best_rmse and not has_nan:
            best_rmse = smoothed_rmse
            best_params = copy.deepcopy(new_params)
    
    # Restore original params
    apply_params(filt, sm, original_params)
    
    return {
        'optimal_params': best_params,
        'best_rmse': best_rmse,
        'original_params': original_params,
        'history': history,
        'n_iterations': len(history),
    }


def main():
    print("=" * 60)
    print("EM Algorithm for RB-PF with Kronecker Structure")
    print("=" * 60)
    
    # 1. Generate observations and true states
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    observations, true_states = generate_observations(subkey, T=200)
    print(f"Generated observations: {observations.t.shape[0]} timesteps")
    
    # 2. Run EM algorithm
    em_result = run_em(
        key=key,
        observations=observations,
        true_states=true_states,
        n_particles=100,
        n_smooth_trajectories=10,
        max_iter=10,
        tol=1e-3,
        verbose=True,
    )
    
    # 3. Save optimal params to JSON
    os.makedirs('./rbpf/outputs', exist_ok=True)
    op = em_result['optimal_params']
    json_params = {
        'optimal_params': {
            'kappa': op['kappa'],
            'phi': op['phi'],
            'mu_0': jnp.array(op['mu_0']).tolist(),
            'Gamma_0': jnp.array(op['Gamma_0']).tolist(),
            'B': jnp.array(op['B']).tolist(),
            'R': jnp.array(op['R']).tolist(),
            'Sigma_0': jnp.array(op['Sigma_0']).tolist(),
        },
        'em_history': em_result['history'],
        'n_iterations': em_result['n_iterations'],
    }
    with open('./rbpf/outputs/smoothed_params_em.json', 'w') as f:
        json.dump(json_params, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EM completed in {em_result['n_iterations']} iterations")
    print(f"Saved optimal params to ./rbpf/outputs/smoothed_params_em.json")
    print(f"{'='*60}")
    
    # Print convergence history
    print(f"\nConvergence History:")
    print(f"  {'Iter':>4}  {'RMSE':>8}  {'Dist':>10}  {'kappa':>8}  {'phi':>8}")
    for h in em_result['history']:
        print(f"  {h['iteration']:>4}  {h['smoothed_rmse']:>8.4f}  {h['param_distance']:>10.6f}  {h['kappa']:>8.6f}  {h['phi']:>8.6f}")
    
    # 4. Final pass: Run filter with optimal params and show predicted states
    print(f"\n{'='*60}")
    print("Final Pass: Filter with EM-Optimal Params")
    print(f"{'='*60}")
    
    # Apply optimal params
    apply_params(filt, sm, em_result['optimal_params'])
    
    # Run original filter (restore first, then run for baseline)
    apply_params(filt, sm, original_params)
    key, subkey = jax.random.split(key)
    states_orig = run_rbpf(subkey, observations, n_particles=100)
    
    # Run optimized filter
    apply_params(filt, sm, em_result['optimal_params'])
    key, subkey = jax.random.split(key)
    states_opt = run_rbpf(subkey, observations, n_particles=100)
    
    # Compute filter means
    def compute_filter_mean(states_history):
        T_f = states_history.particles.x.shape[0]
        means = []
        for i in range(T_f):
            log_w = states_history.log_weights[i]
            w = jnp.exp(log_w - jax.nn.logsumexp(log_w))
            x_mean = jnp.sum(states_history.particles.x[i] * w[:, None, None], axis=0)
            means.append(x_mean)
        return jnp.stack(means)
    
    filter_mean_orig = compute_filter_mean(states_orig)
    filter_mean_opt = compute_filter_mean(states_opt)
    
    T_filter = filter_mean_orig.shape[0]
    true_aligned = true_states[1:T_filter + 1]
    
    rmse_orig = float(jnp.sqrt(jnp.mean((filter_mean_orig - true_aligned) ** 2)))
    rmse_opt = float(jnp.sqrt(jnp.mean((filter_mean_opt - true_aligned) ** 2)))
    
    print(f"\n  Filter RMSE (original params):  {rmse_orig:.4f}")
    print(f"  Filter RMSE (EM-optimal params): {rmse_opt:.4f}")
    print(f"  Improvement:                     {rmse_orig - rmse_opt:.4f} ({(rmse_orig - rmse_opt) / rmse_orig * 100:.1f}%)")
    
    # Per-feature RMSE
    print(f"\n  Per-Feature RMSE:")
    print(f"  {'Feature':>8}  {'Original':>10}  {'Optimized':>10}  {'Improvement':>12}")
    for d in range(D):
        r_orig = float(jnp.sqrt(jnp.mean((filter_mean_orig[:, d, :] - true_aligned[:, d, :]) ** 2)))
        r_opt = float(jnp.sqrt(jnp.mean((filter_mean_opt[:, d, :] - true_aligned[:, d, :]) ** 2)))
        print(f"  F{d:>5}  {r_orig:>10.4f}  {r_opt:>10.4f}  {r_orig - r_opt:>12.4f}")
    
    # Plot predicted states: True vs Original Filter vs EM-Optimized Filter
    import matplotlib.pyplot as plt
    
    T = T_filter + 1
    n_features = D
    
    # Track observed timesteps
    observed_mask = {d: [] for d in range(n_features)}
    n_obs = observations.x1_index.shape[0]
    for i in range(n_obs):
        t = i + 1
        x1_idx = int(observations.x1_index[i].item())
        x2_idx = int(observations.x2_index[i].item())
        observed_mask[x1_idx].append(t)
        observed_mask[x2_idx].append(t)
    
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 28), sharex=True)
    
    for feat_idx in range(n_features):
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
        
        # EM-optimized filter
        att_opt = [float(true_states[0, feat_idx, 0])]
        def_opt = [float(true_states[0, feat_idx, 1])]
        for i in range(T - 1):
            att_opt.append(float(filter_mean_opt[i, feat_idx, 0]))
            def_opt.append(float(filter_mean_opt[i, feat_idx, 1]))
        att_opt = jnp.array(att_opt)
        def_opt = jnp.array(def_opt)
        
        # Attack
        ax = axes[feat_idx, 0]
        ax.plot(range(T), att_true, 'b-', alpha=0.5, lw=1.5, label='True')
        ax.plot(range(T), att_orig, 'r-', alpha=0.5, lw=1.5, label='Filter (orig)')
        ax.plot(range(T), att_opt, 'g-', alpha=0.8, lw=1.5, label='Filter (EM-opt)')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax.scatter(obs_times, [float(att_true[t]) for t in obs_times],
                       c='blue', marker='x', s=50, zorder=5)
        ax.set_ylabel(f'F{feat_idx}')
        if feat_idx == 0:
            ax.set_title('Attack')
        if feat_idx == n_features - 1:
            ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Defense
        ax = axes[feat_idx, 1]
        ax.plot(range(T), def_true, 'b-', alpha=0.5, lw=1.5, label='True')
        ax.plot(range(T), def_orig, 'r-', alpha=0.5, lw=1.5, label='Filter (orig)')
        ax.plot(range(T), def_opt, 'g-', alpha=0.8, lw=1.5, label='Filter (EM-opt)')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax.scatter(obs_times, [float(def_true[t]) for t in obs_times],
                       c='blue', marker='x', s=50, zorder=5)
        if feat_idx == 0:
            ax.set_title('Defense')
        if feat_idx == n_features - 1:
            ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'EM-Optimized RBPF (Blue=True, Red=Filter-orig, Green=Filter-EM-opt, X=Observed)\n'
        f'RMSE: orig={rmse_orig:.4f}, EM-opt={rmse_opt:.4f}',
        y=1.0
    )
    plt.tight_layout()
    plt.savefig('./rbpf/outputs/rbpf_results_em_optimized.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to ./rbpf/outputs/rbpf_results_em_optimized.png")
    
    # Restore original params
    apply_params(filt, sm, original_params)


if __name__ == "__main__":
    main()