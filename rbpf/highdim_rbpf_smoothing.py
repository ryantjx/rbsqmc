"""
High Dimensional RBPF Smoothing using FFBSi (Forward Filtering Backward Sampling)

Based on PROOF_V4.md Section 3: RB-PF Smoothing
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

from highdim_rbpf_filter import (
    ParticleState, State, D, P, H, INIT_KAPPA, INIT_MU, 
    INIT_GAMMA, INIT_B, INIT_R
)
from generate_observations import Observation


class SmoothedState(NamedTuple):
    """A single smoothed state sample."""
    x: jax.Array  # Smoothed state (D, P)
    gamma: jax.Array  # Smoothed covariance (D, D)


def compute_predictive_mean_cov(
    state: ParticleState,
    t_next: int
) -> tuple[jax.Array, jax.Array]:
    """
    Compute predictive mean and covariance for transition from time t to t+1.
    
    From PROOF_V4.md:
    mu_{t+1|t} = mu_0 + Phi_{t+1} (mu_t - mu_0)
    P_{t+1|t} = Phi_{t+1} P_t Phi_{t+1}^T + Q_{t+1}
              = Phi_{t+1} (Gamma_t \otimes B) Phi_{t+1}^T + (Gamma_0 \otimes B) 
                - Phi_{t+1} (Gamma_0 \otimes B) Phi_{t+1}^T
              = Gamma_{t+1}^- \otimes B
    
    Args:
        state: Particle state at time t (with gamma = Gamma_t)
        t_next: Time t+1 (for computing delta_t)
    
    Returns:
        mu_pred: Predictive mean (D, P)
        gamma_pred: Predictive Gamma_{t+1}^- (D, D)
    """
    # Compute per-feature delta_t and phi_t
    delta_t = t_next - state.last_observed
    phi_t = jnp.diag(jnp.exp(-INIT_KAPPA * delta_t))
    
    # Predictive mean: mu_{t+1|t} = mu_0 + phi_t @ (mu_t - mu_0)
    mu_pred = INIT_MU + phi_t @ (state.x - INIT_MU)
    
    # Predictive gamma: Gamma_{t+1}^- = phi_t @ Gamma_t @ phi_t^T + Gamma_0 - phi_t @ Gamma_0 @ phi_t^T
    gamma_pred = phi_t @ state.gamma @ phi_t.T + INIT_GAMMA - phi_t @ INIT_GAMMA @ phi_t.T
    
    return mu_pred, gamma_pred


def compute_backward_weights(
    key: jax.Array,
    states_history: list[State],
    x_next_smoothed: jax.Array,
    t: int
) -> jax.Array:
    """
    Compute backward weights for sampling at time t given smoothed state at t+1.
    
    From PROOF_V4.md:
    w_{t|t+1}^{(i)} \propto w_t^{(i)} * N(X_{t+1}^* | mu_{t+1|t}^{(i)}, P_{t+1|t}^{(i)})
    
    Args:
        key: PRNG key
        states_history: List of filtered states
        x_next_smoothed: Smoothed state at time t+1 (D, P)
        t: Current time index
    
    Returns:
        Normalized backward weights (N,)
    """
    state_t = states_history[t]
    t_next = t + 1
    
    n_particles = len(state_t.log_weights)
    log_weights = state_t.log_weights
    particles = state_t.particles
    
    # Compute log transition probabilities for each particle
    log_transitions = jnp.zeros(n_particles)
    
    for i in range(n_particles):
        # Get particle i at time t
        particle_i = jax.tree.map(lambda x: x[i], particles)
        
        # Compute predictive distribution p(X_{t+1} | X_t^{(i)})
        mu_pred, gamma_pred = compute_predictive_mean_cov(particle_i, t_next)
        
        # Transition covariance: P_{t+1|t} = Gamma_{t+1}^- \otimes B
        # For computing likelihood, we need the full covariance
        Sigma_pred = jnp.kron(gamma_pred, INIT_B)
        
        # Flatten for multivariate normal
        mu_pred_flat = mu_pred.flatten()
        x_next_flat = x_next_smoothed.flatten()
        
        # Compute log probability
        log_prob = -0.5 * (
            (x_next_flat - mu_pred_flat).T @ jnp.linalg.solve(Sigma_pred, x_next_flat - mu_pred_flat) +
            jnp.log(jnp.linalg.det(Sigma_pred)) +
            D * P * jnp.log(2 * jnp.pi)
        )
        log_transitions = log_transitions.at[i].set(log_prob)
    
    # Backward weights: w_{t|t+1}^{(i)} \propto w_t^{(i)} * p(X_{t+1}^* | X_t^{(i)})
    log_backward_weights = log_weights + log_transitions
    
    # Normalize
    log_backward_weights = log_backward_weights - jax.nn.logsumexp(log_backward_weights)
    
    return jnp.exp(log_backward_weights)


def compute_backward_gain(
    gamma_t: jax.Array,
    gamma_pred: jax.Array,
    t: int,
    t_next: int
) -> jax.Array:
    """
    Compute backward sampling gain J_t using Kronecker structure.
    
    From PROOF_V4.md Appendix 3.1:
    J_t = Sigma_t * Phi_{t+1}^T * P_{t+1|t}^{-1}
        = Gamma_t * phi_{t+1}^T * (Gamma_{t+1}^-)^{-1} \otimes I_P
    
    Args:
        gamma_t: Filtered Gamma_t (D, D)
        gamma_pred: Predictive Gamma_{t+1}^- (D, D)
        t: Current time
        t_next: Next time
    
    Returns:
        J_t_gain: Gain matrix in Gamma space (D, D)
    """
    # Compute phi_{t+1}
    # Note: We need the delta_t that was used for prediction
    # For simplicity, assume uniform time steps or compute from stored last_observed
    # Here we use the fact that phi_t = exp(-kappa * delta_t) * I_D
    # For the backward gain, we need phi_{t+1} which depends on the specific particle
    
    # For the general case, we compute:
    # J_t^Gamma = Gamma_t @ phi_{t+1}^T @ inv(Gamma_{t+1}^-)
    
    # Since phi_t is diagonal: phi_t = diag(exp(-kappa * delta_t))
    # We need to reconstruct this. For now, assume we can compute it from the context.
    
    # Actually, looking at PROOF_V4.md more carefully:
    # J_t = Gamma_t * phi_{t+1}^T * (Gamma_{t+1}^-)^{-1} \otimes I_P
    
    # The gain in Gamma space:
    # For simplicity in the diagonal phi case:
    # phi_{t+1} = diag(exp(-kappa * delta_t_next))
    
    # This requires knowing delta_t for each feature, which depends on last_observed
    # For the general implementation, we'll compute this per-particle
    
    # Placeholder: return identity (will be refined)
    return jnp.eye(D)


def backward_sample_step(
    key: jax.Array,
    states_history: list[State],
    x_next_smoothed: jax.Array,
    t: int
) -> tuple[jax.Array, jax.Array]:
    """
    Perform one backward sampling step from t+1 to t.
    
    From PROOF_V4.md:
    1. Sample component I_t ~ Categorical(w_{t|t+1})
    2. Compute RTS backward sampling:
       - J_t = Sigma_t * Phi_{t+1}^T * P_{t+1|t}^{-1}
       - m_{t|t+1} = mu_t + J_t * (X_{t+1}^* - mu_{t+1|t})
       - Sigma_{t|t+1} = Sigma_t - J_t * P_{t+1|t} * J_t^T
    3. Sample X_t^* ~ N(m_{t|t+1}, Sigma_{t|t+1})
    
    Args:
        key: PRNG key
        states_history: List of filtered states
        x_next_smoothed: Smoothed state at time t+1 (D, P)
        t: Current time index
    
    Returns:
        x_t_smoothed: Smoothed state at time t (D, P)
        gamma_t_smoothed: Smoothed Gamma at time t (D, D)
    """
    key, subkey = jax.random.split(key)
    
    # Step 1: Sample component index I_t
    backward_weights = compute_backward_weights(subkey, states_history, x_next_smoothed, t)
    key, subkey = jax.random.split(key)
    I_t = jax.random.categorical(subkey, jnp.log(backward_weights))
    
    # Get selected particle at time t
    state_t = states_history[t]
    particle_i = jax.tree.map(lambda x: x[I_t], state_t.particles)
    
    # Step 2: Compute RTS backward sampling
    t_next = t + 1
    
    # Compute predictive for selected particle
    mu_pred, gamma_pred = compute_predictive_mean_cov(particle_i, t_next)
    
    # Compute backward gain using Kronecker structure
    # J_t = Gamma_t * phi_{t+1}^T * (Gamma_{t+1}^-)^{-1} \otimes I_P
    
    # First, compute phi_{t+1} for this particle
    delta_t = t_next - particle_i.last_observed
    phi_diag = jnp.exp(-INIT_KAPPA * delta_t)
    
    # J_t^Gamma = Gamma_t @ diag(phi) @ inv(Gamma_{t+1}^-)
    # Since phi is diagonal, this is: Gamma_t @ diag(phi) @ inv(gamma_pred)
    gamma_inv = jnp.linalg.inv(gamma_pred)
    J_gamma = particle_i.gamma @ jnp.diag(phi_diag) @ gamma_inv
    
    # Backward mean: m_{t|t+1} = mu_t + J_t * (X_{t+1}^* - mu_{t+1|t})
    # In the Kronecker structure: m = mu + (J_gamma \otimes I_P) * vec(X_{t+1}^* - mu_pred)
    # Which is: m = mu + J_gamma @ (X_{t+1}^* - mu_pred)
    diff = x_next_smoothed - mu_pred
    m_t_given_t1 = particle_i.x + J_gamma @ diff
    
    # Backward covariance: Sigma_{t|t+1} = Sigma_t - J_t * P_{t+1|t} * J_t^T
    # In Gamma space: Gamma_{t|t+1} = Gamma_t - J_gamma @ Gamma_{t+1}^- @ J_gamma^T
    gamma_t_given_t1 = particle_i.gamma - J_gamma @ gamma_pred @ J_gamma.T
    
    # Ensure positive definiteness
    gamma_t_given_t1 = (gamma_t_given_t1 + gamma_t_given_t1.T) / 2
    
    # Step 3: Sample X_t^* ~ N(m_{t|t+1}, Sigma_{t|t+1})
    # Sigma_{t|t+1} = Gamma_{t|t+1} \otimes B
    Sigma_t_given_t1 = jnp.kron(gamma_t_given_t1, INIT_B)
    
    key, subkey = jax.random.split(key)
    x_t_flat = jax.random.multivariate_normal(
        subkey, 
        m_t_given_t1.flatten(), 
        Sigma_t_given_t1
    )
    x_t_smoothed = x_t_flat.reshape(D, P)
    
    return x_t_smoothed, gamma_t_given_t1


def initialize_backward_sampling(
    key: jax.Array,
    states_history: list[State]
) -> tuple[jax.Array, jax.Array]:
    """
    Initialize backward sampling at time T.
    
    From PROOF_V4.md:
    1. Sample I_T ~ Categorical(w_T)
    2. X_T^{E,*} = mu_T^{E,(I_T)} (degenerate)
    3. X_T^{R,*} ~ N(m_T^{R|E,(I_T)}, P_T^{RR|E})
    
    For simplicity, we sample the full state from the mixture.
    
    Args:
        key: PRNG key
        states_history: List of filtered states
    
    Returns:
        x_T_smoothed: Initial smoothed state at time T (D, P)
        gamma_T_smoothed: Initial smoothed Gamma at time T (D, D)
    """
    T = len(states_history) - 1
    state_T = states_history[T]
    
    key, subkey = jax.random.split(key)
    
    # Step 1: Sample component I_T
    log_weights = state_T.log_weights
    I_T = jax.random.categorical(subkey, log_weights - jax.nn.logsumexp(log_weights))
    
    # Get selected particle
    particle_i = jax.tree.map(lambda x: x[I_T], state_T.particles)
    
    # Step 2 & 3: Sample smoothed state
    # The filtered state has:
    # - x: full state mean (D, P)
    # - gamma: full Gamma (D, D)
    # For smoothing initialization, we use the filtered state as the mean
    # and sample from the conditional distribution
    
    # The filtering distribution is:
    # p(X_T | y_{1:T}) ~ N(mu_T^{(I_T)}, Sigma_T^{(I_T)})
    # where Sigma_T = Gamma_T \otimes B
    
    Sigma_T = jnp.kron(particle_i.gamma, INIT_B)
    
    key, subkey = jax.random.split(key)
    x_T_flat = jax.random.multivariate_normal(
        subkey,
        particle_i.x.flatten(),
        Sigma_T
    )
    x_T_smoothed = x_T_flat.reshape(D, P)
    
    return x_T_smoothed, particle_i.gamma


def run_rbpf_smoothing(
    key: jax.Array,
    states_history: list[State],
    n_samples: int = 1
) -> list[SmoothedState]:
    """
    Run RB-PF smoothing using FFBSi algorithm.
    
    From PROOF_V4.md Section 3:
    1. Initialize at time T by sampling from filtered mixture
    2. For t = T-1 down to 0:
       - Compute backward weights
       - Sample component
       - Apply RTS backward sampling
       - Sample smoothed state
    
    Args:
        key: PRNG key
        states_history: List of filtered states from run_rbpf
        n_samples: Number of smoothed trajectories to generate
    
    Returns:
        List of SmoothedState for each time step (single trajectory)
    """
    T = len(states_history)
    
    # Store smoothed trajectory
    smoothed_trajectory = []
    
    # Step 1: Initialize at time T
    key, subkey = jax.random.split(key)
    x_T, gamma_T = initialize_backward_sampling(subkey, states_history)
    
    # We'll build the trajectory backwards, then reverse it
    backward_trajectory = [(x_T, gamma_T)]
    
    # Step 2: Backward simulation for t = T-1 down to 0
    x_next = x_T
    for t in range(T - 2, -1, -1):
        key, subkey = jax.random.split(key)
        x_t, gamma_t = backward_sample_step(subkey, states_history, x_next, t)
        backward_trajectory.append((x_t, gamma_t))
        x_next = x_t
    
    # Reverse to get forward trajectory
    backward_trajectory.reverse()
    
    smoothed_trajectory = [
        SmoothedState(x=x, gamma=gamma)
        for x, gamma in backward_trajectory
    ]
    
    return smoothed_trajectory


def validate_smoothing_results(
    smoothed_trajectory: list[SmoothedState],
    true_states: jax.Array,
    states_history: list[State] = None
) -> dict:
    """
    Validate smoothing results against true states.
    
    Args:
        smoothed_trajectory: List of SmoothedState
        true_states: True latent states (T, D, P)
        states_history: Optional filtered states for comparison
    
    Returns:
        Metrics dictionary
    """
    print(f"\n{'='*60}")
    print("RBPF SMOOTHING VALIDATION RESULTS")
    print(f"{'='*60}")
    
    T = len(smoothed_trajectory)
    
    metrics = {
        'rmse': [],
        'mae': [],
        'rmse_vs_filtered': [] if states_history else None
    }
    
    print(f"\n1. Trajectory Comparison (T={T} time steps)")
    
    for t in range(T):
        smoothed = smoothed_trajectory[t]
        x_true = true_states[t]
        
        # Errors for smoothed estimate
        rmse = jnp.sqrt(jnp.mean((smoothed.x - x_true)**2))
        mae = jnp.mean(jnp.abs(smoothed.x - x_true))
        
        metrics['rmse'].append(float(rmse))
        metrics['mae'].append(float(mae))
        
        # Compare with filtered estimate if available
        if states_history:
            filtered_state = states_history[t]
            weights = jnp.exp(filtered_state.log_weights - jax.nn.logsumexp(filtered_state.log_weights))
            x_filtered = jnp.sum(filtered_state.particles.x * weights[:, None, None], axis=0)
            rmse_filt = jnp.sqrt(jnp.mean((x_filtered - x_true)**2))
            metrics['rmse_vs_filtered'].append(float(rmse_filt))
    
    # Print summary statistics
    print(f"\n2. Smoothing Error Metrics:")
    print(f"   RMSE: mean={jnp.mean(jnp.array(metrics['rmse'])):.4f}, "
          f"max={jnp.max(jnp.array(metrics['rmse'])):.4f}")
    print(f"   MAE:  mean={jnp.mean(jnp.array(metrics['mae'])):.4f}, "
          f"max={jnp.max(jnp.array(metrics['mae'])):.4f}")
    
    if states_history:
        print(f"\n3. Filtered vs Smoothed:")
        print(f"   Filtered RMSE: mean={jnp.mean(jnp.array(metrics['rmse_vs_filtered'])):.4f}")
        print(f"   Smoothed RMSE: mean={jnp.mean(jnp.array(metrics['rmse'])):.4f}")
        improvement = jnp.mean(jnp.array(metrics['rmse_vs_filtered'])) - jnp.mean(jnp.array(metrics['rmse']))
        print(f"   Improvement: {improvement:.4f} ({improvement/jnp.mean(jnp.array(metrics['rmse_vs_filtered']))*100:.1f}%)")
    
    print(f"\n{'='*60}")
    
    return metrics


def plot_smoothing_results(
    smoothed_trajectory: list[SmoothedState],
    true_states: jax.Array,
    states_history: list[State],
    output_path: str = "./rbpf/outputs/rbpf_smoothing_results.png"
):
    """Plot smoothing results comparing filtered, smoothed, and true states."""
    import matplotlib.pyplot as plt
    
    T = len(smoothed_trajectory)
    n_features = D
    
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 24), sharex=True)
    
    for feat_idx in range(n_features):
        # Extract trajectories
        att_true = true_states[:, feat_idx, 0]
        def_true = true_states[:, feat_idx, 1]
        
        att_smoothed = jnp.array([s.x[feat_idx, 0] for s in smoothed_trajectory])
        def_smoothed = jnp.array([s.x[feat_idx, 1] for s in smoothed_trajectory])
        
        # Extract filtered estimates
        att_filtered = []
        def_filtered = []
        for state in states_history:
            weights = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
            x_mean = jnp.sum(state.particles.x * weights[:, None, None], axis=0)
            att_filtered.append(x_mean[feat_idx, 0])
            def_filtered.append(x_mean[feat_idx, 1])
        att_filtered = jnp.array(att_filtered)
        def_filtered = jnp.array(def_filtered)
        
        # Plot Attack
        ax_att = axes[feat_idx, 0]
        ax_att.plot(range(T), att_true, 'b-', alpha=0.7, label='True', linewidth=2)
        ax_att.plot(range(T), att_filtered, 'g--', alpha=0.5, label='Filtered')
        ax_att.plot(range(T), att_smoothed, 'r-', alpha=0.7, label='Smoothed')
        ax_att.set_ylabel(f'F{feat_idx}')
        if feat_idx == 0:
            ax_att.set_title('Attack')
            ax_att.legend(loc='upper right')
        if feat_idx == n_features - 1:
            ax_att.set_xlabel('Time')
        ax_att.grid(True, alpha=0.3)
        
        # Plot Defense
        ax_def = axes[feat_idx, 1]
        ax_def.plot(range(T), def_true, 'b-', alpha=0.7, label='True', linewidth=2)
        ax_def.plot(range(T), def_filtered, 'g--', alpha=0.5, label='Filtered')
        ax_def.plot(range(T), def_smoothed, 'r-', alpha=0.7, label='Smoothed')
        if feat_idx == 0:
            ax_def.set_title('Defense')
            ax_def.legend(loc='upper right')
        if feat_idx == n_features - 1:
            ax_def.set_xlabel('Time')
        ax_def.grid(True, alpha=0.3)
    
    plt.suptitle('RBPF Smoothing (Blue=True, Green=Filtered, Red=Smoothed)', y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    """Example usage of RBPF smoothing."""
    from highdim_rbpf_filter import run_rbpf, generate_observations
    import jax
    
    key = jax.random.PRNGKey(42)
    
    # 1. Generate observations and true states
    key, subkey = jax.random.split(key)
    observations, true_states = generate_observations(subkey, T=100)
    
    # 2. Run filtering
    print("Running RBPF filtering...")
    key, subkey = jax.random.split(key)
    states_history = run_rbpf(subkey, observations, n_particles=100)
    
    # 3. Run smoothing
    print("\nRunning RBPF smoothing...")
    key, subkey = jax.random.split(key)
    smoothed_trajectory = run_rbpf_smoothing(subkey, states_history, n_samples=1)
    
    # 4. Validate results
    metrics = validate_smoothing_results(smoothed_trajectory, true_states, states_history)
    
    # 5. Plot results
    plot_smoothing_results(smoothed_trajectory, true_states, states_history)
    
    return smoothed_trajectory, metrics


if __name__ == "__main__":
    main()
