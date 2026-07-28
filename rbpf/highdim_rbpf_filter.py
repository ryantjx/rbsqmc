"""
High Dimensional RBPF with asynchronous observations

"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import NamedTuple
import cuthbertlib
import jax
import jax.numpy as jnp
from generate_observations import Observation, generate_observations

D = 10  # feature dimension (F in PROOF_V4)
P = 2   # latent state dimension (att/def)
H = 2   # observation dimension

# Model parameters
INIT_KAPPA = 0.1  # OU mean reversion rate
DELTA_T = 1.0     # Time step
INIT_MU = jnp.zeros((D, P))  # Mean matrix (D x P)
INIT_GAMMA = jnp.eye(D)      # Feature covariance (D x D)
INIT_B = jnp.eye(P)          # Latent state covariance (P x P)
INIT_SIGMA = jnp.kron(INIT_GAMMA, INIT_B)  # Full covariance (DP x DP)

INIT_R = jnp.eye(H)

class ParticleState(NamedTuple):
    x : jax.Array  # Current state (D x P)
    x_observed: jax.Array  # Sampled state (H x P)
    gamma: jax.Array  # Current feature covariance (D x D)
    last_observed: jax.Array  # Last observed time for each feature (D,)

class State(NamedTuple):
    particles: ParticleState # vmap over N particles
    log_weights: jax.Array  # Log weights for each particle (N,)
    key: jax.Array  # PRNG key for randomness

def init_sample(key: jax.Array, params: dict[str, jax.Array]):
    """
    Step 1: Initialize
    """
    Sigma_0 = jnp.kron(INIT_GAMMA, INIT_B)
    x_flat = jax.random.multivariate_normal(key, INIT_MU.flatten(), Sigma_0)
    x_mean = x_flat.reshape(D, P)
    return ParticleState(
        x=x_mean, # initial state
        x_observed=jnp.zeros((H, P)), # update on first observation
        gamma=INIT_GAMMA.copy(), # initial feature covariance
        last_observed=jnp.zeros(D, dtype=jnp.int32) # all features start at t=0
    )

def propagate_and_sample(
        key: jax.Array,
        state: ParticleState,
        observations: Observation, 
    ) -> ParticleState:
    """
    Steps 2-4: Prediction, bootstrap sampling
    """

    # 1. Compute per-feature delta_t and phi_t
    delta_t = observations.t - state.last_observed
    phi_t = jnp.diag(jnp.exp(-INIT_KAPPA * delta_t))
    
    # 2. Predictive mean and covariance
    predicted_mean = INIT_MU + phi_t @ (state.x - INIT_MU)
    predicted_gamma = (phi_t @ state.gamma @ phi_t.T + INIT_GAMMA - phi_t @ INIT_GAMMA @ phi_t.T) 
    predicted_covariance = jnp.kron(predicted_gamma, INIT_B)

    # 3. extract observed indices - Sigma_EE is just the feature covariance gamma_EE
    obs_indices = jnp.array([observations.x1_index, observations.x2_index])
    mu_E = predicted_mean[obs_indices]  # (H, P)
    
    # Sigma_EE = kron(gamma_EE, B) is the observed part of the full covariance (HP x HP)
    gamma_EE = predicted_gamma[jnp.ix_(obs_indices, obs_indices)]
    Sigma_EE = jnp.kron(gamma_EE, INIT_B)

    # 4. Sample from the full joint distribution
    # x_E_flat ~ N(mu_E.flatten(), Sigma_EE) where Sigma_EE = kron(gamma_EE, B)
    key, subkey = jax.random.split(key)
    x_E_flat = jax.random.multivariate_normal(subkey, mu_E.flatten(), Sigma_EE)
    x_E = x_E_flat.reshape(H, P)

    # 5. Update full state: replace observed indices with sampled values
    x_update = predicted_mean.at[obs_indices].set(x_E)

    last_observed_updated = state.last_observed.at[obs_indices].set(observations.t)

    return ParticleState(
        x=x_update,
        x_observed=x_E,
        gamma=state.gamma,
        last_observed=last_observed_updated
    )

def kalman_update(
        state: ParticleState,
        observation: Observation
    ) -> ParticleState:
    """
    Step 6: Kalman update for unobserved features
    """
    # Extract indices
    observed_indices = jnp.array([observation.x1_index, observation.x2_index])
    all_indices = jnp.arange(D)
    unobserved_indices = jnp.setdiff1d(all_indices, observed_indices)

    # Current predictive mean and gamma
    mu_pred = state.x
    gamma_pred = state.gamma

    gamma_EE = gamma_pred[jnp.ix_(observed_indices, observed_indices)]
    gamma_RR = gamma_pred[jnp.ix_(unobserved_indices, unobserved_indices)]
    gamma_RE = gamma_pred[jnp.ix_(unobserved_indices, observed_indices)]

    # Gamma RE / Gamma EE 
    K = gamma_RE @ jnp.linalg.inv(gamma_EE)  # (F-H, H)

    mu_E = mu_pred[observed_indices]
    mu_R = mu_pred[unobserved_indices]

    mu_R_given_E = mu_R + K @ (state.x_observed - mu_E)
    x_updated = mu_pred.at[observed_indices].set(state.x_observed)
    x_updated = x_updated.at[unobserved_indices].set(mu_R_given_E)

    # gamma_RR|E = gamma_RR - gamma_RE @ inv(gamma_EE) @ gamma_RE.T
    gamma_RR_given_E = gamma_RR - K @ gamma_RE.T  # (D-H, D-H)
    
    # Build full updated gamma - Schur complement structure
    gamma_updated = jnp.zeros_like(gamma_pred)
    # EE block: observed-observed (keep original)
    gamma_updated = gamma_updated.at[jnp.ix_(observed_indices, observed_indices)].set(gamma_EE)
    # RR block: unobserved-unobserved (conditional covariance)
    gamma_updated = gamma_updated.at[jnp.ix_(unobserved_indices, unobserved_indices)].set(gamma_RR_given_E)
    # RE and ER blocks: cross terms
    gamma_RE_updated = K @ gamma_EE
    gamma_updated = gamma_updated.at[jnp.ix_(unobserved_indices, observed_indices)].set(gamma_RE_updated)
    gamma_updated = gamma_updated.at[jnp.ix_(observed_indices, unobserved_indices)].set(gamma_RE_updated.T)


    return ParticleState(
        x=x_updated,
        gamma=gamma_updated,
        x_observed=state.x_observed,
        last_observed=state.last_observed
    )

def log_potential(state_prev: ParticleState, state_curr: ParticleState, observation: Observation) -> jax.Array:
    """
    Step 5: Compute log weight from observation likelihood.
    """
    # Observation model: y_t = [att_i - def_j, att_j - def_i] + noise
    
    # Extract sampled states for observed features
    x1_state = state_curr.x_observed[0]  # [att, def] for x1
    x2_state = state_curr.x_observed[1]  # [att, def] for x2
    
    # Expected observation
    expected_y = jnp.array([
        x1_state[0] - x2_state[1],  # att_i - def_j
        x2_state[0] - x1_state[1]   # att_j - def_i
    ])
    
    # Observation noise covariance
    Sigma_obs = INIT_R  # (H, H)
    
    # Log likelihood of actual observation
    y_actual = jnp.array([observation.y1_index, observation.y2_index])
    
    diff = y_actual - expected_y
    log_likelihood = -0.5 * (diff.T @ jnp.linalg.solve(Sigma_obs, diff) + 
                             jnp.log(jnp.linalg.det(Sigma_obs)) + 
                             H * jnp.log(2 * jnp.pi))
    return log_likelihood

def run_rbpf(
    key: jax.Array, 
    observations: Observation,
    n_particles: int = 100,
) -> list[State]:
    """
    cuthbert returns a State object, but store as list for simplicity.
    """
    key, init_key = jax.random.split(key)
    # Initialize particles

    # vmap init_sample over all particles
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(lambda k: init_sample(k, {}))(init_keys)
    log_weights = jnp.zeros(n_particles)
    
    states_history = []

    for obs in observations:
        # Steps 2-4: Propagate and sample observed features
        key, prop_key = jax.random.split(key)
        prop_keys = jax.random.split(prop_key, n_particles)
        particles = jax.vmap(
            lambda k, p: propagate_and_sample(k, p, obs)
        )(prop_keys, particles)

        # Step 5: Compute weights
        log_potentials = jax.vmap(
            lambda p: log_potential(p, p, obs)
        )(particles)
        log_weights = log_potentials  # After resampling, weights are just potentials

        # Step 6: Kalman update (condition unobserved on observed)
        particles = jax.vmap(lambda p: kalman_update(p, obs))(particles)

        # Step 7: Systematic resampling using cuthbert
        key, resample_key = jax.random.split(key)
        indices, log_weights, particles = cuthbertlib.resampling.systematic.resampling(
            resample_key, log_weights, particles, n_particles
        )

        states_history.append(State(particles=particles, log_weights=log_weights, key=key))

    return states_history

def validate_rbpf_results(
    states_history: list[State],
    true_states: jax.Array,
    observations: list[Observation],
    n_particles: int
) -> dict:
    """
    Validate RBPF filtering results against true states.
    
    Returns metrics dictionary with filtering performance.
    """
    print(f"\n{'='*60}")
    print("RBPF VALIDATION RESULTS")
    print(f"{'='*60}")
    
    T = len(states_history)
    metrics = {
        'rmse': [],
        'mae': [],
        'ess': [],
        'max_weight': [],
        'particle_spread': []
    }
    
    # 1. Overall trajectory comparison
    print(f"\n1. Trajectory Comparison (T={T} time steps)")
    print(f"   Number of particles: {n_particles}")
    
    for t, state in enumerate(states_history):
        # Get particle states and weights
        particles_x = state.particles.x  # (N, D, P)
        log_weights = state.log_weights    # (N,)
        weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
        
        # Compute weighted mean estimate
        x_mean = jnp.sum(particles_x * weights[:, None, None], axis=0)  # (D, P)
        
        # True state at time t
        x_true = true_states[t]  # (D, P)
        
        # Errors
        rmse = jnp.sqrt(jnp.mean((x_mean - x_true)**2))
        mae = jnp.mean(jnp.abs(x_mean - x_true))
        
        metrics['rmse'].append(float(rmse))
        metrics['mae'].append(float(mae))
        
        # Effective Sample Size
        ess = 1.0 / jnp.sum(weights**2)
        metrics['ess'].append(float(ess))
        
        # Max weight (diagnostic for degeneracy)
        max_w = jnp.max(weights)
        metrics['max_weight'].append(float(max_w))
        
        # Particle spread (std across particles)
        spread = jnp.mean(jnp.std(particles_x, axis=0))
        metrics['particle_spread'].append(float(spread))
    
    # Print summary statistics
    print(f"\n2. Error Metrics:")
    print(f"   RMSE: mean={jnp.mean(jnp.array(metrics['rmse'])):.4f}, "
          f"max={jnp.max(jnp.array(metrics['rmse'])):.4f}")
    print(f"   MAE:  mean={jnp.mean(jnp.array(metrics['mae'])):.4f}, "
          f"max={jnp.max(jnp.array(metrics['mae'])):.4f}")
    
    print(f"\n3. Particle Diversity:")
    print(f"   ESS: mean={jnp.mean(jnp.array(metrics['ess'])):.1f} / {n_particles} particles")
    print(f"   Max weight: mean={jnp.mean(jnp.array(metrics['max_weight'])):.4f}")
    print(f"   Particle spread: mean={jnp.mean(jnp.array(metrics['particle_spread'])):.4f}")
    
    # 4. Check specific observed features
    print(f"\n4. Observed Feature Accuracy:")
    for t in [0, T//2, T-1]:  # Sample a few time steps
        obs = observations[t]
        state = states_history[t]
        
        # Get estimated states for observed features
        weights = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
        x_mean = jnp.sum(state.particles.x * weights[:, None, None], axis=0)
        
        x1_est = x_mean[obs.x1_index]
        x2_est = x_mean[obs.x2_index]
        x1_true = true_states[t, obs.x1_index]
        x2_true = true_states[t, obs.x2_index]
        
        print(f"\n   Time {t}:")
        print(f"      Feature {obs.x1_index}: est=[{x1_est[0]:.3f}, {x1_est[1]:.3f}], "
              f"true=[{x1_true[0]:.3f}, {x1_true[1]:.3f}]")
        print(f"      Feature {obs.x2_index}: est=[{x2_est[0]:.3f}, {x2_est[1]:.3f}], "
              f"true=[{x2_true[0]:.3f}, {x2_true[1]:.3f}]")
        
        # Check if observation prediction matches
        y_pred = jnp.array([x1_est[0] - x2_est[1], x2_est[0] - x1_est[1]])
        y_true = jnp.array([obs.y1_index, obs.y2_index])
        print(f"      y_pred=[{y_pred[0]:.3f}, {y_pred[1]:.3f}], "
              f"y_true=[{y_true[0]:.3f}, {y_true[1]:.3f}]")
    
    print(f"\n{'='*60}")
    
    return metrics


def plot_rbpf_results(
    states_history: list[State],
    true_states: jax.Array,
    observations: list[Observation],
    feature_idx: int = 0
):
    """Plot time series for all features (10 rows x 2 columns for Attack/Defense).
    
    Only shows markers when the feature was actually observed.
    """
    import matplotlib.pyplot as plt
    
    T = len(states_history)
    n_features = D  # Total number of features
    
    # Create figure with 10 rows (features) and 2 columns (Attack/Defense)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 24), sharex=True)
    
    # Track which timesteps each feature was observed
    observed_mask = {d: [] for d in range(n_features)}  # feature -> list of timesteps
    for t, obs in enumerate(observations):
        x1_idx = int(obs.x1_index.item())
        x2_idx = int(obs.x2_index.item())
        observed_mask[x1_idx].append(t)
        observed_mask[x2_idx].append(t)
    
    for feat_idx in range(n_features):
        # Extract true trajectory for this feature
        att_true = true_states[:, feat_idx, 0]  # Attack
        def_true = true_states[:, feat_idx, 1]  # Defense
        
        # Extract estimated trajectory
        att_est = []
        def_est = []
        for state in states_history:
            weights = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
            x_mean = jnp.sum(state.particles.x * weights[:, None, None], axis=0)
            att_est.append(x_mean[feat_idx, 0])
            def_est.append(x_mean[feat_idx, 1])
        att_est = jnp.array(att_est)
        def_est = jnp.array(def_est)
        
        # Plot Attack (column 0)
        ax_att = axes[feat_idx, 0]
        ax_att.plot(range(T), att_true, 'b-', alpha=0.5, label='True')
        ax_att.plot(range(T), att_est, 'r-', alpha=0.5, label='Est')
        # Mark observations with 'x'
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            att_true_vals = [float(att_true[t]) for t in obs_times]
            att_est_vals = [float(att_est[t]) for t in obs_times]
            ax_att.scatter(obs_times, att_true_vals, 
                          c='blue', marker='x', s=50, zorder=5)
            ax_att.scatter(obs_times, att_est_vals, 
                          c='red', marker='x', s=50, zorder=5)
        ax_att.set_ylabel(f'F{feat_idx}')
        if feat_idx == 0:
            ax_att.set_title('Attack')
        if feat_idx == n_features - 1:
            ax_att.set_xlabel('Time')
        ax_att.legend(loc='upper right', fontsize=8)
        ax_att.grid(True, alpha=0.3)
        
        # Plot Defense (column 1)
        ax_def = axes[feat_idx, 1]
        ax_def.plot(range(T), def_true, 'b-', alpha=0.5, label='True')
        ax_def.plot(range(T), def_est, 'r-', alpha=0.5, label='Est')
        # Mark observations with 'x'
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            def_true_vals = [float(def_true[t]) for t in obs_times]
            def_est_vals = [float(def_est[t]) for t in obs_times]
            ax_def.scatter(obs_times, def_true_vals, 
                          c='blue', marker='x', s=50, zorder=5)
            ax_def.scatter(obs_times, def_est_vals, 
                          c='red', marker='x', s=50, zorder=5)
        if feat_idx == 0:
            ax_def.set_title('Defense')
        if feat_idx == n_features - 1:
            ax_def.set_xlabel('Time')
        ax_def.legend(loc='upper right', fontsize=8)
        ax_def.grid(True, alpha=0.3)
    
    plt.suptitle('RBPF Time Series (Blue=True, Red=Est, X=Observed)', y=1.0)
    plt.tight_layout()
    plt.savefig('./rbpf/outputs/rbpf_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to ./rbpf/outputs/rbpf_results.png")

def save_rbpf_data(observations, states_history, true_states, filename_prefix="./rbpf/outputs/rbpf_output"):
    """Save RBPF output to Parquet files for analysis."""
    import polars as pl
    
    # Convert observations to DataFrame
    obs_data = {
        't': [int(obs.t) if hasattr(obs.t, 'item') else int(obs.t) for obs in observations],
        'x1_index': [int(obs.x1_index.item() if hasattr(obs.x1_index, 'item') else obs.x1_index) for obs in observations],
        'x2_index': [int(obs.x2_index.item() if hasattr(obs.x2_index, 'item') else obs.x2_index) for obs in observations],
        'y1_index': [float(obs.y1_index.item() if hasattr(obs.y1_index, 'item') else obs.y1_index) for obs in observations],
        'y2_index': [float(obs.y2_index.item() if hasattr(obs.y2_index, 'item') else obs.y2_index) for obs in observations],
    }
    obs_df = pl.DataFrame(obs_data)
    obs_df.write_parquet(f"{filename_prefix}_observations.parquet")
    print(f"Saved observations to {filename_prefix}_observations.parquet")
    
    # Convert states_history to DataFrame
    # Each row is a timestep, with particle data as arrays
    states_data = []
    for t, state in enumerate(states_history):
        # Get weighted mean for each feature
        weights = jnp.exp(state.log_weights - jax.nn.logsumexp(state.log_weights))
        x_mean = jnp.sum(state.particles.x * weights[:, None, None], axis=0)
        
        row = {'timestep': t}
        # Add mean estimates for each feature
        for d in range(D):
            row[f'f{d}_att_mean'] = float(x_mean[d, 0])
            row[f'f{d}_def_mean'] = float(x_mean[d, 1])
        # Add ESS
        ess = float(1.0 / jnp.sum(weights**2))
        row['ess'] = ess
        row['max_weight'] = float(jnp.max(weights))
        states_data.append(row)
    
    states_df = pl.DataFrame(states_data)
    states_df.write_parquet(f"{filename_prefix}_states.parquet")
    print(f"Saved states to {filename_prefix}_states.parquet")
    
    # Save true states
    true_data = []
    for t in range(true_states.shape[0]):
        row = {'timestep': t}
        for d in range(D):
            row[f'f{d}_att_true'] = float(true_states[t, d, 0])
            row[f'f{d}_def_true'] = float(true_states[t, d, 1])
        true_data.append(row)
    
    true_df = pl.DataFrame(true_data)
    true_df.write_parquet(f"{filename_prefix}_true_states.parquet")
    print(f"Saved true states to {filename_prefix}_true_states.parquet")
    
    return obs_df, states_df, true_df

def main():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    # 1. Generate observations and true states
    observations, true_states = generate_observations(subkey, T=200)
    
    # 2. Run RBPF
    key, subkey = jax.random.split(key)
    states_history = run_rbpf(subkey, observations, n_particles=100)
    
    # 3. Validate results
    metrics = validate_rbpf_results(states_history, true_states, observations, n_particles=100)
    
    # 4. Plot results
    plot_rbpf_results(states_history, true_states, observations)
    
    # 5. Save data for notebook analysis
    save_rbpf_data(observations, states_history, true_states)
    
    return states_history, metrics

if __name__ == "__main__":
    main()