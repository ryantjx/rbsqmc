import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
from typing import NamedTuple

D = 10  # feature dimension (F in PROOF_V4)
P = 2   # latent state dimension (att/def)
H = 2   # observation dimension

INIT_KAPPA = 0.1  # OU mean reversion rate
INIT_MU = jnp.zeros((D, P))  # Mean matrix (D x P)
INIT_GAMMA = jnp.eye(D)      # Feature covariance (D x D)
INIT_B = jnp.eye(P)          # Latent state covariance (P x P)
INIT_SIGMA = jnp.kron(INIT_GAMMA, INIT_B)  # Full covariance (DP x DP)

class Observation(NamedTuple):
    t : int
    x1_index : jax.Array
    x1_prev_index : jax.Array
    x2_index : jax.Array
    x2_prev_index : jax.Array
    y1_index : jax.Array
    y2_index : jax.Array

def generate_observations(key: jax.Array, T: int = 200):
    """
    Generate observations for T time steps based on the model in PROOF_V4.md.
    
    State: X_t in R^(D x P) where D=10 features, P=2 (att, def)
    Observation: y_t in R^H where H=2 (bivariate Gaussian)
    
    The OU-process dynamics: X_t = mu_0 + Phi_t (X_{t-1} - mu_0) + epsilon_t
    where Phi_t = phi_t \\otimes I_P, phi_t = exp(-kappa * delta_t) * I_D
    and Q_t = Sigma_0 - Phi_t Sigma_0 Phi_t^T
    
    Returns:
        observations: list of Observation namedtuples
        true_states: array of shape (T, D, P) - the latent states
    """
    key, key_init = jax.random.split(key)

    # init state: X_0 ~ N(mu_0, Sigma_0) where Sigma_0 = Gamma_0 \otimes B
    Sigma_0 = jnp.kron(INIT_GAMMA, INIT_B)  # (D*D, P*P)
    mu0_flat = INIT_MU.flatten()  # (D*P,)
    x0_flat = jax.random.multivariate_normal(key_init, mean=mu0_flat, cov=Sigma_0)  # (D*P,)
    x_curr = x0_flat.reshape(D, P)  # (D, P)

    observations = []
    true_states = jnp.zeros((T, D, P))
    true_states = true_states.at[0].set(x_curr)
    true_gammas = jnp.zeros((T, D, D))    # Feature covariances Gamma_t
    true_gammas = true_gammas.at[0].set(INIT_GAMMA.copy())

    last_observed = jnp.zeros(D, dtype=jnp.int32)  # All features start at t=0

    for t in range(1, T + 1):
        # 1. propagate state using OU-process dynamics
        x_curr = true_states[t-1]
        delta_t = t - last_observed  # (D,)
        phi_t = jnp.diag(jnp.exp(-INIT_KAPPA * delta_t))  # (D, D)
        predicted_mean = INIT_MU + phi_t @ (x_curr - INIT_MU)  # (D, P)

        gamma_curr = true_gammas[t-1]
        predicted_gamma = phi_t @ gamma_curr @ phi_t.T + INIT_GAMMA - phi_t @ INIT_GAMMA @ phi_t.T

        # sample process noise for all features
        Phi_t = jnp.kron(phi_t, jnp.eye(P))  # (D*P, D*P)
        Q_t = jnp.kron(predicted_gamma, INIT_B)  # (D*P, D*P)
        key, key_process = jax.random.split(key)
        epsilon_flat = jax.random.multivariate_normal(key_process, mean=jnp.zeros(D*P), cov=Q_t)  # (D*P,)
        epsilon = epsilon_flat.reshape(D, P)  # (D, P)

        x_next = predicted_mean + epsilon  # (D, P)

        # store the propagated state and covariance
        true_states = true_states.at[t].set(x_next)
        true_gammas = true_gammas.at[t].set(predicted_gamma)
        
        # 2. Sample for 2 random features (x1, x2) to observe at time t
        key, key_indices = jax.random.split(key)
        x1_idx, x2_idx = jax.random.choice(key_indices, D, shape=(2,), replace=False)
        x1_state = x_next[x1_idx]
        x2_state = x_next[x2_idx]

        obs_mean = jnp.array([x1_state[0] - x2_state[1], x2_state[0] - x1_state[1]]) # ()
        y_t = obs_mean + jax.random.multivariate_normal(key, mean=jnp.zeros(H), cov=jnp.eye(H))  # (H,)
        observations.append(
            Observation(
                t=t,
                x1_index=x1_idx,
                x1_prev_index=last_observed[x1_idx],
                x2_index=x2_idx,
                x2_prev_index=last_observed[x2_idx],
                y1_index=y_t[0],
                y2_index=y_t[1]
            )
        )
        # update last observed times for x1 and x2
        last_observed = last_observed.at[x1_idx].set(t)
        last_observed = last_observed.at[x2_idx].set(t)

    return observations, true_states

def verify_observations(observations, true_states, T=200):
    """Check observations were generated correctly."""
    print(f"Number of observations: {len(observations)} (expected: {T})")
    
    # Check first few observations
    for i, obs in enumerate(observations[:3]):
        print(f"\n--- Observation {i} ---")
        print(f"  t={obs.t}, x1_idx={obs.x1_index}, x2_idx={obs.x2_index}")
        print(f"  y=[{obs.y1_index:.3f}, {obs.y2_index:.3f}]")
        
        # Verify y matches the formula: [att_i - def_j, att_j - def_i]
        x1_state = true_states[obs.t, obs.x1_index]  # [att, def] for x1
        x2_state = true_states[obs.t, obs.x2_index]  # [att, def] for x2
        expected_y = jnp.array([
            x1_state[0] - x2_state[1],  # att_i - def_j
            x2_state[0] - x1_state[1]   # att_j - def_i
        ])
        print(f"  Expected y (without noise): [{expected_y[0]:.3f}, {expected_y[1]:.3f}]")
        print(f"  Difference (should be small): {jnp.abs(obs.y1_index - expected_y[0]):.3f}, {jnp.abs(obs.y2_index - expected_y[1]):.3f}")
    
    # Check indices are valid
    all_x1 = jnp.array([obs.x1_index for obs in observations])
    all_x2 = jnp.array([obs.x2_index for obs in observations])
    assert jnp.all((all_x1 >= 0) & (all_x1 < D)), "Invalid x1 indices"
    assert jnp.all((all_x2 >= 0) & (all_x2 < D)), "Invalid x2 indices"
    print("\n✓ All indices valid")
    
    # Check no NaN or Inf
    all_y = jnp.array([(obs.y1_index, obs.y2_index) for obs in observations])
    assert not jnp.any(jnp.isnan(all_y)), "NaN in observations"
    assert not jnp.any(jnp.isinf(all_y)), "Inf in observations"
    print("✓ No NaN or Inf values")
    
    # Check y values are reasonable (mean should be around 0, std around sqrt(2) due to att-def)
    print(f"\nObservation statistics:")
    print(f"  y1 mean: {jnp.mean(all_y[:, 0]):.3f}, std: {jnp.std(all_y[:, 0]):.3f}")
    print(f"  y2 mean: {jnp.mean(all_y[:, 1]):.3f}, std: {jnp.std(all_y[:, 1]):.3f}")

def main():
    key = jax.random.PRNGKey(0)
    observations, true_states = generate_observations(key, T=200)
    print(f"Generated {len(observations)} observations and {true_states.shape} true states.")

    verify_observations(observations, true_states, T=200)
if __name__ == "__main__":
    main()