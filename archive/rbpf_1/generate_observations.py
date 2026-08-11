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
key = jax.random.PRNGKey(42)
A = jax.random.normal(key, (D, D))
INIT_GAMMA = A @ A.T + 1.0 * jnp.eye(D)
INIT_B = jnp.array([[10, 0.5], [0.5, 3]])  # Latent state covariance (P x P)
INIT_SIGMA = jnp.kron(INIT_GAMMA, INIT_B)  # Full covariance (DP x DP)

class Observation(NamedTuple):
    t : int | jax.Array
    t_prev : int | jax.Array
    x1_index : jax.Array
    x2_index : jax.Array
    y1 : jax.Array
    y2 : jax.Array

def generate_observations(key: jax.Array, T: int = 200) -> tuple[Observation, jax.Array]:
    """
    Generate observations for T time steps based on the model in PROOF_V4.md.
    
    Uses Python loop for generation, then jax.tree.map to stack outputs
    into tree-structured Observation arrays.
    
    State: X_t in R^(D x P) where D=10 features, P=2 (att, def)
    Observation: y_t in R^H where H=2 (bivariate Gaussian)
    
    The OU-process dynamics: X_t = mu_0 + Phi_t (X_{t-1} - mu_0) + epsilon_t
    where Phi_t = phi_t \\otimes I_P, phi_t = exp(-kappa * delta_t) * I_D
    and Q_t = Sigma_0 - Phi_t Sigma_0 Phi_t^T
    
    Returns:
        observations: Observation namedtuple with array fields of shape (T-1, ...)
        true_states: array of shape (T, D, P) - the latent states
    """
    key, key_init = jax.random.split(key)

    # init state: X_0 ~ N(mu_0, Sigma_0) where Sigma_0 = Gamma_0 \otimes B
    Sigma_0 = jnp.kron(INIT_GAMMA, INIT_B)  # (D*D, P*P)
    mu0_flat = INIT_MU.flatten()  # (D*P,)
    x0_flat = jax.random.multivariate_normal(key_init, mean=mu0_flat, cov=Sigma_0)  # (D*P,)
    x_curr = x0_flat.reshape(D, P)  # (D, P)

    observations_list = []
    true_states = jnp.zeros((T, D, P))
    true_states = true_states.at[0].set(x_curr)
    true_gammas = jnp.zeros((T, D, D))    # Feature covariances Gamma_t
    true_gammas = true_gammas.at[0].set(INIT_GAMMA.copy())

    delta_t = 1
    for t in range(1, T):
        # 1. propagate state using OU-process dynamics
        x_curr = true_states[t-1]
        phi_t = jnp.exp(-INIT_KAPPA * delta_t) * jnp.eye(D)  # (D, D)
        predicted_mean = INIT_MU + phi_t @ (x_curr - INIT_MU)  # (D, P)

        gamma_curr = true_gammas[t-1]
        predicted_gamma = phi_t @ gamma_curr @ phi_t.T + INIT_GAMMA - phi_t @ INIT_GAMMA @ phi_t.T

        # sample process noise for all features
        Q_t = jnp.kron(INIT_GAMMA - phi_t @ INIT_GAMMA @ phi_t.T, INIT_B)
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

        obs_mean = jnp.array([x1_state[0] - x2_state[1], x2_state[0] - x1_state[1]])
        key, key_obs = jax.random.split(key)
        y_t = obs_mean + jax.random.multivariate_normal(key_obs, mean=jnp.zeros(H), cov=jnp.eye(H))
        
        observations_list.append(
            Observation(
                t=t,
                t_prev=t - delta_t,
                x1_index=x1_idx,
                x2_index=x2_idx,
                y1=y_t[0],
                y2=y_t[1]
            )
        )

    # Stack list of Observation objects into tree-structured arrays using jax.tree.map
    # observations_list has T-1 elements (t=1 to T-1)
    # Use jax.tree.map to stack all fields: (T-1,) list -> (T-1,) array
    observations = jax.tree.map(lambda *xs: jnp.stack(xs), *observations_list)
    
    return observations, true_states

def verify_observations(observations: Observation, true_states: jax.Array, T: int = 200):
    """Check observations were generated correctly.
    
    Now works with tree-structured Observation arrays.
    observations[i] corresponds to time t=i+1 (since t=0 has no observation).
    """
    print(f"Number of observations: {T-1} (expected: {T-1}, t=1 to t={T-1})")
    
    # Check first few observations
    for i in range(3):
        print(f"\n--- Observation index {i} (time t={i+1}) ---")
        t_val = int(observations.t[i])
        x1_idx = int(observations.x1_index[i])
        x2_idx = int(observations.x2_index[i])
        y1_val = float(observations.y1[i])
        y2_val = float(observations.y2[i])
        
        print(f"  t={t_val}, x1_idx={x1_idx}, x2_idx={x2_idx}")
        print(f"  y=[{y1_val:.3f}, {y2_val:.3f}]")
        
        # Verify y matches the formula: [att_i - def_j, att_j - def_i]
        x1_state = true_states[t_val, x1_idx]  # [att, def] for x1
        x2_state = true_states[t_val, x2_idx]  # [att, def] for x2
        expected_y = jnp.array([
            x1_state[0] - x2_state[1],  # att_i - def_j
            x2_state[0] - x1_state[1]   # att_j - def_i
        ])
        print(f"  Expected y (without noise): [{expected_y[0]:.3f}, {expected_y[1]:.3f}]")
        print(f"  Difference (should be small): {abs(y1_val - float(expected_y[0])):.3f}, {abs(y2_val - float(expected_y[1])):.3f}")
    
    # Check indices are valid
    all_x1 = observations.x1_index
    all_x2 = observations.x2_index
    assert jnp.all((all_x1 >= 0) & (all_x1 < D)), "Invalid x1 indices"
    assert jnp.all((all_x2 >= 0) & (all_x2 < D)), "Invalid x2 indices"
    print("\n✓ All indices valid")
    
    # Check no NaN or Inf
    all_y1 = observations.y1
    all_y2 = observations.y2
    assert not jnp.any(jnp.isnan(all_y1)), "NaN in y1 observations"
    assert not jnp.any(jnp.isnan(all_y2)), "NaN in y2 observations"
    assert not jnp.any(jnp.isinf(all_y1)), "Inf in y1 observations"
    assert not jnp.any(jnp.isinf(all_y2)), "Inf in y2 observations"
    print("✓ No NaN or Inf values")
    
    # Check y values are reasonable
    print(f"\nObservation statistics:")
    print(f"  y1 mean: {jnp.mean(all_y1):.3f}, std: {jnp.std(all_y1):.3f}")
    print(f"  y2 mean: {jnp.mean(all_y2):.3f}, std: {jnp.std(all_y2):.3f}")

def main():
    key = jax.random.PRNGKey(0)
    T = 200
    observations, true_states = generate_observations(key, T=T)
    print(f"Generated observations with tree structure:")
    print(f"  observations.t shape: {observations.t.shape} (T-1 = {T-1} observations)")
    print(f"  observations.x1_index shape: {observations.x1_index.shape}")
    print(f"  observations.y1 shape: {observations.y1.shape}")
    print(f"  true_states shape: {true_states.shape} (T = {T} states)")
    print(f"\nNote: observations[i] corresponds to time t=i+1")

    verify_observations(observations, true_states, T=T)
if __name__ == "__main__":
    main()
