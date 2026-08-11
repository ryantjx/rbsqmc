import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
from archive.rbpf_1.highdim_rbpf_filter import (
    ParticleState, State, D, P, H, INIT_KAPPA, INIT_MU, 
    INIT_GAMMA, INIT_B, INIT_SIGMA
)
from archive.rbpf_1.generate_observations import Observation


def adaptive_smoother(
    observations,
    start_value,
    confidence=None,
    blend=0.35,
):
    """Smooth a noisy 1D sequence with a confidence-weighted recursive update.

    This helper is intentionally lightweight and does not depend on the RBPF
    state machinery. It is useful as a small fallback when a sequence needs a
    simple denoising step while preserving meaningful changes.

    Args:
        observations: Sequence of noisy measurements.
        start_value: Initial estimate for the first point.
        confidence: Optional per-step confidence values. Higher values trust the
            new observation more strongly.
        blend: Strength of the update. Typical values are in the range 0.1 to 0.6.

    Returns:
        A list of smoothed estimates with the same length as observations.
    """
    if not observations:
        return []

    observation_values = list(observations)

    if confidence is None:
        confidence_values = [1.0] * len(observation_values)
    else:
        confidence_values = list(confidence)
        if len(confidence_values) != len(observation_values):
            raise ValueError("confidence must have the same length as observations")

    if blend < 0:
        raise ValueError("blend must be non-negative")

    smoothed = []
    prev_estimate = float(start_value)

    for obs, conf in zip(observation_values, confidence_values):
        obs_value = float(obs)
        conf_value = float(conf)

        weight = blend * conf_value / (1.0 + conf_value)
        correction = (obs_value - prev_estimate) * weight

        updated_estimate = prev_estimate + correction
        momentum = 0.0
        if smoothed:
            momentum = 0.08 * (updated_estimate - smoothed[-1])

        prev_estimate = updated_estimate + momentum
        smoothed.append(prev_estimate)

    return smoothed


def backward_sample_trajectory(
    key: jax.Array,
    states_historical: State,
    observations: Observation
) -> jax.Array:
    """
    Sample one complete smoothed trajectory using backward sampling.
    
    Works with tree-structured observations where observations[i] is at time t=i+1.
    
    Algorithm:
    1. Sample X_T^* from filtering distribution at time T
    2. For t = T-1, ..., 1:
       - Sample X_t^* ~ p(X_t | X_{t+1}^*, y_{1:t})
    
    Returns: Array of shape (T, D, P) where T is the number of filtering steps
    """
    T_filter = states_historical.particles.x.shape[0]  # Number of filtering steps (T-1)
    T_total = T_filter + 1  # Total time steps including t=0
    
    # Initialize trajectory array (includes t=0)
    trajectory = jnp.zeros((T_total, D, P))
    
    # Split keys for each time step
    keys = jax.random.split(key, T_filter)
    
    # === Step 1: Sample terminal state X_T^* at time T = T_filter ===
    # Get state at last filtering step (time T_filter)
    state_T = jax.tree.map(lambda x: x[-1], states_historical)
    # Extract last observation (at time T_filter = T-1)
    obs_T = jax.tree.map(lambda x: x[-1], observations)
    X_T = sample_terminal_state(keys[0], state_T, obs_T)
    trajectory = trajectory.at[T_filter].set(X_T)
    
    # === Step 2: Backward recursion ===
    # states_historical[i] is at time t=i+1
    # observations[i] is at time t=i+1
    X_next = X_T
    for i in range(T_filter-2, -1, -1):  # From second-to-last down to first
        # Get state at filtering step i (time t=i+1)
        state_i = jax.tree.map(lambda x: x[i], states_historical)
        
        # Get observation at time t=i+2 (for transition to t=i+1)
        # observations[i+1] is at time t=i+2
        obs_next = jax.tree.map(lambda x: x[i+1], observations)
        
        # Sample X_{i+1}^* given X_{i+2}^*
        X_i = backward_sampling(
            key=keys[i+1],
            state_t=state_i,
            observation_t_plus_1=obs_next,
            X_star_t_plus_1=X_next
        )
        
        
        trajectory = trajectory.at[i+1].set(X_i)
        X_next = X_i
    
    return trajectory


def run_rbpf_smoother(
    key: jax.Array,
    states_historical: State,
    observations: Observation,
    n_smooth_trajectories: int = 10
) -> tuple[jax.Array, dict]:
    """
    Forward Filtering Backward Sampling (FFBSi) for RB-PF
    
    Args:
        observations: Observation namedtuple with array fields (tree-structured)
    
    Returns:
        smoothed_trajectories: (n_smooth_trajectories, T, D, P)
        smoothed_params: dict with smoothed estimates
    """
    keys = jax.random.split(key, n_smooth_trajectories)

    # Backward sampling for each trajectory
    smoothed_trajectories = jax.vmap(
        lambda k: backward_sample_trajectory(k, states_historical, observations)
    )(keys)  # (n_smooth_trajectories, T, D, P)

    # Compute smoothed parameters (optimal estimates)
    smoothed_params = compute_smoothed_parameters(smoothed_trajectories)
    
    return smoothed_trajectories, smoothed_params

def sample_terminal_state(
    key: jax.Array,
    state_T: State,
    observation_T: Observation  # Need this to know which features were observed
) -> jax.Array:
    """
    Initialize backward trajectory at time T.
    """
    key, cat_key, gauss_key = jax.random.split(key, 3)
    
    # Sample particle
    log_weights = state_T.log_weights
    I_T = jax.random.categorical(cat_key, log_weights - jax.nn.logsumexp(log_weights))
    
    mu_T = state_T.particles.x[I_T]  # (D, P) - filtered mean
    gamma_T = state_T.particles.gamma[I_T]  # (D, D) - feature covariance
    
    # Get observed indices from observation_T
    observed_indices = jnp.array([observation_T.x1_index, observation_T.x2_index])
    all_indices = jnp.arange(D)
    remaining_indices = jnp.setdiff1d(all_indices, observed_indices)
    
    # X_T^{E,*} is deterministic (already sampled)
    X_T_E = state_T.particles.x_observed[I_T]  # (H, P)
    
    # For X_T^{R,*}, sample from conditional N(mu_T^{R|E}, Sigma_T^{RR|E})
    # Extract submatrices
    mu_T_R = mu_T[remaining_indices]  # (D-H, P)
    gamma_T_RR = gamma_T[jnp.ix_(remaining_indices, remaining_indices)]  # (D-H, D-H)
    
    # Conditional covariance: Sigma_T^{RR|E} = kron(gamma_T_RR, B)
    Sigma_T_RR_given_E = jnp.kron(gamma_T_RR, INIT_B)  # ((D-H)*P, (D-H)*P)
    
    # Sample remaining features
    mu_T_R_flat = mu_T_R.flatten()
    X_T_R_flat = jax.random.multivariate_normal(
        gauss_key, mu_T_R_flat, Sigma_T_RR_given_E
    )
    X_T_R = X_T_R_flat.reshape(-1, P)  # (D-H, P)
    
    # Combine E and R into full state
    X_T_star = jnp.zeros((D, P))
    X_T_star = X_T_star.at[observed_indices].set(X_T_E)
    X_T_star = X_T_star.at[remaining_indices].set(X_T_R)
    
    return X_T_star

def compute_particle_log_likelihood(
        mu_t_i: jax.Array, 
        gamma_t_i: jax.Array,
        mu_0_flat: jax.Array,
        Phi_t: jax.Array,
        Q: jax.Array,
        X_star_t_plus_1_flat: jax.Array,
    ) -> jax.Array:
    """
    Compute log N(X_{t+1}^* | mu_{t+1|t}^(i), P_{t+1|t}^(i)) for particle i
    """
    # Flatten particle's mean
    mu_t_flat = mu_t_i.flatten()  # (D*P,)
    
    # Predictive mean: mu_{t+1|t} = mu_0 + Phi (mu_t - mu_0)
    mu_pred_flat = mu_0_flat + Phi_t @ (mu_t_flat - mu_0_flat)
    
    # Predictive covariance: P = Phi @ Sigma_t @ Phi^T + Q
    Sigma_t = jnp.kron(gamma_t_i, INIT_B)  # (D*P, D*P)
    P_pred = Phi_t @ Sigma_t @ Phi_t.T + Q
    
    # Compute log likelihood using Cholesky
    diff = X_star_t_plus_1_flat - mu_pred_flat
    
    # Use Cholesky for numerical stability
    L = jnp.linalg.cholesky(P_pred + 1e-6 * jnp.eye(D * P))
    # Solve L @ y = diff
    y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    quadratic = jnp.sum(y ** 2)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    
    log_likelihood = -0.5 * (quadratic + log_det + D * P * jnp.log(2 * jnp.pi))
    return log_likelihood
    

def backward_sampling(
    key: jax.Array,
    state_t: State,
    # state_t_plus_1: State,
    observation_t_plus_1: Observation,
    # observation_t: Observation,
    X_star_t_plus_1: jax.Array
) -> jax.Array:
    """
    Backward sampling from p(X_t | X_{t+1}^*, y_{1:t}).
    
    From PROOF_V4.md Section 3:
    1. Compute backward weights w_{t|t+1}^(i)
    2. Sample component index
    3. Sample state from conditional distribution
    """
    key, weight_key, categorical_key, gaussian_key = jax.random.split(key, 4)
    # n_particles = state_t.log_weights.shape[0]
    log_weights_t = state_t.log_weights
    # weights_t = jnp.exp(log_weights_t - jax.nn.logsumexp(log_weights_t))

    X_star_t_plus_1_flat = X_star_t_plus_1.flatten()  # (D*P,)

    # compute Phi_t and Q for the selected particle
    # delta_t should be positive: time from t to t+1
    delta_t = observation_t_plus_1.t - observation_t_plus_1.t_prev
    phi_t = jnp.exp(-INIT_KAPPA * delta_t) * jnp.eye(D)  # (D, D)
    Phi_t = jnp.kron(phi_t, jnp.eye(P))  # (D*P, D*P)
    Q = INIT_SIGMA - Phi_t @ INIT_SIGMA @ Phi_t.T

    mu_0_flat = INIT_MU.flatten()  # (D*P,)


    # === Step 1: Compute backward weights using vmap ===
    log_likelihoods = jax.vmap(
        compute_particle_log_likelihood, 
        in_axes=(0, 0, None, None, None, None)
    )(
        state_t.particles.x, state_t.particles.gamma, mu_0_flat, Phi_t, Q, X_star_t_plus_1_flat
    )  # (N,)
    
    log_backward_weights = log_weights_t + log_likelihoods
    log_backward_weights = log_backward_weights - jax.nn.logsumexp(log_backward_weights)

    # === Step 2: Sample component index I_t ~ Categorical(w_{t|t+1}) ===
    I_t = jax.random.categorical(categorical_key, log_backward_weights)

    # === Step 3: Compute RTS smoothed moments for selected particle ===
    mu_t_i = state_t.particles.x[I_t]  # (D, P)
    mu_t_flat = mu_t_i.flatten()  # (D*P,)

    gamma_t_i = state_t.particles.gamma[I_t]  # (D, D)
    Sigma_t = jnp.kron(gamma_t_i, INIT_B)  # (D*P, D*P)

    # Predictive mean and covariance
    mu_pred_flat = mu_0_flat + Phi_t @ (mu_t_flat - mu_0_flat)  # (D*P,)
    P_pred = Phi_t @ Sigma_t @ Phi_t.T + Q


    # Smoother Gain with regularization for numerical stability
    P_pred_reg = P_pred + 1e-6 * jnp.eye(D * P)
    J_t = Sigma_t @ Phi_t.T @ jnp.linalg.inv(P_pred_reg)

    # Smoothed mean 
    m_t_given_t1_flat = mu_t_flat + J_t @ (X_star_t_plus_1_flat - mu_pred_flat)  # (D*P,)

    # Smoothed covariance with regularization for numerical stability
    Sigma_t_given_t1 = Sigma_t - J_t @ P_pred @ J_t.T  # (D*P, D*P)
    Sigma_t_given_t1 = Sigma_t_given_t1 + 1e-6 * jnp.eye(D * P)  # Ensure positive definite
    
    # Ensure positive definiteness by taking symmetric part
    Sigma_t_given_t1 = 0.5 * (Sigma_t_given_t1 + Sigma_t_given_t1.T)

    # === Step 4: Sample X_t^* ~ N(m_{t|t+1}, Sigma_{t|t+1}) ===
    X_t_star_flat = jax.random.multivariate_normal(
        gaussian_key, m_t_given_t1_flat, Sigma_t_given_t1
    )
    X_t_star = X_t_star_flat.reshape(D, P)
    
    return X_t_star

def compute_smoothed_parameters(
    smoothed_trajectories: jax.Array
) -> dict:
    """
    Compute smoothed parameter estimates from trajectories.
    
    Args:
        smoothed_trajectories: (n_trajectories, T, D, P)
        
    Returns:
        dict with:
        - 'mean': (T, D, P) - smoothed mean
        - 'std': (T, D, P) - smoothed std deviation
        - 'cov': (T, D, D) - feature covariance over time
    """
    # Mean over trajectories
    smoothed_mean = jnp.mean(smoothed_trajectories, axis=0)  # (T, D, P)
    
    # Std deviation
    smoothed_std = jnp.std(smoothed_trajectories, axis=0)  # (T, D, P)
    
    # Compute feature covariance at each time
    T = smoothed_trajectories.shape[1]
    covariances = []
    for t in range(T):
        # Get all trajectories at time t: (n_traj, D, P)
        X_t = smoothed_trajectories[:, t, :, :]  # (n_traj, D, P)
        
        # Reshape to (n_traj, D*P)
        X_t_flat = X_t.reshape(X_t.shape[0], -1)
        
        # Compute covariance
        cov_t = jnp.cov(X_t_flat.T)  # (D*P, D*P)
        covariances.append(cov_t)
    
    return {
        'mean': smoothed_mean,
        'std': smoothed_std,
        'cov': jnp.stack(covariances),  # (T, D*P, D*P)
        'gamma': compute_gamma_from_trajectories(smoothed_trajectories)
    }


def compute_gamma_from_trajectories(
    smoothed_trajectories: jax.Array
) -> jax.Array:
    """
    Extract feature covariance (gamma) from smoothed trajectories.
    
    Gamma_t describes covariance between features (D x D).
    For each time t, we estimate gamma from the P latent dimensions.
    """
    n_traj, T, D, P = smoothed_trajectories.shape
    
    # Reshape to (n_traj, T, D, P)
    # For each feature d, we have P values across trajectories
    
    gammas = []
    for t in range(T):
        # X_t: (n_traj, D, P)
        X_t = smoothed_trajectories[:, t, :, :]
        
        # Compute covariance across features
        # Reshape to (n_traj*P, D) - treat each latent dim as separate sample
        X_reshaped = X_t.transpose(0, 2, 1).reshape(-1, D)  # (n_traj*P, D)
        
        # Gamma_t: (D, D)
        gamma_t = jnp.cov(X_reshaped.T)
        gammas.append(gamma_t)
    
    return jnp.stack(gammas)  # (T, D, D)

def validate_smoother(
    smoothed_trajectories: jax.Array,  # (n_trajectories, T, D, P)
    true_states: jax.Array,
    filter_states: State  # Single stacked state with time dimension
):
    """
    Validate smoothed trajectories against true states and filter estimates.
    """
    n_traj, T, _, _ = smoothed_trajectories.shape
    
    print(f"\n{'='*60}")
    print("RBPF SMOOTHER VALIDATION")
    print(f"{'='*60}")
    print(f"Number of smoothed trajectories: {n_traj}")
    print(f"Time steps: {T}")
    
    # Compute smoothed mean
    smoothed_mean = jnp.mean(smoothed_trajectories, axis=0)  # (T, D, P)
    
    # Compare to filter mean - extract from stacked state
    filter_means = []
    for t in range(T):
        log_weights = filter_states.log_weights[t]
        weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
        x_mean = jnp.sum(filter_states.particles.x[t] * weights[:, None, None], axis=0)
        filter_means.append(x_mean)
    filter_mean = jnp.stack(filter_means)  # (T, D, P)
    
    # RMSE comparison - need to align true_states with filter output
    # true_states has shape (T_total, D, P) where T_total may be larger
    true_states_aligned = true_states[:T]  # Take first T timesteps
    
    smse_rmse = jnp.sqrt(jnp.mean((smoothed_mean - true_states_aligned)**2))
    filter_rmse = jnp.sqrt(jnp.mean((filter_mean - true_states_aligned)**2))
    
    print(f"\nRMSE (smoothed): {smse_rmse:.4f}")
    print(f"RMSE (filter):   {filter_rmse:.4f}")
    print(f"Improvement:     {filter_rmse - smse_rmse:.4f}")
    
    return {
        'smoothed_rmse': float(smse_rmse),
        'filter_rmse': float(filter_rmse),
        'rmse': float(smse_rmse)  # For compatibility
    }

def m_step(
    smoothed_trajectories: jax.Array,
    observations: Observation,
    true_states: jax.Array,
) -> dict:
    """
    M-step of EM: Estimate optimal model parameters from smoothed trajectories.
    
    Parameters to estimate:
    - kappa: OU mean reversion rate
    - mu_0: Initial mean (D, P)
    - Gamma_0: Initial feature covariance (D, D)
    - B: Latent state covariance (P, P)
    - R: Observation noise covariance (H, H)
    
    Args:
        smoothed_trajectories: (n_traj, T, D, P)
        observations: Observation namedtuple
        true_states: (T, D, P) - used for observation prediction residuals
    
    Returns:
        dict with estimated parameters
    """
    n_traj, T, _, _ = smoothed_trajectories.shape
    
    # Use smoothed mean as the representative trajectory
    smoothed_mean = jnp.mean(smoothed_trajectories, axis=0)  # (T, D, P)
    
    # === 1. Estimate mu_0 ===
    # mu_0 = mean of smoothed states at t=0
    mu_0 = smoothed_mean[0]  # (D, P)
    
    # === 2. Estimate kappa ===
    # OU process: X_{t+1} - mu_0 = phi * (X_t - mu_0) + noise
    # where phi = exp(-kappa * delta_t)
    # Regress (X_{t+1} - mu_0) on (X_t - mu_0) to estimate phi
    # Then kappa = -ln(phi) / delta_t
    
    delta_t = 1.0  # Time step
    X_prev = smoothed_mean[:-1]  # (T-1, D, P)
    X_next = smoothed_mean[1:]   # (T-1, D, P)
    
    # Flatten to (T-1, D*P) for regression
    X_prev_flat = (X_prev - mu_0).reshape(-1, D * P)
    X_next_flat = (X_next - mu_0).reshape(-1, D * P)
    
    # Least squares: phi = (X_prev^T X_prev)^{-1} X_prev^T X_next
    # But phi is scalar (same for all features), so use element-wise ratio
    # phi = sum(X_next * X_prev) / sum(X_prev^2)
    phi_numerator = jnp.sum(X_next_flat * X_prev_flat)
    phi_denominator = jnp.sum(X_prev_flat ** 2)
    phi = phi_numerator / (phi_denominator + 1e-10)
    
    # Ensure phi is in (0, 1) for valid OU process
    phi = jnp.clip(phi, 1e-6, 1.0 - 1e-6)
    kappa = -jnp.log(phi) / delta_t
    
    # === 3. Estimate Gamma_0 ===
    # Gamma_0 = feature-level covariance (D x D)
    # Use OU process residuals across all timesteps for robust estimation
    # Residual: r_t = X_{t+1} - mu_0 - phi * (X_t - mu_0)
    # Cov(r_t) = (1 - phi^2) * Gamma_0 ⊗ B
    # Extract Gamma_0 from the feature-level covariance of residuals
    
    residuals = X_next - mu_0 - phi * (X_prev - mu_0)  # (T-1, D, P)
    
    # For Gamma_0: average over P dimensions to get D x D feature covariance
    # Reshape residuals to (T-1, D, P), then for each p, compute cov across D
    Gamma_0 = jnp.zeros((D, D))
    for p in range(P):
        r_p = residuals[:, :, p]  # (T-1, D)
        Gamma_0 += jnp.cov(r_p.T)  # (D, D)
    Gamma_0 /= P
    # Remove the (1 - phi^2) scaling
    Gamma_0 = Gamma_0 / (1 - phi**2 + 1e-10)
    Gamma_0 = Gamma_0 + 1e-6 * jnp.eye(D)
    
    # === 4. Estimate B ===
    # B = within-feature covariance (P x P)
    # From the same residuals, average over D features to get P x P
    B_est = jnp.zeros((P, P))
    for d in range(D):
        r_d = residuals[:, d, :]  # (T-1, P)
        B_est += jnp.cov(r_d.T)  # (P, P)
    B_est /= D
    # Remove the (1 - phi^2) scaling
    B_est = B_est / (1 - phi**2 + 1e-10)
    B_est = B_est + 1e-6 * jnp.eye(P)
    
    # === 5. Estimate R ===
    # R = covariance of observation residuals
    # y_t = [att_i - def_j, att_j - def_i] + noise
    # residual = y_t - predicted_y_t
    n_obs = observations.t.shape[0]
    residuals = []
    for i in range(n_obs):
        t = i + 1  # Actual time
        x1_idx = int(observations.x1_index[i].item())
        x2_idx = int(observations.x2_index[i].item())
        
        # Smoothed state at time t
        x_t = smoothed_mean[t]
        x1_state = x_t[x1_idx]  # [att, def]
        x2_state = x_t[x2_idx]
        
        # Predicted observation
        y_pred = jnp.array([
            x1_state[0] - x2_state[1],
            x2_state[0] - x1_state[1]
        ])
        
        # Actual observation
        y_actual = jnp.array([observations.y1[i], observations.y2[i]])
        
        residuals.append(y_actual - y_pred)
    
    residuals = jnp.stack(residuals)  # (n_obs, H)
    R_est = jnp.cov(residuals.T)  # (H, H)
    R_est = R_est + 1e-6 * jnp.eye(H)
    
    # Compute Sigma_0 = Gamma_0 ⊗ B
    Sigma_0 = jnp.kron(Gamma_0, B_est)
    
    return {
        'kappa': float(kappa),
        'mu_0': mu_0,
        'Gamma_0': Gamma_0,
        'B': B_est,
        'R': R_est,
        'Sigma_0': Sigma_0,
        'phi': float(phi),
    }


def plot_rbpf_smoothed_results(
    states_history: State,
    true_states: jax.Array,
    observations: Observation,
    smoothed_mean: jax.Array,
):
    """Plot time series for all features with True, Filter, and Smoothed overlays.
    
    Saves to ./rbpf/outputs/rbpf_results_smoothed.png
    """
    import matplotlib.pyplot as plt
    
    T = states_history.particles.x.shape[0] + 1  # +1 because states_history starts from t=1
    n_features = D
    
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 24), sharex=True)
    
    # Track which timesteps each feature was observed
    observed_mask = {d: [] for d in range(n_features)}
    n_obs = observations.x1_index.shape[0]
    for i in range(n_obs):
        t = i + 1
        x1_idx = int(observations.x1_index[i].item())
        x2_idx = int(observations.x2_index[i].item())
        observed_mask[x1_idx].append(t)
        observed_mask[x2_idx].append(t)
    
    for feat_idx in range(n_features):
        # True trajectory
        att_true = true_states[:, feat_idx, 0]
        def_true = true_states[:, feat_idx, 1]
        
        # Filter estimate (prepend true state at t=0)
        att_filter = [float(true_states[0, feat_idx, 0])]
        def_filter = [float(true_states[0, feat_idx, 1])]
        for i in range(T - 1):
            log_weights = states_history.log_weights[i]
            weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
            x_mean = jnp.sum(states_history.particles.x[i] * weights[:, None, None], axis=0)
            att_filter.append(float(x_mean[feat_idx, 0]))
            def_filter.append(float(x_mean[feat_idx, 1]))
        att_filter = jnp.array(att_filter)
        def_filter = jnp.array(def_filter)
        
        # Smoothed mean
        att_smoothed = smoothed_mean[:, feat_idx, 0]
        def_smoothed = smoothed_mean[:, feat_idx, 1]
        
        # Plot Attack (column 0)
        ax_att = axes[feat_idx, 0]
        ax_att.plot(range(T), att_true, 'b-', alpha=0.5, label='True')
        ax_att.plot(range(T), att_filter, 'r-', alpha=0.5, label='Filter')
        ax_att.plot(range(T), att_smoothed, 'g-', alpha=0.7, label='Smoothed')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax_att.scatter(obs_times, [float(att_true[t]) for t in obs_times],
                          c='blue', marker='x', s=50, zorder=5)
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
        ax_def.plot(range(T), def_filter, 'r-', alpha=0.5, label='Filter')
        ax_def.plot(range(T), def_smoothed, 'g-', alpha=0.7, label='Smoothed')
        if observed_mask[feat_idx]:
            obs_times = observed_mask[feat_idx]
            ax_def.scatter(obs_times, [float(def_true[t]) for t in obs_times],
                          c='blue', marker='x', s=50, zorder=5)
        if feat_idx == 0:
            ax_def.set_title('Defense')
        if feat_idx == n_features - 1:
            ax_def.set_xlabel('Time')
        ax_def.legend(loc='upper right', fontsize=8)
        ax_def.grid(True, alpha=0.3)
    
    plt.suptitle('RBPF Time Series (Blue=True, Red=Filter, Green=Smoothed, X=Observed)', y=1.0)
    plt.tight_layout()
    plt.savefig('./rbpf/outputs/rbpf_results_smoothed.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to ./rbpf/outputs/rbpf_results_smoothed.png")

def main():
    from archive.rbpf_1.highdim_rbpf_filter import run_rbpf, generate_observations
    
    # Run filter first
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    observations, true_states = generate_observations(subkey, T=200)
    
    key, subkey = jax.random.split(key)
    states_history = run_rbpf(
        key=subkey, observations=observations, n_particles=100)

    # Run smoother
    key, smoother_subkey = jax.random.split(key)
    smoothed_trajectories, smoothed_params = run_rbpf_smoother(
        key=smoother_subkey, states_historical=states_history, observations=observations, n_smooth_trajectories=10
    )
    
    # Save smoothed parameters to outputs
    import os
    import json
    os.makedirs('./rbpf/outputs', exist_ok=True)
    
    # Save as .npy files
    jnp.save('./rbpf/outputs/smoothed_mean.npy', smoothed_params['mean'])
    jnp.save('./rbpf/outputs/smoothed_std.npy', smoothed_params['std'])
    jnp.save('./rbpf/outputs/smoothed_gamma.npy', smoothed_params['gamma'])
    
    # === M-step: Estimate optimal model parameters ===
    print("\n=== M-step: Estimating optimal model parameters ===")
    optimal_params = m_step(smoothed_trajectories, observations, true_states)
    
    print(f"  kappa:   {optimal_params['kappa']:.4f}")
    print(f"  phi:     {optimal_params['phi']:.4f}")
    print(f"  mu_0:    shape={optimal_params['mu_0'].shape}")
    print(f"  Gamma_0: shape={optimal_params['Gamma_0'].shape}")
    print(f"  B:       shape={optimal_params['B'].shape}")
    print(f"  R:       shape={optimal_params['R'].shape}")
    
    # Save as JSON - only optimal initial parameters for initialization
    json_params = {
        'optimal_params': {
            'kappa': optimal_params['kappa'],
            'phi': optimal_params['phi'],
            'mu_0': jnp.array(optimal_params['mu_0']).tolist(),
            'Gamma_0': jnp.array(optimal_params['Gamma_0']).tolist(),
            'B': jnp.array(optimal_params['B']).tolist(),
            'R': jnp.array(optimal_params['R']).tolist(),
            'Sigma_0': jnp.array(optimal_params['Sigma_0']).tolist(),
        }
    }
    with open('./rbpf/outputs/smoothed_params.json', 'w') as f:
        json.dump(json_params, f, indent=2)
    
    print("Saved optimal initial params to ./rbpf/outputs/smoothed_params.json")
    
    # Validate
    metrics = validate_smoother(smoothed_trajectories, true_states, states_history)
    
    # Plot results with smoothed overlay
    smoothed_mean = smoothed_params['mean']
    plot_rbpf_smoothed_results(states_history, true_states, observations, smoothed_mean)
    
    print("============================= Final Smoother Metrics =============================")
    print(f"RMSE: {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()