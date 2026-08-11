import jax
import jax.numpy as jnp
import numpy as np

from rbpf.src.utils import RBPFFootballResults, EMParams, FootballResults
from rbpf.src.data import WORLDCUP_2026_TEAMS, get_results, ACTIVE_TEAMS
from rbpf.src.helpers import default_init_params, generate_augmented_data, params_to_dict
from rbpf.src.model import run_filter, compute_gamma_trajectory
from rbpf.src.bivariate_poisson import loglik

import os
import json
import cuthbertlib
import optax
from tqdm import tqdm
from rbpf.src.graphic import plot_log_likelihood_history

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

MAX_GOALS = 8
N = 100


def _sample_psd_gaussian(
    key: jax.Array,
    mean: jax.Array,
    covariance: jax.Array,
) -> jax.Array:
    """Sample from a PSD Gaussian, preserving exact zero-variance directions.

    The RBPF smoothing covariances (Gamma ⊗ B) are positive-semidefinite, not
    positive-definite: observed teams have exact zero covariance per
    ALGORITHM.md §4.1.1. `jax.random.multivariate_normal` (Cholesky-based)
    returns NaN on such singular matrices. Here we eigendecompose, clip tiny
    eigenvalues to zero, and sample noise only in the nonzero-variance
    directions — observed (zero-variance) teams stay exactly at their mean.

    We are sampling from a covariance matrix that has 0 for some covariances.
    """
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    eigvals = jnp.clip(eigvals, 0.0)
    noise = jax.random.normal(key, mean.shape)
    return mean + eigvecs @ (jnp.sqrt(eigvals) * noise)


def smoother_rts(
    filtered_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    num_teams: int,
    key: jax.Array
):
    """
    Rauch-Tung-Striebel (RTS) smoother for the RBPF model.
    """
    n_particles = filtered_states.particles.x.shape[1]
    K = filtered_states.particles.x.shape[3]  # 2 (attack/defence)
    dim = num_teams * K  # total state dimension

    # 1. Sample Terminal States
    key, cat_key, sample_key = jax.random.split(key, 3)
    log_w_T = filtered_states.log_weights[-1]  # (N,)
    I_T = jax.random.categorical(cat_key, log_w_T)  # scalar

    gamma_T = model_inputs.gamma_t[-1]  # (M, M)  -- NOT particles.gamma
    # gamma_T is PSD with exact-zero rows for observed teams (ALGORITHM.md
    # §4.1.1). Use the PSD-aware sampler so observed teams stay at their mean
    # instead of jittering (which would give them artificial variance).
    Sigma_T = jnp.kron(gamma_T, params.B)  # (2M, 2M)
    mu_T = filtered_states.particles.x[-1, I_T]  # (M, 2)
    X_T_STAR = _sample_psd_gaussian(
        sample_key, mu_T.flatten(), Sigma_T
    ).reshape(num_teams, K)  # (M, 2)

    # 2. Backward Sampling
    def backward_sampling(carry, xs):
        X_next_star, key = carry  # X_next_star: (M, 2)
        key, cat_key, sample_key = jax.random.split(key, 3)

        particles_t, log_w_t, gamma_t, gamma_pred_t1, timestamp_tplus1, timestamp_prev_tplus1 = xs

        # --- 2.5: Backward weights ---
        # Transition from t -> t+1. Since timestamp_prev[t+1] = timestamp[t],
        # dt = timestamp[t+1] - timestamp[t] = timestamp[t+1] - timestamp_prev[t+1].
        dt = timestamp_tplus1 - timestamp_prev_tplus1
        phi = jnp.exp(-params.kappa * dt) * jnp.eye(num_teams)  # (M, M)
        pred_mean = params.mean_0 + phi @ (particles_t - params.mean_0)  # (N, M, 2)
        pred_Sigma = jnp.kron(gamma_pred_t1, params.B)  # (2M, 2M)

        deltas = (X_next_star - pred_mean).reshape(n_particles, -1)  # (N, 2M)
        quad = jnp.sum(
            deltas * jnp.linalg.solve(pred_Sigma, deltas.T).T, axis=-1
        )  # (N,)
        log_transition = (
            -0.5 * quad
            - 0.5 * jnp.log(jnp.linalg.det(pred_Sigma))
            - 0.5 * dim * jnp.log(2 * jnp.pi)
        )  # (N,)
        log_backward_weights = log_w_t + log_transition  # (N,)

        I_t = jax.random.categorical(cat_key, log_backward_weights)  # scalar

        # --- 2.6: RTS gain ---
        J_gamma = gamma_t @ phi.T @ jnp.linalg.pinv(gamma_pred_t1)  # (M, M)
        J = jnp.kron(J_gamma, jnp.eye(K))  # (2M, 2M)  -- I_K, NOT B

        diff = (X_next_star - pred_mean[I_t]).flatten()  # (2M,)
        mu_RTS = particles_t[I_t] + (J @ diff).reshape(num_teams, K)  # (M, 2)

        Gamma_RTS = gamma_t - J_gamma @ gamma_pred_t1 @ J_gamma.T  # (M, M)
        Gamma_RTS = 0.5 * (Gamma_RTS + Gamma_RTS.T)  # Ensure symmetry
        Sigma_RTS = jnp.kron(Gamma_RTS, params.B)  # (2M, 2M)

        X_t_star = _sample_psd_gaussian(
            sample_key, mu_RTS.flatten(), Sigma_RTS
        ).reshape(num_teams, K)  # (M, 2)
        return (X_t_star, key), X_t_star

    xs_particles = filtered_states.particles.x[:-1]       # (T-1, N, M, K)
    xs_log_weights = filtered_states.log_weights[:-1]      # (T-1, N)
    xs_gamma_t = model_inputs.gamma_t[:-1]                 # (T-1, M, M)
    xs_gamma_pred_tplus1 = model_inputs.gamma_pred_t[1:]       # (T-1, M, M)
    xs_timestamp_tplus1 = model_inputs.timestamp[1:]        # (T-1,)
    xs_timestamp_prev_tplus1 = model_inputs.timestamp_prev[1:]  # (T-1,)
    _, smoothed_rest = jax.lax.scan(
        f=backward_sampling,
        init=(X_T_STAR, sample_key),
        xs=(xs_particles, xs_log_weights, xs_gamma_t, xs_gamma_pred_tplus1,
            xs_timestamp_tplus1, xs_timestamp_prev_tplus1),
        reverse=True
    )
    # reverse=True returns times 0..T-2 in chronological order; append terminal state
    smoothed_states = jnp.concatenate(
        [smoothed_rest, X_T_STAR[None]], axis=0
    )  # (T, M, 2)
    return smoothed_states

def E_step(
    params: EMParams,
    model_inputs: RBPFFootballResults,
    num_teams: int,
    n_particles: int,
    key: jax.Array
):
    """
    E-step: Forward Filter Backward Sampling (FFBSi)
    """
    key, filter_key, smoother_key = jax.random.split(key, 3)
    print(f"Running E-step: Forward Filter Backward Sampling (FFBSi)")
    print(f"  num_teams = {num_teams}, n_particles = {n_particles}")
    print(f"  Start time: {model_inputs.timestamp[0]}, End time: {model_inputs.timestamp[-1]}")
    
    filtered_states, _ = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        num_teams=num_teams,
        n_particles=n_particles
    )
    smoothed_states = smoother_rts(
        filtered_states=filtered_states,
        model_inputs=model_inputs,
        params=params,
        num_teams=num_teams,
        key=smoother_key
    )
    return filtered_states, smoothed_states, filtered_states.log_normalizing_constant[-1]

def loss_fn(params: EMParams, smoothed_states: jnp.ndarray, model_inputs: RBPFFootballResults):
    n_observations = smoothed_states.shape[0]
    num_teams = smoothed_states.shape[1]
    K = smoothed_states.shape[2]  # 2 (attack/defence)
    dim = num_teams * K  # total state dimension

    observation_indices = jnp.arange(n_observations)
    transition_indices = jnp.arange(1, n_observations)

    # Observation Loss: -sum_t log p(y_t | x_t)   (bivariate Poisson)
    def obs_step(observation_index):
        return loglik(
            y=jnp.array([model_inputs.home_score[observation_index], model_inputs.away_score[observation_index]]),
            x_i=smoothed_states[observation_index, model_inputs.home_team_id[observation_index]],
            x_j=smoothed_states[observation_index, model_inputs.away_team_id[observation_index]],
            alpha=params.alpha,
            beta=params.beta,
            max_goals=MAX_GOALS,
            scale=1.0
        )
    obs_losses = jax.vmap(obs_step)(observation_indices)
    sum_observation_loss = -jnp.sum(obs_losses)

    # Transition Loss: -sum_t log p(x_t | x_{t-1})  (OU process, Kronecker covariance)
    dts = model_inputs.timestamp[transition_indices] - model_inputs.timestamp_prev[transition_indices]
    phis = jnp.exp(-params.kappa * dts)  # scalar phi_t = exp(-kappa*dt), I_M implied

    def transition_step(observation_index, phi):
        # mu_{t|t-1} = mu_0 + Phi_t (x_{t-1} - mu_0),  Phi_t = phi_t * I_M  in team space
        pred_mean = params.mean_0 + phi * (smoothed_states[observation_index - 1] - params.mean_0)  # (M, K)
        diff = smoothed_states[observation_index] - pred_mean  # (M, K)

        # Q_t = (Gamma_0 - phi_t @ Gamma_0 @ phi_t.T) ⊗ B,  phi_t = phi * I_M
        # (1 - phi^2) >= 0 always (phi = exp(-kappa dt) <= 1). It is exactly 0
        # when phi = 1 (dt = 0, consecutive same-timestamp matches), which makes
        # Q singular and solve()/slogdet() return inf/nan. Clamp the scale to a
        # small positive floor as a float32 underflow guard — this is NOT an
        # added structural covariance (Q_gamma is PD wherever phi < 1).
        scale = jnp.clip(1.0 - phi**2, 1e-6, None)
        Q_gamma = scale * params.gamma_0
        Q_gamma = 0.5 * (Q_gamma + Q_gamma.T)  # ensure symmetry
        Q_full = jnp.kron(Q_gamma, params.B)  # (2M, 2M)

        diff_flat = diff.reshape(-1)  # (2M,)
        quad = diff_flat @ jnp.linalg.solve(Q_full, diff_flat)
        # Use slogdet for a numerically stable log-determinant: the raw det()
        # underflows to 0.0 in float32 for these high-dimensional matrices,
        # making log(det) = -inf. sign is +1 for PSD Q.
        sign, log_det = jnp.linalg.slogdet(Q_full)
        return -0.5 * quad - 0.5 * log_det - 0.5 * dim * jnp.log(2 * jnp.pi)

    transition_losses = jax.vmap(transition_step)(transition_indices, phis)
    sum_transition_loss = -jnp.sum(transition_losses)

    return sum_observation_loss + sum_transition_loss

def _symmetrize(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise a square matrix (for PSD-constrained params like gamma_0, B)."""
    return 0.5 * (x + x.T)


def _constrain(params: EMParams) -> EMParams:
    """Apply validity constraints so parameters stay in their support.

    - kappa >= 0 (OU mean-reversion rate).
    - alpha, beta unconstrained real.
    - gamma_0, B symmetrised (and kept positive-definite via projection below).
    """
    kappa = jnp.maximum(params.kappa, 1e-6)
    gamma_0 = _symmetrize(params.gamma_0)
    B = _symmetrize(params.B)
    return EMParams(
        mean_0=params.mean_0,
        gamma_0=gamma_0,
        B=B,
        kappa=kappa,
        alpha=params.alpha,
        beta=params.beta,
    )

def M_step(
    smoothed_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    prev_params: EMParams,
    learning_rate: float,
    n_gradient_steps: int
):
    """
    M-step: Update parameters via scale-aware ADAM with a cosine schedule.

    The loss is a sum over ~O(10^4) matches, so its magnitude is huge and a
    single global learning rate either under- or over-shoots the different
    parameter blocks (alpha/beta vs gamma_0/B/kappa live on very different
    scales). We therefore:
      1. use per-parameter learning rates via ``multi_transform``,
      2. decay the LR with a cosine schedule over ``n_gradient_steps``,
      3. run the full gradient budget (no premature early-stop) while tracking
         the best params by a *relative* improvement threshold,
      4. project the result back onto the valid (symmetric / non-neg) support.
    """
    # --- JIT-compiled value and gradient (over the 5 learned fields) ---
    def _loss_and_grad(carry):
        return loss_fn(
            EMParams(
                mean_0=prev_params.mean_0,
                gamma_0=carry["gamma_0"],
                B=carry["B"],
                kappa=carry["kappa"],
                alpha=carry["alpha"],
                beta=carry["beta"],
            ),
            smoothed_states,
            model_inputs,
        )

    value_and_grad_fn = jax.jit(jax.value_and_grad(_loss_and_grad, argnums=0))

    # Initial parameter blocks (dict so optax.multi_transform labels align).
    carry = {
        "gamma_0": prev_params.gamma_0,
        "B": prev_params.B,
        "kappa": prev_params.kappa,
        "alpha": prev_params.alpha,
        "beta": prev_params.beta,
    }

    # --- Per-parameter learning rates (scale-aware) ---
    # gamma_0 and B are O(1) matrices; kappa/alpha/beta are scalars on
    # different scales. alpha/beta dominate through the observation term, so
    # give them the base LR; gamma_0/B/kappa use smaller multipliers.
    base = learning_rate
    lr_mapping = {
        "gamma_0": base * 0.5,
        "B": base * 0.5,
        "kappa": base * 1.0,
        "alpha": base * 1.0,
        "beta": base * 1.0,
    }
    transforms = {
        "gamma_0": optax.adam(lr_mapping["gamma_0"]),
        "B": optax.adam(lr_mapping["B"]),
        "kappa": optax.adam(lr_mapping["kappa"]),
        "alpha": optax.adam(lr_mapping["alpha"]),
        "beta": optax.adam(lr_mapping["beta"]),
    }
    param_labels = {
        "gamma_0": "gamma_0",
        "B": "B",
        "kappa": "kappa",
        "alpha": "alpha",
        "beta": "beta",
    }
    optimizer = optax.chain(
        optax.multi_transform(transforms, param_labels),
        optax.scale_by_schedule(
            optax.cosine_decay_schedule(-1.0, n_gradient_steps)
        ),
    )
    opt_state = optimizer.init(carry)

    # --- Gradient descent loop ---
    best_loss = None
    best_carry = carry
    best_step = -1
    patience = max(10, n_gradient_steps // 5)
    no_improve_counter = 0

    for step in range(n_gradient_steps):
        loss, grads = value_and_grad_fn(carry)
        loss_val = float(loss)

        updates, opt_state = optimizer.update(grads, opt_state, carry)
        carry = optax.apply_updates(carry, updates)

        # Track best by relative improvement (loss is ~1e4, so absolute tol
        # would never fire; use a relative threshold).
        improved = (best_loss is None) or (loss_val < best_loss * (1 - 1e-4))
        if improved:
            best_loss = loss_val
            best_carry = carry
            best_step = step
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if step % 10 == 0 or step == n_gradient_steps - 1:
            print(f"      M-step [{step:3d}/{n_gradient_steps}] loss={loss_val:.4f}")

        # Only early-stop once we've had some movement; require the loss to
        # actually be improving relative to the best seen so far.
        if no_improve_counter >= patience and best_step >= 5:
            print(f"      M-step early stop at step {step} (no relative improvement "
                  f"for {patience} steps); best at step {best_step}.")
            break

    if best_loss is None:
        best_carry = carry

    # Project back onto the valid support.
    final = _constrain(EMParams(
        mean_0=prev_params.mean_0,
        gamma_0=best_carry["gamma_0"],
        B=best_carry["B"],
        kappa=best_carry["kappa"],
        alpha=best_carry["alpha"],
        beta=best_carry["beta"],
    ))
    print(f"      M-step done: loss {best_loss:.4f} -> best at step {best_step}, "
          f"kappa={float(final.kappa):.5f} alpha={float(final.alpha):.5f} "
          f"beta={float(final.beta):.5f}")
    return final

def run_EM(
    model_inputs: FootballResults,
    init_params: EMParams,
    num_teams: int,
    n_particles: int = 10,
    n_epochs: int = 10,
    n_gradient_steps: int = 10,
    learning_rate: float = 1e-3,
    key: jax.Array = jax.random.PRNGKey(42)
) -> tuple[EMParams, jnp.ndarray]:
    key = jax.random.PRNGKey(42)
    params = init_params
    
    # initialize gamma trajectory
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs, 
        gamma_0=params.gamma_0, 
        kappa=params.kappa, 
        num_teams=num_teams
    )

    augmented_results = generate_augmented_data(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain
    )

    log_likelihood_history = []
    
    for epoch in tqdm(range(n_epochs)):
        key, e_key = jax.random.split(key)
        # 1. run E step to get expected sufficient statistics
        print("    E-step: Running filtering and backward sampling...")
        _, smoothed_states, log_marginal = E_step(
            params=params, 
            model_inputs=augmented_results, 
            num_teams=num_teams,
            n_particles=n_particles,
            key=e_key
        )
        log_likelihood_history.append(log_marginal)
        # 2. run M step to update parameters
        print("    M-step: Updating parameters...")
        # assign the updated parameters to params for the next iteration
        params = M_step(
            smoothed_states=smoothed_states, 
            model_inputs=augmented_results, 
            prev_params=params, 
            learning_rate=learning_rate, n_gradient_steps=n_gradient_steps
        )

    print("EM completed. Final parameters:")
    print("  kappa:", params.kappa)
    print("  alpha:", params.alpha)
    print("  beta:", params.beta)
    print("  B:", params.B)
    print("  gamma_0:", params.gamma_0.shape)
    print("  mean_0:", params.mean_0.shape)

    return params, jnp.array(log_likelihood_history)

def main():
    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01", 
        end_date="2025-12-31", 
        max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS, # ACTIVE_TEAMS
    )
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, team_id_to_name=team_id_to_name)

    # gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
    #     model_inputs=model_inputs, 
    #     gamma_0=params.gamma_0, 
    #     kappa=params.kappa, 
    #     num_teams=NUM_TEAMS
    # )
    # augmented_results = generate_augmented_data(
    #     model_inputs=model_inputs,
    #     gamma_updated=gamma_updated,
    #     gamma_pred=gamma_pred,
    #     kalman_gain=kalman_gain
    # )
    
    # smoothed_states = E_step(
    #     params=params,
    #     model_inputs=augmented_results,
    #     num_teams=NUM_TEAMS,
    #     n_particles=10,
    #     key=key
    # )
    # print("Smoothed states shape:", smoothed_states.shape)
    try:
        final_params, log_marginal_likelihoods = run_EM(
            model_inputs=model_inputs,
            init_params=params,
            num_teams=NUM_TEAMS,
            n_particles=N,
            n_epochs=3,
            n_gradient_steps=10,
            learning_rate=0.01,
            key=key
        )
    except Exception as e:
        print("Error during EM run:", e)
        raise
    # save parameters to JSON
    output_path = "./rbpf/outputs/smoothing"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + "/em_params.json", "w") as f:
        json.dump(params_to_dict(final_params), f, indent=2)
    with open(output_path + "/log_marginal_likelihoods.json", "w") as f:
        json.dump(np.asarray(log_marginal_likelihoods).tolist(), f, indent=2)

    # plot log marginal likelihoods
    plot_log_likelihood_history(log_marginal_likelihoods.tolist(), output_path=output_path + "/log_marginal_likelihoods.png")

if __name__ == "__main__":
    main()