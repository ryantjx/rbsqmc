import jax
import jax.numpy as jnp
import numpy as np

from rbpf.src.utils import RBPFFootballResults, EMParams, FootballResults
from rbpf.src.data import WORLDCUP_2026_TEAMS, get_results, ACTIVE_TEAMS
from rbpf.src.helpers import default_init_params, generate_augmented_data, params_to_dict
from rbpf.src.model import run_filter, compute_gamma_trajectory
from rbpf.src.bivariate_poisson import loglik

from tqdm import tqdm
import json
from rbpf.src.graphic import plot_log_likelihood_history
import os
import cuthbertlib
import optax

jax.config.update("jax_platforms", "cpu")

MAX_GOALS = 8
N = 100

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
    Sigma_T = jnp.kron(gamma_T, params.B)  # (2M, 2M)
    mu_T = filtered_states.particles.x[-1, I_T]  # (M, 2)
    X_T_STAR = jax.random.multivariate_normal(
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

        X_t_star = jax.random.multivariate_normal(
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
        Q_gamma = params.gamma_0 - phi**2 * params.gamma_0
        Q_gamma = 0.5 * (Q_gamma + Q_gamma.T)  # ensure symmetry
        Q_full = jnp.kron(Q_gamma, params.B)  # (2M, 2M)

        diff_flat = diff.reshape(-1)  # (2M,)
        quad = diff_flat @ jnp.linalg.solve(Q_full, diff_flat)
        return -0.5 * quad - 0.5 * jnp.log(jnp.linalg.det(Q_full)) - 0.5 * dim * jnp.log(2 * jnp.pi)

    transition_losses = jax.vmap(transition_step)(transition_indices, phis)
    sum_transition_loss = -jnp.sum(transition_losses)

    return sum_observation_loss + sum_transition_loss

def M_step(
    smoothed_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    num_teams: int,
    prev_params: EMParams,
    learning_rate: float,
    n_gradient_steps: int
):  
    """
    M-step: Update parameters using ADAM gradient descent.
    """
    # --- JIT-compiled value and gradient ---
    value_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_fn, argnums=0)
    )
    # --- Initialize optimizer ---
    optimizer = optax.adam(learning_rate)
    params = (
        prev_params.gamma_0,
        prev_params.B,
        prev_params.kappa,
        prev_params.alpha,
        prev_params.beta,
    )
    opt_state = optimizer.init(params)

    # --- Gradient descent loop ---
    best_loss = jnp.inf
    best_params = params
    no_improve_counter = 0

    for step in range(n_gradient_steps):
        loss, grads = value_and_grad_fn(
            EMParams(
                mean_0=prev_params.mean_0,
                gamma_0=params[0],
                B=params[1],
                kappa=params[2],
                alpha=params[3],
                beta=params[4],
            ),
            smoothed_states,
            model_inputs
        )
        # grads is an EMParams NamedTuple; optimise the learned fields.
        grad_tuple = (
            grads.gamma_0,
            grads.B,
            grads.kappa,
            grads.alpha,
            grads.beta,
        )
        updates, opt_state = optimizer.update(grad_tuple, opt_state)
        params = optax.apply_updates(params, updates)

        loss_val = float(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        
        if no_improve_counter >= 5:
            print(f"Early stopping at step {step} due to no improvement in loss.")
            break

    return EMParams(
        gamma_0=best_params[0],
        B=best_params[1],
        kappa=best_params[2],
        alpha=best_params[3],
        beta=best_params[4],
        mean_0=prev_params.mean_0,  # mean_0 is fixed
    )

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
            num_teams=num_teams,
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