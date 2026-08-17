
import jax
import jax.numpy as jnp
import os
from tqdm import tqdm
import optax
import json

from rbpf.src.model import run_filter
from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
from rbpf.src.helpers import (
    default_init_params,
    generate_rbpf_trajectory,
    save_params,
    encode_EM_params,
    decode_EM_params,
)
from rbpf.src.utils import RBPFState, RBPFFootballResults, EMParams, RawEMParams, FootballResults
from rbpf.src.bivariate_poisson import loglik
from rbpf.src.graphic import plot_all, plot_log_marginal_likelihood_curve, plot_all_smoothing

jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

def _stable_cholesky(matrix: jnp.ndarray, jitter: float = 1e-6) -> jnp.ndarray:
    """Cholesky robust to tiny negative eigenvalues from float32 roundoff.

    The covariance matrices from compute_gamma_trajectory are structurally PSD
    (unobserved block is PD, observed block is exactly 0), but float32 noise
    can leave the zero eigenvalues at ~-5e-10, which makes
    jnp.linalg.cholesky silently return NaN. A small diagonal ridge scaled to
    the matrix norm dominates this noise without affecting the signal.
    """
    matrix = 0.5 * (matrix + matrix.T)
    diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    scale = jnp.maximum(jnp.max(diag, axis=-1, keepdims=True), 1.0)
    n = matrix.shape[-1]
    # Broadcast jitter to the matrix's batch shape.
    eye = jnp.eye(n, dtype=matrix.dtype)
    if matrix.ndim > 2:
        jitter_diag = (jitter * scale)[..., None] * eye
    else:
        jitter_diag = (jitter * scale) * eye
    return jnp.linalg.cholesky(matrix + jitter_diag)


def gaussian_kron_logpdf(x, mean, gamma, B):
    M, K = gamma.shape[0], B.shape[0]
    residual = x - mean                          # (..., M, 2)

    L_gamma = _stable_cholesky(gamma)         # (M, M)
    L_B = _stable_cholesky(B)                 # (2, 2)

    # log|Gamma (x) B| = K log|Gamma| + M log|B|
    log_det = (
        K * 2.0 * jnp.sum(jnp.log(jnp.diag(L_gamma)))
        + M * 2.0 * jnp.sum(jnp.log(jnp.diag(L_B)))
    )

    # Whiten: solve L_gamma Z = R over the M axis, then L_B W = Z^T over the K axis.
    # quad = tr(B^{-1} R^T Gamma^{-1} R) = sum(W^2)
    Z = jax.scipy.linalg.solve_triangular(L_gamma, residual, lower=True)  # (..., M, 2)
    Z_T = jnp.swapaxes(Z, -1, -2)                # (..., 2, M)
    W = jax.scipy.linalg.solve_triangular(L_B, Z_T, lower=True)            # (..., 2, M)
    quad = jnp.sum(W * W, axis=(-1, -2))        # (...,)

    return -0.5 * (M * K * jnp.log(2.0 * jnp.pi) + log_det + quad)


def sample_multivariate_normal_kron(key, mean, gamma, B):
    L_gamma = _stable_cholesky(gamma)         # (M, M)
    L_B = _stable_cholesky(B)                 # (2, 2)
    noise = jax.random.normal(key, mean.shape)   # (M, 2)
    # mean + L_gamma @ noise @ L_B^T
    return mean + L_gamma @ noise @ L_B.T

def rbpf_backward_sampling_fn(
    key: jax.Array,
    filtered_states: RBPFState,
    params: EMParams,
    model_inputs_rbpf: RBPFFootballResults,
) -> RBPFState:
    # Returns X^*_{0:T} = (X_0^*, X_1^*, ..., X_T^*) given the filtered states and the model inputs
    # 1. Sample X_T^* from the filtered states at terminal time T
    # 2. For t = T-1, ..., 0:
    #   1. Compute backward kernal using w_t^{(i)} and mu_t^{(i)}
    #   2. sample index I_t \sim w_{t \mid t+1}^}{i}
    #   3. Draw X_t^* \sim N(\mu_{t \mid t+1}^{(I_t)}, \Sigma_{t \mid t+1})
    key, sample_key = jax.random.split(key)
    T = filtered_states.particles.x.shape[0] - 1

    gamma_filtered = jnp.concatenate([params.gamma_0[None], model_inputs_rbpf.gamma])  # (T+1, M, M)
    dt = model_inputs_rbpf.timestamp - model_inputs_rbpf.timestamp_prev               # (T,)
    phi = jnp.exp(-params.kappa * dt)                                                  # (T,)

    # ---- 1. Terminal: sample X_T^* ----
    I_T = jax.random.categorical(key, filtered_states.log_weights[-1])     # scalar index
    mu_T = filtered_states.particles.x[-1, I_T]
    # Covariance is shared per timestep (deterministic), so use the full matrix,
    # NOT indexed by the particle I_T.
    X_T_star = sample_multivariate_normal_kron(key=sample_key, mean=mu_T, gamma=gamma_filtered[-1], B=params.B)  # (M, 2)

    # ---- 2. Backward scan over t = T-1, ..., 0 ----
    # Per-step keys so each scan iteration draws fresh randomness.
    step_keys = jax.random.split(key, T)          # for categorical index draws
    sample_keys = jax.random.split(sample_key, T) # for state draws

    xs = (
        filtered_states.particles.x[:-1],   # (T, N, M, 2)  mu_t
        filtered_states.log_weights[:-1],   # (T, N)         w_t
        gamma_filtered[:-1],                # (T, M, M)      Sigma_t
        model_inputs_rbpf.gamma_pred,       # (T, M, M)      Sigma_{t+1|t}
        phi,                                # (T,)           phi_t
        step_keys,                          # (T, 2)         index keys
        sample_keys,                        # (T, 2)         draw keys
    )

    def backward_sampling_fn(carry, xs_t):
        X_t1_star = carry                     # (M, 2)  the t+1 particle, brought backward
        mu_t, log_w_t, Sigma_t, Sigma_t1_pred, phi_t, idx_key, draw_key = xs_t

        # OU prediction of the mean at t+1 given filter state at t
        mu_pred_next = params.mean_0 + phi_t * (mu_t - params.mean_0)      # (N, M, 2)

        # Backward kernel weights: w_{t|t+1}^(i) ∝ w_t^(i) * N(X_{t+1}^* | mu_pred_next^(i), Sigma_{t+1|t})
        log_transition = gaussian_kron_logpdf(
            X_t1_star[None], mu_pred_next, Sigma_t1_pred, params.B
        )                                                                  # (N,)
        log_backward_weights = log_w_t + log_transition                    # (N,)

        # Sample the component index I_t
        I_t = jax.random.categorical(idx_key, log_backward_weights)

        # RTS smoother gain and shared conditional covariance
        J_t = (phi_t * Sigma_t) @ jnp.linalg.inv(Sigma_t1_pred)            # (M, M)
        Sigma_cond = Sigma_t - J_t @ Sigma_t1_pred @ J_t.T                 # (M, M)
        # RTS conditional mean for the chosen component
        mu_cond = mu_t[I_t] + J_t @ (X_t1_star - mu_pred_next[I_t])        # (M, 2)
        # Draw X_t^* from the shared conditional covariance
        X_t_star = sample_multivariate_normal_kron(
            key=draw_key, mean=mu_cond, gamma=Sigma_cond, B=params.B
        )                                                                  # (M, 2)

        return X_t_star, X_t_star

    # With reverse=True, jax.lax.scan processes xs[T-1], ..., xs[0] but returns
    # ys in FORWARD order: ys[i] corresponds to xs[i] (timestep i). So the output
    # is already indexed 0, 1, ..., T-1 — no extra reversal needed.
    _, X_star_0_to_Tm1 = jax.lax.scan(
        backward_sampling_fn,
        X_T_star,
        xs,
        reverse=True
    )
    # Append the terminal sample at t=T: [X_0, X_1, ..., X_{T-1}, X_T]
    X_star = jnp.concatenate([X_star_0_to_Tm1, X_T_star[None]], axis=0)  # (T+1, M, 2)

    return RBPFState(x=X_star)

def rbpf_backward_smoothing(
    key: jax.Array,
    n_smoothed_trajectories: int,
    filtered_states: RBPFState,
    params: EMParams,
    model_inputs_rbpf: RBPFFootballResults,
):
    # Returns N smoothed trajectories X^*_{0:T} = (X_0^*, X_1^*, ..., X_T^*) given the filtered states and the model inputs
    # 1. For n = 1, ..., N:
    #   1. Sample X_{0:T}^{*(n)} using rbpf_backward_sampling_fn
    # 2. Return the N smoothed trajectories
    keys = jax.random.split(key, n_smoothed_trajectories)
    smoothed_trajectories = jax.vmap(
        rbpf_backward_sampling_fn,
        in_axes=(0, None, None, None),
        out_axes=0
    )(keys, filtered_states, params, model_inputs_rbpf)  # (N, T+1, M, 2)

    return smoothed_trajectories

def E_step(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    n_smoothed_trajectories : int,
    max_goals: int
):
    filter_key, smoother_key = jax.random.split(key, 2)
    # 1. Run RBPF forward pass
    filtered_states, model_inputs_rbpf = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=n_particles,
        max_goals=max_goals
    )
    # 2. Perform backward smoothing via backward sampling fn. Skip implementation using cuthbert because we are exploring a trajectories from a sampled state instead of using previous particles. This creates the noise that will enable the EM algorithm to converge.
    # 3. return the smoothed trajectories for M-step
    smoothed_trajectories = rbpf_backward_smoothing(
        key=smoother_key,
        n_smoothed_trajectories=n_smoothed_trajectories,
        filtered_states=filtered_states,
        params=params,
        model_inputs_rbpf=model_inputs_rbpf
    )
    return smoothed_trajectories, filtered_states.log_normalizing_constant[-1]

def loss_fn(
        params: EMParams, 
        smoothed_trajectory: RBPFState, 
        model_inputs: FootballResults,
        max_goals: int
    ):
    # initial, transition, and observation
    def _loss_init(x_0):
            return gaussian_kron_logpdf(x_0, params.mean_0, params.gamma_0, params.B)

    # ---- transition: log p(X_t | X_{t-1}) via OU ----
    def _loss_transition(x_prev, x_next, t):
        dt = model_inputs.timestamp[t] - model_inputs.timestamp_prev[t]
        phi = jnp.exp(-params.kappa * dt)
        pred_mean = params.mean_0 + phi * (x_prev - params.mean_0)
        q_gamma = (1.0 - phi**2) * params.gamma_0
        return gaussian_kron_logpdf(x_next, pred_mean, q_gamma, params.B)

    # ---- observation: sum bivariate-Poisson loglik over matches on day t ----
    def _loss_observation(x_t, t):
        home_id = model_inputs.matches.home_id[t]
        away_id = model_inputs.matches.away_id[t]
        home_score = model_inputs.matches.home_score[t]
        away_score = model_inputs.matches.away_score[t]
        valid = model_inputs.match_mask[t]

        def match_body(total, match):
            h, a, yh, ya, v = match
            ll = loglik(
                jnp.array([yh, ya]), x_t[h], x_t[a],
                params.alpha, params.beta, max_goals, 1.0,
            )
            return total + jnp.where(v, ll, 0.0), None

        total, _ = jax.lax.scan(
            match_body, 0.0,
            (home_id, away_id, home_score, away_score, valid),
        )
        return total
    init = _loss_init(smoothed_trajectory.x[0])
    def _single_trajectory_loss(carry, t):
        # carry is the accumulated scalar loss; states come from the trajectory.
        trans = _loss_transition(
            x_prev=smoothed_trajectory.x[t - 1],
            x_next=smoothed_trajectory.x[t],
            t=t,
        )
        obs = _loss_observation(
            x_t=smoothed_trajectory.x[t],
            t=t,
        )
        return carry + trans + obs, None
    total, _ = jax.lax.scan(
        _single_trajectory_loss,
        init,
        jnp.arange(1, smoothed_trajectory.x.shape[0]),
    )
    return -total  # negative log-likelihood for minimization

def run_EM(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    n_smoothed_trajectories: int,
    num_epochs: int,
    learning_rate: float,
    max_goals: int,
):
    # 1. Run the E-step to get N smoothed trajectories
    # 2. Average the gradients of the loss function w.r.t. the parameters for all N smoothed trajectories
    # 3. Update the parameters using the averaged gradients
    # 4. Repeat until convergence
    # 5. Return the optimized parameters

    # Optimize in UNCONSTRAINED space (RawEMParams). decode_EM_params always
    # produces a PSD gamma_0, positive kappa, and diagonal PSD B, which keeps
    # the forward filter numerically stable (no NaN from non-PSD covariances).
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(raw_params)

    # Track decoded params (EMParams) per epoch.
    params_history = jax.tree_util.tree_map(
        lambda x: x[None], decode_EM_params(raw_params, fixed_mean_0)
    )

    print(f"[run_EM] Starting EM: {num_epochs} epochs, lr={learning_rate}, "
          f"N_particles={n_particles}, N_trajectories={n_smoothed_trajectories}")

    log_marginal_likelihood_history = []
    for epoch in range(num_epochs):
        # Decode the current unconstrained params into identified EMParams.
        params = decode_EM_params(raw_params, fixed_mean_0)

        # E-step
        print(f"  [Epoch {epoch+1}/{num_epochs}] Running E-step (filter + backward sampling)...")
        smoothed_trajectories, log_marginal_likelihood = E_step(
            key=key,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            n_smoothed_trajectories=n_smoothed_trajectories,
            max_goals=max_goals
        )
        print(f"  [Epoch {epoch+1}/{num_epochs}] E-step done. log marginal = {log_marginal_likelihood:.4f}")

        # M-step
        # 1. Compute gradients w.r.t. the RAW (unconstrained) params.
        #    loss_fn takes decoded EMParams, so we differentiate through decode.
        def _raw_loss(raw, traj):
            decoded = decode_EM_params(raw, fixed_mean_0)
            return loss_fn(decoded, traj, model_inputs, max_goals)

        print(f"  [Epoch {epoch+1}/{num_epochs}] Running M-step (gradient + Adam update)...")
        loss_grads = jax.vmap(lambda traj: jax.grad(_raw_loss)(raw_params, traj))(smoothed_trajectories)
        # negative mean so that we minimize the negative log-likelihood
        avg_grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), loss_grads)

        # 2. Update the raw parameters using the averaged gradients
        updates, opt_state = optimizer.update(avg_grad, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)

        # Track the log marginal likelihood
        log_marginal_likelihood_history.append(log_marginal_likelihood)
        print(f"  [Epoch {epoch+1}/{num_epochs}] M-step done.")

        params = decode_EM_params(raw_params, fixed_mean_0)
        params_history = jax.tree_util.tree_map(
            lambda track, new: jnp.concatenate([track, new[None]], axis=0), params_history, params
        )

    print(f"[run_EM] EM complete. Final log marginal = {log_marginal_likelihood_history[-1]:.4f}")
    # current best params, historical parameters and log marginal likelihood history
    return params, params_history, log_marginal_likelihood_history

def main():
    ############################# MODEL TRAINING PIPELINE #############################
    start_date = "2000-01-01"  # date of first game
    end_date = "2025-12-31"
    teams_only = WORLDCUP_2026_TEAMS  # only include teams that qualified for the 2026 World Cup
    MAX_GOALS = 8
    N_particles = 1000
    N_smoothed_trajectories = 1000
    learning_rate = 1e-4
    epochs = 3
    key = jax.random.PRNGKey(0)
    ############################################
    # 1. Load the football results data
    df, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=teams_only
    )
    print(f"Loaded football results data from {start_date} to {end_date}. Number of unique dates: {len(df['date'].unique())}. Number of unique teams: {len(team_id_to_name)}.")

    # 2. run EM algorithm to optimize the parameters
    print("[main] Running EM...")
    latest_params, params_history, log_marginal_likelihood_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=default_init_params(len(team_id_to_name)),
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        num_epochs=epochs,
        learning_rate=learning_rate,
        max_goals=MAX_GOALS
    )
    print("[main] EM finished.")

    # 3. Save the optimized parameters to a file
    save_path = "./rbpf/outputs/smoothing/"
    os.makedirs(save_path, exist_ok=True)
    save_params(latest_params, save_path + "optimized_params.json")
    print("[main] Saved optimized params.")

    # Save the run configuration
    run_config = {
        "start_date": start_date,
        "end_date": end_date,
        "n_particles": N_particles,
        "n_smoother_paths": N_smoothed_trajectories,
        "n_epochs": epochs,
        "learning_rate": learning_rate,
        "max_goals": MAX_GOALS,
        "seed": 0,
        "m_step": "adam",
        "output_dir": save_path,
    }
    with open(save_path + "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print("[main] Saved run config.")

    # 4. Run RBPF filter with the optimized parameters to get the final state estimates
    print("[main] Running final filter with optimized params...")
    filtered_states, model_inputs_rbpf = run_filter(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=N_particles,
        max_goals=MAX_GOALS
    )
    print(f"[main] Final filter done. Final log marginal = {filtered_states.log_normalizing_constant[-1]:.4f}")

    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=save_path + "/filter"
    )
    print("[main] Saved filter plots.")

    # 5. Plot the log marginal likelihood curve over epochs
    log_marginal_likelihood_history.append(filtered_states.log_normalizing_constant[-1])  # append final log marginal likelihood
    plot_log_marginal_likelihood_curve(
        log_marginal_likelihoods=log_marginal_likelihood_history,
        save_path=save_path + "/em_log_marginal_likelihood_curve.png"
    )
    print("[main] Saved EM log marginal likelihood curve.")

    # 6. Run a final smoothing pass with the optimized params and plot
    print("[main] Running final smoothing pass...")
    final_smoothed, _ = E_step(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        max_goals=MAX_GOALS,
    )
    plot_all_smoothing(
        smoothed_trajectories=final_smoothed.x,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=10,
        save_path=save_path + "/smoothing",
    )
    print("[main] Saved smoothing plots.")
if __name__ == "__main__":
    main()