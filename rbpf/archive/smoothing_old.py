import json
import jax
import jax.numpy as jnp
import numpy as np

from cuthbert.smc.backward_sampler import build_smoother
from cuthbertlib.smc.smoothing import exact_sampling
import cuthbertlib

from rbpf.src.model import run_filter, compute_gamma_trajectory
from rbpf.utils import RBPFState, RBPFFootballResults, FootballResults, EMParams
from rbpf.src.bivariate_poisson import loglik
from rbpf.src.graphic import plot_log_likelihood_history

jax.config.update("jax_platforms", "cpu")

N = 10000
MAX_GOALS = 8


def _sample_psd_gaussian(
    key: jax.Array,
    mean: jax.Array,
    covariance: jax.Array,
) -> jax.Array:
    """Sample from a PSD Gaussian, preserving exact zero-variance directions."""
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    eigvals = jnp.clip(eigvals, 0.0)
    noise = jax.random.normal(key, mean.shape)
    return mean + eigvecs @ (jnp.sqrt(eigvals) * noise)


def _log_psd_gaussian(deltas: jax.Array, covariance: jax.Array) -> jax.Array:
    """Evaluate a Gaussian log density on the support of a PSD covariance.

    Predictive covariances can be singular when consecutive observations have
    the same timestamp.  This uses the pseudo-determinant and pseudo-inverse
    without adding diagonal jitter; points outside the covariance support get
    log density -inf.
    """
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(eigvals)))
    rank_tol = 1e-7 * scale
    positive = eigvals > rank_tol

    projected = deltas @ eigvecs
    inv_eigvals = jnp.where(positive, 1.0 / eigvals, 0.0)
    quad = jnp.sum(projected**2 * inv_eigvals[None, :], axis=-1)
    null_sq = jnp.sum(
        jnp.where(positive[None, :], 0.0, projected**2), axis=-1
    )

    rank = jnp.sum(positive)
    log_pdet = jnp.sum(
        jnp.where(positive, jnp.log(jnp.maximum(eigvals, rank_tol)), 0.0)
    )
    log_norm = rank * jnp.log(2.0 * jnp.pi) + log_pdet
    log_density = -0.5 * (quad + log_norm)

    support_tol = (1e-5 * scale) ** 2
    return jnp.where(null_sq <= support_tol, log_density, -jnp.inf)

# ---------------------------------------------------------------------------
# E-step: Forward filter + Backward sampling
# ---------------------------------------------------------------------------

# def _joint_log_potential(
#     state_prev: RBPFState,
#     state_curr: RBPFState,
#     model_inputs: RBPFFootballResults,
#     alpha: float,
#     beta: float,
#     scale: float,
#     max_goals: int,
#     init_mean: jax.Array,
#     init_B: jax.Array,
#     init_kappa: float,
#     init_gamma: jax.Array,
#     num_teams: int,
# ) -> jax.Array:
#     """Joint log potential for the backward sampler.

#     Computes: log p(y_t | x_t) + log p(x_t | x_{t-1})

#     $\log G_t(x_{t-1}, x_t) + \log M_t(x_t \mid x_{t-1})$

#     The observation likelihood is the bivariate Poisson.
#     The transition density is the OU process with Kronecker covariance.
#     """
#     # log likelihood weights w_t
#     y = jnp.array([model_inputs.home_score, model_inputs.away_score])
#     x_i = state_curr.x[model_inputs.home_team_id]
#     x_j = state_curr.x[model_inputs.away_team_id]
#     log_obs = loglik(y, x_i, x_j, alpha=alpha, beta=beta, max_goals=max_goals, scale=scale)

#     # log transition density weights, log p(x_{t+1}^* | μ_{t+1|t}, Σ_{t+1|t})
#     dt = model_inputs.timestamp - model_inputs.timestamp_prev
#     phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
#     # \mu_{t+1 \mid t}
#     pred_mean = init_mean + phi_t @ (state_prev.x - init_mean)

#     # Σ_{t+1 \mid t} = Γ_{t+1 \mid t} ⊗ B
#     pred_sigma = jnp.kron(model_inputs.gamma_pred_t, init_B)
#     pred_sigma_reg = pred_sigma + 1e-6 * jnp.eye(pred_sigma.shape[0])
#     log_det = jnp.log(jnp.linalg.det(pred_sigma_reg))

#     delta = (state_curr.x - pred_mean).flatten()
#     # Quadratic form: (x - μ)^T Σ^{-1} (x - μ)
#     quad_form = delta @ jnp.linalg.solve(pred_sigma_reg, delta)
#     dim = state_curr.x.size
#     # log transition = -0.5 * (quad_form + log_det + dim * log(2π)) since x_t ~ N(μ_{t+1|t}, Σ_{t+1|t})
#     log_trans = -0.5 * (quad_form + log_det + dim * jnp.log(2 * jnp.pi))

#     return log_obs + log_trans

# def _backward_log_potential(
#     t: int,
#     dt: jax.Array,
#     X_next_star: jax.Array,
#     filtered_states: cuthbertlib.types.ArrayTree,
#     model_inputs: RBPFFootballResults,
#     init_mean: jax.Array,
#     init_B: jax.Array,
#     init_kappa: float,
#     num_teams: int,
# ):
#     phi = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
#     pred_means = init_mean + phi @ (filtered_states.particles.x[t] - init_mean)  # (N, M, 2)
#     pred_sigma = jnp.kron(model_inputs.gamma_pred_t[t+1], init_B)
#     pred_sigma_reg = pred_sigma + 1e-6 * jnp.eye(pred_sigma.shape[0])

#     delta = (X_next_star - pred_means).flatten()  # (N, M, 2)
#     quad = delta @ jnp.linalg.solve(pred_sigma_reg, delta)
#     dim = filtered_states.particles.x[t].size
#     log_trans = -0.5 * (quad + jnp.log(jnp.linalg.det(pred_sigma_reg)) + dim * jnp.log(2 * jnp.pi))
#     return log_trans

def smoother_rts(
    filtered_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    init_mean: jax.Array,
    init_B: jax.Array,
    init_kappa: float,
    num_teams: int,
    key: jax.Array,
):
    """
    Custom RTS backward sampler — ALGORITHM.md §4.1.

    Uses jax.lax.scan with reverse=True for the backward pass.
    Returns smoothed trajectory of shape (T, M, K).
    """
    K = 2
    N = filtered_states.particles.x.shape[1]
    T = filtered_states.particles.x.shape[0]
    dim = num_teams * K

    # --- Step 1: Sample terminal state X_T^* ---
    key, cat_key, sample_key = jax.random.split(key, 3)
    log_w_T = filtered_states.log_weights[-1]  # (N,)
    I_T = jax.random.categorical(cat_key, log_w_T)  # scalar

    # Σ_T = Γ_T ⊗ B.  Do not add diagonal jitter: zero rows in Γ_T encode
    # observed particle coordinates whose conditional variance is exactly 0.
    gamma_T = model_inputs.gamma_t[-1]  # (M, M)
    sigma_T = jnp.kron(gamma_T, init_B)  # (MK, MK)
    mu_T = filtered_states.particles.x[-1, I_T]  # (M, K)
    X_T_star = _sample_psd_gaussian(
        sample_key, mu_T.flatten(), sigma_T
    ).reshape(num_teams, K)

    # --- Step 2: Backward recursion via jax.lax.scan(reverse=True) ---
    # carry = (X_next_star, step_key) — pass both state and PRNG key through scan

    xs_particles = filtered_states.particles.x[:-1]       # (T-1, N, M, K)
    xs_log_weights = filtered_states.log_weights[:-1]      # (T-1, N)
    xs_gamma_t = model_inputs.gamma_t[:-1]                 # (T-1, M, M)
    xs_gamma_pred_t1 = model_inputs.gamma_pred_t[1:]       # (T-1, M, M)
    xs_ts_t1 = model_inputs.timestamp[1:]                  # (T-1,)
    xs_ts_t = model_inputs.timestamp[:-1]                  # (T-1,)

    def backward_step(carry, xs):
        """Single backward step. carry = (X_{t+1}^*, key), output = X_t^*"""
        X_next_star, key = carry
        particles_t, log_w_t, gamma_t, gamma_pred_t1, ts_t1, ts_t = xs

        # --- 2.5: Backward weights ---
        dt = ts_t1 - ts_t
        phi = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
        pred_means = init_mean + phi @ (particles_t - init_mean)  # (N, M, K)

        pred_sigma = jnp.kron(gamma_pred_t1, init_B)

        deltas = (X_next_star[None, :, :] - pred_means).reshape(N, dim)
        log_trans = _log_psd_gaussian(deltas, pred_sigma)
        log_bw = log_w_t + log_trans

        key, cat_key, sample_key = jax.random.split(key, 3)
        I_t = jax.random.categorical(cat_key, log_bw)

        # --- 2.6: RTS gain ---
        J_gamma = gamma_t @ phi.T @ jnp.linalg.pinv(gamma_pred_t1)
        J = jnp.kron(J_gamma, jnp.eye(K))

        diff = (X_next_star - pred_means[I_t]).flatten()
        mu_cond = particles_t[I_t] + (J @ diff).reshape(num_teams, K)

        gamma_cond = gamma_t - J_gamma @ gamma_pred_t1 @ J_gamma.T
        gamma_cond = 0.5 * (gamma_cond + gamma_cond.T)
        sigma_cond = jnp.kron(gamma_cond, init_B)

        # --- 2.7: Sample from the exact PSD conditional covariance ---
        X_t_star = _sample_psd_gaussian(
            sample_key, mu_cond.flatten(), sigma_cond
        ).reshape(num_teams, K)

        return (X_t_star, key), X_t_star

    _, smoothed_rest = jax.lax.scan(
        f=backward_step,
        init=(X_T_star, key),
        xs=(xs_particles, xs_log_weights, xs_gamma_t,
            xs_gamma_pred_t1, xs_ts_t1, xs_ts_t),
        reverse=True,
    )

    # reverse=True returns outputs in chronological order:
    # [X_0^*, ..., X_{T-2}^*, X_{T-1}^*].
    smoothed_x = jnp.concatenate([smoothed_rest, X_T_star[None]], axis=0)  # (T, M, K)
    return smoothed_x

def E_step(
    params: EMParams,
    model_inputs: RBPFFootballResults,
    key: jax.Array,
    num_teams: int,
) -> tuple:
    """E-step: Forward filter (jitted) + Backward sampling (jitted).

    Returns:
        (filtered_states, smoothed_states, log_marginal_likelihood)
    """
    # 1. Forward filter — returns filtered_states + augmented_results (with gamma trajectories)
    key, filter_key = jax.random.split(key)
    filtered_states, augmented_results = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        gamma=params.gamma_0,
        kappa=params.kappa,
        num_teams=num_teams,
        n=N,
        mean=params.mean_0,
        B=params.B,
        alpha=params.alpha,
        beta=params.beta,
        friendly_scale=params.friendly_scale,
    )

    # 2. The filter history contains one bookkeeping prior state followed by
    # one state per observation.  RTS should smooth the posterior states, so
    # remove only that initial prior while retaining observation 0.
    smoother_filtered_states = filtered_states._replace(
        particles=filtered_states.particles._replace(
            x=filtered_states.particles.x[1:]
        ),
        log_weights=filtered_states.log_weights[1:],
    )
    smoother_inputs = jax.tree.map(lambda x: x[1:], augmented_results)

    # 3. RTS backward sampler — uses aligned gamma_t and gamma_pred_t.
    key, smoother_key = jax.random.split(key)
    smoothed_states = smoother_rts(
        filtered_states=smoother_filtered_states,
        model_inputs=smoother_inputs,
        init_mean=params.mean_0,
        init_B=params.B,
        init_kappa=params.kappa,
        num_teams=num_teams,
        key=smoother_key,
    )
    log_marginal = float(filtered_states.log_normalizing_constant[-1])

    return filtered_states, smoothed_states, log_marginal

# ---------------------------------------------------------------------------
# M-step: Update parameters from smoothed trajectories
# ---------------------------------------------------------------------------

def M_step(
    smoothed_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    num_teams: int,
    prev_params: EMParams,
    n_mstep_iterations: int = 100,
    learning_rate: float = 0.05,
    patience: int = 20,
) -> EMParams:
    """
    Gradient based M-step to update parameters from smoothed trajectories.

    Optimizes: κ, Γ₀ (via Cholesky), B (via Cholesky, trace-normalized),
               α, β, scale (via log).
    Fixed: μ₀

    Returns updated EMParams.
    """
    import optax

    K = 2
    dim = num_teams * K
    mu_0 = prev_params.mean_0  # FIXED

    # --- Extract initial values from prev_params ---
    gamma_0_prev = prev_params.gamma_0
    B_prev = prev_params.B

    # Cholesky of Γ₀: L_gamma such that Γ₀ = L_gamma @ L_gamma.T
    L_gamma_prev = jnp.linalg.cholesky(gamma_0_prev + 1e-6 * jnp.eye(num_teams))
    # Convert to unconstrained: diagonal -> log, off-diagonal -> as-is
    L_gamma_init = L_gamma_prev.at[jnp.diag_indices(num_teams)].set(
        jnp.log(jnp.diag(L_gamma_prev))
    )

    # Cholesky of B: L_B such that B = L_B @ L_B.T
    L_B_prev = jnp.linalg.cholesky(B_prev + 1e-6 * jnp.eye(K))
    L_B_init = L_B_prev.at[jnp.diag_indices(K)].set(
        jnp.log(jnp.diag(L_B_prev))
    )

    # Log-space for positivity-constrained scalars
    log_kappa_init = jnp.log(prev_params.kappa)
    log_scale_init = jnp.log(prev_params.friendly_scale)

    # --- Smoothed states as constants (not differentiated) ---
    all_x = jnp.array(smoothed_states)  # (T, M, K) — treat as fixed data

    # --- Data arrays ---
    timestamps = model_inputs.timestamp
    timestamps_prev = model_inputs.timestamp_prev
    home_ids = model_inputs.home_team_id
    away_ids = model_inputs.away_team_id
    home_scores = model_inputs.home_score
    away_scores = model_inputs.away_score

    n_observations = timestamps.shape[0]
    if all_x.shape[0] != n_observations:
        raise ValueError(
            "Expected one smoothed state per observation"
        )
    observation_indices = jnp.arange(n_observations)
    transition_indices = jnp.arange(1, n_observations)

    # --- Precompute Sigma_0 = Γ₀ ⊗ B for transition loss ---
    def loss_fn(L_gamma_raw, L_B_raw, log_kappa, alpha, beta, log_scale):
        kappa = jnp.exp(log_kappa)
        scale = jnp.exp(log_scale)

        # Reconstruct Γ₀ (SPD)
        L_gamma = L_gamma_raw.at[jnp.diag_indices(num_teams)].set(
            jnp.clip(jnp.exp(jnp.diag(L_gamma_raw)), 1e-4, 1e4)
        )
        gamma_0 = L_gamma @ L_gamma.T  # (M, M)

        # Reconstruct B (SPD, trace = K)
        L_B = L_B_raw.at[jnp.diag_indices(K)].set(
            jnp.clip(jnp.exp(jnp.diag(L_B_raw)), 1e-4, 1e4)
        )
        B_raw = L_B @ L_B.T  # (K, K)
        B = B_raw * K / jnp.trace(B_raw)  # normalize trace(B) = K

        # --- Transition loss ---
        # Q_t = (Γ₀ - φ_t Γ₀ φ_t) ⊗ B, where φ_t = exp(-κ dt) * I_M
        # Transition o goes from all_x[o - 1] to all_x[o], using observation
        # o's elapsed time.  Observation 0 has no preceding transition in
        # the smoothed posterior-state trajectory.
        dts = timestamps[transition_indices] - timestamps_prev[transition_indices]
        phis = jnp.exp(-kappa * dts)  # (T-1,)
        # Clip phi to avoid dt=0 singularity (phi=1 → Q_t=0)
        phis = jnp.clip(phis, None, 0.9999)

        Sigma_0 = jnp.kron(gamma_0, B)  # (MK, MK)
        Sigma_0_reg = Sigma_0 + 1e-4 * jnp.eye(dim)
        Sigma_0_inv = jnp.linalg.inv(Sigma_0_reg)
        sign, log_det_Sigma_0 = jnp.linalg.slogdet(Sigma_0_reg)

        def trans_step(obs_idx, phi_val):
            pred_mean = mu_0 + phi_val * (all_x[obs_idx - 1] - mu_0)  # (M, K)
            diff = (all_x[obs_idx] - pred_mean).flatten()  # (MK,)
            # Q_t = (1 - φ²) * Σ₀ since φ is scalar * I_M
            denom = 1.0 - phi_val ** 2
            quad = diff @ Sigma_0_inv @ diff / denom
            log_det_Q = jnp.log(denom) * dim + log_det_Sigma_0
            return -0.5 * (quad + log_det_Q + dim * jnp.log(2 * jnp.pi))

        trans_losses = jax.vmap(trans_step)(transition_indices, phis)
        trans_loss = -jnp.sum(trans_losses)

        # --- Observation loss ---
        def obs_step(obs_idx):
            h_id = home_ids[obs_idx]
            a_id = away_ids[obs_idx]
            y = jnp.array([home_scores[obs_idx], away_scores[obs_idx]])
            x_i = all_x[obs_idx, h_id]
            x_j = all_x[obs_idx, a_id]
            return loglik(y, x_i, x_j, alpha=alpha, beta=beta,
                          max_goals=MAX_GOALS, scale=scale)

        obs_losses = jax.vmap(obs_step)(observation_indices)
        obs_loss = -jnp.sum(obs_losses)

        return trans_loss + obs_loss

    # --- JIT-compiled value and gradient ---
    value_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    )

    # --- Initialize optimizer ---
    optimizer = optax.adam(learning_rate)
    params = (L_gamma_init, L_B_init, log_kappa_init,
              prev_params.alpha, prev_params.beta, log_scale_init)
    opt_state = optimizer.init(params)

    # --- Optimization loop ---
    best_loss = float('inf')
    best_params = params
    no_improve = 0

    for i in range(n_mstep_iterations):
        loss, grads = value_and_grad_fn(*params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_val = float(loss)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params
            no_improve = 0
        else:
            no_improve += 1

        if (i + 1) % 10 == 0:
            L_g, L_b, lk, a, b, ls = params
            print(f"    M-step iter {i+1}/{n_mstep_iterations}: "
                  f"loss={loss_val:.2f}, κ={float(jnp.exp(lk)):.4f}, "
                  f"α={float(a):.4f}, β={float(b):.4f}, "
                  f"scale={float(jnp.exp(ls)):.4f}")

        if no_improve >= patience:
            print(f"    Early stopping at M-step iteration {i+1}")
            break

    # --- Recover best parameters ---
    L_gamma_best, L_B_best, log_kappa_best, alpha_best, beta_best, log_scale_best = best_params

    # Reconstruct Γ₀
    L_gamma_final = L_gamma_best.at[jnp.diag_indices(num_teams)].set(
        jnp.exp(jnp.diag(L_gamma_best))
    )
    gamma_0_new = L_gamma_final @ L_gamma_final.T

    # Reconstruct B (trace-normalized)
    L_B_final = L_B_best.at[jnp.diag_indices(K)].set(
        jnp.exp(jnp.diag(L_B_best))
    )
    B_raw_final = L_B_final @ L_B_final.T
    B_new = B_raw_final * K / jnp.trace(B_raw_final)

    return EMParams(
        mean_0=mu_0,                    # FIXED
        gamma_0=gamma_0_new,            # estimated
        B=B_new,                        # estimated (trace = K)
        kappa=float(jnp.exp(log_kappa_best)),
        alpha=float(alpha_best),
        beta=float(beta_best),
        friendly_scale=float(jnp.exp(log_scale_best)),
    )


def run_EM(
    model_inputs: RBPFFootballResults,
    init_params: EMParams,
    num_teams: int,
    n_epochs: int = 5,
    n_mstep_iterations: int = 50,
    mstep_lr: float = 0.05,
) -> tuple[EMParams, list]:
    """
    Run EM for n_epochs.
    
    - n_epochs: number of EM iterations
    - n_mstep_iterations: number of gradient descent steps in M-step
    - mstep_lr: learning rate for M-step gradient descent
    """
    key = jax.random.PRNGKey(42)
    params = init_params
    log_marginal_history = []

    for epoch in range(n_epochs):
        print(f"\n{'='*60}")
        print(f"EM Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*60}")

        # E-step
        print("  E-step: filtering + backward sampling...")
        key, e_key = jax.random.split(key)
        filtered_states, smoothed_states, log_marginal = E_step(
            params, model_inputs, e_key, num_teams,
        )
        log_marginal_history.append(log_marginal)
        print(f"  Log marginal likelihood: {log_marginal:.4f}")

        # M-step
        print("  M-step: gradient descent...")
        params = M_step(
            smoothed_states, model_inputs, num_teams, params,
            n_mstep_iterations=n_mstep_iterations,
            learning_rate=mstep_lr,
        )
        print(f"  Updated kappa={params.kappa:.4f}, alpha={params.alpha:.4f}, "
              f"beta={params.beta:.4f}, scale={params.friendly_scale:.4f}")
        print(f"  (mean_0 fixed, gamma_0 and B estimated)")

    print(f"\n{'='*60}")
    print("EM COMPLETE")
    print(f"{'='*60}")
    print(f"Log marginal likelihood history: {log_marginal_history}")
    print(f"\nFinal parameters:")
    print(f"  kappa = {params.kappa:.4f}")
    print(f"  alpha = {params.alpha:.4f}")
    print(f"  beta = {params.beta:.4f}")
    print(f"  scale = {params.friendly_scale:.4f}")
    print(f"  B = {np.array(params.B)}")
    print(f"  gamma_0 shape = {params.gamma_0.shape}")
    print(f"  mean_0 shape = {params.mean_0.shape} [FIXED]")

    # Save optimal parameters
    import os
    from rbpf.src.helpers import save_params
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "smoothing_params.json")
    save_params(params, output_path)

    # Also save log marginal history
    history_path = os.path.join(os.path.dirname(__file__), "outputs", "em_log_marginal_history.json")
    with open(history_path, "w") as f:
        json.dump(log_marginal_history, f, indent=2)
    print(f"Saved log marginal history to {history_path}")

    plot_path = os.path.join(
        os.path.dirname(__file__), "outputs", "em_log_likelihood_history.png"
    )
    plot_log_likelihood_history(log_marginal_history, plot_path)
    print(f"Saved log-likelihood plot to {plot_path}")

    return params, log_marginal_history


def main():
    import os
    from rbpf.src.data import WORLDCUP_2026_TEAMS, get_results
    from rbpf.src.helpers import default_init_params

    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01", end_date="2025-12-31", max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, key)

    print(f"Teams: {NUM_TEAMS}, Matches: {len(model_inputs.timestamp)}")
    print(f"Initial: κ={params.kappa}, α={params.alpha}, β={params.beta}, scale={params.friendly_scale}")

    final_params, log_marginal_history = run_EM(
        model_inputs,
        params,
        NUM_TEAMS,
        n_epochs=15,
        n_mstep_iterations=50,
        mstep_lr=0.01,
    )

if __name__ == "__main__":
    main()
