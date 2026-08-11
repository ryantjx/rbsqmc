import jax
import jax.numpy as jnp
import numpy as np

from rbpf.src.utils import RBPFFootballResults, EMParams, FootballResults
from rbpf.src.data import WORLDCUP_2026_TEAMS, get_results, ACTIVE_TEAMS
from rbpf.src.helpers import default_init_params, generate_augmented_data, params_to_dict
from rbpf.src.model import (
    run_filter,
    compute_gamma_trajectory,
    _sample_psd_gaussian,
)
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
    # dim = num_teams * K  # total state dimension

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

        # pred_Sigma = kron(gamma_pred_t1, B) is PSD but can be singular /
        # slightly indefinite in float32 (gamma_pred_t1 has near-zero and tiny
        # negative eigenvalues for unobserved / same-day teams). A Cholesky-based
        # solve or raw log-det would produce inf/nan. Use slogdet for a stable
        # log-det and a small PSD clamp before the solve.
        gamma_pred_reg = gamma_pred_t1 + 1e-8 * jnp.eye(num_teams)
        pred_Sigma = jnp.kron(gamma_pred_reg, params.B)  # (2M, 2M)

        deltas = (X_next_star - pred_mean).reshape(n_particles, -1)  # (N, 2M)
        quad = jnp.sum(
            deltas * jnp.linalg.solve(pred_Sigma, deltas.T).T, axis=-1
        )  # (N,)
        sign, log_det = jnp.linalg.slogdet(pred_Sigma)  # stable log-det
        dim = K * K
        log_transition = (
            -0.5 * quad
            - 0.5 * log_det
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

        # Deterministic transition when dt = 0 (phi = I, Q = 0): the state is
        # preserved, so X_t^* = X_{t+1}^* with no sampling (consistent with the
        # zero-stochastic-cost handling in loss_fn's transition_step).
        deterministic = dt == 0.0
        X_t_sampled = _sample_psd_gaussian(
            sample_key, mu_RTS.flatten(), Sigma_RTS
        ).reshape(num_teams, K)  # (M, 2)
        X_t_star = jnp.where(deterministic, X_next_star, X_t_sampled)
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
    # dim = num_teams * K  # total state dimension
    dim = 2 * K

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
        # check if the phi is close to 1.0. If phi is close to 1.0, this means that there is no time difference.,
        scale = 1.0 - phi**2
        deterministic = scale <= 1e-8

        # Full Gaussian density for the stochastic (phi < 1) case.
        Q_gamma = jnp.maximum(scale, 1e-8) * params.gamma_0
        Q_gamma = 0.5 * (Q_gamma + Q_gamma.T)  # ensure symmetry
        Q_full = jnp.kron(Q_gamma, params.B)  # (2M, 2M)

        diff_flat = diff.reshape(-1)  # (2M,)
        quad = diff_flat @ jnp.linalg.solve(Q_full, diff_flat)
        # slogdet is numerically stable (raw det() underflows to 0 in float32).
        sign, log_det = jnp.linalg.slogdet(Q_full)
        log_density = -0.5 * quad - 0.5 * log_det - 0.5 * dim * jnp.log(2 * jnp.pi)

        return jnp.where(deterministic, 0.0, log_density)

    transition_losses = jax.vmap(transition_step)(transition_indices, phis)
    sum_transition_loss = -jnp.sum(transition_losses)

    return sum_observation_loss + sum_transition_loss

def _symmetrize(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise a square matrix (for PSD-constrained params like gamma_0, B)."""
    return 0.5 * (x + x.T)


# Minimum eigenvalue floor for the projected covariances gamma_0 and B.
#
# gamma_0/B are used to build transition covariances Q = scale * (gamma_0 ⊗ B)
# and smoother pred/Sigma matrices, which are inverted (solve) and whose
# log-determinants are taken. A pure PSD projection (clip eigvals to 0) would
# leave them singular, so Q could still be non-invertible and log_det -> -inf.
# Flooring the eigenvalues keeps them strictly PD (full-rank), guaranteeing Q
# is invertible and log_det stays finite. Empirically the M-step drives
# gamma_0 indefinite (min eig ~ -0.44), so this floor is load-bearing.
_EIGEN_FLOOR = 1e-4

# Lower bound for the OU mean-reversion rate kappa.
#
# kappa -> 0 makes phi = exp(-kappa*dt) -> 1, so Q = (1 - phi^2)*gamma_0 -> 0
# (a degenerate transition covariance). That sends log_det(Q) -> -inf and the
# transition loss explodes (observed as the 1e7-1e8 loss spikes each epoch).
# Flooring kappa away from zero keeps Q meaningfully non-singular. The default
# init is kappa=0.0039; a floor of 1e-3 keeps the model in the well-conditioned
# regime while still allowing slow mean reversion.
_KAPPA_MIN = 1e-3


def _project_psd(x: jnp.ndarray, floor: float = _EIGEN_FLOOR) -> jnp.ndarray:
    """Project a symmetric matrix onto the positive-definite cone.

    Eigen-decompose, clamp eigenvalues to be >= ``floor`` (> 0), and rebuild.
    Guarantees a full-rank, strictly PD matrix whose log-determinant and solve
    are well defined.
    """
    x = _symmetrize(x)
    eigvals, eigvecs = jnp.linalg.eigh(x)
    eigvals = jnp.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def _gamma0_from_cholesky(L: jnp.ndarray, num_teams: int) -> jnp.ndarray:
    """Build a PD gamma_0 = L L^T from an unconstrained M x M factor.

    ``L`` is a free matrix. We take its lower triangle, keep the diagonal
    positive via ``softplus``, zero the strictly-upper triangle, and form
    ``gamma_0 = L L^T``. Because L is full-rank lower-triangular with positive
    diagonal, gamma_0 is positive-definite by construction and the map from the
    free entries of L to gamma_0 is smooth and surjective onto the PD cone.
    """
    L_low = jnp.tril(L)
    diag = jax.nn.softplus(jnp.diag(L_low))  # > 0, strictly
    L_low = L_low.at[jnp.diag_indices(num_teams)].set(diag)
    return L_low @ L_low.T


def _cholesky_from_gamma0(gamma_0: jnp.ndarray, num_teams: int) -> jnp.ndarray:
    """Inverse map: a free M x M factor encoding the PD matrix gamma_0.

    ``L`` is a lower-triangular Cholesky factor of the PD ``gamma_0`` with a
    softplus-wrapped diagonal, padded to a full M x M free array (upper triangle
    is arbitrary/zero and ignored by ``_gamma0_from_cholesky``). This is the
    starting point for the M-step optimizer.
    """
    L = jnp.linalg.cholesky(gamma_0)  # lower-triangular, positive diagonal
    diag = L[jnp.diag_indices(num_teams)]
    L_free = jnp.zeros_like(gamma_0)
    L_free = L_free.at[jnp.tril_indices(num_teams)].set(
        L[jnp.tril_indices(num_teams)]
    )
    # invert softplus on the diagonal so reconstructing gamma_0 recovers it:
    # softplus(x) = diag  =>  x = log(exp(diag) - 1)
    L_free = L_free.at[jnp.diag_indices(num_teams)].set(
        jnp.log(jnp.exp(diag) - 1.0 + 1e-10)
    )
    return L_free


def _constrain(params: EMParams) -> EMParams:
    """Apply validity constraints so parameters stay in their support.

    - kappa >= _KAPPA_MIN (OU mean-reversion rate; keeps the transition
      covariance non-degenerate).
    - alpha, beta unconstrained real.
    - gamma_0, B projected onto the positive-definite cone (full-rank, so the
      transition covariance Q and the smoother covariances stay invertible and
      their log-determinants finite).

    Note: during the M-step gamma_0 is Cholesky-parameterized (so it stays PD
    automatically); this projection is retained as a safety net for params
    constructed outside the optimizer (e.g. hand-loaded values).
    """
    kappa = jnp.maximum(params.kappa, _KAPPA_MIN)
    gamma_0 = _project_psd(params.gamma_0)
    B = _project_psd(params.B)
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

    Returns:
        tuple[EMParams, float, float, list[float]]: (final_params, loss_start,
        loss_best, loss_trace). loss_start is the objective evaluated at
        ``prev_params`` (start of the M-step), loss_best is the best objective
        achieved during the loop, and loss_trace is the objective at every
        gradient step.
    """
    # --- JIT-compiled value and gradient (over the 5 learned fields) ---
    num_teams = prev_params.gamma_0.shape[0]

    def _loss_and_grad(carry):
        # Reconstruct a PD gamma_0 = L L^T from the free factor L.
        gamma_0 = _gamma0_from_cholesky(carry["L"], num_teams)
        return loss_fn(
            EMParams(
                mean_0=prev_params.mean_0,
                gamma_0=gamma_0,
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
    # gamma_0 is represented by a free Cholesky factor L (PD by construction);
    # this is what the optimizer moves. B/kappa/alpha/beta are as before.
    carry = {
        "L": _cholesky_from_gamma0(prev_params.gamma_0, num_teams),
        "B": prev_params.B,
        "kappa": prev_params.kappa,
        "alpha": prev_params.alpha,
        "beta": prev_params.beta,
    }

    # --- Per-parameter learning rates (scale-aware) ---
    # gamma_0 (via L) and B are O(1) matrices; kappa/alpha/beta are scalars on
    # different scales. alpha/beta dominate through the observation term, so
    # give them the base LR; gamma_0/B/kappa use smaller multipliers.
    base = learning_rate
    # transition loss dominates compared to observation loss. scale gamma_0/B/kappa down to avoid overshooting.
    lr_mapping = {
        "L": base * 0.03, # scale down for L (gamma_0) to avoid overshooting
        "B": base * 0.03, # scale down for B to avoid overshooting
        "kappa": base * 0.03, # scale down for kappa to avoid overshooting
        "alpha": base * 1.0,
        "beta": base * 1.0,
    }
    transforms = {
        "L": optax.adam(lr_mapping["L"]),
        "B": optax.adam(lr_mapping["B"]),
        "kappa": optax.adam(lr_mapping["kappa"]),
        "alpha": optax.adam(lr_mapping["alpha"]),
        "beta": optax.adam(lr_mapping["beta"]),
    }
    param_labels = {
        "L": "L",
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
    loss_start = None  # objective at prev_params (start of the M-step)
    loss_trace = []    # objective at every gradient step (for diagnostics)
    patience = max(10, n_gradient_steps // 5)
    no_improve_counter = 0

    for step in range(n_gradient_steps):
        loss, grads = value_and_grad_fn(carry)
        loss_val = float(loss)
        loss_trace.append(loss_val)

        # First evaluation is the objective at the starting params.
        if loss_start is None:
            loss_start = loss_val

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

    # Reconstruct gamma_0 from the best free factor L and project onto support.
    best_gamma_0 = _gamma0_from_cholesky(best_carry["L"], num_teams)
    final = _constrain(EMParams(
        mean_0=prev_params.mean_0,
        gamma_0=best_gamma_0,
        B=best_carry["B"],
        kappa=best_carry["kappa"],
        alpha=best_carry["alpha"],
        beta=best_carry["beta"],
    ))
    print(f"      M-step done: loss {best_loss:.4f} -> best at step {best_step}, "
          f"kappa={float(final.kappa):.5f} alpha={float(final.alpha):.5f} "
          f"beta={float(final.beta):.5f}")
    return final, loss_start, best_loss, loss_trace

def run_EM(
    model_inputs: FootballResults,
    init_params: EMParams,
    num_teams: int,
    n_particles: int = 10,
    n_epochs: int = 10,
    n_gradient_steps: int = 10,
    learning_rate: float = 1e-3,
    key: jax.Array = jax.random.PRNGKey(42)
) -> tuple[EMParams, jnp.ndarray, dict]:
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
    mstep_start_history = []
    mstep_end_history = []
    mstep_loss_traces = []  # full loss trajectory per epoch (every gradient step)

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
        params, loss_start, loss_best, loss_trace = M_step(
            smoothed_states=smoothed_states, 
            model_inputs=augmented_results, 
            prev_params=params, 
            learning_rate=learning_rate, n_gradient_steps=n_gradient_steps
        )
        # Track the M-step objective (loss = -log L) at the start, the best
        # point reached within this epoch, and the full per-step trajectory.
        mstep_start_history.append(loss_start)
        mstep_end_history.append(loss_best)
        mstep_loss_traces.append(loss_trace)

    print("EM completed. Final parameters:")
    print("  kappa:", params.kappa)
    print("  alpha:", params.alpha)
    print("  beta:", params.beta)
    print("  B:", params.B)
    print("  gamma_0:", params.gamma_0.shape)
    print("  mean_0:", params.mean_0.shape)

    # Return log-marginal (E-step) plus the M-step objective diagnostics,
    # including the full per-gradient-step loss trajectory for every epoch.
    em_diagnostics = {
        "mstep_loss_start": jnp.array(mstep_start_history),
        "mstep_loss_end": jnp.array(mstep_end_history),
        "mstep_loss_trace": mstep_loss_traces,  # list of per-epoch loss lists
    }
    return params, jnp.array(log_likelihood_history), em_diagnostics

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
        final_params, log_marginal_likelihoods, em_diagnostics = run_EM(
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