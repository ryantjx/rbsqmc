"""EM for RBPF with Kronecker structure — E-step (smoothing) and M-step (parameter update).

Implements the Forward Filter Backward Sampling (FFBSi) algorithm from PROOF_V5 §4
for parameter estimation via Expectation Maximization.

E-step: Forward filter → backward sample → smoothed trajectory samples X*_{0:T}
M-step: Maximize Q(θ; q*) w.r.t. θ using smoothed samples → update κ, α, β, B, Γ_0, μ_0

Output: JSON files of parameters at each EM epoch.
"""

import json
import os
from functools import partial
from typing import NamedTuple

from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import cuthbert
import cuthbertlib
from cuthbert.smc.backward_sampler import build_smoother
from cuthbertlib.smc.smoothing import exact_sampling

from bivariate_poisson import loglik
from data import download_results, FootballResults, read_results, WORLDCUP_2026_TEAMS
from model import (
    RBPFState, RBPFFootballResults,
    init_sample, propagate_sample, _log_potential,
    build_rbpf_filter, compute_gamma_trajectory, run_filter,
)

jax.config.update("jax_platforms", "cuda") # Script to run GPU

N = 10000
MAX_GOALS = 8


class EMParams(NamedTuple):
    """All parameters that EM optimizes."""
    init_mean: jax.Array      # (M, 2)
    init_gamma: jax.Array     # (M, M)
    init_B: jax.Array         # (2, 2)
    init_kappa: float
    init_alpha: float
    init_beta: float
    init_friendly_scale: float


def params_to_dict(params: EMParams) -> dict:
    """Convert EMParams to a JSON-serializable dict."""
    return {
        "init_mean": np.array(params.init_mean).tolist(),
        "init_gamma": np.array(params.init_gamma).tolist(),
        "init_B": np.array(params.init_B).tolist(),
        "init_kappa": float(params.init_kappa),
        "init_alpha": float(params.init_alpha),
        "init_beta": float(params.init_beta),
        "init_friendly_scale": float(params.init_friendly_scale),
    }


def params_from_dict(d: dict) -> EMParams:
    """Convert a JSON dict back to EMParams."""
    return EMParams(
        init_mean=jnp.array(d["init_mean"]),
        init_gamma=jnp.array(d["init_gamma"]),
        init_B=jnp.array(d["init_B"]),
        init_kappa=d["init_kappa"],
        init_alpha=d["init_alpha"],
        init_beta=d["init_beta"],
        init_friendly_scale=d["init_friendly_scale"],
    )


def save_params(params: EMParams, path: str):
    """Save parameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(params_to_dict(params), f, indent=2)
    print(f"Saved parameters to {path}")


def load_params(path: str) -> EMParams:
    """Load parameters from a JSON file."""
    with open(path, "r") as f:
        d = json.load(f)
    return params_from_dict(d)


# ---------------------------------------------------------------------------
# E-step: Forward filter + Backward sampling
# ---------------------------------------------------------------------------

def _joint_log_potential(
    state_prev: RBPFState,
    state_curr: RBPFState,
    model_inputs: RBPFFootballResults,
    alpha: float,
    beta: float,
    scale: float,
    max_goals: int,
    init_mean: jax.Array,
    init_B: jax.Array,
    init_kappa: float,
    init_gamma: jax.Array,
    num_teams: int,
) -> jax.Array:
    """Joint log potential for the backward sampler.

    Computes: log p(y_t | x_t) + log p(x_t | x_{t-1})

    The observation likelihood is the bivariate Poisson.
    The transition density is the OU process with Kronecker covariance.
    """
    # --- Observation likelihood ---
    y = jnp.array([model_inputs.home_score, model_inputs.away_score])
    x_i = state_curr.x[model_inputs.home_team_id]
    x_j = state_curr.x[model_inputs.away_team_id]
    log_obs = loglik(y, x_i, x_j, alpha=alpha, beta=beta, max_goals=max_goals, scale=scale)

    # --- Transition density: log N(x_t | μ + Φ(x_{t-1} - μ), Q_t) ---
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
    pred_mean = init_mean + phi_t @ (state_prev.x - init_mean)

    # Q_t = (Γ_0 - φ Γ_0 φ) ⊗ B
    Q_t_gamma = init_gamma - phi_t @ init_gamma @ phi_t.T
    Q_t = jnp.kron(Q_t_gamma, init_B)
    Q_t_reg = Q_t + 1e-6 * jnp.eye(Q_t.shape[0])

    diff = (state_curr.x - pred_mean).flatten()
    log_det_Q = jnp.log(jnp.linalg.det(Q_t_reg))
    quad_form = diff @ jnp.linalg.solve(Q_t_reg, diff)
    dim = state_curr.x.size
    log_trans = -0.5 * (quad_form + log_det_Q + dim * jnp.log(2 * jnp.pi))

    return log_obs + log_trans


def E_step(
    params: EMParams,
    results: FootballResults,
    key: jax.Array,
    num_teams: int,
) -> tuple:
    """E-step: Forward filter (jitted) + Backward sampling (jitted).

    Returns:
        (filtered_states, smoothed_states, log_marginal_likelihood)
    """
    # 1. Forward filter — uses the jitted run_filter from model.py
    key, filter_key = jax.random.split(key)
    filtered_states = run_filter(
        key=filter_key,
        results=results,
        init_gamma=params.init_gamma,
        init_kappa=params.init_kappa,
        num_teams=num_teams,
        n=N,
        init_mean=params.init_mean,
        init_B=params.init_B,
        init_alpha=params.init_alpha,
        init_beta=params.init_beta,
        init_friendly_scale=params.init_friendly_scale,
    )

    # 2. Compute gamma trajectory (needed for smoother's model_inputs)
    gamma_trajectory = compute_gamma_trajectory(
        results, params.init_gamma, params.init_kappa, num_teams,
    )

    # 3. Build and run the backward sampler (smoother)
    smoother_obj = build_smoother(
        log_potential=partial(
            _joint_log_potential,
            alpha=params.init_alpha,
            beta=params.init_beta,
            scale=params.init_friendly_scale,
            max_goals=MAX_GOALS,
            init_mean=params.init_mean,
            init_B=params.init_B,
            init_kappa=params.init_kappa,
            init_gamma=params.init_gamma,
            num_teams=num_teams,
        ),
        backward_sampling_fn=exact_sampling.simulate,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
        n_smoother_particles=N,
    )

    key, smoother_key = jax.random.split(key)
    smoothed_states = cuthbert.smoothing.smoother(
        smoother_obj, filtered_states, key=smoother_key,
    )

    log_marginal = float(filtered_states.log_normalizing_constant[-1])

    return filtered_states, smoothed_states, log_marginal


# ---------------------------------------------------------------------------
# M-step: Update parameters from smoothed trajectories
# ---------------------------------------------------------------------------

def M_step(
    smoothed_states,
    results: FootballResults,
    num_teams: int,
    prev_params: EMParams,
) -> EMParams:
    """M-step: Update parameters from smoothed particle trajectories.

    All inner loops are vectorized with jax.vmap for JIT compatibility.

    Returns:
        Updated EMParams.
    """
    all_x = smoothed_states.particles.x  # (T+1, N, M, 2)
    T_plus_1 = all_x.shape[0]
    N_particles = all_x.shape[1]
    K = 2

    # --- 1. Update μ_0: empirical mean of smoothed states at t=0 ---
    new_init_mean = jnp.mean(all_x[0], axis=0)  # (M, 2)

    # --- 2. Update Γ_0: empirical covariance between teams at t=0 ---
    x0_centered = all_x[0] - new_init_mean[None, :, :]  # (N, M, 2)
    # Vectorized: Γ_0 = (1/K) Σ_k X_k^T X_k / N
    gamma_0_sum = jnp.einsum("nmk,nml->kl", x0_centered, x0_centered)  # wait, wrong dims
    # x0_centered: (N, M, K), we want (M, M) averaged over K
    # Cov over teams for each k: (M, M) = X_k^T X_k / N
    gamma_0_sum = jnp.zeros((num_teams, num_teams))
    for k in range(K):
        x0_k = x0_centered[:, :, k]  # (N, M)
        gamma_0_sum += x0_k.T @ x0_k / N_particles
    new_init_gamma = gamma_0_sum / K
    new_init_gamma = new_init_gamma + 1e-6 * jnp.eye(num_teams)

    # --- 3. Update B: average within-team covariance ---
    # B = (1/N) Σ_n Cov_within(X_0^n) = (1/N) Σ_n (1/M) Σ_m x_centered[n,m,:] x_centered[n,m,:]^T
    # Vectorized: B = einsum('nmk,nml->kl', x0_centered, x0_centered) / (N * M)
    new_init_B = jnp.einsum("nmk,nml->kl", x0_centered, x0_centered) / (N_particles * num_teams)
    new_init_B = new_init_B + 1e-6 * jnp.eye(K)

    # --- 4. Update κ: vectorized grid search over transition likelihood ---
    timestamps = results.timestamp
    timestamps_prev = results.timestamp_prev

    def neg_log_trans_kappa(kappa_val):
        """Vectorized negative log transition likelihood for a single kappa."""
        # Subsample time steps for speed
        t_indices = jnp.arange(1, min(T_plus_1, 200))
        dts = timestamps[t_indices] - timestamps_prev[t_indices]
        phis = jnp.exp(-kappa_val * dts)  # (T_sub,)

        # Q_t for each step: (Γ_0 - φ Γ_0 φ) ⊗ B — but φ is scalar per step
        # Q_gamma_t = Γ_0 - φ_t * Γ_0 * φ_t = Γ_0 (1 - φ_t^2)  (since φ is scalar)
        # Actually φ_t = exp(-κ dt) * I_M, so φ Γ_0 φ = φ^2 Γ_0
        Q_gamma = new_init_gamma[None, :, :] * (1.0 - phis[:, None, None] ** 2)  # (T_sub, M, M)
        Q_full = jnp.kron(Q_gamma[0], new_init_B)  # (KM, KM) — same B for all, Q_gamma differs
        # But Q differs per step... we need per-step Q
        # Q_t = (Γ_0 - φ^2 Γ_0) ⊗ B = (1 - φ^2) Γ_0 ⊗ B = (1 - φ^2) (Γ_0 ⊗ B)
        # Since Γ_0 ⊗ B is fixed, Q_t = (1 - φ_t^2) * Sigma_0
        Sigma_0 = jnp.kron(new_init_gamma, new_init_B)  # (KM, KM)
        Sigma_0_reg = Sigma_0 + 1e-6 * jnp.eye(Sigma_0.shape[0])
        Sigma_0_inv = jnp.linalg.inv(Sigma_0_reg)
        log_det_Sigma_0 = jnp.log(jnp.linalg.det(Sigma_0_reg))
        dim = num_teams * K

        def step_loss(t_idx, phi_val):
            pred_means = new_init_mean + phi_val * (all_x[t_idx - 1] - new_init_mean)  # (N, M, K)
            diffs = (all_x[t_idx] - pred_means).reshape(N_particles, -1)  # (N, KM)
            quad = jnp.einsum("nd,dd,nd->n", diffs, Sigma_0_inv, diffs)  # (N,)
            log_det_Q = jnp.log(1.0 - phi_val ** 2) * dim + log_det_Sigma_0
            return jnp.sum(-0.5 * (quad / (1.0 - phi_val ** 2 + 1e-10) + log_det_Q + dim * jnp.log(2 * jnp.pi)))

        # Vectorize over time steps
        step_losses = jax.vmap(step_loss)(t_indices, phis)
        return -jnp.sum(step_losses)

    kappa_candidates = jnp.linspace(0.01, 10.0, 30)
    kappa_losses = jax.vmap(neg_log_trans_kappa)(kappa_candidates)
    new_kappa = float(kappa_candidates[jnp.argmin(kappa_losses)])

    # --- 5. Update α, β: vectorized grid search over observation likelihood ---
    home_ids = results.home_team_id
    away_ids = results.away_team_id
    home_scores = results.home_score
    away_scores = results.away_score

    def neg_log_obs_alpha_beta(alpha_val, beta_val):
        """Vectorized negative log observation likelihood for a single (α, β)."""
        t_indices = jnp.arange(1, min(T_plus_1, 200))

        def step_obs(t_idx):
            h_id = home_ids[t_idx]
            a_id = away_ids[t_idx]
            y = jnp.array([home_scores[t_idx], away_scores[t_idx]])

            def particle_obs(n):
                x_i = all_x[t_idx, n, h_id]
                x_j = all_x[t_idx, n, a_id]
                return loglik(y, x_i, x_j, alpha=alpha_val, beta=beta_val,
                              max_goals=MAX_GOALS, scale=prev_params.init_friendly_scale)

            return jax.vmap(particle_obs)(jnp.arange(N_particles)).sum()

        return -jax.vmap(step_obs)(t_indices).sum()

    alpha_candidates = jnp.linspace(-2.0, 2.0, 15)
    beta_candidates = jnp.linspace(-8.0, 0.0, 15)

    # Vectorize over both alpha and beta using nested vmap
    def eval_beta(alpha_val):
        return jax.vmap(lambda b: neg_log_obs_alpha_beta(alpha_val, b))(beta_candidates)

    all_losses = jax.vmap(eval_beta)(alpha_candidates)  # (n_alpha, n_beta)
    best_idx = jnp.unravel_index(jnp.argmin(all_losses), all_losses.shape)
    best_alpha = float(alpha_candidates[best_idx[0]])
    best_beta = float(beta_candidates[best_idx[1]])

    return EMParams(
        init_mean=new_init_mean,
        init_gamma=new_init_gamma,
        init_B=new_init_B,
        init_kappa=new_kappa,
        init_alpha=best_alpha,
        init_beta=best_beta,
        init_friendly_scale=prev_params.init_friendly_scale,
    )


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------

def run_em(
    results: FootballResults,
    init_params: EMParams,
    num_teams: int,
    n_epochs: int = 10,
    output_dir: str = "./outputs",
) -> tuple[EMParams, list]:
    """Run the EM algorithm for n_epochs.

    Saves parameter JSON files at each epoch and the final result.

    Returns:
        (final_params, log_marginal_history)
    """
    os.makedirs(output_dir, exist_ok=True)
    key = jax.random.PRNGKey(42)

    params = init_params
    log_marginal_history = []

    for epoch in tqdm(range(n_epochs), desc="EM epochs", unit="epoch"):
        print(f"\n{'='*60}")
        print(f"EM Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*60}")

        # E-step
        print("  E-step: filtering + backward sampling...")
        key, e_key = jax.random.split(key)
        filtered_states, smoothed_states, log_marginal = E_step(
            params, results, e_key, num_teams,
        )
        log_marginal_history.append(log_marginal)
        print(f"  Log marginal likelihood: {log_marginal:.4f}")

        # Save current parameters
        save_params(params, f"{output_dir}/em_params_epoch_{epoch}.json")

        # M-step
        print("  M-step: updating parameters...")
        params = M_step(smoothed_states, results, num_teams, params)
        print(f"  Updated κ={params.init_kappa:.4f}, α={params.init_alpha:.4f}, β={params.init_beta:.4f}")

    # Save final parameters
    save_params(params, f"{output_dir}/em_params_final.json")

    return params, log_marginal_history


def main():
    # data, results, team_id_to_name = download_results(
    #     start_date="2000-01-01", end_date="2025-12-31", max_goals=MAX_GOALS,
    # )
    data, results, team_id_to_name = read_results(
        start_date="2000-01-01", end_date="2025-12-31", max_goals=MAX_GOALS, teams_only=WORLDCUP_2026_TEAMS,
    )

    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)

    # Initialize parameters
    A = jax.random.normal(key, (NUM_TEAMS, NUM_TEAMS))
    INIT_GAMMA = A @ A.T + 1.0 * jnp.eye(NUM_TEAMS)
    B = jax.random.normal(key, (2, 2))
    INIT_B = B @ B.T + 1.0 * jnp.eye(2)

    init_params = EMParams(
        init_mean=jnp.zeros((NUM_TEAMS, 2)),
        init_gamma=INIT_GAMMA,
        init_B=INIT_B,
        init_kappa=2.0,
        init_alpha=0.2,
        init_beta=-4.0,
        init_friendly_scale=2.0,
    )

    # Save initial parameters
    save_params(init_params, "./outputs/em_params_init.json")

    # Run EM
    final_params, log_marginal_history = run_em(
        results, init_params, NUM_TEAMS, n_epochs=15, output_dir="./outputs",
    )

    print(f"\n{'='*60}")
    print("EM COMPLETE")
    print(f"{'='*60}")
    print(f"Log marginal likelihood history: {log_marginal_history}")
    print(f"\nFinal parameters:")
    print(f"  κ = {final_params.init_kappa:.4f}")
    print(f"  α = {final_params.init_alpha:.4f}")
    print(f"  β = {final_params.init_beta:.4f}")
    print(f"  B = {np.array(final_params.init_B)}")
    print(f"  Γ_0 shape = {final_params.init_gamma.shape}")
    print(f"  μ_0 shape = {final_params.init_mean.shape}")


if __name__ == "__main__":
    main()