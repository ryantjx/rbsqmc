"""EM for RBPF with Kronecker structure — V2: Fixed κ, B, Γ₀. Estimate α, β, μ₀.

Fixes the degeneracy issue from smoothing.py by:
  - Fixing κ = 1.0 (OU reversion rate, breaks the κ→0 feedback loop)
  - Fixing B = I₂ (within-team covariance, prevents collapse)
  - Γ₀ is randomly sampled at initialization (A@A.T + I) and kept fixed
  - Γ_t for t > 0 is computed deterministically from Γ₀ via compute_gamma_trajectory
    (PROOF_V5 §3.1, step 1 — Kalman prediction/update equations)
  - Estimating α, β via grid search
  - Estimating μ₀ via empirical mean of smoothed particles

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

jax.config.update("jax_platforms", "cuda")  # Script to run GPU

N = 1000
MAX_GOALS = 8

# --- Fixed parameters ---
FIXED_KAPPA = 1.0  # OU reversion rate (half-life ~0.7 days)
FIXED_B = jnp.eye(2)  # Within-team covariance = identity


class EMParams(NamedTuple):
    """Parameters that EM optimizes (V2: κ and B are fixed)."""
    init_mean: jax.Array      # (M, 2)
    init_gamma: jax.Array     # (M, M)
    init_B: jax.Array         # (2, 2) — fixed, not updated
    init_kappa: float         # fixed, not updated
    init_alpha: float
    init_beta: float
    init_friendly_scale: float


def params_to_dict(params: EMParams) -> dict:
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
    with open(path, "w") as f:
        json.dump(params_to_dict(params), f, indent=2)
    print(f"Saved parameters to {path}")


def load_params(path: str) -> EMParams:
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
    """Joint log potential for the backward sampler."""
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
    Q_t_reg = Q_t + 1e-3 * jnp.eye(Q_t.shape[0])

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
    """E-step: Forward filter + Backward sampling."""
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

    gamma_trajectory = compute_gamma_trajectory(
        results, params.init_gamma, params.init_kappa, num_teams,
    )

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
# M-step: Update Γ₀, α, β, μ₀ (κ and B are fixed)
# ---------------------------------------------------------------------------

def M_step(
    smoothed_states,
    results: FootballResults,
    num_teams: int,
    prev_params: EMParams,
) -> EMParams:
    """M-step: Update μ₀, α, β. κ, B, and Γ₀ are fixed.

    Γ₀ is initialized at the start and propagated forward via
    compute_gamma_trajectory (PROOF_V5 §3.1, step 1). It is NOT
    re-estimated empirically from particles — Γ_t is a deterministic
    function of Γ₀ through the Kalman prediction/update equations.
    """
    all_x = smoothed_states.particles.x  # (T+1, N, M, 2)
    T_plus_1 = all_x.shape[0]
    N_particles = all_x.shape[1]
    K = 2

    # --- 1. Update μ_0: empirical mean of smoothed states at t=0 ---
    new_init_mean = jnp.mean(all_x[0], axis=0)  # (M, 2)

    # --- 2. Γ_0 is fixed (initialized at start, not re-estimated) ---
    # Γ_t for t > 0 is computed via compute_gamma_trajectory from Γ_0
    new_init_gamma = prev_params.init_gamma

    # --- 3. B is fixed (not updated) ---
    new_init_B = FIXED_B

    # --- 4. κ is fixed (not updated) ---
    new_kappa = FIXED_KAPPA

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

    alpha_candidates = jnp.linspace(-5.0, 5.0, 25)
    beta_candidates = jnp.linspace(-10.0, 2.0, 25)

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
    n_epochs: int = 15,
    output_dir: str = "./outputs_gpu_v2",
) -> tuple[EMParams, list]:
    """Run the EM algorithm for n_epochs."""
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
        print(f"  Updated α={params.init_alpha:.4f}, β={params.init_beta:.4f}")
        print(f"  (κ={params.init_kappa:.4f} [fixed], B=I₂ [fixed], Γ₀ [fixed])")

    # Save final parameters
    save_params(params, f"{output_dir}/em_params_final.json")

    return params, log_marginal_history


def main():
    data, results, team_id_to_name = read_results(
        start_date="2000-01-01", end_date="2025-12-31", max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
    )

    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)

    # Initialize parameters
    # Γ_0: non-diagonal covariance between teams
    A = jax.random.normal(key, (NUM_TEAMS, NUM_TEAMS))
    INIT_GAMMA = A @ A.T + 1.0 * jnp.eye(NUM_TEAMS)

    init_params = EMParams(
        init_mean=jnp.zeros((NUM_TEAMS, 2)),
        init_gamma=INIT_GAMMA,
        init_B=FIXED_B,           # Fixed: B = I₂
        init_kappa=FIXED_KAPPA,   # Fixed: κ = 1.0
        init_alpha=0.2,
        init_beta=-4.0,
        init_friendly_scale=2.0,
    )

    # Save initial parameters
    save_params(init_params, "./outputs_gpu_v2/em_params_init.json")

    # Run EM
    final_params, log_marginal_history = run_em(
        results, init_params, NUM_TEAMS, n_epochs=15, output_dir="./outputs_gpu_v2",
    )

    print(f"\n{'='*60}")
    print("EM COMPLETE (V2: fixed κ=1.0, B=I₂)")
    print(f"{'='*60}")
    print(f"Log marginal likelihood history: {log_marginal_history}")
    print(f"\nFinal parameters:")
    print(f"  κ = {final_params.init_kappa:.4f} [FIXED]")
    print(f"  α = {final_params.init_alpha:.4f}")
    print(f"  β = {final_params.init_beta:.4f}")
    print(f"  B = {np.array(final_params.init_B)} [FIXED]")
    print(f"  Γ_0 shape = {final_params.init_gamma.shape}")
    print(f"  μ_0 shape = {final_params.init_mean.shape}")


if __name__ == "__main__":
    main()