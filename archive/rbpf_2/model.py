import jax
import jax.numpy as jnp
import numpy as np
import cuthbert
import cuthbertlib
from functools import partial
from typing import NamedTuple

from archive.rbpf_2.bivariate_poisson import loglik
from data import download_results, FootballResults

jax.config.update("jax_platforms", "cpu")

N = 10
MAX_GOALS = 8
class RBPFState(NamedTuple):
    x: jax.Array

class RBPFFootballResults(NamedTuple):
    match_index_id : jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    home_team_id: jax.Array
    away_team_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array
    gamma_t: jax.Array

def init_sample(
        key : jax.Array, 
        model_inputs : FootballResults, 
        init_mean : jnp.ndarray, 
        init_gamma : jnp.ndarray, 
        init_B : jnp.ndarray
    ) -> RBPFState:
    x_flat = jax.random.multivariate_normal(
        key=key,
        mean=init_mean.flatten(),
        cov=jnp.kron(init_gamma, init_B),
    )
    x = x_flat.reshape(init_mean.shape)
    return RBPFState(x=x)

def propagate_sample(
        key: jax.Array, 
        state: RBPFState, 
        model_inputs: FootballResults, 
        init_mean: jnp.ndarray,
        init_B: jnp.ndarray,
        init_kappa: float,
        num_teams: int):
    # 1. Propagate all states (OU process)
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
    pred_mean = init_mean + phi_t @ (state.x - init_mean)
    pred_gamma = model_inputs.gamma_t
    # Q_t = init_gamma - phi_t @ init_gamma @ phi_t.T
    # pred_gamma = (phi_t @ state.gamma @ phi_t.T + Q_t)
    # pred_gamma = (phi_t @ state.gamma @ phi_t.T + init_gamma - phi_t @ init_gamma @ phi_t.T)

    obs_indices = jnp.array([model_inputs.home_team_id, model_inputs.away_team_id])

    # 2. Sample for observed teams
    mu_E = pred_mean[obs_indices]                              # (2, 2)
    gamma_EE = pred_gamma[jnp.ix_(obs_indices, obs_indices)]   # (2, 2)
    # Add jitter to prevent singular gamma_EE when dt=0 and teams were observed
    # in the previous step (zeroed rows make gamma_EE singular)
    gamma_EE = gamma_EE + 1e-6 * jnp.eye(gamma_EE.shape[0])
    Sigma_EE = jnp.kron(gamma_EE, init_B)                      # (4, 4)

    key, subkey = jax.random.split(key)
    x_E_flat = jax.random.multivariate_normal(subkey, mu_E.flatten(), Sigma_EE)
    x_E = x_E_flat.reshape(mu_E.shape)                         # (2, 2)

    # 3. Kalman update for ALL teams (vmap-safe, no setdiff1d)
    # Cross-covariance: all teams vs observed teams
    gamma_RE_full = pred_gamma[:, obs_indices]                 # (num_teams, 2)

    # Kalman gain for all teams: K = Gamma_RE @ inv(Gamma_EE)
    K_full = jnp.linalg.solve(gamma_EE.T, gamma_RE_full.T).T  # (num_teams, 2)

    # Correction for all teams
    correction = K_full @ (x_E - mu_E)                         # (num_teams, 2)
    x_update = pred_mean + correction
    # Overwrite observed teams with sampled values
    x_update = x_update.at[obs_indices].set(x_E)

    # Covariance update for all teams
    gamma_updated = pred_gamma - K_full @ gamma_RE_full.T      # (num_teams, num_teams)

    # Zero out observed rows/cols (uncertainty captured in the sample)
    obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
    keep_mask = jnp.outer(~obs_mask, ~obs_mask)                # (num_teams, num_teams)
    gamma_updated = gamma_updated * keep_mask

    return RBPFState(x=x_update)

# wrapper function to index the observed teams
def _log_potential(
    state_prev: RBPFState,
    state: RBPFState,
    model_inputs: FootballResults,
    alpha: float,
    beta: float,
    scale: float,
    max_goals: int,
) -> jax.Array:
    y = jnp.array([model_inputs.home_score, model_inputs.away_score])
    x_i = state.x[model_inputs.home_team_id]
    x_j = state.x[model_inputs.away_team_id]
    return loglik(y, x_i, x_j, alpha=alpha, beta=beta, max_goals=max_goals, scale=scale)

def build_rbpf_filter(
        n: int,
        init_mean: jnp.ndarray,
        init_gamma: jnp.ndarray,
        init_B: jnp.ndarray,
        init_kappa: float,
        init_alpha: float,
        init_beta: float,
        init_friendly_scale: float,
        num_teams: int,
    ) -> cuthbert.inference.Filter:
    rbpf = cuthbert.smc.particle_filter.build_filter(
        init_sample=partial(init_sample,
            init_mean=init_mean,
            init_gamma=init_gamma,
            init_B=init_B
        ),
        propagate_sample=partial(
            propagate_sample,
            init_mean=init_mean,
            # init_gamma=init_gamma,
            init_B=init_B,
            init_kappa=init_kappa,
            num_teams=num_teams
        ),
        log_potential=partial(_log_potential,
                            alpha=init_alpha,
                            beta=init_beta,
                            scale=init_friendly_scale,
                            max_goals=MAX_GOALS
        ),
        n_filter_particles=n,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )
    return rbpf

def compute_gamma_trajectory(model_inputs, init_gamma, init_kappa, num_teams):
    """Deterministic gamma evolution. One copy per time step, NOT per particle."""
    def gamma_step(prev_gamma, model_input):
        dt = model_input.timestamp - model_input.timestamp_prev
        phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
        Q_t = init_gamma - phi_t @ init_gamma @ phi_t.T
        pred_gamma = phi_t @ prev_gamma @ phi_t.T + Q_t

        obs_indices = jnp.array([model_input.home_team_id, model_input.away_team_id])
        gamma_EE = pred_gamma[jnp.ix_(obs_indices, obs_indices)]
        # Add jitter to prevent singular gamma_EE when dt=0 and teams were observed
        # in the previous step (zeroed rows make gamma_EE singular)
        gamma_EE = gamma_EE + 1e-6 * jnp.eye(gamma_EE.shape[0])
        gamma_RE_full = pred_gamma[:, obs_indices]
        K_full = jnp.linalg.solve(gamma_EE.T, gamma_RE_full.T).T
        gamma_updated = pred_gamma - K_full @ gamma_RE_full.T

        obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
        gamma_updated = gamma_updated * jnp.outer(~obs_mask, ~obs_mask)

        return gamma_updated, pred_gamma   # carry post-update, store predictive

    rest_inputs = jax.tree.map(lambda x: x[1:], model_inputs)
    _, pred_gamma_traj = jax.lax.scan(gamma_step, init_gamma, rest_inputs)
    return jnp.concatenate([init_gamma[None], pred_gamma_traj])  # (T+1, M, M)

@partial(jax.jit, static_argnames=("num_teams", "n"))
def run_filter(key, results, init_gamma, init_kappa, num_teams, n,
               init_mean, init_B, init_alpha, init_beta, init_friendly_scale):
    gamma_trajectory = compute_gamma_trajectory(results, init_gamma, init_kappa, num_teams)
    augmented_results = RBPFFootballResults(
        match_index_id=results.match_index_id,
        timestamp=results.timestamp,
        timestamp_prev=results.timestamp_prev,
        home_team_id=results.home_team_id,
        away_team_id=results.away_team_id,
        home_score=results.home_score,
        away_score=results.away_score,
        gamma_t=gamma_trajectory
    )
    rbpf = build_rbpf_filter(
        n=n,
        init_mean=init_mean,
        init_gamma=init_gamma,
        init_B=init_B,
        init_kappa=init_kappa,
        init_alpha=init_alpha,
        init_beta=init_beta,
        init_friendly_scale=init_friendly_scale,
        num_teams=num_teams
    )
    # Prepare the initial state from the first time step's model inputs
    init_state = rbpf.init_prepare(
        jax.tree.map(lambda x: x[0], augmented_results), key=key,
    )
    # Filter over the remaining T steps (exclude the initial state at index 0)
    rest_inputs = jax.tree.map(lambda x: x[1:], augmented_results)
    filtered_states = cuthbert.filtering.filter(rbpf, rest_inputs, init_state, key=key)
    return filtered_states


def main():
    
    data, results, team_id_to_name = download_results(start_date="2020-01-01", end_date = "2025-12-31", max_goals=MAX_GOALS)

    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head())
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail())
    NUM_TEAMS = len(team_id_to_name)
    INIT_MEAN = jnp.zeros((NUM_TEAMS, 2))
    key = jax.random.PRNGKey(42)
    # Gamma_0 in R^{M x M}: covariance between teams (must be PD)
    A = jax.random.normal(key, (NUM_TEAMS, NUM_TEAMS))
    INIT_GAMMA = A @ A.T + 1.0 * jnp.eye(NUM_TEAMS) # ensure positive definite
    B = jax.random.normal(key, (2, 2))
    INIT_B = B @ B.T + 1.0 * jnp.eye(2) # ensure positive definite
    INIT_KAPPA = 2
    INIT_ALPHA = 0.2
    INIT_BETA = -4.0
    INIT_FRIENDLY_SCALE = 2.0

    print("Initial mean:", INIT_MEAN.shape)
    print("Initial gamma:", INIT_GAMMA.shape)
    # print(INIT_GAMMA[0])
    print("Initial B:", INIT_B.shape)
    print(INIT_B)
    print(f"Kappa: {INIT_KAPPA}, Alpha: {INIT_ALPHA}, Beta: {INIT_BETA}, Friendly Scale: {INIT_FRIENDLY_SCALE}" )

    # print("Building RBPF filter...")
    # gamma_trajectory = compute_gamma_trajectory(results, INIT_GAMMA, INIT_KAPPA, NUM_TEAMS)
    # augmented_results = RBPFFootballResults(
    #     match_index_id=results.match_index_id,
    #     timestamp=results.timestamp,
    #     timestamp_prev=results.timestamp_prev,
    #     home_team_id=results.home_team_id,
    #     away_team_id=results.away_team_id,
    #     home_score=results.home_score,
    #     away_score=results.away_score,
    #     gamma_t=gamma_trajectory
    # )
    # # Build the filter with the data-dependent params
    # rbpf = build_rbpf_filter(
    #     n=N,
    #     init_mean=INIT_MEAN,
    #     init_gamma=INIT_GAMMA,
    #     init_B=INIT_B,
    #     init_kappa=INIT_KAPPA,
    #     init_alpha=INIT_ALPHA,
    #     init_beta=INIT_BETA,
    #     init_friendly_scale=INIT_FRIENDLY_SCALE,
    #     num_teams=NUM_TEAMS,
    # )

    # print("Running filter...")
    # # Run the filter
    # key, filter_key = jax.random.split(key)
    # filtered_states = cuthbert.filtering.filter(rbpf, augmented_results, key=filter_key)
    # print("Filtering complete.")

    print("Running filter (optimized)...")
    key, filter_key = jax.random.split(key)
    filtered_states, gamma_trajectory = run_filter(
        key=filter_key,
        results=results,
        init_mean=INIT_MEAN,
        init_gamma=INIT_GAMMA,
        init_B=INIT_B,
        init_kappa=INIT_KAPPA,
        init_alpha=INIT_ALPHA,
        init_beta=INIT_BETA,
        init_friendly_scale=INIT_FRIENDLY_SCALE,
        num_teams=NUM_TEAMS,
        n=N
    )

    # --- Analyze results ---
    # result.particles.x     -> (T+1, N, NUM_TEAMS, 2)
    # result.particles.gamma -> (T+1, N, NUM_TEAMS, NUM_TEAMS)
    # result.log_weights     -> (T+1, N)

    print(f"\nResult shapes:")
    print(f"  obeservations:   {len(results.timestamp)}")
    print(f"  particles.x:     {filtered_states.particles.x.shape}")
    # print(f"  particles.gamma: {filtered_states.particles.gamma.shape}")
    print(f"  particles.gamma: {gamma_trajectory.shape}")
    print(f"  log_weights:     {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_constant: {filtered_states.log_normalizing_constant.shape}")

    # Compute weighted mean of the final state
    from archive.rbpf_2.graphic import generate_all_plots, weighted_mean

    final_log_w = filtered_states.log_weights[-1]
    final_x_mean = weighted_mean(filtered_states.particles.x[-1], final_log_w)
    # Gamma is precomputed (one copy per step, not per particle)
    final_gamma = gamma_trajectory[-1]  # (NUM_TEAMS, NUM_TEAMS)

    print(f"\nFinal state (weighted mean) — x shape: {final_x_mean.shape}")
    print(f"Final gamma — shape: {final_gamma.shape}")

    # Print top 5 teams by attack strength
    final_att = np.array(final_x_mean[:, 0])
    top5_idx = np.argsort(final_att)[::-1][:5]
    print("\nTop 5 teams by attack strength:")
    for tid in top5_idx:
        name = team_id_to_name.get(int(tid), f"Team {tid}")
        print(f"  {name}: attack={final_att[tid]:.3f}, defense={final_x_mean[tid, 1]:.3f}")

    # Generate all diagnostic plots
    print("\nGenerating plots...")
    generate_all_plots(
        filtered_states,
        gamma_trajectory,
        team_id_to_name,
        output_dir="./outputs/base",
        max_teams=10,
    )

    # Save results to parquet for analysis and parameter estimation
    print("\nSaving results to parquet...")
    from archive.rbpf_2.graphic import save_results
    save_results(
        filtered_states,
        gamma_trajectory,
        team_id_to_name,
        output_dir="./outputs/base",
    )

if __name__ == "__main__":
    main()