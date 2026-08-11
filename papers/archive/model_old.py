import jax
import jax.numpy as jnp
import numpy as np
import cuthbert
import cuthbertlib
from functools import partial
from typing import NamedTuple

from rbpf.src.bivariate_poisson import loglik
from rbpf.src.data import download_results, FootballResults
from rbpf.src.helpers import default_init_params
from rbpf.utils import RBPFState, RBPFFootballResults

jax.config.update("jax_platforms", "cpu")

N = 100
MAX_GOALS = 8

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
    pred_gamma = model_inputs.gamma_pred_t
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
    # kalman_gain = jnp.linalg.solve(gamma_EE.T, gamma_RE_full.T).T  # (num_teams, 2)
    kalman_gain = model_inputs.kalman_gain_t

    # Correction for all teams
    correction = kalman_gain @ (x_E - mu_E)                         # (num_teams, 2)
    x_update = pred_mean + correction
    # Overwrite observed teams with sampled values
    x_update = x_update.at[obs_indices].set(x_E)

    # # Covariance update for all teams
    # gamma_updated = pred_gamma - kalman_gain @ gamma_RE_full.T      # (num_teams, num_teams)

    # # Zero out observed rows/cols (uncertainty captured in the sample)
    # obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
    # keep_mask = jnp.outer(~obs_mask, ~obs_mask)                # (num_teams, num_teams)
    # gamma_updated = gamma_updated * keep_mask

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

def compute_gamma_trajectory(
        model_inputs: RBPFFootballResults, 
        init_gamma: jnp.ndarray, 
        init_kappa: float, 
        num_teams: int
    ):
    """
    Gamma evolution does not depend on particle states, so we can compute the gamma trajectory deterministically.
    """
    def gamma_step(gamma_prev, model_input):
        # gamma step for jax.lax.scan
        dt = model_input.timestamp - model_input.timestamp_prev
        phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
        Q_t = init_gamma - phi_t @ init_gamma @ phi_t.T
        gamma_pred = phi_t @ gamma_prev @ phi_t.T + Q_t
        obs_indices = jnp.array([model_input.home_team_id, model_input.away_team_id])
        gamma_EE = gamma_pred[jnp.ix_(obs_indices, obs_indices)]
        # # Add jitter to prevent singular gamma_EE when dt=0 and teams were observed
        # # in the previous step (zeroed rows make gamma_EE singular)
        gamma_EE = gamma_EE + 1e-6 * jnp.eye(gamma_EE.shape[0])
        gamma_RE_full = gamma_pred[:, obs_indices]
        kalman_gain = jnp.linalg.solve(gamma_EE.T, gamma_RE_full.T).T
        # Kalman update for all teams
        gamma_updated = gamma_pred - kalman_gain @ gamma_RE_full.T
        # Zero out observed rows/cols (uncertainty captured in the sample)
        obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
        gamma_updated = gamma_updated * jnp.outer(~obs_mask, ~obs_mask)
        return gamma_updated, (gamma_updated, gamma_pred, kalman_gain)   # carry post-update, store predictive
    _, (gamma_updated, gamma_pred, kalman_gain) = jax.lax.scan(
        f=gamma_step, init=init_gamma, xs=model_inputs    # no [1:] slicing
    )
    return gamma_updated, gamma_pred, kalman_gain
# def compute_gamma_trajectory(model_inputs, init_gamma, init_kappa, num_teams):
#     """Deterministic gamma evolution. One copy per time step, NOT per particle."""
#     def gamma_step(prev_gamma, model_input):
#         dt = model_input.timestamp - model_input.timestamp_prev
#         phi_t = jnp.exp(-init_kappa * dt) * jnp.eye(num_teams)
#         Q_t = init_gamma - phi_t @ init_gamma @ phi_t.T
#         pred_gamma = phi_t @ prev_gamma @ phi_t.T + Q_t

#         obs_indices = jnp.array([model_input.home_team_id, model_input.away_team_id])
#         gamma_EE = pred_gamma[jnp.ix_(obs_indices, obs_indices)]
#         # Add jitter to prevent singular gamma_EE when dt=0 and teams were observed
#         # in the previous step (zeroed rows make gamma_EE singular)
#         gamma_EE = gamma_EE + 1e-6 * jnp.eye(gamma_EE.shape[0])
#         gamma_RE_full = pred_gamma[:, obs_indices]
#         K_full = jnp.linalg.solve(gamma_EE.T, gamma_RE_full.T).T
#         gamma_updated = pred_gamma - K_full @ gamma_RE_full.T

#         obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
#         gamma_updated = gamma_updated * jnp.outer(~obs_mask, ~obs_mask)

#         return gamma_updated, pred_gamma   # carry post-update, store predictive

#     rest_inputs = jax.tree.map(lambda x: x[1:], model_inputs)
#     _, pred_gamma_traj = jax.lax.scan(gamma_step, init_gamma, rest_inputs)
#     return jnp.concatenate([init_gamma[None], pred_gamma_traj])  # (T+1, M, M)

@partial(jax.jit, static_argnames=("num_teams", "n"))
def run_filter(
    key : jax.Array, 
    model_inputs: FootballResults, 
    gamma: jnp.ndarray, 
    kappa: float, 
    num_teams: int, 
    n: int,
    mean: jnp.ndarray, 
    B: jnp.ndarray, 
    alpha: float, 
    beta: float, 
    friendly_scale: float
):
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(model_inputs, gamma, kappa, num_teams)
    augmented_results = RBPFFootballResults(
        match_index_id=model_inputs.match_index_id,
        timestamp=model_inputs.timestamp,
        timestamp_prev=model_inputs.timestamp_prev,
        home_team_id=model_inputs.home_team_id,
        away_team_id=model_inputs.away_team_id,
        home_score=model_inputs.home_score,
        away_score=model_inputs.away_score,
        gamma_t=gamma_updated,
        gamma_pred_t=gamma_pred,
        kalman_gain_t=kalman_gain
    )
    rbpf = build_rbpf_filter(
        n=n,
        init_mean=mean,
        init_gamma=gamma,
        init_B=B,
        init_kappa=kappa,
        init_alpha=alpha,
        init_beta=beta,
        init_friendly_scale=friendly_scale,
        num_teams=num_teams
    )
    # Prepare the initial state from the first time step's model inputs
    init_state = rbpf.init_prepare(
        jax.tree.map(lambda x: x[0], augmented_results), key=key,
    )
    # Filter over the remaining T steps (exclude the initial state at index 0)
    rest_inputs = jax.tree.map(lambda x: x[1:], augmented_results)
    filtered_states = cuthbert.filtering.filter(rbpf, rest_inputs, init_state, key=key)
    return filtered_states, augmented_results


def main():
    from rbpf.src.data import WORLDCUP_2026_TEAMS
    data, model_inputs, team_id_to_name = download_results(start_date="1950-01-01", end_date = "2026-01-01", max_goals=MAX_GOALS, teams_only=WORLDCUP_2026_TEAMS)

    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head())
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail())
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, key)

    print("Running filter (optimized)...")
    key, filter_key = jax.random.split(key)
    filtered_states, augmented_results = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        mean=params.mean_0,
        gamma=params.gamma_0,
        B=params.B,
        kappa=params.kappa,
        alpha=params.alpha,
        beta=params.beta,
        friendly_scale=params.friendly_scale,
        num_teams=NUM_TEAMS,
        n=N
    )
    print(f"\nResult shapes:")
    print(f"  obeservations:   {len(model_inputs.timestamp)}")
    print(f"  particles.x:     {filtered_states.particles.x.shape}")
    print(filtered_states.particles.x[-1][0])  # last time step's particle states
    # print(f"  particles.gamma: {filtered_states.particles.gamma.shape}")
    print(f"  particles.gamma: {augmented_results.gamma_t.shape}")
    print(f"  log_weights:     {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_constant: {filtered_states.log_normalizing_constant.shape}")

    from rbpf.src.graphic import plot_top_strengths, plot_correlation_matrix, plot_log_normalizing_constant, plot_correlation_extremes
    import os
    path = os.path.join(os.path.dirname(__file__), "outputs", "graphic")
    os.makedirs(path, exist_ok=True)
    plot_top_strengths(filtered_states, team_id_to_name, top_n=10, save_path=os.path.join(path, "top_strengths.png"))
    plot_correlation_matrix(augmented_results, team_id_to_name, save_path=os.path.join(path, "correlation_matrix.png"))
    plot_log_normalizing_constant(filtered_states, save_path=os.path.join(path, "log_normalizing_constant.png"))
    plot_correlation_extremes(augmented_results, team_id_to_name, top_n=5, save_path=os.path.join(path, "correlation_extremes.png"))

if __name__ == "__main__":
    main()