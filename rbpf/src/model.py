import os
import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from functools import partial

from rbpf.src.bivariate_poisson import loglik
from rbpf.src.data import get_results, FootballResults
from rbpf.src.helpers import default_init_params
from rbpf.src.utils import RBPFState, RBPFFootballResults, EMParams

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

N = 10
MAX_GOALS = 8

def init_sample(
    key: jax.Array, 
    model_inputs: FootballResults, 
    init_mean: jnp.ndarray
) -> RBPFState:
    return RBPFState(x=init_mean)

def propagate_sample(
        key: jax.Array, 
        state: RBPFState, 
        model_inputs: FootballResults, 
        mean: jnp.ndarray,
        B: jnp.ndarray,
        kappa: float,
        num_teams: int):
    # 1. Propagate all states (OU process)
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi_t = jnp.exp(-kappa * dt) * jnp.eye(num_teams)
    pred_mean = mean + phi_t @ (state.x - mean)
    pred_gamma = model_inputs.gamma_pred_t

    obs_indices = jnp.array([model_inputs.home_team_id, model_inputs.away_team_id])

    # 2. Sample for observed teams
    mu_E = pred_mean[obs_indices]                              # (2, 2)
    gamma_EE = pred_gamma[jnp.ix_(obs_indices, obs_indices)]   # (2, 2)
    # Add jitter to prevent singular gamma_EE when dt=0 and teams were observed
    # in the previous step (zeroed rows make gamma_EE singular)
    gamma_EE = gamma_EE + 1e-6 * jnp.eye(gamma_EE.shape[0])
    Sigma_EE = jnp.kron(gamma_EE, B)                      # (4, 4)

    key, subkey = jax.random.split(key)
    x_E_flat = jax.random.multivariate_normal(subkey, mu_E.flatten(), Sigma_EE)
    x_E = x_E_flat.reshape(mu_E.shape)                         # (2, 2)

    # 3. Kalman update for ALL teams (vmap-safe, no setdiff1d)
    # Cross-covariance: all teams vs observed teams
    gamma_RE_full = pred_gamma[:, obs_indices]
    kalman_gain = model_inputs.kalman_gain_t
    x_update = pred_mean + kalman_gain @ (x_E - mu_E) 
    # Overwrite observed teams with sampled values
    x_update = x_update.at[obs_indices].set(x_E)
    return RBPFState(x=x_update)

# wrapper function to index the observed teams
def _log_potential(
    state_prev: RBPFState,
    state: RBPFState,
    model_inputs: FootballResults,
    alpha: float,
    beta: float,
    max_goals: int,
) -> jax.Array:
    y = jnp.array([model_inputs.home_score, model_inputs.away_score])
    x_i = state.x[model_inputs.home_team_id]
    x_j = state.x[model_inputs.away_team_id]
    return loglik(y, x_i, x_j, alpha=alpha, beta=beta, max_goals=max_goals, scale=1.0)

def build_rbpf_filter(
        params: EMParams,
        n: int,
        num_teams: int,
    ) -> cuthbert.inference.Filter:
    rbpf = cuthbert.smc.particle_filter.build_filter(
        init_sample=partial(init_sample,
            init_mean=params.mean_0
        ),
        propagate_sample=partial(
            propagate_sample,
            mean=params.mean_0,
            B=params.B,
            kappa=params.kappa,
            num_teams=num_teams
        ),
        log_potential=partial(_log_potential,
                            alpha=params.alpha,
                            beta=params.beta,
                            max_goals=MAX_GOALS
        ),
        n_filter_particles=n,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )
    return rbpf

def compute_gamma_trajectory(
        model_inputs: FootballResults, 
        gamma_0: jnp.ndarray, 
        kappa: float, 
        num_teams: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Gamma evolution does not depend on particle states, so we can compute the gamma trajectory deterministically.
    """
    def gamma_step(gamma_prev, model_input):
        # gamma step for jax.lax.scan
        dt = model_input.timestamp - model_input.timestamp_prev
        phi_t = jnp.exp(-kappa * dt) * jnp.eye(num_teams)

        Q_t = gamma_0 - phi_t @ gamma_0 @ phi_t.T
        gamma_pred = phi_t @ gamma_prev @ phi_t.T + Q_t
        gamma_pred = 0.5 * (gamma_pred + gamma_pred.T) # ensure symmetry

        obs_indices = jnp.array([
            model_input.home_team_id, 
            model_input.away_team_id
        ])

        gamma_EE = gamma_pred[jnp.ix_(obs_indices, obs_indices)]
        gamma_EE = 0.5 * (gamma_EE + gamma_EE.T)
        gamma_RE = gamma_pred[:, obs_indices]

        # Handles both positive-definite and structurally singular gamma_EE.
        K = gamma_RE @ jnp.linalg.pinv(gamma_EE)

        gamma_updated = gamma_pred - K @ gamma_RE.T
        gamma_updated = 0.5 * (gamma_updated + gamma_updated.T)

        obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
        keep_mask = jnp.outer(~obs_mask, ~obs_mask)
        gamma_updated = gamma_updated * keep_mask

        return gamma_updated, (gamma_updated, gamma_pred, K)
    
    _, (gamma_updated, gamma_pred, kalman_gain) = jax.lax.scan(
        f=gamma_step, init=gamma_0, xs=model_inputs    # no [1:] slicing
    )
    return gamma_updated, gamma_pred, kalman_gain

@partial(jax.jit, static_argnames=("num_teams", "n_particles"))
def run_filter(
    key : jax.Array, 
    model_inputs: RBPFFootballResults, 
    params: EMParams,
    # mean_0: jnp.ndarray,
    # gamma_0: jnp.ndarray, 
    # kappa: float,
    # B: jnp.ndarray, 
    # alpha: float, 
    # beta: float,
    num_teams: int, 
    n_particles: int,
):
    rbpf = build_rbpf_filter(
        n=n_particles,
        params=params,
        num_teams=num_teams
    )
    # Prepare the initial state from the first time step's model inputs
    init_state = rbpf.init_prepare(
        jax.tree.map(lambda x: x[0], model_inputs), key=key,
    )
    # Filter over the remaining T steps (exclude the initial state at index 0)
    rest_inputs = jax.tree.map(lambda x: x[1:], model_inputs)
    filtered_states = cuthbert.filtering.filter(rbpf, rest_inputs, init_state, key=key)
    return filtered_states, model_inputs


def main():
    from rbpf.src.data import WORLDCUP_2026_TEAMS, ACTIVE_TEAMS
    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01", 
        end_date = "2026-01-01", 
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=ACTIVE_TEAMS
    )

    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(5))
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, team_id_to_name=team_id_to_name)
    
    # data.to_csv("rbpf/data/results.csv", index=False)

    print("Running filter (optimized)...")
    key, filter_key = jax.random.split(key)
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs, 
        gamma_0=params.gamma_0, 
        kappa=params.kappa, 
        num_teams=NUM_TEAMS
    )
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
    try:
        filtered_states, augmented_results = run_filter(
            key=filter_key,
            model_inputs=augmented_results,
            params=params,
            num_teams=NUM_TEAMS,
            n_particles=N
        )
    except Exception as e:
        print("Error during filtering:", e)
        return
    print(f"\nResult shapes:")
    print(f"  obeservations:   {len(model_inputs.timestamp)}")
    print(f"  particles.x:     {filtered_states.particles.x.shape}")
    print(filtered_states.particles.x[-1][0])  # last time step's particle states
    # print(f"  particles.gamma: {filtered_states.particles.gamma.shape}")
    print(f"  particles.gamma: {augmented_results.gamma_t.shape}")
    print(f"  log_weights:     {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_constant: {filtered_states.log_normalizing_constant.shape}")

    import os
    from rbpf.src.graphic import plot_all
    path = os.path.join(os.path.dirname(__file__), "outputs", "graphic")
    plot_all(filtered_states, augmented_results, team_id_to_name, top_n=5, save_path=path)
    
if __name__ == "__main__":
    main()