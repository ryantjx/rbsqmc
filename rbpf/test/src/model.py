import os
import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from functools import partial

from rbpf.test.src.bivariate_poisson import loglik
from rbpf.test.src.data import get_results, FootballResults
from rbpf.test.src.helpers import default_init_params
from rbpf.test.src.utils import RBPFState, RBPFFootballResults, EMParams

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
    init_mean: jnp.ndarray,
) -> RBPFState:
    return RBPFState(x=init_mean)


def _sample_psd_gaussian(
    key: jax.Array,
    mean: jax.Array,
    covariance: jax.Array,
) -> jax.Array:
    """Sample from a PSD Gaussian, preserving exact zero-variance directions.

    The filtered posterior covariances are positive-semidefinite, not
    positive-definite: observed teams have exact zero variance after the
    Schur-complement marginalization (their rows/cols are zeroed). A
    Cholesky-based ``jax.random.multivariate_normal`` returns NaN on such
    singular matrices. Here we eigendecompose, clip tiny eigenvalues to zero,
    and sample noise only in the nonzero-variance directions — observed
    (zero-variance) teams stay exactly at their mean.

    Used for the smoothing draws of ``X_T^*`` (terminal) and the backward RTS
    states, where the posterior covariance is singular. Also used as a safety
    net in the filter's ``propagate_sample`` for the rare ``dt == 0``
    re-observation case where the observed block can be singular.
    """
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    eigvals = jnp.clip(eigvals, 0.0)
    noise = jax.random.normal(key, mean.shape)
    return mean + eigvecs @ (jnp.sqrt(eigvals) * noise)


def propagate_sample(
    key: jax.Array,
    state: RBPFState,
    model_inputs: RBPFFootballResults,
    mean: jnp.ndarray,
    gamma_Q: jnp.ndarray,
    B_Q: jnp.ndarray,
    num_teams: int,
):
    """Random-walk propagation with Rao-Blackwellized Kalman update.

    The random-walk transition has no mean reversion, so the prediction mean is
    simply the previous state: ``mu_{t|t-1} = mu_{t-1|t-1}``. The prediction
    covariance ``Sigma_{t|t-1}`` and the Kalman gain ``K_t`` are precomputed
    deterministically in ``compute_covariance_trajectory`` and carried in
    ``model_inputs``.

    We sample only the observed block (home + away teams, 4 dims) from the
    prediction Gaussian, then condition the remaining (Rao-Blackwellized)
    teams on it via the Kalman gain.
    """
    # 1. Random-walk prediction: mean unchanged (no phi_t / mean reversion).
    pred_mean = state.x  # (M, 2)

    sigma_pred = model_inputs.sigma_pred_t  # (2M, 2M)

    h = model_inputs.home_team_id
    a = model_inputs.away_team_id
    obs_indices_flat = jnp.array([2 * h, 2 * h + 1, 2 * a, 2 * a + 1])  # (4,)

    # 2. Sample the observed block.
    mu_E = pred_mean.reshape(-1)[obs_indices_flat]  # (4,)
    Sigma_EE = sigma_pred[jnp.ix_(obs_indices_flat, obs_indices_flat)]  # (4, 4)

    key, subkey = jax.random.split(key)
    # PSD-aware sampler: robust to the rare dt==0 re-observation case where
    # Sigma_EE is singular (a team's posterior variance was zeroed). Zero-
    # variance directions stay exactly at the mean.
    x_E_flat = _sample_psd_gaussian(subkey, mu_E, Sigma_EE)  # (4,)

    # 3. Kalman update for ALL teams (Rao-Blackwellization).
    kalman_gain = model_inputs.kalman_gain_t  # (2M, 4)
    x_update_flat = pred_mean.reshape(-1) + kalman_gain @ (x_E_flat - mu_E)  # (2M,)
    # Overwrite observed teams with the sampled values.
    x_update_flat = x_update_flat.at[obs_indices_flat].set(x_E_flat)
    x_update = x_update_flat.reshape(num_teams, 2)  # (M, 2)
    return RBPFState(x=x_update)


def _log_potential(
    state_prev: RBPFState,
    state: RBPFState,
    model_inputs: RBPFFootballResults,
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
        init_sample=partial(
            init_sample,
            init_mean=params.mean_0,
        ),
        propagate_sample=partial(
            propagate_sample,
            mean=params.mean_0,
            gamma_Q=params.gamma_Q,
            B_Q=params.B_Q,
            num_teams=num_teams,
        ),
        log_potential=partial(
            _log_potential,
            alpha=params.alpha,
            beta=params.beta,
            max_goals=MAX_GOALS,
        ),
        n_filter_particles=n,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )
    return rbpf


@partial(jax.jit, static_argnames=("num_teams",))
def compute_covariance_trajectory(
    model_inputs: FootballResults,
    sigma_0: jnp.ndarray,
    gamma_Q: jnp.ndarray,
    B_Q: jnp.ndarray,
    num_teams: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Deterministic covariance trajectory for the random-walk model.

    The covariance evolution does not depend on the particle states, so we can
    compute the full ``2M x 2M`` trajectory with a single ``lax.scan``:

        Sigma_{t|t-1} = Sigma_{t-1|t-1} + dt_t * Q,   Q = gamma_Q (x) B_Q
        K_t           = Sigma_{t|t-1}[:, O] pinv(Sigma_{t|t-1}[O, O])
        Sigma_{t|t}   = Sigma_{t|t-1} - K_t Sigma_{t|t-1}[O, :]

    where ``O`` is the observed block (home + away teams, 4 dims). The observed
    teams' posterior rows/cols are zeroed (Schur-complement marginalization).

    Returns:
        (sigma_updated, sigma_pred, kalman_gain), each of shape
        ``(T, 2M, 2M)`` / ``(T, 2M, 4)``.
    """
    Q = jnp.kron(gamma_Q, B_Q)  # (2M, 2M)
    dim = 2 * num_teams

    def sigma_step(sigma_prev, model_input):
        dt = model_input.timestamp - model_input.timestamp_prev
        sigma_pred = sigma_prev + dt * Q
        sigma_pred = 0.5 * (sigma_pred + sigma_pred.T)  # ensure symmetry

        h = model_input.home_team_id
        a = model_input.away_team_id
        obs_indices_flat = jnp.array([2 * h, 2 * h + 1, 2 * a, 2 * a + 1])

        sigma_EE = sigma_pred[jnp.ix_(obs_indices_flat, obs_indices_flat)]
        sigma_EE = 0.5 * (sigma_EE + sigma_EE.T)
        sigma_RE = sigma_pred[:, obs_indices_flat]

        # Handles both positive-definite and structurally singular sigma_EE.
        K = sigma_RE @ jnp.linalg.pinv(sigma_EE)

        sigma_updated = sigma_pred - K @ sigma_RE.T
        sigma_updated = 0.5 * (sigma_updated + sigma_updated.T)

        obs_mask = jnp.zeros(dim, dtype=bool).at[obs_indices_flat].set(True)
        keep_mask = jnp.outer(~obs_mask, ~obs_mask)
        sigma_updated = sigma_updated * keep_mask

        return sigma_updated, (sigma_updated, sigma_pred, K)

    _, (sigma_updated, sigma_pred, kalman_gain) = jax.lax.scan(
        f=sigma_step, init=sigma_0, xs=model_inputs
    )
    return sigma_updated, sigma_pred, kalman_gain


@partial(jax.jit, static_argnames=("num_teams", "n_particles"))
def run_filter(
    key: jax.Array,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    num_teams: int,
    n_particles: int,
):
    rbpf = build_rbpf_filter(
        n=n_particles,
        params=params,
        num_teams=num_teams,
    )
    # Prepare the initial state from the first time step's model inputs.
    init_state = rbpf.init_prepare(
        jax.tree.map(lambda x: x[0], model_inputs), key=key,
    )
    # Filter over the remaining T steps (exclude the initial state at index 0).
    rest_inputs = jax.tree.map(lambda x: x[1:], model_inputs)
    filtered_states = cuthbert.filtering.filter(rbpf, rest_inputs, init_state, key=key)
    return filtered_states, model_inputs


def main():
    from rbpf.test.src.data import WORLDCUP_2026_TEAMS, ACTIVE_TEAMS
    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01",
        end_date="2026-01-01",
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=ACTIVE_TEAMS,
    )

    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(5))
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, team_id_to_name=team_id_to_name)

    print("Running filter (random-walk)...")
    key, filter_key = jax.random.split(key)
    sigma_updated, sigma_pred, kalman_gain = compute_covariance_trajectory(
        model_inputs=model_inputs,
        sigma_0=params.sigma_0,
        gamma_Q=params.gamma_Q,
        B_Q=params.B_Q,
        num_teams=NUM_TEAMS,
    )
    augmented_results = RBPFFootballResults(
        match_index_id=model_inputs.match_index_id,
        timestamp=model_inputs.timestamp,
        timestamp_prev=model_inputs.timestamp_prev,
        home_team_id=model_inputs.home_team_id,
        away_team_id=model_inputs.away_team_id,
        home_score=model_inputs.home_score,
        away_score=model_inputs.away_score,
        sigma_t=sigma_updated,
        sigma_pred_t=sigma_pred,
        kalman_gain_t=kalman_gain,
    )
    try:
        filtered_states, augmented_results = run_filter(
            key=filter_key,
            model_inputs=augmented_results,
            params=params,
            num_teams=NUM_TEAMS,
            n_particles=N,
        )
    except Exception as e:
        print("Error during filtering:", e)
        return
    print(f"\nResult shapes:")
    print(f"  observations:   {len(model_inputs.timestamp)}")
    print(f"  particles.x:     {filtered_states.particles.x.shape}")
    print(filtered_states.particles.x[-1][0])  # last time step's particle states
    print(f"  particles.sigma: {augmented_results.sigma_t.shape}")
    print(f"  log_weights:     {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_constant: {filtered_states.log_normalizing_constant.shape}")

    import os
    from rbpf.test.src.graphic import plot_all
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphic")
    plot_all(filtered_states, augmented_results, team_id_to_name, top_n=5, save_path=path)


if __name__ == "__main__":
    main()
