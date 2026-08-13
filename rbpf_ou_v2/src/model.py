import os
import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from functools import partial

from rbpf_ou_v2.src.bivariate_poisson import loglik
from rbpf_ou_v2.src.data import get_results, FootballResults
from rbpf_ou_v2.src.helpers import default_init_params, kron_sample_psd
from rbpf_ou_v2.src.utils import RBPFState, RBPFFootballResults, EMParams

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
    gamma_0: jnp.ndarray,
    B: jnp.ndarray,
) -> RBPFState:
    """Sample the initial state from the prior X_0 ~ N(init_mean, gamma_0 (x) B).

    ``cuthbert``'s ``init_prepare`` vmaps this over ``n_filter_particles`` keys,
    so each particle gets an independent draw from the prior. Without the
    dispersion the filter would start fully collapsed at the mean.
    """
    num_teams = init_mean.shape[0]
    K = init_mean.shape[1]  # 2 (attack/defence)
    x = kron_sample_psd(key, init_mean.flatten(), gamma_0, B).reshape(num_teams, K)
    return RBPFState(x=x)


def _sample_psd_gaussian(
    key: jax.Array,
    mean: jax.Array,
    covariance: jax.Array,
) -> jax.Array:
    """Sample from a PSD Gaussian via a differentiable Cholesky reparameterization.

    The filtered posterior covariances are positive-semidefinite, not
    positive-definite: observed teams have exact zero variance after the
    Schur-complement marginalization (their rows/cols are zeroed). A plain
    ``jax.random.multivariate_normal`` returns NaN on such singular matrices.

    We use the reparameterization trick ``x = mean + L @ noise`` with a
    Cholesky factor ``L`` of ``covariance + jitter*I``. The small diagonal
    jitter makes the matrix strictly positive-definite so the Cholesky gradient
    is finite (an eigendecomposition-based sampler has a NaN gradient at the
    zero-variance boundary). The jitter is small enough that zero-variance
    directions stay essentially at the mean, but the gradient through the
    covariance is preserved, which is what the direct-GD trainer needs to
    learn ``gamma_0``/``B``/``kappa``.
    """
    covariance = 0.5 * (covariance + covariance.T)
    n = covariance.shape[0]
    # Jitter to a strictly-PD matrix so the Cholesky gradient is finite.
    covariance = covariance + 1e-6 * jnp.eye(n)
    L = jnp.linalg.cholesky(covariance)
    noise = jax.random.normal(key, mean.shape)
    return mean + L @ noise


def propagate_sample(
    key: jax.Array,
    state: RBPFState,
    model_inputs: RBPFFootballResults,
    mean: jnp.ndarray,
    B: jnp.ndarray,
    kappa: float,
    num_teams: int,
):
    """OU (scalar-phi AR(1)) propagation with Rao-Blackwellized Kalman update.

    The OU transition has mean reversion: ``mu_{t|t-1} = mu + phi_t (X_{t-1} - mu)``
    with ``phi_t = exp(-kappa * dt)``. The prediction team covariance
    ``Gamma_{t|t-1}`` and the Kalman gain ``K_t`` are precomputed
    deterministically in ``compute_gamma_trajectory`` and carried in
    ``model_inputs``.

    We sample only the observed block (home + away teams, 4 dims) from the
    prediction Gaussian ``Sigma_EE = gamma_EE (x) B``, then condition the
    remaining (Rao-Blackwellized) teams on it via the Kalman gain in team space.
    """
    # 1. OU prediction: mean-revert toward mu with phi_t = exp(-kappa*dt).
    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi_t = jnp.exp(-kappa * dt)
    pred_mean = mean + phi_t * (state.x - mean)  # (M, 2)

    gamma_pred = model_inputs.gamma_pred_t  # (M, M)

    h = model_inputs.home_team_id
    a = model_inputs.away_team_id
    obs_indices = jnp.array([h, a])  # (2,)

    # 2. Sample the observed block (home + away, 4 dims).
    # mu_E: (2, 2) -> flatten to (4,)
    mu_E = pred_mean[obs_indices]  # (2, 2)
    gamma_EE = gamma_pred[jnp.ix_(obs_indices, obs_indices)]  # (2, 2)
    Sigma_EE = jnp.kron(gamma_EE, B)  # (4, 4)

    key, subkey = jax.random.split(key)
    # PSD-aware sampler: robust to the rare dt==0 re-observation case where
    # Sigma_EE is singular (a team's posterior variance was zeroed). Zero-
    # variance directions stay exactly at the mean.
    x_E_flat = _sample_psd_gaussian(subkey, mu_E.flatten(), Sigma_EE)  # (4,)
    x_E = x_E_flat.reshape(2, 2)  # (2, 2)

    # 3. Kalman update for ALL teams (Rao-Blackwellization) in team space.
    kalman_gain = model_inputs.kalman_gain_t  # (M, 2)
    x_update = pred_mean + kalman_gain @ (x_E - mu_E)  # (M, 2)
    # Overwrite observed teams with the sampled values.
    x_update = x_update.at[obs_indices].set(x_E)
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
            gamma_0=params.gamma_0,
            B=params.B,
        ),
        propagate_sample=partial(
            propagate_sample,
            mean=params.mean_0,
            B=params.B,
            kappa=params.kappa,
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
def compute_gamma_trajectory(
    model_inputs: FootballResults,
    gamma_0: jnp.ndarray,
    kappa: float,
    num_teams: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Deterministic team-covariance trajectory for the OU (scalar-phi AR(1)) model.

    The covariance evolution does not depend on the particle states, so we can
    compute the ``M x M`` team-covariance trajectory with a single ``lax.scan``.
    With the shared attack/defence factor ``B``, the OU prediction is a convex
    combination of the current posterior and the stationary covariance:

        phi_t = exp(-kappa * dt)
        Gamma_{t|t-1} = phi_t^2 Gamma_{t-1|t-1} + (1 - phi_t^2) gamma_0
        K_t           = Gamma_{t|t-1}[:, O] pinv(Gamma_{t|t-1}[O, O])
        Gamma_{t|t}   = Gamma_{t|t-1} - K_t Gamma_{t|t-1}[O, :]

    where ``O`` is the observed block (home + away teams, 2 dims in team space).
    The observed teams' posterior rows/cols are zeroed (Schur-complement
    marginalization).

    The convex combination is PD by construction (sum of two PD matrices with
    positive weights) and team-specific (heavily-observed teams have small
    posterior covariance, so their transition noise is small).

    Returns:
        (gamma_updated, gamma_pred, kalman_gain), each of shape
        ``(T, M, M)`` / ``(T, M, 2)``.
    """
    def gamma_step(gamma_prev, model_input):
        dt = model_input.timestamp - model_input.timestamp_prev
        phi_t = jnp.exp(-kappa * dt)
        # OU convex-combination prediction (PD by construction).
        gamma_pred = phi_t**2 * gamma_prev + (1.0 - phi_t**2) * gamma_0
        gamma_pred = 0.5 * (gamma_pred + gamma_pred.T)  # ensure symmetry

        h = model_input.home_team_id
        a = model_input.away_team_id
        obs_indices = jnp.array([h, a])  # (2,)

        gamma_EE = gamma_pred[jnp.ix_(obs_indices, obs_indices)]
        gamma_EE = 0.5 * (gamma_EE + gamma_EE.T)
        gamma_RE = gamma_pred[:, obs_indices]

        # Handles both positive-definite and structurally singular gamma_EE.
        # Jitter to a strictly-PD matrix so the pinv gradient is finite (the
        # direct-GD trainer backprops through this Kalman gain).
        gamma_EE = gamma_EE + 1e-6 * jnp.eye(2)
        K = gamma_RE @ jnp.linalg.pinv(gamma_EE)

        gamma_updated = gamma_pred - K @ gamma_RE.T
        gamma_updated = 0.5 * (gamma_updated + gamma_updated.T)

        obs_mask = jnp.zeros(num_teams, dtype=bool).at[obs_indices].set(True)
        keep_mask = jnp.outer(~obs_mask, ~obs_mask)
        gamma_updated = gamma_updated * keep_mask

        return gamma_updated, (gamma_updated, gamma_pred, K)

    _, (gamma_updated, gamma_pred, kalman_gain) = jax.lax.scan(
        f=gamma_step, init=gamma_0, xs=model_inputs
    )
    return gamma_updated, gamma_pred, kalman_gain


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
    from rbpf_ou_v2.src.data import WORLDCUP_2026_TEAMS, ACTIVE_TEAMS
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

    print("Running filter (OU)...")
    key, filter_key = jax.random.split(key)
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa,
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
        gamma_t=gamma_updated,
        gamma_pred_t=gamma_pred,
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
    print(f"  particles.gamma: {augmented_results.gamma_t.shape}")
    print(f"  log_weights:     {filtered_states.log_weights.shape}")
    print(f"  log_normalizing_constant: {filtered_states.log_normalizing_constant.shape}")

    import numpy as np
    from rbpf_ou_v2.src.graphic import plot_all
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphic")
    plot_all(filtered_states, augmented_results, team_id_to_name, top_n=5, save_path=path)

    # --- Save final filter states and full correlation matrix to outputs_gpu ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)

    x_final = np.asarray(filtered_states.particles.x[-1])  # (N, M, 2)
    final_mean = x_final.mean(axis=0)  # (M, 2)
    np.save(os.path.join(out_dir, "final_filter_states.npy"), final_mean)

    gamma_final = np.asarray(augmented_results.gamma_t[-1])  # (M, M)
    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)
    np.save(os.path.join(out_dir, "correlation_matrix.npy"), corr)

    team_names = [team_id_to_name[i] for i in range(NUM_TEAMS)]
    np.save(os.path.join(out_dir, "team_names.npy"), np.array(team_names))
    print(f"Saved final filter states and correlation matrix under {out_dir}/")


if __name__ == "__main__":
    main()
