from rbpf.src.model import MAX_GOALS, compute_gamma_trajectory, run_filter
from rbpf.src.utils import EMParams, RBPFState, RBPFFootballResults, FootballResults
from rbpf.src.helpers import default_init_params, generate_rbpf_trajectory, save_em_results
from rbpf.src.bivariate_poisson import loglik
from rbpf.src.helpers import _cholesky_from_psd, _psd_from_cholesky
from rbpf.src.graphic import plot_log_marginal_likelihood_curve, plot_loss_components

import jax
import cuthbert
import cuthbertlib
import jax.numpy as jnp
from functools import partial
import optax
from cuthbert.smc.backward_sampler import build_smoother
from cuthbertlib.smc.smoothing.exact_sampling import simulate as exact_sampling_simulate

#################### E-step: RBPF filtering and smoothing ###################

def _kron_quad(A, B, V):
    """v_i^T (A (x) B)^{-1} v_i for each row v_i of V.

    V: (N, M*K) where each row is vec_C(S_i) of an (M, K) matrix S_i.
    A: (M, M), B: (K, K). Returns (N,).
    """
    K = B.shape[0]
    N = V.shape[0]
    S = V.reshape(N, -1, K)  # (N, M, K)
    B_inv = jnp.linalg.inv(B)  # (K, K), K is tiny
    A_inv_S = jnp.linalg.solve(A, S)  # (N, M, K)  == A^{-1} S
    St_Ainv_S = jnp.matmul(S.transpose(0, 2, 1), A_inv_S)  # (N, K, K)
    # tr(S^T A^{-1} S B^{-T}) = sum((S^T A^{-1} S) * B^{-1})
    return jnp.sum(St_Ainv_S * B_inv[None], axis=(-2, -1))


def _kron_logdet(A, B):
    """logdet(A (x) B) = K logdet(A) + M logdet(B). A: (M, M), B: (K, K)."""
    M = A.shape[0]
    K = B.shape[0]
    _, logdet_A = jnp.linalg.slogdet(A)
    _, logdet_B = jnp.linalg.slogdet(B)
    return K * logdet_A + M * logdet_B

def _log_potential_terms(
    x_prev: jax.Array,
    x: jax.Array,
    model_inputs: RBPFFootballResults,
    params: EMParams,
):
    """Decompose ``log_potential`` into its two terms.

    Returns ``(log_transition, log_obs)`` where:
      - ``log_transition`` = log p(state | state_prev) (OU transition density)
      - ``log_obs`` = log G_t(y_t | state) (bivariate Poisson observation term)
    """
    # --- transition term: log p(state | state_prev) ---
    # log OU process transition density
    num_teams = x.shape[0]
    dim = num_teams * 2 # 2M

    dt = model_inputs.timestamp - model_inputs.timestamp_prev
    phi = jnp.exp(-params.kappa * dt)
    pred_mean = params.mean_0 + phi * (x_prev - params.mean_0)   # (M,2)

    scale = jnp.maximum(1.0 - phi**2, 1e-8)
    diff = x - pred_mean                                        # (M,2)

    quad = _kron_quad(scale * params.gamma_0, params.B, diff.reshape(1, -1))[0]
    log_det = _kron_logdet(scale * params.gamma_0, params.B)
    log_transition = jax.lax.cond(
        dt <= 1e-8, # if dt is hit lower limit i.e. same day, then log transition is 0.0, else compute log transition
        lambda _: 0.0,
        lambda _: -0.5*quad - 0.5*log_det - 0.5*dim*jnp.log(2*jnp.pi),
        operand=None,
    )
    # --- observation term: log G_t(y_t | state) ---
    y = jnp.array([model_inputs.home_score, model_inputs.away_score])  # (2,) scalars
    x_i = x[model_inputs.home_team_id]   # (2,) attack/def of home team
    x_j = x[model_inputs.away_team_id]   # (2,) attack/def of away team
    log_obs = loglik(
        y=y,
        x_i=x_i,
        x_j=x_j,
        alpha=params.alpha,
        beta=params.beta,
        max_goals=MAX_GOALS,
        scale=1.0,
    )

    return log_transition, log_obs   # (scalar, scalar)


def log_potential(
    x_prev: jax.Array,
    x: jax.Array,
    model_inputs: RBPFFootballResults,
    params: EMParams
) -> jax.Array:
    log_transition, log_obs = _log_potential_terms(x_prev, x, model_inputs, params)
    return log_transition + log_obs   # scalar

def E_step(
    params: EMParams,
    model_inputs: FootballResults,
    n_particles: int,
    n_smoother_particles: int,
    num_teams: int,
    key: jax.Array,
):
    key, filter_key = jax.random.split(key)
    # initialize covariance trajectory
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa
    )

    model_inputs_rbpf = generate_rbpf_trajectory(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain
    )
    filtered_states, _ = run_filter(
        key=filter_key,
        model_inputs=model_inputs_rbpf,
        params=params,
        num_teams=num_teams,
        n_particles=n_particles,
    )
    
    smoother_obj = build_smoother(
        log_potential=partial(
            lambda sp, s, mi, p: log_potential(sp.x, s.x, mi, p),
            p=params,
        ),
        backward_sampling_fn=exact_sampling_simulate,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
        n_smoother_particles=n_smoother_particles,
    )
    smoothed_states = cuthbert.smoother(smoother_obj, filtered_states, model_inputs_rbpf, False, key)
    return smoothed_states, filtered_states, model_inputs_rbpf

################### M-step: EM parameter estimation ###################

def _inverse_wishart_log_prior(A, nu, S):
    """
    log p(A) for A ~ IW(nu, S), up to a constant (dropped for MAP).
    https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/WishartTriL
    check in the future for implementation.
    """
    
    d = A.shape[0]
    _, logdet = jnp.linalg.slogdet(A)
    # -0.5*(nu + d + 1)*log|A| - 0.5*tr(S A^{-1})
    return -0.5 * (nu + d + 1) * logdet - 0.5 * jnp.trace(S @ jnp.linalg.inv(A))

def _complete_log_likelihood_terms(params, X, model_inputs):
    """Term-by-term decomposition of the complete-data log likelihood.

    ``X`` has shape ``(T, num_teams, 2)`` — one smoothed trajectory.

    Returns a tuple ``(init_ll, trans_ll, obs_ll)`` where:
      - ``init_ll`` = log p(X_0 | mean_0, gamma_0 (x) B)  (initial-state term)
      - ``trans_ll`` = sum_t log p(X_t | X_{t-1})         (OU transition term)
      - ``obs_ll``   = sum_t log G_t(y_t | X_t)           (observation term)
    """
    n_obs = X.shape[0]
    num_teams = X.shape[1]
    dim = num_teams * 2

    # init term: log p(X_0 | mean_0, gamma_0 (x) B)
    diff0 = (X[0] - params.mean_0).reshape(-1)
    Sigma0 = jnp.kron(params.gamma_0, params.B)
    quad0 = diff0 @ jnp.linalg.solve(Sigma0, diff0)
    _, log_det0 = jnp.linalg.slogdet(Sigma0)
    init_ll = -0.5*quad0 - 0.5*log_det0 - 0.5*dim*jnp.log(2*jnp.pi)

    # sum log_potential over all transitions t=1..T-1, keeping terms separate
    def step(t):
        # slice model_inputs to a single time step (scalars)
        mi = jax.tree.map(lambda a: a[t], model_inputs)
        return _log_potential_terms(X[t-1], X[t], mi, params)
    trans_ll, obs_ll = jax.vmap(step)(jnp.arange(1, n_obs))
    trans_ll = jnp.sum(trans_ll)
    obs_ll = jnp.sum(obs_ll)

    return init_ll, trans_ll, obs_ll


# def _complete_log_likelihood(params, X, model_inputs):
#     init_ll, trans_ll, obs_ll = _complete_log_likelihood_terms(params, X, model_inputs)
#     # total loss
#     return init_ll + trans_ll + obs_ll

def M_step(
    smoothed_states: jax.Array,
    model_inputs: RBPFFootballResults,
    prev_params: EMParams,
    learning_rate: float = 1e-3,
    n_gradient_steps: int = 1,
    # nu_gamma: float | None = None,
    # nu_B: float | None = None,
    # S_gamma: jax.Array | None = None,
    # S_B: jax.Array | None = None,
):
    """M-step: maximize the MAP objective (complete-data LL + inverse-Wishart priors).

    ``nu_gamma``/``nu_B`` are the inverse-Wishart degrees of freedom and
    ``S_gamma``/``S_B`` the scale matrices for the priors on ``gamma_0`` and
    ``B``. These are fixed hyperparameters (NOT optimized). If ``None``, they
    default to a weak prior centered on the previous parameters.
    """
    # smoothed_states.particles.x: (T+1, M, num_teams, 2) -> (M, T+1, num_teams, 2)
    smoothed_trajectories = smoothed_states.particles.x.transpose(1, 0, 2, 3)
    num_teams = smoothed_trajectories.shape[2]

    # --- default hyperparameters: weak prior centered on prev params ---
    # if nu_gamma is None:
    #     nu_gamma = num_teams + 2
    # if nu_B is None:
    #     nu_B = 4
    # if S_gamma is None:
    #     S_gamma = nu_gamma * prev_params.gamma_0
    # if S_B is None:
    #     S_B = nu_B * prev_params.B

    def _loss_and_components(carry):
        gamma_0 = _psd_from_cholesky(carry["L_gamma0"], num_teams)
        B = _psd_from_cholesky(carry["L_B"], 2)
        params = EMParams(mean_0=prev_params.mean_0, gamma_0=gamma_0, B=B,
                        kappa=carry["kappa"], alpha=carry["alpha"], beta=carry["beta"])
        init_ll, trans_ll, obs_ll = jax.vmap(
            lambda X: _complete_log_likelihood_terms(params, X, model_inputs)
        )(jax.lax.stop_gradient(smoothed_trajectories))

        init_mean = jnp.mean(init_ll)
        trans_mean = jnp.mean(trans_ll)
        obs_mean = jnp.mean(obs_ll)

        # from scipy.stats import invwishart
        # prior_gamma = invwishart.logpdf(gamma_0, df=nu_gamma, scale=S_gamma)
        # prior_B = invwishart.logpdf(B, df=nu_B, scale=S_B)
        # --- inverse-Wishart priors on the covariance factors (MAP) ---
        prior_gamma = _inverse_wishart_log_prior(gamma_0, nu_gamma, S_gamma)
        prior_B = _inverse_wishart_log_prior(B, nu_B, S_B)
        prior = prior_gamma + prior_B

        total = init_mean + trans_mean + obs_mean + prior
        return total, (init_mean, trans_mean, obs_mean, prior)

    def _total_loss(carry):
        total, _ = _loss_and_components(carry)
        return -total  # minimize negative complete log-likelihood

    value_and_grad_fn = jax.jit(jax.value_and_grad(_total_loss, argnums=0))

    # Initial parameter blocks (Cholesky-parameterized so gamma_0, B stay PD).
    carry = {
        "L_gamma0": _cholesky_from_psd(prev_params.gamma_0, num_teams),
        "L_B": _cholesky_from_psd(prev_params.B, 2),
        "kappa": prev_params.kappa,
        "alpha": prev_params.alpha,
        "beta": prev_params.beta,
    }
    # create the chain for optimization
    # 1. clip gradients to avoid exploding gradients
    # 2. use Adam optimizer
    # 3. cosine LR schedule - multiplies by cosine-decaying factor (1.0 -> 0.0) over n_gradient_steps
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), # 1
        optax.adam(learning_rate), # 2
        optax.scale_by_schedule( # 3
            optax.cosine_decay_schedule(1.0, n_gradient_steps)
        ),
    )
    # init with params
    opt_state = optimizer.init(carry)

    # per step function
    def _step(carry, _):
        opt_state, params_carry = carry
        # calculate loss and gradients
        loss, grads = value_and_grad_fn(params_carry)
        # component breakdown at the current (pre-update) parameters
        _, (init_mean, trans_mean, obs_mean, prior) = _loss_and_components(params_carry)
        # update optimizer state and params
        updates, opt_state = optimizer.update(grads, opt_state, params_carry)
        # apply updates to the next set of parameters
        params_carry = optax.apply_updates(params_carry, updates)
        return (opt_state, params_carry), (loss, init_mean, trans_mean, obs_mean, prior)

    # run the optimization loop for n_gradient_steps
    (_, best_carry), loss_trace = jax.lax.scan(
        _step, (opt_state, carry), jnp.arange(n_gradient_steps)
    )
    # loss_trace: (total, init, transition, observation, prior), each (n_gradient_steps,)
    loss_trace = jnp.stack(loss_trace, axis=1)  # (n_gradient_steps, 5)

    final_params = EMParams(
        mean_0=prev_params.mean_0,
        gamma_0=_psd_from_cholesky(best_carry["L_gamma0"], num_teams),
        B=_psd_from_cholesky(best_carry["L_B"], 2),
        kappa=best_carry["kappa"],
        alpha=best_carry["alpha"],
        beta=best_carry["beta"],
    )
    return final_params, loss_trace

@partial(jax.jit, static_argnames=("n_smoother_particles",))
def _eval_loss_components(
    smoothed_states: jax.Array,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    n_smoother_particles: int,
    nu_gamma: float | None = None,
    nu_B: float | None = None,
    S_gamma: jax.Array | None = None,
    S_B: jax.Array | None = None,
):
    """Mean complete-data log-likelihood component scores over smoothed paths.

    ``smoothed_states.particles.x`` has shape ``(T+1, M, num_teams, 2)`` and is
    rearranged to ``(M, T+1, num_teams, 2)`` so each row is one smoothed
    trajectory. Returns the mean (over M smoother particles) of each term:
    ``(init_mean, trans_mean, obs_mean)`` plus the inverse-Wishart prior score.
    """
    trajectories = smoothed_states.particles.x.transpose(1, 0, 2, 3)
    init_ll, trans_ll, obs_ll = jax.vmap(
        lambda X: _complete_log_likelihood_terms(params, X, model_inputs)
    )(jax.lax.stop_gradient(trajectories))

    # --- inverse-Wishart prior score (defaults match M_step) ---
    num_teams = params.gamma_0.shape[0]
    if nu_gamma is None:
        nu_gamma = num_teams + 2
    if nu_B is None:
        nu_B = 4
    if S_gamma is None:
        S_gamma = nu_gamma * params.gamma_0
    if S_B is None:
        S_B = nu_B * params.B
    prior = _inverse_wishart_log_prior(params.gamma_0, nu_gamma, S_gamma) + \
        _inverse_wishart_log_prior(params.B, nu_B, S_B)

    return (
        jnp.mean(init_ll),
        jnp.mean(trans_ll),
        jnp.mean(obs_ll),
        prior,
    )

def run_EM(
    n_epochs: int,
    init_params: EMParams,
    model_inputs: FootballResults,
    n_particles: int,
    n_smoother_particles: int,
    num_teams: int,
    learning_rate: float = 1e-3,
    n_gradient_steps: int = 1,
    key: jax.Array = jax.random.PRNGKey(0),
    nu_gamma: float | None = None,
    nu_B: float | None = None,
    S_gamma: jax.Array | None = None,
    S_B: jax.Array | None = None,
):
    """
    
    Diagnostics
    1. Plot log-likelihood over epochs
    2. 
    """
    params = init_params

    best_log_marginal = -jnp.inf
    best_params = params

    # init params_track: add a leading axis to each param
    params_track = jax.tree.map(lambda x: x[None], params)

    log_marginal_likelihoods = []
    # per-epoch complete-data loss components (init / transition / obs / prior)
    loss_components = {"init": [], "transition": [], "observation": [], "prior": []}
    # per-gradient-step loss trace for each epoch: (n_epochs, n_gradient_steps, 5)
    # columns = [total, init, transition, observation, prior]
    loss_traces = []

    for epoch in range(n_epochs):
        key, e_key = jax.random.split(key)

        # --- E-step: filter + smoother ---
        smoothed_states, filtered_states, model_inputs_rbpf = E_step(
            params=params,
            model_inputs=model_inputs,
            n_particles=n_particles,
            num_teams=num_teams,
            n_smoother_particles=n_smoother_particles,
            key=e_key,
        )
        log_marginal_likelihoods.append(
            filtered_states.log_normalizing_constant[-1]
        )

        # --- diagnostic: loss contribution of each term in the complete-data LL ---
        init_mean, trans_mean, obs_mean, prior = _eval_loss_components(
            smoothed_states,
            model_inputs_rbpf,
            params,
            n_smoother_particles,
            nu_gamma=nu_gamma,
            nu_B=nu_B,
            S_gamma=S_gamma,
            S_B=S_B,
        )
        loss_components["init"].append(init_mean)
        loss_components["transition"].append(trans_mean)
        loss_components["observation"].append(obs_mean)
        loss_components["prior"].append(prior)

        # --- M-step: update params ---
        new_params, loss_trace = M_step(
            smoothed_states=smoothed_states,
            model_inputs=model_inputs_rbpf,
            prev_params=params,
            learning_rate=learning_rate,
            n_gradient_steps=n_gradient_steps,
            nu_gamma=nu_gamma,
            nu_B=nu_B,
            S_gamma=S_gamma,
            S_B=S_B,
        )
        loss_traces.append(loss_trace)

        # --- track params: concatenate new params along leading axis ---
        params_track = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[None]], axis=0),
            params_track,
            new_params,
        )

        # --- carry the updated params forward to the next E-step ---
        params = new_params

        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"log_marginal = {log_marginal_likelihoods[-1]:.4f} | "
              f"complete-LL init = {init_mean:.4f}, "
              f"transition = {trans_mean:.4f}, "
              f"observation = {obs_mean:.4f}, "
              f"prior = {prior:.4f}")
        if log_marginal_likelihoods[-1] > best_log_marginal:
            best_log_marginal = log_marginal_likelihoods[-1]
            best_params = new_params

    loss_components = jax.tree.map(lambda v: jnp.array(v), loss_components)
    loss_traces = jnp.stack(loss_traces, axis=0)  # (n_epochs, n_gradient_steps, 5)
    return (
        best_params,
        params_track,
        jnp.array(log_marginal_likelihoods),
        loss_components,
        loss_traces,
    )


def main():
    # --- Load data ---
    from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
    data, model_inputs, team_id_to_name = get_results(
        start_date="2000-01-01",
        end_date="2026-01-01",
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(3))
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(3))
    num_teams = len(team_id_to_name)
    key = jax.random.PRNGKey(42)

    params = default_init_params(num_teams=num_teams, team_id_to_name=None)

    # --- run configuration (single source of truth, also written to metadata) ---
    n_epochs = 20
    n_particles = 100
    n_smoother_particles = 100
    learning_rate = 1e-3
    n_gradient_steps = 1
    nu_gamma = num_teams + 2
    nu_B = 4

    best_params, params_track, log_marginal_likelihoods, loss_components, loss_traces = run_EM(
        n_epochs=n_epochs,
        init_params=params,
        model_inputs=model_inputs,
        n_particles=n_particles,
        n_smoother_particles=n_smoother_particles,
        num_teams=num_teams,
        learning_rate=learning_rate,
        n_gradient_steps=n_gradient_steps,
        key=key,
        # inverse-Wishart prior hyperparameters (fixed, not optimized)
        nu_gamma=nu_gamma,
        nu_B=nu_B,
    )

    # --- Save results and plot diagnostics ---
    import os
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "smoothing"
    )
    save_em_results(
        best_params=best_params,
        log_marginal_likelihoods=log_marginal_likelihoods,
        output_dir=output_dir,
        extra={
            "n_epochs": n_epochs,
            "n_particles": n_particles,
            "n_smoother_particles": n_smoother_particles,
            "num_teams": num_teams,
            "learning_rate": learning_rate,
            "n_gradient_steps": n_gradient_steps,
            "nu_gamma": nu_gamma,
            "nu_B": nu_B,
            # per-epoch complete-data loss contribution of each term
            "loss_components": {
                k: jnp.asarray(v).tolist() for k, v in loss_components.items()
            },
            # per-gradient-step loss trace per epoch: (n_epochs, n_gradient_steps, 5)
            # columns = [total, init, transition, observation, prior]
            "loss_traces": jnp.asarray(loss_traces).tolist(),
        },
    )
    plot_log_marginal_likelihood_curve(
        log_marginal_likelihoods,
        save_path=os.path.join(output_dir, "em_log_marginal_likelihood_curve.png"),
    )
    plot_loss_components(
        loss_traces,
        save_path=os.path.join(output_dir, "em_loss_components_curve.png"),
    )
    print("Done.")
if __name__ == "__main__":
    main()