from functools import partial
import os
import json

import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from cuthbertlib.resampling import autodiff
import optax
from datetime import datetime

from rbsqmc.src.utils.type import EMParams, FootballResults, RBPFFootballResults, RawEMParams
from rbsqmc.src.utils.helpers import (
    decode_EM_params,
    encode_EM_params,
    default_init_params,
    save_params,
    resolve_teams,
    log_inverse_wishart_kernel,
)
from rbsqmc.src.data.data import get_results, WORLDCUP_2026_TEAMS
from rbsqmc.src.model.model import init_sample, propagate_sample, _log_potential, compute_gamma_trajectory, generate_rbpf_trajectory

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def run_filter_unbiased(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
) -> tuple[jax.Array, RBPFFootballResults]:


    key, init_key, filter_key = jax.random.split(key, 3)

    gamma, gamma_pred, gamma_observed, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa,
        num_teams=params.mean_0.shape[0],
    )
    model_inputs_rbpf = generate_rbpf_trajectory(
        model_inputs=model_inputs,
        gamma=gamma,
        gamma_pred=gamma_pred,
        gamma_observed=gamma_observed,
        kalman_gain=kalman_gain
    )
    rbpf = cuthbert.smc.particle_filter.build_filter(
        init_sample=partial(
            init_sample,
            mean_0=params.mean_0,
            # gamma_0=params.gamma_0,
            # B=params.B
        ),
        propagate_sample=partial(
            propagate_sample,
            mean=params.mean_0,
            B=params.B,
            kappa=params.kappa,
        ),
        log_potential=partial(
            _log_potential,
            alpha=params.alpha,
            beta=params.beta,
            max_goals=max_goals,
        ),
        n_filter_particles=n_particles,
        resampling_fn=autodiff.stop_gradient_decorator(
            cuthbertlib.resampling.systematic.resampling
        ),
    )

    # prepare initial state for the filter. use first value to prepare, but pass the full model_inputs_rbpf to the filter, which will handle the time steps and propagate the state accordingly.
    init_state = rbpf.init_prepare(model_inputs=jax.tree.map(lambda x: x[0], model_inputs_rbpf),
        key=init_key,
    )
    # since init_state is only used for preparing the filter, we pass the full model_inputs_rbpf to the filter, which will handle the time steps and propagate the state accordingly.
    filtered_states = cuthbert.filtering.filter(
        filter_obj=rbpf,
        model_inputs=model_inputs_rbpf,
        init_state=init_state,
        key=filter_key,
    )
    return filtered_states, model_inputs_rbpf

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_fn(
    keys : jax.Array,
    model_inputs: FootballResults,
    params : EMParams,
    n_particles : int,
    max_goals: int,
    # gamma_0_prior: jax.Array | None = None,
    # gamma_prior_dof: float = 5.0,
    # gamma_prior_strength: float = 1.0,
):
    """Negative log marginal likelihood, averaged over ``n_reps`` filter replicas.

    Args:
        keys: (n_reps, 2) array of independent PRNG keys.
        model_inputs: FootballResults (raw inputs; RBPF trajectory built inside
            ``run_filter_unbiased``).
        params: EMParams to score.
        # gamma_0_prior: optional scale matrix for an inverse-Wishart prior on
        #     ``gamma_0``. If given, a regularizer ``-strength * log p(gamma_0)``
        #     is added to the loss, pulling ``gamma_0`` toward ``gamma_0_prior``
        #     (which widens team separation if its scale is larger).
        # gamma_prior_dof: inverse-Wishart degrees of freedom.
        # gamma_prior_strength: multiplicative strength of the prior term.

    Returns:
        ``-mean(log Z)`` plus the (optional) gamma_0 prior regularizer.
    """
    logz = jax.vmap(
        lambda k: run_filter_unbiased(
            k, model_inputs, params, n_particles, max_goals
        )[0].log_normalizing_constant[-1]
    )(keys)
    loss = -jnp.mean(logz)
    # if gamma_0_prior is not None:
    #     loss = loss - gamma_prior_strength * log_inverse_wishart_kernel(
    #         params.gamma_0, gamma_0_prior, gamma_prior_dof
    #     )
    return loss

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_and_grad_raw(
        keys: jax.Array,
        model_inputs: FootballResults,
        raw_params: RawEMParams,
        fixed_mean_0: jax.Array,
        n_particles: int,
        max_goals: int,
        # gamma_0_prior: jax.Array | None = None,
        # gamma_prior_dof: float = 5.0,
        # gamma_prior_strength: float = 1.0,
):
    """Value and gradient of ``loss_fn`` w.r.t. unconstrained raw params.

    ``mean_0`` is fixed (not optimized): the decoded ``EMParams`` always uses
    ``fixed_mean_0``. The gradient flows through ``decode_EM_params`` so it is
    taken with respect to ``raw_params``, which is what the optimizer updates.
    """
    def _loss(raw):
        params = decode_EM_params(raw, fixed_mean_0=fixed_mean_0)
        return loss_fn(
            keys=keys, 
            model_inputs=model_inputs, 
            params=params, 
            n_particles=n_particles, 
            max_goals=max_goals,
            # gamma_0_prior, 
            # gamma_prior_dof, 
            # gamma_prior_strength,
        )
    return jax.value_and_grad(_loss)(raw_params)

def logmarginal_maximize(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    max_goals: int,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    n_reps: int = 4,
    # gamma_0_prior: jax.Array | None = None,
    # gamma_prior_dof: float = 5.0,
    # gamma_prior_strength: float = 1.0,
):
    """Maximize the log marginal likelihood ``log Z`` with Adam (optax).

    ``mean_0`` is held fixed; the remaining identified parameters
    (``gamma_0``, ``B``, ``kappa``, ``alpha``, ``beta``) are optimized in the
    unconstrained ``RawEMParams`` space via ``encode``/``decode_EM_params``.
    Gradients are the unbiased stop-gradient (Fisher) estimates from
    ``run_filter_unbiased``. The learning rate is cosine-annealed from
    ``learning_rate`` to 0 over ``n_epochs``.

    Args:
        gamma_0_prior: optional inverse-Wishart scale for a ``gamma_0`` prior.
            If None, defaults to the initial ``params.gamma_0`` (shrinkage
            toward the starting value).
        gamma_prior_dof: inverse-Wishart degrees of freedom.
        gamma_prior_strength: multiplicative strength of the prior term.

    Returns:
        (best_params, history) where ``best_params`` is the decoded ``EMParams``
        achieving the lowest loss over training, and ``history`` is the
        (regularized) negative loss per epoch.
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)
    # if gamma_0_prior is None:
    #     gamma_0_prior = params.gamma_0

    # Cosine-anneal the learning rate from `learning_rate` down to 0 over the
    # training run. This lets Adam take large early steps and shrink to small
    # refinements near the end, improving convergence (matches smoothing_v2).
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_epochs,
    )
    # Use Adam to optimize the unconstrained raw parameters. The gradient is
    optimizer = optax.adam(schedule)
    # Initialize the optimizer state with the raw parameters. The optimizer will
    opt_state = optimizer.init(raw_params)

    import time as _time
    logz_history = []
    best_logz = -jnp.inf
    best_params = params
    epoch_times = []  # seconds per epoch, for ETA estimation
    total_start = _time.perf_counter()

    for epoch in range(n_epochs):
        epoch_start = _time.perf_counter()
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_reps)

        loss, grads = loss_and_grad_raw(
            keys=keys,
            model_inputs=model_inputs,
            raw_params=raw_params,
            fixed_mean_0=fixed_mean_0,
            n_particles=n_particles,
            max_goals=max_goals,
            # gamma_0_prior=gamma_0_prior,
            # gamma_prior_dof=gamma_prior_dof,
            # gamma_prior_strength=gamma_prior_strength,
        )
        loss = float(loss)
        logz = -loss
        logz_history.append(logz)

        if logz > best_logz:
            best_logz = logz
            best_params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)

        # Update parameters with Adam
        updates, opt_state = optimizer.update(grads, opt_state)
        # Apply updates to raw_params (unconstrained space)
        raw_params = optax.apply_updates(raw_params, updates)

        epoch_secs = _time.perf_counter() - epoch_start
        epoch_times.append(epoch_secs)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            elapsed = _time.perf_counter() - total_start
            avg_sec = sum(epoch_times) / len(epoch_times)
            remaining = avg_sec * (n_epochs - (epoch + 1))
            print(
                f"[epoch {epoch:4d}] logZ = {logz:.4f}  (best {best_logz:.4f})  "
                f"[{epoch_secs:6.1f}s this epoch, {elapsed:6.1f}s elapsed, "
                f"ETA {remaining:6.1f}s]",
                flush=True,
            )

    total_sec = _time.perf_counter() - total_start
    print(f"[optimization] finished {n_epochs} epochs in {total_sec:.1f}s "
          f"(avg {total_sec / max(n_epochs, 1):.1f}s/epoch)", flush=True)
    return best_params, jnp.asarray(logz_history)

def main():
    date_text = datetime.now().strftime("gd_%Y%m%d_%H%M%S")
    output_dir = f"rbpf/outputs/{date_text}/"

    cfg = {
        "start_date": "1900-01-01",
        "end_date": "2026-01-01",
        "n_particles": 50,          # N
        "max_goals": 8,               # MAX_GOALS
        "seed": 0,                    # PRNG seed
        # optimization
        "n_epochs": 200,
        "learning_rate": 1e-3,
        "n_reps": 10,
        # data / output
        "include_friendly": False,
        "teams": "worldcup2026",
        "output_dir": output_dir,
    }
    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)
    data, model_inputs, team_id_to_name = get_results(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
    )
    num_teams = len(team_id_to_name)
    print(f"Extracted data from {cfg['start_date']} to {cfg['end_date']}, with {num_teams} teams and {len(data)} dates.")
    print("Number of teams:", num_teams)

    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)

    ############ FILTER ################
    key, filter_key = jax.random.split(key, 2)
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )
    baseline_logz = float(filtered_states.log_normalizing_constant[-1])


    ############# LOG MARGINALIZATION OPTIMIZATION ################
    key, opt_key = jax.random.split(key, 2)
    best_params, logz_history = logmarginal_maximize(
        key=opt_key,
        params=params,
        model_inputs=model_inputs,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        # gamma_0_prior=gamma_0_prior,
        # gamma_prior_dof=cfg.get("gamma_prior_dof", 5.0),
        # gamma_prior_strength=cfg.get("gamma_prior_strength", 1.0),
    )

    ############## FILTER WITH BEST PARAMETERS ################
    key, final_filter_key = jax.random.split(key, 2)
    final_states, final_model_inputs_rbpf = run_filter_unbiased(
        key=final_filter_key,
        model_inputs=model_inputs,
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    ############## SAVE RESULTS ################
    os.makedirs(cfg["output_dir"], exist_ok=True)

    from rbsqmc.src.utils.graphic import (
        # save_filter_states,
        plot_all,
        plot_log_marginal_likelihood_curve,
    )
    plot_all(
        filtered_states=final_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=cfg["output_dir"],
        timestamps=data["date"].to_numpy(),
        params=best_params,
    )
    plot_log_marginal_likelihood_curve(
        logz_history,
        save_path=os.path.join(cfg["output_dir"], "optimization_logZ_curve.png"),
    )
    print(f"Smoothing Optimization completed. Saved results to {cfg['output_dir']}")
if __name__ == "__main__":
    main()