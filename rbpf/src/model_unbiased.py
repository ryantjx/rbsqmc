

from functools import partial
import os
import json

import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from cuthbertlib.resampling import autodiff
import optax

from rbpf.src.utils import EMParams, FootballResults, RBPFFootballResults, RawEMParams
from rbpf.src.helpers import decode_EM_params, encode_EM_params, default_init_params, save_params, resolve_teams
from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS, ACTIVE_TEAMS
from rbpf.src.model import init_sample, propagate_sample, _log_potential, compute_gamma_trajectory, generate_rbpf_trajectory

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

# @partial(jax.jit, static_argnames=("n_particles", "max_goals"))
# def neg_log_marginal_likelihood(key, model_inputs, params, n_particles, max_goals):
#     filtered_states, _ = run_filter_unbiased(
#         key, model_inputs, params, n_particles, max_goals
#     )
#     return -filtered_states.log_normalizing_constant[-1]

@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_fn(
    keys : jax.Array,
    model_inputs: FootballResults,
    params : EMParams,
    n_particles : int,
    max_goals: int,
):
    """Negative log marginal likelihood, averaged over ``n_reps`` filter replicas.

    Args:
        keys: (n_reps, 2) array of independent PRNG keys.
        model_inputs: FootballResults (raw inputs; RBPF trajectory built inside
            ``run_filter_unbiased``).
        params: EMParams to score.

    Returns:
        ``-mean(log Z)`` where ``log Z`` is the filter's accumulated log
        marginal likelihood across replicas.
    """
    logz = jax.vmap(
        lambda k: run_filter_unbiased(
            k, model_inputs, params, n_particles, max_goals
        )[0].log_normalizing_constant[-1]
    )(keys)
    return -jnp.mean(logz)


@partial(jax.jit, static_argnames=("n_particles", "max_goals"))
def loss_and_grad_raw(
        keys: jax.Array,
        model_inputs: FootballResults,
        raw_params: RawEMParams,
        fixed_mean_0: jax.Array,
        n_particles: int,
        max_goals: int,
):
    """Value and gradient of ``loss_fn`` w.r.t. unconstrained raw params.

    ``mean_0`` is fixed (not optimized): the decoded ``EMParams`` always uses
    ``fixed_mean_0``. The gradient flows through ``decode_EM_params`` so it is
    taken with respect to ``raw_params``, which is what the optimizer updates.
    """
    def _loss(raw):
        params = decode_EM_params(raw, fixed_mean_0=fixed_mean_0)
        return loss_fn(keys, model_inputs, params, n_particles, max_goals)
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
):
    """Maximize the log marginal likelihood ``log Z`` with Adam (optax).

    ``mean_0`` is held fixed; the remaining identified parameters
    (``gamma_0``, ``B``, ``kappa``, ``alpha``, ``beta``) are optimized in the
    unconstrained ``RawEMParams`` space via ``encode``/``decode_EM_params``.
    Gradients are the unbiased stop-gradient (Fisher) estimates from
    ``run_filter_unbiased``. The learning rate is cosine-annealed from
    ``learning_rate`` to 0 over ``n_epochs``.

    Returns:
        (best_params, history) where ``best_params`` is the decoded ``EMParams``
        achieving the highest ``log Z`` over training, and ``history`` is the
        ``log Z`` per epoch.
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)

    # Cosine-anneal the learning rate from `learning_rate` down to 0 over the
    # training run. This lets Adam take large early steps and shrink to small
    # refinements near the end, improving convergence (matches smoothing_v2).
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_epochs,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(raw_params)

    history = []
    best_logz = -jnp.inf
    best_params = params

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_reps)

        loss, grads = loss_and_grad_raw(
            keys=keys,
            model_inputs=model_inputs,
            raw_params=raw_params,
            fixed_mean_0=fixed_mean_0,
            n_particles=n_particles,
            max_goals=max_goals,
        )
        loss = float(loss)
        logz = -loss
        history.append(logz)

        if logz > best_logz:
            best_logz = logz
            best_params = decode_EM_params(raw_params, fixed_mean_0=fixed_mean_0)

        updates, opt_state = optimizer.update(grads, opt_state)
        raw_params = optax.apply_updates(raw_params, updates)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[epoch {epoch:4d}] logZ = {logz:.4f}  (best {best_logz:.4f})", flush=True)

    return best_params, jnp.asarray(history)

def main():
    ######################
    # --- Config ---
    # Defaults; overridden by a config file when RBSQMC_CONFIG is set
    # (e.g. by the Colab GPU bootstrap).
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
        "output_dir": "rbpf/outputs/filter_unbiased",
    }
    import os as _os
    _cfg_path = _os.environ.get("RBSQMC_CONFIG")
    if _cfg_path:
        with open(_cfg_path) as _f:
            _cfg_override = json.load(_f)
        cfg.update(_cfg_override)
    #####################
    start_date = cfg["start_date"]
    end_date = cfg["end_date"]
    N = cfg["n_particles"]
    MAX_GOALS = cfg["max_goals"]
    out_dir = cfg["output_dir"]
    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)

    data, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
    )
    num_teams = len(team_id_to_name)
    print(f"Extracted data from {start_date} to {end_date}, with {num_teams} teams and {len(data)} dates.")
    print("Number of teams:", num_teams)

    ######################

    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)
    key, filter_key = jax.random.split(key, 2)
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=filter_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=N,
        max_goals=MAX_GOALS,
    )
    baseline_logz = float(filtered_states.log_normalizing_constant[-1])
    print(f"Finished initial filtering with default parameters. logZ = {baseline_logz:.4f}")
    ######################
    key, optimize_key = jax.random.split(key, 2)
    best_params, logz_history = logmarginal_maximize(
        key=optimize_key,
        model_inputs=model_inputs,
        params=params,
        n_particles=N,
        max_goals=MAX_GOALS,
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
    )

    best_logz = float(jnp.max(logz_history))
    print(f"\nOptimization finished.")
    print(f"  baseline logZ = {baseline_logz:.4f}")
    print(f"  best     logZ = {best_logz:.4f}   (improvement {best_logz - baseline_logz:+.4f})")

    # Ensure the output directory exists, then save the best fitted parameters.
    os.makedirs(out_dir, exist_ok=True)
    save_params(best_params, path=os.path.join(out_dir, "params_unbiased.json"))

    ######################
    # Final filter with the best (optimized) parameters, saved to the output dir.
    key, final_filter_key = jax.random.split(key)
    final_states, final_model_inputs_rbpf = run_filter_unbiased(
        key=final_filter_key,
        model_inputs=model_inputs,
        params=best_params,
        n_particles=N,
        max_goals=MAX_GOALS,
    )
    final_logz = float(final_states.log_normalizing_constant[-1])
    print(f"Final filter (best params). logZ = {final_logz:.4f}")

    # Write filter outputs (.npz) and plots into the output directory.
    from rbpf.src.graphic import (
        save_filter_states,
        plot_all,
        plot_log_marginal_likelihood_curve,
    )
    save_filter_states(
        final_states,
        final_model_inputs_rbpf,
        save_path=os.path.join(out_dir, "filter_states.npz"),
    )
    plot_all(
        filtered_states=final_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=out_dir,
        timestamps=data["date"].to_numpy(),
        params=best_params,
    )
    plot_log_marginal_likelihood_curve(
        logz_history,
        save_path=os.path.join(out_dir, "optimization_logZ_curve.png"),
    )

    # Save an optimization summary JSON.
    summary = {
        "baseline_logZ": baseline_logz,
        "best_logZ": best_logz,
        "final_filter_logZ": final_logz,
        "logZ_history": [float(x) for x in logz_history],
        "n_epochs": int(len(logz_history)),
    }
    with open(os.path.join(out_dir, "optimization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote optimization summary to {os.path.join(out_dir, 'optimization_summary.json')}")

    # Persist the run config alongside the outputs (mirrors smoothing pipeline).
    run_config = {
        "start_date": start_date,
        "end_date": end_date,
        "n_particles": N,
        "max_goals": MAX_GOALS,
        "seed": cfg["seed"],
        "n_epochs": int(cfg["n_epochs"]),
        "learning_rate": float(cfg["learning_rate"]),
        "n_reps": int(cfg["n_reps"]),
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Wrote run config to {os.path.join(out_dir, 'run_config.json')}")

    ######################
    # Prediction step: score upcoming fixtures from the fitted posterior.
    from rbpf.src.predict import run_predictions, _load_fixtures

    fixtures_path = "rbpf/data/fixtures.json"
    if not os.path.isfile(fixtures_path):
        print(f"[predict] fixtures file not found: {fixtures_path}; skipping predictions", flush=True)
    else:
        fixtures = _load_fixtures(fixtures_path)
        print(f"[predict] Loaded {len(fixtures)} fixtures from {fixtures_path}", flush=True)
        pred_result = run_predictions(
            params=best_params,
            team_id_to_name=team_id_to_name,
            fixtures=fixtures,
            max_goals=MAX_GOALS,
            n_particles=N,
            seed=0,
        )
        pred_result["params_path"] = "rbpf/outputs/filter_unbiased/params_unbiased.json"
        pred_result["teams"] = sorted(team_id_to_name.values())
        pred_out = os.path.join(out_dir, "predictions.json")
        with open(pred_out, "w") as f:
            json.dump(pred_result, f, indent=2)
        s = pred_result["summary"]
        print(
            f"[predict] Predictions: {s['n_predictions']} (scored {s['n_scored']}), "
            f"exact-score {s['score_accuracy']}, outcome-accuracy {s['outcome_accuracy']}, "
            f"total-loglik {s['total_log_likelihood']}",
            flush=True,
        )
        print(f"[predict] Wrote predictions to {os.path.abspath(pred_out)}", flush=True)

        # Generate per-match prediction images (heatmaps + outcome breakdowns).
        from rbpf.src.graphic import plot_all_predictions
        plot_all_predictions(
            pred_result,
            max_goals=MAX_GOALS,
            save_path=os.path.join(out_dir, "prediction_plots"),
        )
        print(f"[predict] Wrote prediction plots to {os.path.join(out_dir, 'prediction_plots')}", flush=True)

if __name__ == "__main__":
    main()