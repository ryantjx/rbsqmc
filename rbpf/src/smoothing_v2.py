import json

import optax

from rbpf.src.helpers import decode_EM_params, default_init_params, encode_EM_params, monitor_params, save_params
from rbpf.src.smoothing import _load_run_config, rbpf_backward_smoothing, E_step
from rbpf.src.data import get_results, ACTIVE_TEAMS, WORLDCUP_2026_TEAMS, TEAMS_SMALL
import jax
from rbpf.src.smoothing import loss_fn, E_step
from rbpf.src.model import run_filter
from rbpf.src.utils import FootballResults, EMParams, RawEMParams
from rbpf.src.graphic import plot_all, plot_all_smoothing, plot_log_marginal_likelihood_curve
import os
import jax.numpy as jnp


def run_EM(
    key: jax.Array,
    model_inputs : FootballResults,
    params: EMParams,
    N_particles: int,
    N_smoothed_trajectories: int,
    max_goals: int,
    n_epochs: int,
    learning_rate = 1e-3 # more aggresive learning rates
):
    current_key = key
    fixed_mean_0 = params.mean_0
    raw_params: RawEMParams = encode_EM_params(params)

    # use cosine decay schedule for learning rate
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=n_epochs)
    # use Yogi optimizer for more stable convergence
    optimizer = optax.yogi(schedule)
    opt_state = optimizer.init(raw_params)

    # Track decoded params (EMParams) per epoch.
    params_history = jax.tree_util.tree_map(
        lambda x: x[None], decode_EM_params(raw_params, fixed_mean_0)
    )
    log_marginal_likelihood_history = []

    print("[Init] Parameters before any EM update:", flush=True)
    monitor_params(decode_EM_params(raw_params, fixed_mean_0), prefix="  ")

    for epoch in range(n_epochs):
        print(f"EM Epoch {epoch+1}/{n_epochs}...", flush=True)
        current_key, e_key = jax.random.split(current_key, 2)
        current_params = decode_EM_params(raw_params, fixed_mean_0)

        # 1. run E_step to get the average gradient and log marginal likelihood
        smoothed_trajectories, log_marginal_likelihood, model_inputs_rbpf = E_step(
            key=e_key,
            model_inputs=model_inputs,
            params=current_params,
            n_particles=N_particles,
            n_smoothed_trajectories=N_smoothed_trajectories,
            max_goals=max_goals
        )
        # 2. Run M_step to update the parameters using the average gradient
        # vmap the loss_fn over all smoothed trajectories to get the gradients
        print(f"  [M-step] Computing gradients for {N_smoothed_trajectories} smoothed trajectories...", flush=True)
        # differentiate the loss w.r.t. raw_params by decoding inside the gradient
        loss_grads = jax.vmap(
            lambda traj: jax.grad(
                lambda rp: loss_fn(
                    decode_EM_params(rp, fixed_mean_0), traj, model_inputs_rbpf, max_goals
                )
            )(raw_params)
        )(smoothed_trajectories)
        avg_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), loss_grads)
        print(f"  [M-step] Average gradient computed. Updating parameters...", flush=True)
        # 2.2 Update the raw parameters using the averaged gradients
        # update the raw parameters using the optimizer
        updates, opt_state = optimizer.update(avg_grad, opt_state, raw_params)
        # re-assign raw_params to the updated parameters
        raw_params = optax.apply_updates(raw_params, updates)

        print(f"  [M-step] Parameters updated. Log marginal likelihood: {log_marginal_likelihood:.4f}", flush=True)
        monitor_params(decode_EM_params(raw_params, fixed_mean_0), prefix="  ")
        params_history = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y[None]], axis=0),
            params_history, decode_EM_params(raw_params, fixed_mean_0)
        )
        log_marginal_likelihood_history.append(log_marginal_likelihood)

    return decode_EM_params(raw_params, fixed_mean_0), params_history, log_marginal_likelihood_history

def save_results(
    latest_params: EMParams,
    run_config: dict,
    save_path : str = "rbpf/outputs/smoothing_v2/"
):
    # 3. Save the optimized parameters to a file
    os.makedirs(save_path, exist_ok=True)
    save_params(latest_params, save_path + "optimized_params.json")
    # Save the run configuration
    run_config = {
        "start_date": run_config["start_date"],
        "end_date": run_config["end_date"],
        "n_particles": run_config["n_particles"],
        "n_smoother_paths": run_config["n_smoother_paths"],
        "n_epochs": run_config["n_epochs"],
        "learning_rate": run_config["learning_rate"],
        "max_goals": run_config["max_goals"],
        "seed": run_config["seed"],
        "optax": run_config["optax"],
        "output_dir": save_path,
    }
    with open(save_path + "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

def _resolve_teams(cfg: dict) -> set[str] | None:
    """Resolve the ``teams`` config entry to a set of team names.

    ``teams`` may be:
      - a preset name: ``"teams_small"`` | ``"worldcup2026"`` | ``"active"``
      - an explicit list of team names (e.g. ``["England", "France"]``)
      - ``"all"`` or empty/missing -> None (use all teams)
    """
    from rbpf.src.data import TEAMS_SMALL, WORLDCUP_2026_TEAMS, ACTIVE_TEAMS

    value = cfg.get("teams", "teams_small")
    presets = {
        "teams_small": TEAMS_SMALL,
        "worldcup2026": WORLDCUP_2026_TEAMS,
        "active": ACTIVE_TEAMS,
    }
    if isinstance(value, str):
        name = value.strip().lower()
        if name in presets:
            return presets[name]
        if name in ("", "all", "none"):
            return None
        raise ValueError(
            f"Unknown 'teams' preset '{value}'. Choose from {sorted(presets)} "
            "or pass an explicit list of team names."
        )
    if isinstance(value, (list, tuple, set)):
        return set(value)
    raise ValueError(f"Invalid 'teams' value in config: {value!r}")


def main():
    #############
    cfg = _load_run_config()
    start_date = cfg.get("start_date", "1900-01-01")
    end_date = cfg.get("end_date", "2025-12-31")
    teams_only = _resolve_teams(cfg)  # None means all teams
    max_goals = cfg.get("max_goals", 8)
    N_particles = cfg.get("n_particles", 2500)
    N_smoothed_trajectories = cfg.get("n_smoother_paths", 250)
    learning_rate = cfg.get("learning_rate", 1e-3)
    epochs = cfg.get("n_epochs", 20)
    seed = cfg.get("seed", 0)
    key = jax.random.PRNGKey(seed)
    ##############
    df, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        teams_only=teams_only,
        include_friendly=False
    )
    print(f"Loaded football results data from {start_date} to {end_date}. Number of unique dates: {len(df['date'].unique())}. Number of unique teams: {len(team_id_to_name)}.")

    init_params=default_init_params(len(team_id_to_name))

    # 2. Run EM algorithm to estimate parameters
    best_params, params_history, log_marginal_likelihood_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=init_params,
        N_particles=N_particles,
        N_smoothed_trajectories=N_smoothed_trajectories,
        max_goals=max_goals,
        n_epochs=epochs,
        learning_rate=learning_rate
    )

    # 3. Run the E-step one last time to get the smoothed trajectories using the best parameters
    save_path = "rbpf/outputs/smoothing_v2/"
    os.makedirs(save_path, exist_ok=True)
    save_params(best_params, save_path + "optimized_params.json")
    print("[main] Saved optimized params.")

    # Save the run configuration
    run_config = {
        "start_date": start_date,
        "end_date": end_date,
        "n_particles": N_particles,
        "n_smoother_paths": N_smoothed_trajectories,
        "n_epochs": epochs,
        "learning_rate": learning_rate,
        "max_goals": max_goals,
        "seed": seed,
        "m_step": "adam",
        "output_dir": save_path,
    }
    with open(save_path + "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print("[main] Saved run config.")

    key, filter_key, smoother_key = jax.random.split(key, 3)

    # Run Filter
    filtered_states, model_inputs_rbpf = run_filter(
        key=filter_key,
        model_inputs=model_inputs,
        params=best_params,
        n_particles=N_particles,
        max_goals=max_goals
    )

    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=save_path + "/filter"
    )
    log_marginal_likelihood_history.append(filtered_states.log_normalizing_constant[-1])
    plot_log_marginal_likelihood_curve(
        log_marginal_likelihoods=log_marginal_likelihood_history,
        save_path=save_path + "/em_log_marginal_likelihood_curve.png"
    )
    final_smoothed, log_marginal_likelihood, _ = E_step(
        key=smoother_key,
        model_inputs=model_inputs,
        params=best_params,
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        max_goals=max_goals,
    )
    plot_all_smoothing(
        smoothed_trajectories=final_smoothed.x,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=10,
        save_path=save_path + "/smoothing",
    )
if __name__ == "__main__":
    main()