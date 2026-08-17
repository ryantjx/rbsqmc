"""EM with a BFGS M-step.

This variant keeps the Monte-Carlo E-step and complete-data log-likelihood
estimate from `rbpf.src.smoothing` (the MC estimate is unbiased and correct),
but replaces the fixed-learning-rate Adam M-step with a quasi-Newton BFGS
optimization of the negative ELBO in unconstrained parameter space.

Following the cuthbert EM example
(https://state-space-models.github.io/cuthbert/examples/parameter_estimation_em/):
  1. Optimize in UNCONSTRAINED space via `encode_EM_params` / `decode_EM_params`
     so constraints (PSD gamma_0, positive kappa, diagonal PSD B with bounded
     ratio) are enforced by construction and never produce NaNs.
  2. Use `jax.scipy.optimize.minimize(method="BFGS")` on the flattened
     unconstrained vector for the M-step, which is far more stable than a
     fixed-lr Adam step for small-parameter problems.

The Monte-Carlo ELBO is the average of `loss_fn` over the backward-sampled
smoothing trajectories, exactly as in the Adam version.
"""

import jax
import jax.numpy as jnp
import os

from jax.scipy.optimize import minimize as jax_minimize

from rbpf.src.model import run_filter
from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
from rbpf.src.helpers import (
    default_init_params,
    save_params,
    encode_EM_params,
    decode_EM_params,
)
from rbpf.src.utils import (
    EMParams, FootballResults, RawEMParams, RBPFState, RBPFFootballResults,
)
from rbpf.src.bivariate_poisson import loglik
from rbpf.src.graphic import plot_all, plot_log_marginal_likelihood_curve, plot_all_smoothing

# Reuse the E-step (filter + backward sampling) and loss_fn from smoothing.py.
# smoothing.py now has the fixed backward sampler (stable Cholesky + correct
# scan ordering), so there's no need to duplicate it here.
from rbpf.src.smoothing import E_step, loss_fn

jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

# ---------------------------------------------------------------------------
# Flatten / unflatten RawEMParams <-> 1D vector for jax.scipy.optimize.minimize
# ---------------------------------------------------------------------------
def _raw_to_flat(raw: RawEMParams) -> jnp.ndarray:
    """Flatten a RawEMParams namedtuple into a 1D array."""
    leaves, _ = jax.tree_util.tree_flatten(raw)
    return jnp.concatenate([jnp.ravel(x) for x in leaves])


def _raw_structure(template: RawEMParams):
    """Precompute static structure (shapes, split indices) for unflattening.

    Returns (treedef, shapes, splits) where `splits` is a concrete Python list
    of cumulative offsets, safe to use inside traced functions.
    """
    leaves, treedef = jax.tree_util.tree_flatten(template)
    shapes = [tuple(x.shape) for x in leaves]
    sizes = [int(jnp.size(x)) for x in leaves]
    # Concrete Python ints -> jnp.split won't hit a tracer.
    splits = list(jnp.cumsum(jnp.array(sizes))[:-1].tolist())
    return treedef, shapes, splits


def _flat_to_raw(flat: jnp.ndarray, structure) -> RawEMParams:
    """Inverse of `_raw_to_flat` using precomputed `structure`."""
    treedef, shapes, splits = structure
    parts = jnp.split(flat, splits)
    new_leaves = [p.reshape(s) for p, s in zip(parts, shapes)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ---------------------------------------------------------------------------
# M-step: BFGS on the Monte-Carlo negative ELBO in unconstrained space
# ---------------------------------------------------------------------------
def m_step_bfgs(
    raw_params: RawEMParams,
    smoothed_trajectories: RBPFState,
    model_inputs: FootballResults,
    fixed_mean_0: jax.Array,
    max_goals: int,
) -> tuple[RawEMParams, dict]:
    """One M-step via BFGS.

    Minimizes the mean of `loss_fn` over the smoothed trajectories w.r.t. the
    unconstrained parameters. `loss_fn` already returns the negative
    complete-data log-likelihood, so we minimize its Monte-Carlo average
    (the negative ELBO, up to the entropy term which is independent of theta).
    """
    flat_init = _raw_to_flat(raw_params)
    # Precompute static structure outside the traced loss so jnp.split gets
    # concrete Python split indices.
    structure = _raw_structure(raw_params)

    def _mc_neg_elbo(flat: jax.Array) -> jax.Array:
        raw = _flat_to_raw(flat, structure)
        decoded = decode_EM_params(raw, fixed_mean_0)
        # Average loss over the N smoothed trajectories (Monte-Carlo ELBO).
        per_traj = jax.vmap(
            lambda traj: loss_fn(decoded, traj, model_inputs, max_goals)
        )(smoothed_trajectories)
        return jnp.mean(per_traj)

    result = jax_minimize(_mc_neg_elbo, flat_init, method="BFGS")

    raw_next = _flat_to_raw(result.x, structure)
    diagnostics = {
        "success": bool(result.success),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "final_loss": float(result.fun),
        "status": int(result.status),
    }
    return raw_next, diagnostics


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------
def run_EM(
    key: jax.Array,
    model_inputs: FootballResults,
    params: EMParams,
    n_particles: int,
    n_smoothed_trajectories: int,
    num_epochs: int,
    max_goals: int,
):
    """EM with a BFGS M-step.

    Returns (final_params, params_history, log_marginal_likelihood_history,
             mstep_history).
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)

    # Track decoded params (EMParams) per epoch.
    params_history = jax.tree_util.tree_map(
        lambda x: x[None], decode_EM_params(raw_params, fixed_mean_0)
    )

    print(f"[run_EM/BFGS] Starting EM: {num_epochs} epochs, "
          f"N_particles={n_particles}, N_trajectories={n_smoothed_trajectories}")

    log_marginal_likelihood_history = []
    mstep_history = []

    for epoch in range(num_epochs):
        params = decode_EM_params(raw_params, fixed_mean_0)

        # E-step
        print(f"  [Epoch {epoch+1}/{num_epochs}] Running E-step (filter + backward sampling)...")
        smoothed_trajectories, log_marginal_likelihood = E_step(
            key=key,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            n_smoothed_trajectories=n_smoothed_trajectories,
            max_goals=max_goals,
        )
        print(f"  [Epoch {epoch+1}/{num_epochs}] E-step done. log marginal = {log_marginal_likelihood:.4f}")

        # M-step (BFGS)
        print(f"  [Epoch {epoch+1}/{num_epochs}] Running M-step (BFGS)...")
        raw_params, mstep_diag = m_step_bfgs(
            raw_params=raw_params,
            smoothed_trajectories=smoothed_trajectories,
            model_inputs=model_inputs,
            fixed_mean_0=fixed_mean_0,
            max_goals=max_goals,
        )
        print(f"  [Epoch {epoch+1}/{num_epochs}] M-step done. "
              f"loss={mstep_diag['final_loss']:.4f}, nit={mstep_diag['nit']}, "
              f"success={mstep_diag['success']}")

        log_marginal_likelihood_history.append(log_marginal_likelihood)
        mstep_history.append(mstep_diag)

        params = decode_EM_params(raw_params, fixed_mean_0)
        params_history = jax.tree_util.tree_map(
            lambda track, new: jnp.concatenate([track, new[None]], axis=0),
            params_history, params,
        )

    print(f"[run_EM/BFGS] EM complete. Final log marginal = {log_marginal_likelihood_history[-1]:.4f}")
    return params, params_history, log_marginal_likelihood_history, mstep_history


def main():
    ############################# MODEL TRAINING PIPELINE #############################
    start_date = "2000-01-01"
    end_date = "2025-12-31"
    teams_only = WORLDCUP_2026_TEAMS
    MAX_GOALS = 8
    N_particles = 10000
    N_smoothed_trajectories = 10000
    epochs = 3
    key = jax.random.PRNGKey(0)
    ############################################
    df, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=teams_only,
    )
    print(f"Loaded football results data from {start_date} to {end_date}. "
          f"Number of unique dates: {len(df['date'].unique())}. "
          f"Number of unique teams: {len(team_id_to_name)}.")

    print("[main] Running EM (BFGS M-step)...")
    latest_params, params_history, log_marginal_likelihood_history, mstep_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=default_init_params(len(team_id_to_name)),
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        num_epochs=epochs,
        max_goals=MAX_GOALS,
    )
    print("[main] EM finished.")

    save_path = "./rbpf/outputs/smoothing_bfgs/"
    os.makedirs(save_path, exist_ok=True)
    save_params(latest_params, save_path + "optimized_params.json")
    print("[main] Saved optimized params.")

    # Save the run configuration
    import json
    run_config = {
        "start_date": start_date,
        "end_date": end_date,
        "n_particles": N_particles,
        "n_smoother_paths": N_smoothed_trajectories,
        "n_epochs": epochs,
        "max_goals": MAX_GOALS,
        "seed": 0,
        "m_step": "bfgs",
        "output_dir": save_path,
    }
    with open(save_path + "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print("[main] Saved run config.")

    print("[main] Running final filter with optimized params...")
    filtered_states, model_inputs_rbpf = run_filter(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=N_particles,
        max_goals=MAX_GOALS,
    )
    print(f"[main] Final filter done. Final log marginal = {filtered_states.log_normalizing_constant[-1]:.4f}")

    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=save_path + "/filter",
    )
    print("[main] Saved filter plots.")

    log_marginal_likelihood_history.append(filtered_states.log_normalizing_constant[-1])
    plot_log_marginal_likelihood_curve(
        log_marginal_likelihoods=log_marginal_likelihood_history,
        save_path=save_path + "/em_log_marginal_likelihood_curve.png",
    )
    print("[main] Saved EM log marginal likelihood curve.")

    # 6. Run a final smoothing pass with the optimized params and plot
    print("[main] Running final smoothing pass...")
    final_smoothed, _ = E_step(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        max_goals=MAX_GOALS,
    )
    plot_all_smoothing(
        smoothed_trajectories=final_smoothed.x,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=10,
        save_path=save_path + "/smoothing",
    )
    print("[main] Saved smoothing plots.")


if __name__ == "__main__":
    main()