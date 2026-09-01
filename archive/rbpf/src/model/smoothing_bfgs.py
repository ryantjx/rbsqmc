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
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from archive.rbpf.src.model.model import run_filter
from archive.rbpf.src.data.data import get_results, WORLDCUP_2026_TEAMS
from archive.rbpf.src.utils.helpers import (
    default_init_params,
    save_params,
    encode_EM_params,
    decode_EM_params,
    log_inverse_wishart_kernel,
)
from archive.rbpf.src.utils.type import (
    EMParams, FootballResults, RawEMParams, RBPFState, RBPFFootballResults,
)
from archive.rbpf.src.utils.graphic import plot_all, plot_log_marginal_likelihood_curve, plot_all_smoothing
from archive.rbpf.src.predict.predict import run_predictions_from_config

# Reuse the E-step (filter + backward sampling) and loss_fn from smoothing.py.
# smoothing.py now has the fixed backward sampler (stable Cholesky + correct
# scan ordering), so there's no need to duplicate it here.
from archive.rbpf.src.model.smoothing import E_step, loss_fn

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
    model_inputs_rbpf: RBPFFootballResults,
    fixed_mean_0: jax.Array,
    max_goals: int,
    maxiter: int = 1,
    maxcor: int = 10,
    n_chunks: int = 1,
    gamma_0_prior: jax.Array | None = None,
    gamma_prior_dof: float = 5.0,
) -> tuple[RawEMParams, dict]:
    """One M-step via L-BFGS-B (scipy) on the negative ELBO + gamma_0 prior.

    The loss is the Monte-Carlo negative ELBO (mean of ``loss_fn`` over the
    fixed smoothed trajectories) plus an inverse-Wishart prior on ``gamma_0``.
    The prior is centered on the ORIGINAL ``gamma_0`` (from
    ``default_init_params``), so it shrinks the fitted stationary covariance
    toward its initial value and prevents the optimizer from collapsing to a
    degenerate near-stationary solution with tiny variance.

    Uses scipy.optimize.minimize with L-BFGS-B, which:
      - Limits memory via the limited-memory Hessian approximation (maxcor).
      - Supports maxiter to prevent overshooting (the main cause of divergence).
      - Runs on CPU, but the loss is computed on GPU via JAX with checkpointing.

    The loss uses jax.checkpoint (rematerialization) so Cholesky factors and
    triangular solves are recomputed during backprop instead of being stored,
    avoiding OOM on large n_smoother_paths.

    Batching: when ``n_chunks > 1``, the trajectory vmap is split into
    ``n_chunks`` chunks (``n_traj // n_chunks`` trajectories per chunk). Each
    chunk is still fully parallelised on the GPU, but only one chunk is
    resident in memory at a time, capping peak VRAM.
    """
    # --- Flatten the unconstrained params into a 1D vector for scipy. ---
    flat_init = np.asarray(_raw_to_flat(raw_params))
    structure = _raw_structure(raw_params)

    # --- Checkpointed per-trajectory loss (rematerialize during backprop). ---
    _per_traj_loss = jax.checkpoint(
        lambda raw, traj: loss_fn(
            decode_EM_params(raw, fixed_mean_0), traj, model_inputs_rbpf, max_goals
        ),
        policy=jax.checkpoint_policies.nothing_saveable(),
    )

    # --- JIT value_and_grad for one trajectory. ---
    _per_traj_vg = jax.jit(jax.value_and_grad(_per_traj_loss))

    n_traj = smoothed_trajectories.x.shape[0]
    chunk_size = max(1, n_traj // n_chunks) if n_chunks > 0 else n_traj
    n_chunks_eff = int(np.ceil(n_traj / chunk_size))

    # --- JIT the mean loss/grad over ONE chunk (passed as a runtime arg). ---
    @jax.jit
    def _chunk_mean_loss_and_grad(raw, traj_chunk):
        vals, grads = jax.vmap(lambda traj: _per_traj_vg(raw, traj))(traj_chunk)
        return jnp.mean(vals), jax.tree_util.tree_map(
            lambda g: jnp.mean(g, axis=0), grads
        )

    # --- Accumulate the mean loss/grad over all trajectories in chunks. ---
    def _mean_loss_and_grad(raw, trajectories):
        val_sum = 0.0
        grad_sum = None
        for start in range(0, n_traj, chunk_size):
            end = min(start + chunk_size, n_traj)
            chunk = jax.tree_util.tree_map(lambda x: x[start:end], trajectories)
            v, g = _chunk_mean_loss_and_grad(raw, chunk)
            n = end - start
            val_sum = val_sum + v * n
            if grad_sum is None:
                grad_sum = jax.tree_util.tree_map(lambda gg: gg * n, g)
            else:
                grad_sum = jax.tree_util.tree_map(
                    lambda a, b: a + b * n, grad_sum, g
                )
        return val_sum / n_traj, jax.tree_util.tree_map(
            lambda g: g / n_traj, grad_sum
        )

    # --- Add the gamma_0 prior: inverse-Wishart centered on the ORIGINAL
    #     gamma_0 (shrinkage toward the initialization). ---
    prior_scale = (
        gamma_0_prior
        if gamma_0_prior is not None
        else decode_EM_params(raw_params, fixed_mean_0).gamma_0
    )

    def _prior_loss(raw):
        gamma_0 = decode_EM_params(raw, fixed_mean_0).gamma_0
        # -log p(gamma_0) = -log_inverse_wishart_kernel(gamma_0, prior_scale, dof)
        return -log_inverse_wishart_kernel(gamma_0, prior_scale, gamma_prior_dof)

    # --- Total loss/grad as functions of the flat vector. ---
    def _loss_and_grad(flat):
        raw = _flat_to_raw(flat, structure)
        mean_val, mean_grad = _mean_loss_and_grad(raw, smoothed_trajectories)
        prior_val, prior_grad = jax.value_and_grad(_prior_loss)(raw)
        return mean_val + prior_val, _raw_to_flat(
            jax.tree_util.tree_map(lambda a, b: a + b, mean_grad, prior_grad)
        )

    # --- scipy callbacks: convert flat <-> float / numpy for L-BFGS-B. ---
    def _scipy_loss(flat):
        val, _ = _loss_and_grad(jnp.asarray(flat))
        return float(val)

    def _scipy_grad(flat):
        _, grad = _loss_and_grad(jnp.asarray(flat))
        return np.asarray(grad)

    # --- Run L-BFGS-B. ---
    result = scipy_minimize(
        fun=_scipy_loss,
        x0=flat_init,
        jac=_scipy_grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxcor": maxcor, "ftol": 1e-6, "gtol": 1e-5},
    )

    # --- NaN guard: if the optimizer produced NaN/inf params or a non-finite
    #     loss, fall back to the previous params instead of corrupting the fit. ---
    result_x = np.asarray(result.x)
    result_fun = float(result.fun)
    if not np.all(np.isfinite(result_x)) or not np.isfinite(result_fun):
        print(
            f"    [m_step_bfgs] NaN/inf detected (loss={result_fun}); "
            f"reverting to previous params",
            flush=True,
        )
        raw_next = raw_params
        diagnostics = {
            "success": False,
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "final_loss": result_fun,
            "status": int(result.status),
            "nan_guard": True,
        }
        return raw_next, diagnostics

    # --- Unflatten the result and return diagnostics. ---
    raw_next = _flat_to_raw(jnp.asarray(result.x), structure)
    diagnostics = {
        "success": bool(result.success),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "final_loss": float(result.fun),
        "status": int(result.status),
        "nan_guard": False,
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
    maxiter: int = 1,
    maxcor: int = 10,
    n_chunks: int = 1,
    gamma_0_prior: jax.Array | None = None,
    gamma_prior_dof: float = 5.0,
):
    """EM with a BFGS M-step.

    Returns (final_params, params_history, log_marginal_likelihood_history,
             mstep_history).
    """
    fixed_mean_0 = params.mean_0
    raw_params = encode_EM_params(params)

    # The gamma_0 prior is centered on the ORIGINAL gamma_0 (from
    # default_init_params) unless an explicit prior is supplied.
    if gamma_0_prior is None:
        gamma_0_prior = params.gamma_0

    # Track decoded params (EMParams) per epoch.
    params_history = jax.tree_util.tree_map(
        lambda x: x[None], decode_EM_params(raw_params, fixed_mean_0)
    )

    print(f"[run_EM/BFGS] Starting EM: {num_epochs} epochs, "
          f"N_particles={n_particles}, N_trajectories={n_smoothed_trajectories}",
          flush=True)

    import time as _time
    log_marginal_likelihood_history = []
    mstep_history = []
    epoch_times = []
    total_start = _time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = _time.perf_counter()
        params = decode_EM_params(raw_params, fixed_mean_0)

        # E-step
        print(f"  [Epoch {epoch+1}/{num_epochs}] Running E-step (filter + backward sampling)...", flush=True)
        smoothed_trajectories, log_marginal_likelihood, model_inputs_rbpf = E_step(
            key=key,
            model_inputs=model_inputs,
            params=params,
            n_particles=n_particles,
            n_smoothed_trajectories=n_smoothed_trajectories,
            max_goals=max_goals,
        )
        print(f"  [Epoch {epoch+1}/{num_epochs}] E-step done. log marginal = {log_marginal_likelihood:.4f}", flush=True)

        # M-step (BFGS)
        print(f"  [Epoch {epoch+1}/{num_epochs}] Running M-step (BFGS)...", flush=True)
        raw_params, mstep_diag = m_step_bfgs(
            raw_params=raw_params,
            smoothed_trajectories=smoothed_trajectories,
            model_inputs_rbpf=model_inputs_rbpf,
            fixed_mean_0=fixed_mean_0,
            max_goals=max_goals,
            maxiter=maxiter,
            maxcor=maxcor,
            n_chunks=n_chunks,
            gamma_0_prior=gamma_0_prior,
            gamma_prior_dof=gamma_prior_dof,
        )
        print(f"  [Epoch {epoch+1}/{num_epochs}] M-step done. "
              f"loss={mstep_diag['final_loss']:.4f}, nit={mstep_diag['nit']}, "
              f"success={mstep_diag['success']}", flush=True)

        log_marginal_likelihood_history.append(log_marginal_likelihood)
        mstep_history.append(mstep_diag)

        params = decode_EM_params(raw_params, fixed_mean_0)
        params_history = jax.tree_util.tree_map(
            lambda track, new: jnp.concatenate([track, new[None]], axis=0),
            params_history, params,
        )

        epoch_sec = _time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)
        elapsed = _time.perf_counter() - total_start
        avg_sec = sum(epoch_times) / len(epoch_times)
        remaining = avg_sec * (num_epochs - (epoch + 1))
        print(
            f"  [Epoch {epoch+1}/{num_epochs}] done. "
            f"[{epoch_sec:6.1f}s this epoch, {elapsed:6.1f}s elapsed, ETA {remaining:6.1f}s]",
            flush=True,
        )

    print(f"[run_EM/BFGS] EM complete. Final log marginal = {log_marginal_likelihood_history[-1]:.4f}", flush=True)
    return params, params_history, log_marginal_likelihood_history, mstep_history


def _load_run_config():
    """Load run config from RBSQMC_CONFIG env var, falling back to defaults."""
    config_path = os.environ.get("RBSQMC_CONFIG")
    if config_path and os.path.isfile(config_path):
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"[main] Loaded config from {config_path}")
        return cfg
    print("[main] No RBSQMC_CONFIG set, using hardcoded defaults")
    return {}

def main():
    cfg = _load_run_config()
    ############################# MODEL TRAINING PIPELINE #############################
    start_date = cfg.get("start_date", "1950-01-01")
    end_date = cfg.get("end_date", "2025-12-31")
    teams_only = WORLDCUP_2026_TEAMS
    MAX_GOALS = cfg.get("max_goals", 8)
    N_particles = cfg.get("n_particles", 1000)
    N_smoothed_trajectories = cfg.get("n_smoother_paths", 100)
    epochs = cfg.get("n_epochs", 5)
    seed = cfg.get("seed", 0)
    maxiter = cfg.get("maxiter", 5)
    maxcor = cfg.get("maxcor", 10)
    n_chunks = cfg.get("n_chunks", 1)
    gamma_prior_dof = cfg.get("gamma_prior_dof", 5.0)
    key = jax.random.PRNGKey(seed)
    # Write outputs to the config's output_dir. This must match the Colab
    # orchestrator's download location.
    save_path = cfg.get("output_dir", "./rbpf/outputs/smoothing_bfgs/")
    if not save_path.endswith("/"):
        save_path += "/"
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

    # The gamma_0 prior is centered on the ORIGINAL gamma_0 from
    # default_init_params (shrinkage toward the initialization).
    init_params = default_init_params(len(team_id_to_name))
    print("[main] Running EM (BFGS M-step)...", flush=True)
    latest_params, params_history, log_marginal_likelihood_history, mstep_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=init_params,
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        num_epochs=epochs,
        max_goals=MAX_GOALS,
        maxiter=maxiter,
        maxcor=maxcor,
        n_chunks=n_chunks,
        gamma_0_prior=init_params.gamma_0,
        gamma_prior_dof=gamma_prior_dof,
    )
    print("[main] EM finished.")

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
        "seed": seed,
        "m_step": "bfgs",
        "maxiter": maxiter,
        "maxcor": maxcor,
        "gamma_prior_dof": gamma_prior_dof,
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
    final_smoothed, _, _ = E_step(
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

    # 7. Sequential match predictions using the fitted params.
    run_predictions_from_config(
        cfg=cfg,
        params=latest_params,
        team_id_to_name=team_id_to_name,
        save_path=save_path,
        max_goals=MAX_GOALS,
    )


if __name__ == "__main__":
    main()