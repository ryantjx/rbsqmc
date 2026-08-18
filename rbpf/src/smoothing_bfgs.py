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
from rbpf.src.smoothing import E_step, loss_fn, rbpf_backward_smoothing

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
    maxiter: int = 1,
    maxcor: int = 10,
    n_batch: int = 0,
) -> tuple[RawEMParams, dict]:
    """One M-step via L-BFGS-B (scipy) with bounded iterations.

    Uses scipy.optimize.minimize with L-BFGS-B, which:
      - Limits memory via the limited-memory Hessian approximation (maxcor).
      - Supports maxiter to prevent overshooting (the main cause of divergence).
      - Runs on CPU, but the loss is computed on GPU via JAX with checkpointing.

    The loss uses jax.checkpoint (rematerialization) so Cholesky factors and
    triangular solves are recomputed during backprop instead of being stored,
    avoiding OOM on large n_smoother_paths.
    """
    flat_init = np.asarray(_raw_to_flat(raw_params))
    structure = _raw_structure(raw_params)

    # Checkpointed loss: rematerialize intermediates during backprop
    _loss_fn = jax.checkpoint(
        lambda raw, traj: loss_fn(
            decode_EM_params(raw, fixed_mean_0), traj, model_inputs, max_goals
        ),
        policy=jax.checkpoint_policies.nothing_saveable(),
    )

    # JIT-compiled value_and_grad for the per-trajectory loss
    _per_traj_vg = jax.jit(jax.value_and_grad(
        lambda raw, traj: _loss_fn(raw, traj)
    ))

    if n_batch > 0 and n_batch < smoothed_trajectories.x.shape[0]:
        # Batched: process trajectories in chunks to cap peak GPU memory.
        # Each scipy loss/grad evaluation loops over batches, accumulating
        # the sum of per-batch mean losses and gradients, then divides by
        # the number of batches to get the overall mean.
        n_total = smoothed_trajectories.x.shape[0]
        n_batches = (n_total + n_batch - 1) // n_batch
        print(f"    [m_step_bfgs] Batched gradient: {n_batches} batches of <= {n_batch}")

        def _batched_loss_and_grad(flat):
            raw = _flat_to_raw(flat, structure)
            total_val = jnp.zeros(())
            total_grad: RawEMParams | None = None
            for b_idx in range(n_batches):
                start = b_idx * n_batch
                end = min(start + n_batch, n_total)
                batch = jax.tree_util.tree_map(
                    lambda x: x[start:end], smoothed_trajectories
                )
                vals, grads = jax.vmap(
                    lambda traj: _per_traj_vg(raw, traj)
                )(batch)
                batch_val = jnp.mean(vals)
                batch_grad = jax.tree_util.tree_map(
                    lambda g: jnp.mean(g, axis=0), grads
                )
                total_val = total_val + batch_val
                if total_grad is None:
                    total_grad = batch_grad
                else:
                    total_grad = jax.tree_util.tree_map(
                        lambda acc, g: acc + g, total_grad, batch_grad
                    )
            mean_val = total_val / n_batches
            assert total_grad is not None  # n_batches >= 1
            flat_grad = _raw_to_flat(total_grad)
            return mean_val, flat_grad

        _loss_and_grad = jax.jit(_batched_loss_and_grad)
    else:
        # Non-batched: process all trajectories at once
        @jax.jit
        def _loss_and_grad(flat):
            raw = _flat_to_raw(flat, structure)
            vals, grads = jax.vmap(
                lambda traj: _per_traj_vg(raw, traj)
            )(smoothed_trajectories)
            mean_val = jnp.mean(vals)
            mean_grad = jax.tree_util.tree_map(
                lambda g: jnp.mean(g, axis=0), grads
            )
            flat_grad = _raw_to_flat(mean_grad)
            return mean_val, flat_grad

    def _scipy_loss(flat):
        val, _ = _loss_and_grad(jnp.asarray(flat))
        return float(val)

    def _scipy_grad(flat):
        _, grad = _loss_and_grad(jnp.asarray(flat))
        return np.asarray(grad)

    result = scipy_minimize(
        fun=_scipy_loss,
        x0=flat_init,
        jac=_scipy_grad,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "maxcor": maxcor,
            "ftol": 1e-6,
            "gtol": 1e-5,
        },
    )

    raw_next = _flat_to_raw(jnp.asarray(result.x), structure)
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
    maxiter: int = 1,
    maxcor: int = 10,
    n_batch: int = 0,
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
          f"N_particles={n_particles}, N_trajectories={n_smoothed_trajectories}"
          f"{f', n_batch={n_batch}' if n_batch > 0 else ''}", flush=True)

    log_marginal_likelihood_history = []
    mstep_history = []

    for epoch in range(num_epochs):
        params = decode_EM_params(raw_params, fixed_mean_0)

        # E-step
        if n_batch > 0 and n_batch < n_smoothed_trajectories:
            # Batched E-step: run the forward filter once (shared across all
            # batches), then sample backward trajectories in chunks of
            # n_batch.  Each chunk is moved to CPU (numpy) immediately so
            # peak GPU memory is O(n_batch * T * M) instead of
            # O(n_smoothed_trajectories * T * M).
            filter_key, smoother_key = jax.random.split(key, 2)
            print(f"  [Epoch {epoch+1}/{num_epochs}] Running E-step (forward filter)...", flush=True)
            filtered_states, model_inputs_rbpf = run_filter(
                key=filter_key,
                model_inputs=model_inputs,
                params=params,
                n_particles=n_particles,
                max_goals=max_goals,
            )
            log_marginal_likelihood = filtered_states.log_normalizing_constant[-1]
            print(f"  [Epoch {epoch+1}/{num_epochs}] Filter done. log marginal = {log_marginal_likelihood:.4f}", flush=True)

            n_batches = (n_smoothed_trajectories + n_batch - 1) // n_batch
            batch_keys = jax.random.split(smoother_key, n_batches)
            print(f"  [Epoch {epoch+1}/{num_epochs}] Batched backward sampling: "
                  f"{n_batches} batches of <= {n_batch} trajectories", flush=True)
            all_x = []
            for b_idx in range(n_batches):
                current_size = min(
                    n_batch, n_smoothed_trajectories - b_idx * n_batch
                )
                batch_traj = rbpf_backward_smoothing(
                    key=batch_keys[b_idx],
                    n_smoothed_trajectories=current_size,
                    filtered_states=filtered_states,
                    params=params,
                    model_inputs_rbpf=model_inputs_rbpf,
                )
                # Move to CPU immediately to free GPU memory
                all_x.append(np.asarray(batch_traj.x))
            # Concatenate on CPU and transfer back to device as one array
            smoothed_trajectories = RBPFState(
                x=jnp.asarray(np.concatenate(all_x, axis=0))
            )
            print(f"  [Epoch {epoch+1}/{num_epochs}] E-step done (batched). "
                  f"log marginal = {log_marginal_likelihood:.4f}", flush=True)
        else:
            print(f"  [Epoch {epoch+1}/{num_epochs}] Running E-step (filter + backward sampling)...", flush=True)
            smoothed_trajectories, log_marginal_likelihood = E_step(
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
            model_inputs=model_inputs,
            fixed_mean_0=fixed_mean_0,
            max_goals=max_goals,
            maxiter=maxiter,
            maxcor=maxcor,
            n_batch=n_batch,
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
    N_smoothed_trajectories = cfg.get("n_smoother_paths", 1000)
    epochs = cfg.get("n_epochs", 5)
    seed = cfg.get("seed", 0)
    maxiter = cfg.get("maxiter", 1)
    maxcor = cfg.get("maxcor", 10)
    n_batch = cfg.get("n_batch", 0)
    key = jax.random.PRNGKey(seed)
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

    print("[main] Running EM (BFGS M-step)...", flush=True)
    latest_params, params_history, log_marginal_likelihood_history, mstep_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=default_init_params(len(team_id_to_name)),
        n_particles=N_particles,
        n_smoothed_trajectories=N_smoothed_trajectories,
        num_epochs=epochs,
        max_goals=MAX_GOALS,
        maxiter=maxiter,
        maxcor=maxcor,
        n_batch=n_batch,
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
        "seed": seed,
        "m_step": "bfgs",
        "maxiter": maxiter,
        "maxcor": maxcor,
        "n_batch": n_batch,
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