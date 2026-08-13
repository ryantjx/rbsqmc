"""Direct gradient descent on the log normalizing constant (v2 training).

This replaces the Monte Carlo EM (MCEM) in ``rbpf_ou`` with **direct gradient
descent on the marginal log-likelihood** ``-log Z(theta)``, where

    log Z(theta) = filtered_states.log_normalizing_constant[-1]

is the total log marginal likelihood ``log p(y_{1:T} | theta)`` produced by the
forward particle filter. This is the "differentiable particle filter" approach:

- No E-step / M-step split, so there is no MCEM noise and no per-dimension loss
  scaling (the two causes of the diverging EM in ``rbpf_ou``).
- The objective is a single well-defined scalar ``-log Z(theta)``.
- The posterior is still represented exactly (non-parametrically) by particles,
  so the non-Gaussian bivariate-Poisson posterior is handled exactly.

The particle filter uses ``jax.random`` internally, so ``log Z`` is stochastic.
For a stable gradient we fix the PRNG key per gradient step (a biased-but-
consistent estimator, standard in differentiable particle filtering). The
deterministic covariance trajectory ``compute_gamma_trajectory`` provides a
clean differentiable path through ``gamma_0`` and ``kappa``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from rbpf_ou_v2.src.utils import EMParams, FootballResults
from rbpf_ou_v2.src.helpers import (
    default_init_params,
    generate_augmented_data,
    params_to_dict,
)
from rbpf_ou_v2.src.model import run_filter, compute_gamma_trajectory

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", __import__("os").environ.get("RBSQMC_PLATFORM", "cpu")
)

MAX_GOALS = 8

# Minimum eigenvalue floor for the projected covariances (see `_project_psd`).
_EIGEN_FLOOR = 1e-4

# Lower bound for the OU mean-reversion rate kappa.
#
# kappa -> 0 makes phi = exp(-kappa*dt) -> 1, so the transition covariance
# (1-phi^2)*Sigma_0 -> 0 (a degenerate transition). Flooring kappa away from
# zero keeps the transition covariance meaningfully non-singular, preventing
# the transition log-likelihood from exploding.
_KAPPA_MIN = 0.001

# Upper bound for the OU mean-reversion rate kappa.
#
# International matches are spaced far apart (median gap ~43 days). For team
# quality to persist across that gap, the half-life must be much longer than
# the gap. We cap kappa at 0.002, giving a half-life of ln(2)/0.002 ~= 347 days
# (~1 year), so a team's strength retains ~92% of its value after the median
# 43-day gap.
_KAPPA_MAX = 0.002


def _symmetrize(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise a square matrix (for PSD-constrained params)."""
    return 0.5 * (x + x.T)


def _project_psd(x: jnp.ndarray, floor: float = _EIGEN_FLOOR) -> jnp.ndarray:
    """Project a symmetric matrix onto the positive-definite cone.

    Eigen-decompose, clamp eigenvalues to be >= ``floor`` (> 0), and rebuild.
    Guarantees a full-rank, strictly PD matrix whose log-determinant and solve
    are well defined.
    """
    x = _symmetrize(x)
    eigvals, eigvecs = jnp.linalg.eigh(x)
    eigvals = jnp.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def _psd_from_cholesky(L: jnp.ndarray, n: int) -> jnp.ndarray:
    """Build a PD matrix ``A = L L^T`` from an unconstrained ``n x n`` factor.

    ``L`` is a free matrix. We take its lower triangle, keep the diagonal
    positive via ``softplus``, zero the strictly-upper triangle, and form
    ``A = L L^T``. Because L is full-rank lower-triangular with positive
    diagonal, A is positive-definite by construction and the map from the free
    entries of L to the PD cone is smooth and surjective.

    The diagonal is floored at ``_EIGEN_FLOOR`` (added, not clamped) so the
    matrix cannot drift near-singular during training, which would make the
    filter's Cholesky / pinv unstable and produce NaN. This floor is
    differentiable (unlike an eigendecomposition-based projection).
    """
    L_low = jnp.tril(L)
    diag = jax.nn.softplus(jnp.diag(L_low)) + _EIGEN_FLOOR  # > floor, strictly
    L_low = L_low.at[jnp.diag_indices(n)].set(diag)
    return L_low @ L_low.T


def _cholesky_from_psd(A: jnp.ndarray, n: int) -> jnp.ndarray:
    """Inverse map: a free ``n x n`` factor encoding the PD matrix ``A``.

    ``L`` is a lower-triangular Cholesky factor of the PD ``A`` with a
    softplus-wrapped diagonal, padded to a full ``n x n`` free array (upper
    triangle is arbitrary/zero and ignored by ``_psd_from_cholesky``).
    """
    L = jnp.linalg.cholesky(A)  # lower-triangular, positive diagonal
    diag = L[jnp.diag_indices(n)]
    L_free = jnp.zeros_like(A)
    L_free = L_free.at[jnp.tril_indices(n)].set(
        L[jnp.tril_indices(n)]
    )
    # invert softplus on the diagonal so reconstructing A recovers it:
    # softplus(x) = diag  =>  x = log(exp(diag) - 1)
    L_free = L_free.at[jnp.diag_indices(n)].set(
        jnp.log(jnp.exp(diag) - 1.0 + 1e-10)
    )
    return L_free


def _constrain(params: EMParams) -> EMParams:
    """Apply validity constraints so parameters stay in their support.

    - alpha, beta unconstrained real.
    - kappa clamped to [_KAPPA_MIN, _KAPPA_MAX] (keeps the transition covariance
      non-degenerate while forcing a long mean-reversion half-life so team
      strengths persist between matches).
    - gamma_0, B projected onto the positive-definite cone (full-rank, so the
      transition covariance Q and the smoother covariances stay invertible
      and their log-determinants finite).

    Note: during training these are Cholesky-parameterized (so they stay PD
    automatically); this projection is retained as a safety net for params
    constructed outside the optimizer (e.g. hand-loaded values).
    """
    gamma_0 = _project_psd(params.gamma_0)
    B = _project_psd(params.B)
    kappa = jnp.clip(params.kappa, _KAPPA_MIN, _KAPPA_MAX)
    return EMParams(
        mean_0=params.mean_0,
        gamma_0=gamma_0,
        B=B,
        kappa=kappa,
        alpha=params.alpha,
        beta=params.beta,
    )


def _log_normalizing_constant(
    carry: dict,
    model_inputs: FootballResults,
    num_teams: int,
    n_particles: int,
    key: jax.Array,
) -> jax.Array:
    """Run the forward filter and return the total log marginal likelihood.

    ``log Z(theta) = filtered_states.log_normalizing_constant[-1]``. The
    covariance trajectory is deterministic in ``(gamma_0, kappa)``, giving a
    clean differentiable path; the particle filter itself is stochastic, so the
    PRNG ``key`` is fixed per gradient step for a stable (biased-consistent)
    gradient.
    """
    # Reconstruct PD matrices from free Cholesky factors.
    gamma_0 = _psd_from_cholesky(carry["L_gamma0"], num_teams)
    B = _psd_from_cholesky(carry["L_B"], 2)
    params = EMParams(
        mean_0=carry["mean_0"],
        gamma_0=gamma_0,
        B=B,
        kappa=carry["kappa"],
        alpha=carry["alpha"],
        beta=carry["beta"],
    )

    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa,
        num_teams=num_teams,
    )
    augmented = generate_augmented_data(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain,
    )
    filtered, _ = run_filter(
        key=key,
        model_inputs=augmented,
        params=params,
        num_teams=num_teams,
        n_particles=n_particles,
    )
    return filtered.log_normalizing_constant[-1]


def _loss(carry: dict, model_inputs, num_teams, n_particles, key) -> jax.Array:
    """Negative log marginal likelihood ``-log Z(theta)`` (single scalar)."""
    return -_log_normalizing_constant(carry, model_inputs, num_teams, n_particles, key)


def run_gd(
    model_inputs: FootballResults,
    init_params: EMParams,
    num_teams: int,
    n_particles: int = 100,
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    key: jax.Array = jax.random.PRNGKey(42),
) -> tuple[EMParams, jnp.ndarray, dict]:
    """Train parameters by direct gradient descent on ``-log Z(theta)``.

    Unlike MCEM, there is no E-step / M-step split and no smoothing: we run the
    forward filter once per step, read off the log normalizing constant, and
    backprop through it. The PRNG key is fixed per step for a stable gradient.

    Returns:
        tuple[EMParams, jnp.ndarray, dict]: (final_params, log_marginal_history,
        diagnostics).
    """
    # Initial parameter blocks (dict so optax.multi_transform labels align).
    carry = {
        "mean_0": init_params.mean_0,
        "L_gamma0": _cholesky_from_psd(init_params.gamma_0, num_teams),
        "L_B": _cholesky_from_psd(init_params.B, 2),
        "kappa": init_params.kappa,
        "alpha": init_params.alpha,
        "beta": init_params.beta,
    }

    # --- Per-parameter learning rates ---
    base = learning_rate
    lr_mapping = {
        "L_gamma0": base * 1.0,
        "L_B": base * 1.0,
        "kappa": base * 1.0,
        "alpha": base * 1.0,
        "beta": base * 1.0,
    }
    transforms = {
        "L_gamma0": optax.adam(lr_mapping["L_gamma0"]),
        "L_B": optax.adam(lr_mapping["L_B"]),
        "kappa": optax.adam(lr_mapping["kappa"]),
        "alpha": optax.adam(lr_mapping["alpha"]),
        "beta": optax.adam(lr_mapping["beta"]),
        # mean_0 is fixed (unidentifiable from the likelihood); never update it.
        "mean_0": optax.set_to_zero(),
    }
    param_labels = {
        "L_gamma0": "L_gamma0",
        "L_B": "L_B",
        "kappa": "kappa",
        "alpha": "alpha",
        "beta": "beta",
        "mean_0": "mean_0",
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(transforms, param_labels),
    )
    opt_state = optimizer.init(carry)

    value_and_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda c, k: _loss(c, model_inputs, num_teams, n_particles, k),
            argnums=0,
        )
    )

    log_marginal_history = []
    loss_history = []

    for step in tqdm(range(n_steps)):
        # Fixed key per step for a stable (biased-consistent) gradient.
        step_key = jax.random.fold_in(key, step)
        loss_val, grads = value_and_grad_fn(carry, step_key)
        log_marginal_history.append(float(-loss_val))
        loss_history.append(float(loss_val))

        if step % max(10, 1) == 0 or step == n_steps - 1:
            print(
                f"step={step:05d}, log_marginal={float(-loss_val):.4f}, "
                f"kappa={float(carry['kappa']):.5f} "
                f"alpha={float(carry['alpha']):.5f} beta={float(carry['beta']):.5f}"
            )

        # If a step produced NaN (params drifted into an unstable region),
        # stop early and keep the last finite parameters (do NOT apply the
        # NaN update to carry).
        if not jnp.isfinite(loss_val):
            print(f"step={step:05d}: NaN loss; stopping early.")
            break

        updates, opt_state = optimizer.update(grads, opt_state, carry)
        carry = optax.apply_updates(carry, updates)
        # Clamp kappa to its support inside the loop (the optimizer updates it
        # freely; a negative kappa makes phi = exp(-kappa*dt) > 1 and the
        # transition covariance (1-phi^2)*gamma_0 negative -> NaN).
        carry["kappa"] = jnp.clip(carry["kappa"], _KAPPA_MIN, _KAPPA_MAX)

    # Reconstruct PD matrices from the best free factors and project onto support.
    best_gamma_0 = _psd_from_cholesky(carry["L_gamma0"], num_teams)
    best_B = _psd_from_cholesky(carry["L_B"], 2)
    final = _constrain(EMParams(
        mean_0=init_params.mean_0,
        gamma_0=best_gamma_0,
        B=best_B,
        kappa=carry["kappa"],
        alpha=carry["alpha"],
        beta=carry["beta"],
    ))

    diagnostics = {
        "loss_history": loss_history,
    }
    return final, jnp.array(log_marginal_history), diagnostics
