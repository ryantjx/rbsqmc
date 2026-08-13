from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rbpf_ou.src.utils import RBPFFootballResults, EMParams, FootballResults
from rbpf_ou.src.data import WORLDCUP_2026_TEAMS, get_results, ACTIVE_TEAMS
from rbpf_ou.src.helpers import (
    default_init_params,
    generate_augmented_data,
    params_to_dict,
    kron_sample_psd,
)
from rbpf_ou.src.model import (
    run_filter,
    compute_gamma_trajectory,
)
from rbpf_ou.src.bivariate_poisson import loglik

import os
import json
import cuthbertlib
import optax
from tqdm import tqdm
from rbpf_ou.src.graphic import plot_log_likelihood_history

# Default to CPU locally, but allow the GPU pipeline to force a device via
# the RBSQMC_PLATFORM env var (e.g. RBSQMC_PLATFORM=cuda on a Colab T4).
jax.config.update(
    "jax_platforms", os.environ.get("RBSQMC_PLATFORM", "cpu")
)

MAX_GOALS = 8
N = 100
# Number of independent smoothed trajectories sampled in the E-step for MCEM.
# Averaging the complete-data log-likelihood over these reduces the Monte Carlo
# noise of the 1-sample estimate (which was the root cause of the M-step being
# "stuck at step 0"). Raised from 2 to 8 to cut MCEM Q-function noise.
N_TRAJECTORIES = 8

# Minimum eigenvalue floor for the projected covariances (see `_project_psd`).
_EIGEN_FLOOR = 1e-4

# Upper bound for the OU mean-reversion rate kappa.
#
# kappa controls how fast a team's strength reverts to the population mean:
# the OU half-life is t_1/2 = ln(2)/kappa. A large kappa (e.g. 0.58 from a
# previous run) gives a half-life of ~1.2 days, so a team's attack/defence
# reverts to the mean almost immediately between matches — destroying the
# persistence that makes rankings meaningful. Even kappa = 0.08 (half-life
# ~8.6 days) is too fast: the filtered states bounce around per-match noise
# instead of settling into a stable ranking.
#
# International matches are spaced far apart: the median gap between a team's
# matches is ~43 days (mean ~125 days). For team quality to persist across
# that gap, the half-life must be much longer than the gap. We cap kappa at
# 0.002, giving a half-life of ln(2)/0.002 ~= 347 days (~1 year), so a team's
# strength retains ~92% of its value after the median 43-day gap.
_KAPPA_MAX = 0.00001


# ---------------------------------------------------------------------------
# Kronecker-aware helpers (avoid materializing the full (2M, 2M) covariance).
#
# The full state covariance is always a Kronecker product  Sigma = A (x) B
# with A = (M, M) (team covariance) and B = (K, K) (shared attack/defence
# factor, K = 2). Forming A (x) B is O((MK)^2) memory and O((MK)^3) compute,
# which blows up at M = 228 (ACTIVE_TEAMS). These helpers compute the same
# quantities in factored form using the identities
#     (A (x) B) vec_C(S) = vec_C(A S B^T)          (S is (M, K))
#     logdet(A (x) B)    = K logdet(A) + M logdet(B)
# so memory stays O(M^2) and compute O(M^3) + O(K^3).
# ---------------------------------------------------------------------------
def _kron_quad_form(A, B, V):
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


def _project_psd_small(x: jnp.ndarray, floor: float = 1e-6) -> jnp.ndarray:
    """Project a symmetric matrix onto the PD cone (eigenvalue floor).

    The filtered ``gamma_pred`` can be slightly indefinite in float32 (min
    eigenvalue ~ -1e-5) at large M (ACTIVE_TEAMS). A tiny ``+1e-8 I`` shift is
    not enough to make it PD, so the Kronecker solve/logdet produce NaN. This
    clamps eigenvalues to ``>= floor`` so the factored solve is well defined.
    """
    x = 0.5 * (x + x.T)
    eigvals, eigvecs = jnp.linalg.eigh(x)
    eigvals = jnp.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def _pinv_psd(x: jnp.ndarray, floor: float = 1e-6) -> jnp.ndarray:
    """Robust pseudo-inverse of a PSD matrix via eigendecomposition.

    ``gamma_pred`` is PSD but highly rank-deficient at large M (teams that
    never played have exact-zero variance). ``jnp.linalg.pinv`` (SVD-based) can
    fail to converge on such degenerate matrices. This eigendecomposes, inverts
    only the eigenvalues above ``floor``, and zeroes the rest — a stable
    Moore-Penrose pseudo-inverse for PSD inputs.
    """
    x = 0.5 * (x + x.T)
    eigvals, eigvecs = jnp.linalg.eigh(x)
    inv_eigvals = jnp.where(eigvals > floor, 1.0 / jnp.maximum(eigvals, floor), 0.0)
    return (eigvecs * inv_eigvals) @ eigvecs.T


def _smoother_rts_single(
    filtered_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    num_teams: int,
    key: jax.Array,
):
    """Sample ONE smoothed trajectory via FFBSi (RTS backward sampling).

    OU (scalar-phi AR(1)) transition: ``X_{t+1} = mu + phi (X_t - mu) + eps``
    with ``phi = exp(-kappa*dt)``. The prediction mean mean-reverts toward
    ``mu`` and the RTS gain includes the ``phi`` factor.
    """
    n_particles = filtered_states.particles.x.shape[1]
    K = filtered_states.particles.x.shape[3]  # 2 (attack/defence)
    dim = num_teams * K  # total state dimension (2M)

    # 1. Sample Terminal States
    key, cat_key, sample_key = jax.random.split(key, 3)
    log_w_T = filtered_states.log_weights[-1]  # (N,)
    I_T = jax.random.categorical(cat_key, log_w_T)  # scalar

    # Terminal covariance: Sigma_T = gamma_T (x) B (Kronecker, shared B).
    gamma_T = model_inputs.gamma_t[-1]  # (M, M) filtered posterior team covariance
    # Sigma_T is PSD with exact-zero rows for observed teams (Schur-complement
    # marginalization). Use the PSD-aware Kronecker sampler so observed teams
    # stay at their mean instead of jittering (no (2M, 2M) matrix is formed).
    mu_T = filtered_states.particles.x[-1, I_T]  # (M, 2)
    X_T_STAR = kron_sample_psd(
        sample_key, mu_T.flatten(), gamma_T, params.B
    ).reshape(num_teams, K)  # (M, 2)

    # Derive a fresh key for the backward pass so the terminal draw and the
    # backward trajectory are not correlated.
    key, backward_key = jax.random.split(key)

    # 2. Backward Sampling
    def backward_sampling(carry, xs):
        X_next_star, key = carry  # X_next_star: (M, 2)
        key, cat_key, sample_key = jax.random.split(key, 3)

        particles_t, log_w_t, gamma_t, gamma_pred_t1, timestamp_tplus1, timestamp_prev_tplus1 = xs

        # --- 2.5: Backward weights ---
        # Transition from t -> t+1. OU: prediction mean mean-reverts toward mu
        # with phi = exp(-kappa*dt). Prediction covariance = gamma_pred_t[t+1] (x) B.
        dt = timestamp_tplus1 - timestamp_prev_tplus1
        phi = jnp.exp(-params.kappa * dt)
        pred_mean = params.mean_0 + phi * (particles_t - params.mean_0)  # (N, M, 2)
        # pred_Sigma = gamma_pred_t1 (x) B. Compute the quadratic form and
        # log-det in factored form (no (2M, 2M) matrix). gamma_pred_t1 can be
        # slightly indefinite in float32 at large M, so project it onto the PD
        # cone before the factored solve/logdet.
        gamma_pred_reg = _project_psd_small(gamma_pred_t1)
        deltas = (X_next_star - pred_mean).reshape(n_particles, -1)  # (N, 2M)
        quad = _kron_quad_form(gamma_pred_reg, params.B, deltas)  # (N,)
        log_det = _kron_logdet(gamma_pred_reg, params.B)  # stable log-det
        log_transition = (
            -0.5 * quad
            - 0.5 * log_det
            - 0.5 * dim * jnp.log(2 * jnp.pi)
        )  # (N,)
        log_backward_weights = log_w_t + log_transition  # (N,)

        I_t = jax.random.categorical(cat_key, log_backward_weights)  # scalar

        # --- 2.6: RTS gain ---
        # J_t = Sigma_{t|t} Phi_{t+1}^T Sigma_{t+1|t}^{-1}  (OU RTS gain).
        # With Kronecker structure and scalar phi, using the Kronecker identity
        # (A (x) B)(C (x) D) = (AC) (x) (BD) and pinv(gamma_pred (x) B) =
        # pinv(gamma_pred) (x) pinv(B), we get
        #   J = (gamma_t phi pinv(gamma_pred_t1)) (x) (B pinv(B)) = J_gamma (x) I_K.
        J_gamma = phi * gamma_t @ _pinv_psd(gamma_pred_t1)  # (M, M)
        # (J_gamma (x) I_K) vec_C(S) = vec_C(J_gamma S)  -- Kronecker matvec.
        diff = (X_next_star - pred_mean[I_t])  # (M, K)
        mu_RTS = particles_t[I_t] + (J_gamma @ diff)  # (M, 2)

        Gamma_RTS = gamma_t - J_gamma @ gamma_pred_t1 @ J_gamma.T  # (M, M)
        Gamma_RTS = 0.5 * (Gamma_RTS + Gamma_RTS.T)  # Ensure symmetry

        # Deterministic transition when dt = 0 (phi = 1, Q = 0): the state is
        # preserved, so X_t^* = X_{t+1}^* with no sampling. Use lax.cond so the
        # sampling branch (which inverts Sigma_RTS) is only executed when dt > 0.
        deterministic = dt == 0.0
        X_t_star = jax.lax.cond(
            deterministic,
            lambda _: X_next_star,  # dt == 0: preserve the state exactly
            lambda _: kron_sample_psd(
                sample_key, mu_RTS.flatten(), Gamma_RTS, params.B
            ).reshape(num_teams, K),  # dt > 0: sample from the RTS posterior
            operand=sample_key,
        )
        return (X_t_star, key), X_t_star

    xs_particles = filtered_states.particles.x[:-1]       # (T-1, N, M, K)
    xs_log_weights = filtered_states.log_weights[:-1]      # (T-1, N)
    xs_gamma_t = model_inputs.gamma_t[:-1]                 # (T-1, M, M)
    xs_gamma_pred_tplus1 = model_inputs.gamma_pred_t[1:]   # (T-1, M, M)
    xs_timestamp_tplus1 = model_inputs.timestamp[1:]        # (T-1,)
    xs_timestamp_prev_tplus1 = model_inputs.timestamp_prev[1:]  # (T-1,)
    _, smoothed_rest = jax.lax.scan(
        f=backward_sampling,
        init=(X_T_STAR, backward_key),
        xs=(xs_particles, xs_log_weights, xs_gamma_t, xs_gamma_pred_tplus1,
            xs_timestamp_tplus1, xs_timestamp_prev_tplus1),
        reverse=True,
    )
    # reverse=True returns times 0..T-2 in chronological order; append terminal state
    smoothed_states = jnp.concatenate(
        [smoothed_rest, X_T_STAR[None]], axis=0
    )  # (T, M, 2)
    return smoothed_states


@partial(jax.jit, static_argnames=("num_teams", "n_trajectories"))
def smoother_rts(
    filtered_states: cuthbertlib.types.ArrayTree,
    model_inputs: RBPFFootballResults,
    params: EMParams,
    num_teams: int,
    key: jax.Array,
    n_trajectories: int = N_TRAJECTORIES,
):
    """Sample M independent smoothed trajectories in parallel (FFBSi).

    Vmaps the single-trajectory backward sampler over ``n_trajectories`` keys,
    producing ``(M, T, num_teams, 2)``. This is the E-step for Monte Carlo EM.
    """
    keys = jax.random.split(key, n_trajectories)
    return jax.vmap(
        lambda k: _smoother_rts_single(
            filtered_states, model_inputs, params, num_teams, k
        )
    )(keys)  # (M, T, num_teams, 2)


def _ess_from_log_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    """Effective sample size (ESS) from unnormalized log-weights.

    ``log_weights`` has shape ``(T, N)`` (one row per time step). For each time
    step we compute the standard ESS estimate

        ESS_t = (sum_i w_i)^2 / sum_i w_i^2

    on the *normalized* weights, using the log-sum-exp trick for numerical
    stability. Returns ``(T,)``. A healthy filter keeps ESS well above ``1``
    (ideally a large fraction of ``N``); ESS collapsing toward ``1`` signals
    weight degeneracy (one particle dominating).
    """
    max_log_w = jnp.max(log_weights, axis=1, keepdims=True)
    w = jnp.exp(log_weights - max_log_w)  # (T, N), stable
    sum_w = jnp.sum(w, axis=1)  # (T,)
    ess = sum_w**2 / jnp.sum(w**2, axis=1)  # (T,)
    return ess


def E_step(
    params: EMParams,
    model_inputs: FootballResults,
    num_teams: int,
    n_particles: int,
    key: jax.Array,
    n_trajectories: int = N_TRAJECTORIES,
):
    """
    E-step: Forward Filter Backward Sampling (FFBSi) with M trajectories.

    Returns ``(filtered_states, smoothed_states, log_marginal, augmented_results, ess)``
    where ``ess`` is the per-time-step effective sample size ``(T,)`` of the
    forward filter (a weight-degeneracy diagnostic).
    """
    key, filter_key, smoother_key = jax.random.split(key, 3)
    print(f"Running E-step: Forward Filter Backward Sampling (FFBSi)")
    print(f"  num_teams = {num_teams}, n_particles = {n_particles}, "
          f"n_trajectories = {n_trajectories}")
    print(f"  Start time: {model_inputs.timestamp[0]}, End time: {model_inputs.timestamp[-1]}")

    # initialize covariance trajectory
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa,
        num_teams=num_teams,
    )

    augmented_results = generate_augmented_data(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain,
    )

    filtered_states, _ = run_filter(
        key=filter_key,
        model_inputs=augmented_results,
        params=params,
        num_teams=num_teams,
        n_particles=n_particles,
    )
    smoothed_states = smoother_rts(
        filtered_states=filtered_states,
        model_inputs=augmented_results,
        params=params,
        num_teams=num_teams,
        key=smoother_key,
        n_trajectories=n_trajectories,
    )
    # Weight-degeneracy diagnostic: ESS per time step of the forward filter.
    ess = _ess_from_log_weights(filtered_states.log_weights)  # (T,)
    return filtered_states, smoothed_states, filtered_states.log_normalizing_constant[-1], augmented_results, ess


def _complete_log_likelihood(
    params: EMParams,
    X: jnp.ndarray,  # (T, num_teams, 2) -- one smoothed trajectory
    model_inputs: RBPFFootballResults,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """log p(y, X) for a single trajectory X (complete-data log-likelihood).

    Returns the three components separately, ``(init_ll, obs_ll, transition_ll)``,
    so callers can either sum them (the MCEM objective) or inspect which term
    drives the loss (the per-epoch diagnostic).

    Includes the initial term ``log p(X_0 | mean_0, gamma_0 (x) B)``, the
    transition terms ``log p(X_t | X_{t-1})`` (OU, ``Q = gamma_0 (x) B``), and
    the observation terms ``log p(y_t | X_t)`` (bivariate Poisson).

    Each term is scaled by its size (number of dimensions it spans) so the
    three terms are on a comparable per-dimension scale:

    - ``init_ll``: a single ``2M``-dim Gaussian, scaled by ``2M``.
    - ``obs_ll``: ``T`` observations, each ``2``-dim (home + away goals),
      scaled by ``2 * T``.
    - ``transition_ll``: ``T-1`` transitions, each ``2M``-dim, scaled by
      ``2M * (T-1)``.
    """
    n_observations = X.shape[0]
    num_teams = X.shape[1]
    K = X.shape[2]  # 2 (attack/defence)
    dim = num_teams * K  # FULL state dimension (2M)
    n_transitions = n_observations - 1

    observation_indices = jnp.arange(n_observations)
    transition_indices = jnp.arange(1, n_observations)

    # --- Initial term: log p(X_0 | mean_0, gamma_0 (x) B), scaled by dim ---
    diff0 = (X[0] - params.mean_0)  # (M, K)
    quad0 = _kron_quad_form(params.gamma_0, params.B, diff0.reshape(1, -1))[0]
    log_det0 = _kron_logdet(params.gamma_0, params.B)
    init_ll = (
        -0.5 * quad0 - 0.5 * log_det0 - 0.5 * dim * jnp.log(2 * jnp.pi)
    ) / dim

    # --- Observation log-likelihood: sum_t log p(y_t | x_t), scaled by 2*T ---
    def obs_step(observation_index):
        return loglik(
            y=jnp.array([model_inputs.home_score[observation_index], model_inputs.away_score[observation_index]]),
            x_i=X[observation_index, model_inputs.home_team_id[observation_index]],
            x_j=X[observation_index, model_inputs.away_team_id[observation_index]],
            alpha=params.alpha,
            beta=params.beta,
            max_goals=MAX_GOALS,
            scale=1.0,  # scale is fixed at 1 (unidentifiable with gamma_0)
        )
    obs_ll = jnp.sum(jax.vmap(obs_step)(observation_indices)) / (2.0 * n_observations)

    # --- Transition log-likelihood: sum_t log p(x_t | x_{t-1}), scaled by 2M*(T-1) ---
    # OU (scalar-phi AR(1)): X_t = mu + phi (X_{t-1} - mu) + eps,
    #   phi = exp(-kappa*dt), eps ~ N(0, (1-phi^2) * gamma_0 (x) B).
    dts = model_inputs.timestamp[transition_indices] - model_inputs.timestamp_prev[transition_indices]
    # Q = gamma_0 (x) B (stationary covariance). We keep it factored: the
    # per-step covariance is scale * (gamma_0 (x) B), so the quadratic form
    # and log-det are computed via the Kronecker helpers (no (2M, 2M) matrix).

    def transition_step(observation_index, dt):
        phi = jnp.exp(-params.kappa * dt)
        pred_mean = params.mean_0 + phi * (X[observation_index - 1] - params.mean_0)  # (M, K)
        diff = X[observation_index] - pred_mean  # (M, K)

        deterministic = dt <= 1e-8
        scale = jnp.maximum(1.0 - phi**2, 1e-8)
        # (scale * gamma_0) (x) B  -- factored, no kron.
        quad = _kron_quad_form(scale * params.gamma_0, params.B, diff.reshape(1, -1))[0]
        log_det = _kron_logdet(scale * params.gamma_0, params.B)
        log_density = -0.5 * quad - 0.5 * log_det - 0.5 * dim * jnp.log(2 * jnp.pi)

        return jnp.where(deterministic, 0.0, log_density)

    transition_ll = jnp.sum(jax.vmap(transition_step)(transition_indices, dts)) / (dim * n_transitions)

    return init_ll, obs_ll, transition_ll


def loss_fn(
    params: EMParams,
    smoothed_trajectories: jnp.ndarray,  # (M, T, num_teams, 2)
    model_inputs: RBPFFootballResults,
    gamma_0_prior: float = 0.0,
):
    """MCEM objective: -average over M trajectories of log p(y, X^*).

    This is a proper Monte Carlo estimate of the EM objective
    Q(theta) = E_{p(X|y,theta_old)}[log p(y, X | theta)], averaged over the
    M smoothed trajectories from the E-step. Each term is scaled by its size
    (see ``_complete_log_likelihood``), so the three terms are on a comparable
    per-dimension scale and summed with equal weight.

    ``scale`` is fixed at 1.0 (it is unidentifiable with ``gamma_0``: scaling
    ``x -> a x``, ``gamma_0 -> a^2 gamma_0``, ``scale -> a scale`` leaves the
    likelihood unchanged), so ``gamma_0`` is free to carry the true
    state-variance scale.

    A **diagonal-only** quadratic shrinkage prior on ``gamma_0`` is added to
    prevent EM from inflating the cross-team variance scale to chase per-match
    noise. Only the diagonal (the per-team variance) is penalized, toward 1
    (the correlation-matrix diagonal); the off-diagonal correlations are left
    free. The term is normalized by the number of teams so it is on a
    comparable per-dimension scale to the data term:

        prior = gamma_0_prior * mean_m (gamma_0[m,m] - 1)^2

    ``gamma_0_prior`` is the prior strength (0 disables it).
    """
    init_ll, obs_ll, transition_ll = jax.vmap(
        lambda X: _complete_log_likelihood(params, X, model_inputs)
    )(smoothed_trajectories)  # each (M,)
    data_loss = -jnp.mean(init_ll + obs_ll + transition_ll)

    # Diagonal-only shrinkage prior on gamma_0 toward the correlation scale (1).
    if gamma_0_prior > 0:
        diag = jnp.diag(params.gamma_0)
        prior = gamma_0_prior * jnp.mean((diag - 1.0) ** 2)
    else:
        prior = 0.0
    return data_loss + prior


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
    """
    L_low = jnp.tril(L)
    diag = jax.nn.softplus(jnp.diag(L_low))  # > 0, strictly
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
    - kappa clamped to [0, _KAPPA_MAX] (forces a mean-reversion half-life of at
      least one week so team strengths persist between matches).
    - gamma_0, B projected onto the positive-definite cone (full-rank, so the
      transition covariance Q and the smoother covariances stay invertible
      and their log-determinants finite).

    Note: during the M-step these are Cholesky-parameterized (so they stay PD
    automatically); this projection is retained as a safety net for params
    constructed outside the optimizer (e.g. hand-loaded values).
    """
    gamma_0 = _project_psd(params.gamma_0)
    B = _project_psd(params.B)
    kappa = jnp.clip(params.kappa, 0.0, _KAPPA_MAX)
    return EMParams(
        mean_0=params.mean_0,
        gamma_0=gamma_0,
        B=B,
        kappa=kappa,
        alpha=params.alpha,
        beta=params.beta,
    )


def M_step(
    smoothed_trajectories: jnp.ndarray,  # (M, T, num_teams, 2)
    model_inputs: RBPFFootballResults,
    prev_params: EMParams,
    learning_rate: float,
    n_gradient_steps: int,
    gamma_0_prior: float = 0.0,
):
    """
    M-step: Update parameters via scale-aware ADAM with a cosine schedule.

    The objective is the MCEM loss: -average over the M smoothed trajectories
    of log p(y, X^*). gamma_0 and B are Cholesky-parameterized so they stay
    positive-definite by construction.

    ``gamma_0_prior`` is the strength of a diagonal-only shrinkage prior on
    ``gamma_0`` (see ``loss_fn``), preventing EM from inflating the cross-team
    variance scale to chase per-match noise.

    Returns:
        tuple[EMParams, float, float, list[float]]: (final_params, loss_start,
        loss_best, loss_trace).
    """
    num_teams = prev_params.gamma_0.shape[0]

    def _loss_and_grad(carry):
        # Reconstruct PD matrices from free Cholesky factors.
        gamma_0 = _psd_from_cholesky(carry["L_gamma0"], num_teams)
        B = _psd_from_cholesky(carry["L_B"], 2)
        return loss_fn(
            EMParams(
                mean_0=prev_params.mean_0,
                gamma_0=gamma_0,
                B=B,
                kappa=carry["kappa"],
                alpha=carry["alpha"],
                beta=carry["beta"],
            ),
            smoothed_trajectories,
            model_inputs,
            gamma_0_prior=gamma_0_prior,
        )

    value_and_grad_fn = jax.jit(jax.value_and_grad(_loss_and_grad, argnums=0))

    # Initial parameter blocks (dict so optax.multi_transform labels align).
    carry = {
        "L_gamma0": _cholesky_from_psd(prev_params.gamma_0, num_teams),
        "L_B": _cholesky_from_psd(prev_params.B, 2),
        "kappa": prev_params.kappa,
        "alpha": prev_params.alpha,
        "beta": prev_params.beta,
    }

    # --- Per-parameter learning rates (scale-aware) ---
    base = learning_rate
    lr_mapping = {
        "L_gamma0": base * 1,
        "L_B": base * 1,
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
    }
    param_labels = {
        "L_gamma0": "L_gamma0",
        "L_B": "L_B",
        "kappa": "kappa",
        "alpha": "alpha",
        "beta": "beta",
    }
    # --- Optimizer ---
    optimizer = optax.chain(
        # Clip the global gradient norm to prevent the explosive first step
        # (the transition log-density gradient ~ Q^{-1} can be enormous when
        # gamma_0 is near-singular). This stabilizes the M-step.
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(transforms, param_labels),
        optax.scale_by_schedule(
            optax.cosine_decay_schedule(1.0, n_gradient_steps)
        ),
    )
    opt_state = optimizer.init(carry)

    # --- JIT-compiled gradient-descent loop (single lax.scan) ---
    def _step(carry, step):
        opt_state, params_carry, best_carry, best_loss = carry
        loss, grads = value_and_grad_fn(params_carry)
        updates, opt_state = optimizer.update(grads, opt_state, params_carry)
        params_carry = optax.apply_updates(params_carry, updates)

        # Track best by relative improvement.
        improved = loss < best_loss * (1 - 1e-4)
        best_carry = jax.tree.map(
            lambda b, p: jnp.where(improved, p, b), best_carry, params_carry
        )
        best_loss = jnp.where(improved, loss, best_loss)
        return (opt_state, params_carry, best_carry, best_loss), loss

    # run the gradient-descent loop, tracking the best parameters and loss
    init_carry = (opt_state, carry, carry, jnp.inf)
    (_, _, best_carry, best_loss), loss_trace = jax.lax.scan(
        _step, init_carry, jnp.arange(n_gradient_steps)
    )
    loss_trace = jnp.asarray(loss_trace)  # (n_gradient_steps,)
    loss_start = loss_trace[0]            # objective at prev_params
    loss_best = best_loss                 # best objective over the M-step

    # Reconstruct PD matrices from the best free factors and project onto support.
    best_gamma_0 = _psd_from_cholesky(best_carry["L_gamma0"], num_teams)
    best_B = _psd_from_cholesky(best_carry["L_B"], 2)
    final = _constrain(EMParams(
        mean_0=prev_params.mean_0,
        gamma_0=best_gamma_0,
        B=best_B,
        kappa=best_carry["kappa"],
        alpha=best_carry["alpha"],
        beta=best_carry["beta"],
    ))
    best_step = int(jnp.argmin(loss_trace))
    print(f"      M-step done: loss {float(loss_best):.4f} -> best at step {best_step}, "
          f"kappa={float(final.kappa):.5f} alpha={float(final.alpha):.5f} beta={float(final.beta):.5f}")
    return final, float(loss_start), float(loss_best), loss_trace


def run_EM(
    model_inputs: FootballResults,
    init_params: EMParams,
    num_teams: int,
    n_particles: int = 10,
    n_epochs: int = 10,
    n_gradient_steps: int = 10,
    learning_rate: float = 1e-3,
    n_trajectories: int = N_TRAJECTORIES,
    gamma_0_prior: float = 0.0,
    key: jax.Array = jax.random.PRNGKey(42),
) -> tuple[EMParams, jnp.ndarray, dict]:
    params = init_params

    log_likelihood_history = []
    mstep_start_history = []
    mstep_end_history = []
    mstep_loss_traces = []  # full loss trajectory per epoch (every gradient step)
    ess_history = []        # per-epoch ESS (T,) of the forward filter
    ll_component_history = []  # per-epoch (init, obs, transition) complete-LL means

    for epoch in tqdm(range(n_epochs)):
        key, e_key = jax.random.split(key)
        print(f"    [EM] Epoch {epoch + 1}/{n_epochs} — E-step starting...")
        # 1. run E step to get M smoothed trajectories
        print("    E-step: Running filtering and backward sampling...")
        _, smoothed_trajectories, log_marginal, augmented_results, ess = E_step(
            params=params,
            model_inputs=model_inputs,
            num_teams=num_teams,
            n_particles=n_particles,
            key=e_key,
            n_trajectories=n_trajectories,
        )
        log_likelihood_history.append(log_marginal)
        ess_history.append(ess)
        # Per-component complete-LL breakdown (averaged over trajectories).
        init_ll, obs_ll, transition_ll = jax.vmap(
            lambda X: _complete_log_likelihood(params, X, augmented_results)
        )(smoothed_trajectories)
        ll_component_history.append(
            (float(jnp.mean(init_ll)), float(jnp.mean(obs_ll)), float(jnp.mean(transition_ll)))
        )
        print(f"    [EM] Epoch {epoch + 1} E-step done: log_marginal={float(log_marginal):.4f}, "
              f"ESS[min/mean]={float(jnp.min(ess)):.1f}/{float(jnp.mean(ess)):.1f}")
        print(f"         complete-LL components (init/obs/transition): "
              f"{float(jnp.mean(init_ll)):.4f} / {float(jnp.mean(obs_ll)):.4f} / "
              f"{float(jnp.mean(transition_ll)):.4f}")
        # 2. run M step to update parameters (MCEM over M trajectories)
        print("    M-step: Updating parameters...")
        params, loss_start, loss_best, loss_trace = M_step(
            smoothed_trajectories=smoothed_trajectories,
            model_inputs=augmented_results,
            prev_params=params,
            learning_rate=learning_rate,
            n_gradient_steps=n_gradient_steps,
            gamma_0_prior=gamma_0_prior,
        )
        mstep_start_history.append(loss_start)
        mstep_end_history.append(loss_best)
        mstep_loss_traces.append(loss_trace)
        print(f"    [EM] Epoch {epoch + 1} M-step done: loss {float(loss_best):.4f}")

        # Free GPU/TPU memory between epochs. The E-step materializes the full
        # (T, M, M) gamma trajectories and (T, N, M, K) filter states, which at
        # M=228 (ACTIVE_TEAMS) is ~8GB. The loop variables ``smoothed_trajectories``
        # and ``augmented_results`` keep those device buffers alive into the next
        # epoch, so the second E-step OOMs. Explicitly drop them and clear the
        # JAX compilation cache so epoch N+1 can reuse epoch N's memory.
        del smoothed_trajectories, augmented_results
        jax.clear_caches()

    print("EM completed. Final parameters:")
    print("  kappa:", params.kappa)
    print("  alpha:", params.alpha)
    print("  beta:", params.beta)
    print("  B:", params.B)
    print("  gamma_0:", params.gamma_0.shape)
    print("  mean_0:", params.mean_0.shape)

    em_diagnostics = {
        "mstep_loss_start": jnp.array(mstep_start_history),
        "mstep_loss_end": jnp.array(mstep_end_history),
        "mstep_loss_trace": mstep_loss_traces,  # list of per-epoch loss lists
        "ess": ess_history,                     # list of per-epoch (T,) ESS arrays
        "ll_components": ll_component_history,  # list of (init, obs, transition) tuples
    }
    return params, jnp.array(log_likelihood_history), em_diagnostics


def main():
    data, model_inputs, team_id_to_name = get_results(
        start_date="1950-01-01",
        end_date="2025-12-31",
        max_goals=MAX_GOALS,
        teams_only=WORLDCUP_2026_TEAMS,
        include_friendly=False,
    )
    NUM_TEAMS = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    params = default_init_params(NUM_TEAMS, team_id_to_name=team_id_to_name)

    try:
        final_params, log_marginal_likelihoods, em_diagnostics = run_EM(
            model_inputs=model_inputs,
            init_params=params,
            num_teams=NUM_TEAMS,
            n_particles=N,
            n_epochs=3,
            n_gradient_steps=10,
            learning_rate=0.01,
            key=key,
        )
    except Exception as e:
        print("Error during EM run:", e)
        raise
    # save parameters to JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "smoothing")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + "/em_params.json", "w") as f:
        json.dump(params_to_dict(final_params), f, indent=2)
    with open(output_path + "/log_marginal_likelihoods.json", "w") as f:
        json.dump(np.asarray(log_marginal_likelihoods).tolist(), f, indent=2)

    # plot log marginal likelihoods
    plot_log_likelihood_history(log_marginal_likelihoods.tolist(), output_path=output_path + "/log_marginal_likelihoods.png")


if __name__ == "__main__":
    main()
