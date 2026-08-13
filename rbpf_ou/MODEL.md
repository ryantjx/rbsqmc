# OU-RBPF Model

This is the **Ornstein–Uhlenbeck (scalar-$\phi$ AR(1))** variant of the Rao-Blackwellized Particle Filter (RBPF) for football. It is the successor to the random-walk model documented in [`rbpf/MODEL.md`](../rbpf/MODEL.md). The key difference is the **mean-reverting transition**: instead of a random walk, each team's latent strength $X_t^m$ drifts back toward a shared mean $\mu_0$ at rate $\kappa$.

We model the latent attack/defence strengths of $M$ football teams, $X_t^m = (X_t^{m,\text{att}}, X_t^{m,\text{def}}) \in \mathbb{R}^2$ for team $m$ at time $t$. Observed goals are modelled by a bivariate Poisson distribution.

> **Scope note.** This document is a *math + practical* hybrid: it gives the formal equations (Sections 1–4) and then a practical section (Section 5) describing the actual code layout, the parameter set, and the known issues that currently prevent the EM from converging. The known-issues section is the starting point for the improvement plan.

---

## 1 Setup

**Initial**

$X_0^m = (X_0^{m,\text{att}}, X_0^{m,\text{def}})$ is the initial latent state for team $m$, distributed as

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0), \qquad \Sigma_0 = \Gamma_0 \otimes B$$

where $\Gamma_0 \in \mathbb{R}^{M \times M}$ is the **team covariance** (built from a regional-correlation prior) and $B \in \mathbb{R}^{2 \times 2}$ is the **shared attack/defence factor**. The Kronecker structure $\Sigma = \Gamma \otimes B$ means all teams share the same attack/defence covariance $B$, while the team-to-team correlations live in $\Gamma$.

**Transition (OU / scalar-$\phi$ AR(1))**

$$X_t = \mu_0 + \phi_t (X_{t-1} - \mu_0) + \epsilon_t, \qquad \phi_t = e^{-\kappa \, \Delta_t}, \qquad \epsilon_t \sim \mathcal{N}\bigl(0, (1 - \phi_t^2)\, \Sigma_0\bigr)$$

where $\Delta_t$ is the elapsed time between matches and $\kappa > 0$ is the **mean-reversion rate**. As $\kappa \to 0$, $\phi_t \to 1$ and the transition degenerates toward a random walk; as $\kappa \to \infty$, $\phi_t \to 0$ and the state snaps back to the mean. The transition covariance is a convex combination of the current posterior and the stationary covariance $\Sigma_0$:

$$\Sigma_{t \mid t-1} = \phi_t^2\, \Sigma_{t-1 \mid t-1} + (1 - \phi_t^2)\, \Sigma_0$$

This is positive-definite by construction (a positive-weighted sum of two PD matrices) and team-specific: heavily-observed teams have small posterior covariance, so their transition noise is small.

**Likelihood**

$y_t = (y_t^{\text{h}}, y_t^{\text{a}})$ is the observed goals for the home $X_t^{\text{h}}$ and away $X_t^{\text{a}}$ teams at time $t$.

$$G_t(y_t \mid x_t^{\text{h}}, x_t^{\text{a}}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{\text{h}}}}{y_t^{\text{h}}!} \frac{\lambda_2^{y_t^{\text{a}}}}{y_t^{\text{a}}!} \sum_{k=0}^{\min(y_t^{\text{h}}, y_t^{\text{a}})} \binom{y_t^{\text{h}}}{k} \binom{y_t^{\text{a}}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k$$

where

$$\lambda_1 = \exp\!\Bigl(\alpha + \tfrac{x_t^{\text{att},\text{h}} - x_t^{\text{def},\text{a}}}{\text{scale}}\Bigr), \qquad \lambda_2 = \exp\!\Bigl(\alpha + \tfrac{x_t^{\text{att},\text{a}} - x_t^{\text{def},\text{h}}}{\text{scale}}\Bigr), \qquad \lambda_3 = \exp(\beta)$$

with $\text{scale} = 1.0$ fixed (it is unidentifiable with $\Gamma_0$ — see Section 4). Since only $X_t^{\mathcal{O}_t} = (X_t^{\text{h}}, X_t^{\text{a}})$ enter the likelihood, the remaining latent states are represented analytically as a Gaussian conditional (Rao–Blackwellization).

---

## 2 RB-PF Filter

The filter is a **bootstrap** particle filter (proposal = transition) with a Rao–Blackwellized Gaussian conditional for the unobserved teams. The team-covariance trajectory is **deterministic** — it does not depend on the particle states — so it is precomputed once with a single `lax.scan` (`compute_gamma_trajectory`).

**1 Prediction (OU)**

$$\mu_{t \mid t-1} = \mu_0 + \phi_t (X_{t-1} - \mu_0), \qquad \phi_t = e^{-\kappa \Delta_t}$$

$$\Gamma_{t \mid t-1} = \phi_t^2\, \Gamma_{t-1 \mid t-1} + (1 - \phi_t^2)\, \Gamma_0$$

**2 Bootstrap particle sampling.** The proposal equals the transition, $q(X_t \mid X_{t-1}) = p(X_t \mid X_{t-1})$. Only the observed block (home + away, 4 dims) is sampled from the prediction Gaussian $\Sigma_{EE} = \Gamma_{EE} \otimes B$:

$$X_t^{\mathcal{O}_t} \sim \mathcal{N}\bigl(\mu_{t \mid t-1}^{\mathcal{O}_t},\; \Gamma_{t \mid t-1}^{\mathcal{O}_t \mathcal{O}_t} \otimes B\bigr)$$

**3 Compute weights**

$$\log \tilde{w}_t^{(i)} = \log w_{t-1}^{(i)} + \log G_t\bigl(y_t \mid X_t^{\mathcal{O}_t,(i)}\bigr)$$

**4 Exact marginalization (Rao–Blackwellization).** Condition the remaining teams on the sampled observed block via the Kalman gain in team space:

$$X_t^{\mathcal{R}_t} \mid X_t^{\mathcal{O}_t} \sim \mathcal{N}\bigl(\mu_{t \mid t-1}^{\mathcal{R}_t \mid \mathcal{O}_t},\; \Gamma_{t \mid t-1}^{\mathcal{R}_t \mathcal{R}_t \mid \mathcal{O}_t}\bigr)$$

- $\mu_{t \mid t-1}^{\mathcal{R}_t \mid \mathcal{O}_t} = \mu_{t \mid t-1}^{\mathcal{R}_t} + K_t\, (X_t^{\mathcal{O}_t} - \mu_{t \mid t-1}^{\mathcal{O}_t})$
- $\Gamma_{t \mid t-1}^{\mathcal{R}_t \mathcal{R}_t \mid \mathcal{O}_t} = \Gamma_{t \mid t-1}^{\mathcal{R}_t \mathcal{R}_t} - K_t\, \Gamma_{t \mid t-1}^{\mathcal{O}_t \mathcal{R}_t}$

with $K_t = \Gamma_{t \mid t-1}[:, \mathcal{O}_t]\, \Gamma_{t \mid t-1}[\mathcal{O}_t, \mathcal{O}_t]^{-1}$. The observed teams' posterior rows/cols are zeroed (Schur-complement marginalization), so their posterior variance is exactly zero.

**5 Resampling.** Systematic resampling (`cuthbertlib.resampling.systematic`), applied **unconditionally at every time step**. The resampler resets the log-weights to zero (uniform), so the weights stored at each `t` reflect only the current step's observation potential — degeneracy is per-step, not cumulative.

> **PSD-aware sampling.** The filtered posterior covariances are positive-*semi*-definite (observed teams have exact-zero variance). A Cholesky-based `multivariate_normal` returns NaN on such singular matrices, so the code eigendecomposes, clips tiny eigenvalues to zero, and samples noise only in the nonzero-variance directions (`_sample_psd_gaussian` / `_kron_sample_psd`).

---

## 3 RB-PF Smoothing (FFBSi)

We use **Forward Filtering Backward Simulation** (FFBSi) to sample backwards in time from the smoothing distribution $p(X_{0:T} \mid y_{1:T})$. The smoother samples $M$ independent trajectories in parallel (`smoother_rts`, vmap over `n_trajectories` keys).

**1 Initialization.** Sample particle index $I_T$ from the normalized weights $w_T^{(i)}$, then draw the full terminal state from the corresponding Gaussian:

$$X_T^{*} \sim \mathcal{N}\bigl(\mu_T^{(I_T)},\; \Gamma_T \otimes B\bigr)$$

**2 Backward sampling.** From $T-1$ down to $0$, sample particle index $I_t$ from the backward kernel and draw the full state from the RTS posterior:

$$p(X_t \mid X_{t+1}^*, y_{1:t}) \propto p(X_{t+1}^* \mid X_t)\, p(X_t \mid y_{1:t})$$

The backward weights are

$$w_{t \mid t+1}^{(i)} \propto w_t^{(i)}\, \mathcal{N}\bigl(X_{t+1}^* \mid \mu_{t+1 \mid t}^{(i)},\; \Gamma_{t+1 \mid t} \otimes B\bigr)$$

and the RTS update (with the OU $\phi$ factor) is

$$\mu_{t \mid t+1}^{(i)} = \mu_t^{(i)} + J_t\, \bigl(X_{t+1}^* - \mu_{t+1 \mid t}^{(i)}\bigr)$$

$$\Gamma_{t \mid t+1} = \Gamma_t - J_t\, \Gamma_{t+1 \mid t}\, J_t^T$$

where the RTS gain, using the Kronecker identity $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ and $\text{pinv}(\Gamma \otimes B) = \text{pinv}(\Gamma) \otimes \text{pinv}(B)$, collapses to a team-space gain times identity:

$$J_t = \bigl(\phi_t\, \Gamma_t\, \text{pinv}(\Gamma_{t+1 \mid t})\bigr) \otimes I_2 = J_{\Gamma,t} \otimes I_2$$

so the Kronecker matvec $(J_{\Gamma,t} \otimes I_2)\,\text{vec}_C(S) = \text{vec}_C(J_{\Gamma,t} S)$ never materializes the full $(2M, 2M)$ matrix.

> **Deterministic `dt == 0`.** When two matches share a timestamp ($\Delta_t = 0$, $\phi_t = 1$, $Q = 0$), the state is preserved exactly: $X_t^* = X_{t+1}^*$ with no sampling (`lax.cond` guards the sampling branch).

---

## 4 Parameter Estimation — Monte Carlo EM (MCEM)

Let $\Theta = (\mu_0, \Gamma_0, B, \kappa, \alpha, \beta)$ be the model parameters. We estimate $\Theta$ by **Monte Carlo Expectation-Maximization** (MCEM).

**1 E-step.** Approximate the smoothing distribution $p(X_{0:T} \mid y_{1:T}, \Theta^{(k)})$ via FFBSi (Section 3), then estimate the expected complete log-likelihood by Monte Carlo over the $M$ smoothed trajectories:

$$A(\Theta \mid \Theta^{(k)}) \approx \frac{1}{M} \sum_{i=1}^{M} \log p\bigl(X_{0:T}^{(i)}, y_{1:T} \mid \Theta\bigr)$$

The complete-data log-likelihood for one trajectory is

$$\log p(X_{0:T}, y_{1:T} \mid \Theta) = \underbrace{\log p_{\mu_0, \Gamma_0 \otimes B}(X_0)}_{\text{init}} + \sum_{t=1}^{T} \underbrace{\log p_{\kappa, \Gamma_0 \otimes B}(X_t \mid X_{t-1})}_{\text{transition}} + \sum_{t=1}^{T} \underbrace{\log p_{\alpha, \beta}(y_t \mid X_t^{\mathcal{O}_t})}_{\text{observation}}$$

**2 M-step.** Maximize $A(\Theta \mid \Theta^{(k)})$ with respect to $\Theta$ via **scale-aware ADAM** with a cosine schedule and global-norm gradient clipping. The covariance matrices $\Gamma_0, B$ are **Cholesky-parameterized** so they stay positive-definite by construction; $\kappa$ is clamped to $[0, \text{\_KAPPA\_MAX}] = [0, 0.02]$ to force a mean-reversion half-life of about a month ($t_{1/2} = \ln 2 / \kappa \approx 35$ days), so team quality persists over a tournament.

**Loss scaling.** Each term of the complete-data log-likelihood is divided by the number of dimensions it spans, so the three terms are on a comparable per-dimension scale and summed with equal weight:

- `init_ll`: a single $2M$-dim Gaussian, scaled by $2M$.
- `obs_ll`: $T$ observations, each 2-dim (home + away goals), scaled by $2T$.
- `transition_ll`: $T-1$ transitions, each $2M$-dim, scaled by $2M(T-1)$.

The observation term's influence is instead controlled by the `scale` hyperparameter: smaller `scale` amplifies the team-strength signal in the goal rates, giving the observation term more gradient weight without a manual loss-weighting hack.

**Scale identifiability (resolved).** $\Gamma_0$ and `scale` are jointly unidentifiable: scaling $x \to a x$, $\Gamma_0 \to a^2 \Gamma_0$, `scale` $\to a \cdot$`scale` leaves the likelihood unchanged (team strengths only enter the goal rates as $(x_{\text{att}} - x_{\text{def}})/\text{scale}$, and the Gaussian terms depend on $\Gamma_0$ only through the ratio of the state to its covariance). EM could otherwise shrink $\Gamma_0 \to 0$ along this flat direction, collapsing the state variance to ~0. **Fix:** `scale` is fixed at 1.0 (removed as a parameter), which cuts off the flat direction — EM can no longer shrink `scale` to compensate, so it is forced to keep $\Gamma_0$ at a meaningful scale. `gamma_0` is now free to be estimated from the data.

---

## 5 Practical Reference

### 5.1 Code layout

| File | Responsibility |
|------|----------------|
| `src/model.py` | `propagate_sample` (OU prediction + Kalman update), `compute_gamma_trajectory` (deterministic covariance trajectory via `lax.scan`), `run_filter`, `build_rbpf_filter`. |
| `src/smoothing.py` | `E_step` (FFBSi), `M_step` (optax `multi_transform`), `run_EM`, `_complete_log_likelihood`, `loss_fn`, Kronecker-aware helpers (`_kron_quad_form`, `_kron_logdet`, `_kron_sample_psd`). |
| `src/helpers.py` | `default_init_params` (regional-correlation prior), `params_to_dict` / `params_from_dict`, `save_params` / `load_params`. |
| `src/bivariate_poisson.py` | `loglik` / `loglik_grid` (bivariate Poisson score likelihood). |
| `src/data.py` | `get_results` with `ACTIVE_TEAMS` (228) and `WORLDCUP_2026_TEAMS` (48) sets. |
| `smoothing_gpu.py` | Colab bootstrap + EM core; config loaded from `smoothing_gpu_config.json`. |
| `model_trained.py` | Loads trained params, runs the filter, produces 5 PNGs + CSV. |

### 5.2 Parameter set

| Param | Shape | Meaning | Init |
|-------|-------|---------|------|
| `mean_0` | $(M, 2)$ | shared initial/stationary mean | zeros |
| `gamma_0` | $(M, M)$ | stationary team covariance (regional prior) | regional correlation |
| `B` | $(2, 2)$ | shared attack/defence covariance | $\begin{bmatrix}1 & 0.2\\0.2 & 1\end{bmatrix}$ |
| `kappa` | scalar | OU mean-reversion rate (clamped to $[0, 0.02]$) | 0.01 |
| `alpha` | scalar | baseline scoring rate | 0.2 |
| `beta` | scalar | shared-scoring / correlation rate | −4.0 |

### 5.3 Training configuration (`smoothing_gpu_config.json`)

| Key | Value | Meaning |
|-----|-------|---------|
| `N` | 300 | number of filter particles |
| `n_epochs` | 10 | number of EM epochs |
| `n_gradient_steps` | 30 | ADAM steps per M-step |
| `learning_rate` | 0.01 | base ADAM learning rate |
| `n_trajectories` | 12 | smoothed trajectories per E-step (MCEM) |
| `teams` | `WORLDCUP_2026_TEAMS` | 48-team set (L4 run) |
| `hardware` / `gpu_type` | `gpu` / `L4` | Colab L4 |

The number of smoothed trajectories per E-step, `N_TRAJECTORIES = 8`, is a **module-level constant** in `src/smoothing.py` (the config's `n_trajectories` overrides it via `run_EM`).

**Diagnostics.** Each EM run now writes, into `em_mstep_diagnostics.json`:
- `ess`: per-epoch, per-time-step effective sample size of the forward filter (weight-degeneracy diagnostic).
- `ll_components`: per-epoch means of the init/obs/transition complete-LL terms (which term drives the loss).

A small-M CPU config for fast local iteration lives in `smoothing_gpu_config_cpu_small.json` (48 WC2026 teams); select it with `RBSQMC_CONFIG=...`.

---

## 6 Known Issues (why EM is not converging)

The A100 run (`outputs_gpu_active/`) originally showed three failure signals: the log marginal likelihood was flat/declining across epochs (`[-50513, -50515, -50716, ...]`), the parameters barely moved from their initial values, and the M-step loss reset to ~1.6 at the start of every epoch (a signature of Monte Carlo noise dominating the Q-function estimate). Most of the root causes have since been fixed (see Section 6.2). The one that remains open:

1. **Bootstrap proposal.** The filter uses the transition as the proposal, ignoring the observation. This causes per-step weight degeneracy when the observation is informative (the proposal rarely lands near the observed score). A natural next step is an auxiliary/guided proposal that incorporates the observation (e.g. a locally-optimal or a mixture proposal), at the cost of a more complex importance weight.

### 6.1 Post-convergence issues (L4 run, `outputs_gpu_l4/`)

The L4 run (N=300, n_trajectories=12, `scale` free) converged — log marginal `[-5315, ..., -5303]`, ESS healthy (~220/300), prediction mean log-likelihood −3.41 — but the **rankings were still noise** (Egypt, US, Saudi Arabia, Haiti top; Portugal, Belgium, Switzerland bottom). Two root causes, both now addressed:

1. **`kappa` was too large (0.58).** The OU half-life is $t_{1/2} = \ln 2 / \kappa \approx 1.2$ days, so a team's strength reverted to the mean almost immediately between matches (median gap 1 day, mean 5.3 days). This destroys persistence and makes rankings noise. **Fix:** clamp $\kappa$ to `_KAPPA_MAX = 0.02` (half-life ≈ 35 days) in `_constrain`.
2. **State variance collapsed to ~0.2.** The stationary state variance is $\Gamma_0[i,i] \cdot B[k,k]$. EM drove $\Gamma_0 \to 0$ along the $\Gamma_0$/`scale` scale-identifiability flat direction, so attack/defence states hovered near the mean with almost no spread. **Fix:** `scale` is now fixed at 1.0 (removed as a parameter), cutting off the flat direction so EM keeps $\Gamma_0$ at a meaningful scale (see Section 4).

> **Note on $\mu_0$.** $\mu_0$ is unidentifiable from the likelihood: team strengths only appear in *differences* ($x_i^{\text{att}} - x_j^{\text{def}}$), so shifting all teams' strengths by a constant leaves the goal rates unchanged. It is therefore held fixed at zero.

### 6.2 Resolved issues

The following issues from the original "why EM is not converging" list have been fixed and verified on the L4 run (`outputs_gpu_l4/`):

1. **`N_TRAJECTORIES = 2` was far too low.** The MCEM Q-function estimate was an average over only 2 smoothed trajectories, so it was dominated by Monte Carlo noise. **Fix:** raised to `n_trajectories = 12` in the config (module default `N_TRAJECTORIES = 8`). The L4 run converged with a healthy ESS (~220/300).
2. **`gamma_Q` was dead weight.** It was estimated in the M-step but never used: both the transition log-likelihood (`_complete_log_likelihood`) and the filter's covariance trajectory (`compute_gamma_trajectory`) use `gamma_0`, not `gamma_Q`. **Fix:** `gamma_Q` has been **removed entirely** from the parameter set, the filter, and the M-step — it no longer exists in the model.
3. **Loss-scaling imbalance.** The transition term was scaled by $2M(T-1) \approx 2.3\text{M}$ dimensions while the observation term was scaled by $2T \approx 10\text{K}$, so the M-step effectively ignored `alpha`/`beta`. **Fix:** each term is now scaled by the number of dimensions it spans (per-dimension scaling), putting the three terms on a comparable scale.
4. **`learning_rate = 0.001` + global-norm clip too small.** Combined with the noise, the M-step made negligible progress per epoch. **Fix:** raised `learning_rate` to `0.01` in the config.
5. **`N = 50` particles for $M = 228$ teams was too low.** **Fix:** raised `N` to `300` (and switched to the 48-team `WORLDCUP_2026_TEAMS` set on L4 to avoid OOM).
6. **ESS diagnostics were missing.** The filter did not report ESS, so per-step degeneracy was invisible. **Fix:** now tracked via `_ess_from_log_weights` in `E_step` (see Section 5.3).

### 6.3 Hacks in the codebase

A review of `rbpf_ou/` surfaced a number of ad-hoc workarounds, hardcoded constants, and numerical fudges. They fall into a few families. The most consequential are flagged with ⚠️.

**Numerical fudges (eigenvalue floors & epsilons).** These keep the Kronecker solves/log-dets finite in float32 when covariances are near-singular or slightly indefinite:

- `_EIGEN_FLOOR = 1e-4` (`smoothing.py`) — clamps eigenvalues of projected covariances.
- `_project_psd_small` / `_pinv_psd` default `floor = 1e-6` (`smoothing.py`) — eigenvalue floors for the smoother's prediction covariance and the RTS pseudo-inverse.
- `_kron_sample_psd` clips eigenvalues to `>= 0` (`smoothing.py`) — PSD-aware sampling.
- `_sample_psd_gaussian` clips eigenvalues to `>= 0` (`model.py`) — same idea in the filter.
- `scale = jnp.maximum(1.0 - phi**2, 1e-8)` and `dt <= 1e-8` deterministic branch (`smoothing.py`) — guards the degenerate `dt == 0` transition.
- `jnp.log(jnp.exp(diag) - 1.0 + 1e-10)` in `_cholesky_from_psd` (`smoothing.py`) — inverts softplus on the Cholesky diagonal.
- `std_safe = np.where(std > 1e-10, std, 1.0)` + `np.clip(corr, -1, 1)` (`model_trained.py`) — guards a correlation-matrix plot against zero-variance teams.

**Ad-hoc reparameterizations & clamps.** These pin parameters that EM would otherwise drive to degenerate values:

- `_KAPPA_MAX = 0.02` (`smoothing.py`) — hard-clamps the mean-reversion rate to force a ~1-month half-life so team quality persists. This is a *constraint*, not an estimate: EM is not free to find the true `kappa`. Kept deliberately as a constraint.
- `_psd_from_cholesky` / `_cholesky_from_psd` softplus diagonal (`smoothing.py`) — Cholesky reparameterization so PD matrices stay PD by construction.

**Loss & optimization hacks.** These make the M-step behave:

- ⚠️ Per-dimension loss scaling in `_complete_log_likelihood` (`smoothing.py`) — divides each term by the number of dimensions it spans so the three terms are comparable. This is an arbitrary reweighting that (before the fix) made the M-step ignore the observation terms. A key hurdle for per-parameter learning rates — see Section 6.4.
- `optax.clip_by_global_norm(1.0)` (`smoothing.py`) — global gradient clipping to prevent explosive first steps.
- `improved = loss < best_loss * (1 - 1e-4)` (`smoothing.py`) — arbitrary relative-improvement threshold for tracking the best params.
- `lr_mapping` sets every parameter's learning-rate multiplier to `1.0` (`smoothing.py`) — the "scale-aware" per-parameter LR is effectively dead (all equal). See Section 6.4.

**MCEM / particle-count magic numbers.** Hand-tuned counts that trade noise against compute. Kept, since the right value depends on the GPU/CPU compute budget:

- `N_TRAJECTORIES = 8` (module default) / `n_trajectories = 12` (config) (`smoothing.py`) — number of smoothed trajectories per E-step.
- `N = 100` (`smoothing.py`) / `N = 10` (`model.py`) / `N = 300` (config) — filter particle counts, hand-tuned.

**Filter initialization.** `init_sample` now draws each particle from the prior $X_0 \sim \mathcal{N}(\mu_0, \Gamma_0 \otimes B)$ via `kron_sample_psd` (`model.py`). Previously it returned the mean with no dispersion, so all particles started identical and the filter was fully collapsed at `t=0`.

**Memory workarounds.** `del smoothed_trajectories, augmented_results; jax.clear_caches()` between epochs (`smoothing.py`) — explicit buffer dropping to avoid OOM at `M=228`. Also `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `ALLOCATOR=platform` in `smoothing_gpu.py`. This is safe: dropping the Python references frees the JAX device buffers, and `clear_caches()` only forces a recompile on the next call.

**Data-processing hacks** (`data.py`):
- Hardcoded **Morocco–Senegal 2026-01-18 score fix** — a one-off data correction baked into `get_results`.
- `fillna(-1)` sentinel for missing scores, then `astype(int)`.
- `timestamp_prev = timestamp.shift(1).fillna(0)` — the first match's previous timestamp is arbitrarily 0.

**Colab bootstrap hacks** (`smoothing_gpu.py`):
- `git reset --hard origin/main` — destructive force-reset of the remote checkout on every run. Safe because it is gated to the Colab bootstrap on a fresh/disposable VM and never runs on a local checkout.
- `_DEFAULT_CONFIG` fallback dict, `_candidate_config_paths` stale-config removal, `__file__` fallback to `os.getcwd()`.
- `_repo_root()` checks for a `"rbpf"` directory (the *old* package name) to detect the Colab clone — a stale reference.
- `PRNGKey(42)` fixed seed everywhere (`smoothing.py`, `model.py`, `model_predict.py`, `model_trained.py`).

**Config hacks.** Hardcoded date ranges (`2000-01-01` → `2025-12-31`), `max_goals = 8`, and team sets in the JSON configs; `run_smoothing_colab.sh` parses JSON with `python3` (no `jq` dependency).

> **Note.** Several of these (the eigenvalue floors, PSD-aware sampling, Cholesky reparameterization, `dt == 0` branch) are legitimate numerical robustness measures for a Kronecker-structured filter in float32, not hacks per se. The one that most warrants revisiting is the per-dimension loss scaling (and the dead per-parameter LR) — see Section 6.4.

### 6.4 Open hurdles

**1. Per-parameter learning rates (key hurdle).** The M-step currently uses a single `learning_rate` for every parameter (`lr_mapping` sets all multipliers to `1.0`), so the "scale-aware" per-parameter LR is effectively dead. The natural improvement is to give each parameter its own learning rate — e.g. a smaller LR for the covariance factors (`gamma_0`, `B`) and a larger one for the scalar observation parameters (`alpha`, `beta`, `kappa`). The key hurdle is that the **per-dimension loss scaling** (Section 6.3) already reweights the three loss terms, so the effective gradient magnitudes per parameter are a *combination* of the loss scaling and the LR. Tuning per-parameter LRs therefore interacts with the loss scaling, and the two must be considered together — you cannot set sensible per-parameter LRs without first deciding whether the loss scaling is the right objective.

**2. Collapsed `init_sample` (resolved).** `init_sample` previously returned all particles exactly at `mean_0` with no dispersion, so the filter was fully collapsed at `t=0`. `cuthbert`'s `init_prepare` vmaps `init_sample` over `n_filter_particles` keys, so it is called once per particle — the fix draws each particle from the prior $X_0 \sim \mathcal{N}(\mu_0, \Gamma_0 \otimes B)$ via `kron_sample_psd`, giving independent dispersion across particles. Propagation is handled separately by `propagate_sample` at each subsequent time step.
