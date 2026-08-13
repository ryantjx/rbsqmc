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

with $\text{scale} = 1.0$ in the current code. Since only $X_t^{\mathcal{O}_t} = (X_t^{\text{h}}, X_t^{\text{a}})$ enter the likelihood, the remaining latent states are represented analytically as a Gaussian conditional (Rao–Blackwellization).

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

Let $\Theta = (\mu_0, \Gamma_0, \Gamma_Q, B, \kappa, \alpha, \beta)$ be the model parameters. We estimate $\Theta$ by **Monte Carlo Expectation-Maximization** (MCEM).

**1 E-step.** Approximate the smoothing distribution $p(X_{0:T} \mid y_{1:T}, \Theta^{(k)})$ via FFBSi (Section 3), then estimate the expected complete log-likelihood by Monte Carlo over the $M$ smoothed trajectories:

$$A(\Theta \mid \Theta^{(k)}) \approx \frac{1}{M} \sum_{i=1}^{M} \log p\bigl(X_{0:T}^{(i)}, y_{1:T} \mid \Theta\bigr)$$

The complete-data log-likelihood for one trajectory is

$$\log p(X_{0:T}, y_{1:T} \mid \Theta) = \underbrace{\log p_{\mu_0, \Gamma_0 \otimes B}(X_0)}_{\text{init}} + \sum_{t=1}^{T} \underbrace{\log p_{\kappa, \Gamma_0 \otimes B}(X_t \mid X_{t-1})}_{\text{transition}} + \sum_{t=1}^{T} \underbrace{\log p_{\alpha, \beta}(y_t \mid X_t^{\mathcal{O}_t})}_{\text{observation}}$$

**2 M-step.** Maximize $A(\Theta \mid \Theta^{(k)})$ with respect to $\Theta$ via **scale-aware ADAM** with a cosine schedule and global-norm gradient clipping. The covariance matrices $\Gamma_0, \Gamma_Q, B$ are **Cholesky-parameterized** so they stay positive-definite by construction; $\kappa$ is clamped to $[\text{\_KAPPA\_MIN}, \text{\_KAPPA\_MAX}] = [10^{-3}, 0.1]$ to keep the transition covariance non-degenerate **and** to force a mean-reversion half-life of at least one week ($t_{1/2} = \ln 2 / \kappa \ge 7$ days).

**Loss scaling.** Each term of the complete-data log-likelihood is divided by the number of dimensions it spans, so the three terms are on a comparable per-dimension scale and summed with equal weight:

- `init_ll`: a single $2M$-dim Gaussian, scaled by $2M$.
- `obs_ll`: $T$ observations, each 2-dim (home + away goals), scaled by $2T$.
- `transition_ll`: $T-1$ transitions, each $2M$-dim, scaled by $2M(T-1)$.

The observation term's influence is instead controlled by the `scale` hyperparameter: smaller `scale` amplifies the team-strength signal in the goal rates, giving the observation term more gradient weight without a manual loss-weighting hack.

Two quadratic shrinkage priors keep the covariance parameters from collapsing to degenerate (near-zero) values:

- A prior on $\Gamma_Q$ toward a small scaled identity (retained for backward compatibility; $\Gamma_Q$ is currently held fixed in the M-step, so this prior is effectively inert).
- A prior on $\Gamma_0$ toward the initial regional-correlation prior (`gamma_0_prior`, default 0.1). Without it, EM drives $\Gamma_0 \to 0$, collapsing the stationary state variance so team strengths hover near the mean with almost no spread. This prior is what keeps the attack/defence state variance at a meaningful scale.

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
| `gamma_Q` | $(M, M)$ | transition team covariance factor | small scaled identity |
| `B` | $(2, 2)$ | shared attack/defence covariance | $\begin{bmatrix}1 & 0.2\\0.2 & 1\end{bmatrix}$ |
| `kappa` | scalar | OU mean-reversion rate (clamped to $[10^{-3}, 0.1]$) | 0.01 |
| `alpha` | scalar | baseline scoring rate | 0.2 |
| `beta` | scalar | shared-scoring / correlation rate | −4.0 |
| `scale` | scalar | team-strength influence on goal rates (free param, clamped to $[0, 10]$) | 1.0 |

### 5.3 Training configuration (`smoothing_gpu_config.json`)

| Key | Value | Meaning |
|-----|-------|---------|
| `N` | 300 | number of filter particles |
| `n_epochs` | 10 | number of EM epochs |
| `n_gradient_steps` | 30 | ADAM steps per M-step |
| `learning_rate` | 0.01 | base ADAM learning rate |
| `n_trajectories` | 12 | smoothed trajectories per E-step (MCEM) |
| `gamma_0_prior` | 0.1 | shrinkage prior on $\Gamma_0$ toward the regional prior (keeps state variance from collapsing) |
| `teams` | `WORLDCUP_2026_TEAMS` | 48-team set (L4 run) |
| `hardware` / `gpu_type` | `gpu` / `L4` | Colab L4 |

The number of smoothed trajectories per E-step, `N_TRAJECTORIES = 8`, is a **module-level constant** in `src/smoothing.py` (the config's `n_trajectories` overrides it via `run_EM`).

**Diagnostics.** Each EM run now writes, into `em_mstep_diagnostics.json`:
- `ess`: per-epoch, per-time-step effective sample size of the forward filter (weight-degeneracy diagnostic).
- `ll_components`: per-epoch means of the init/obs/transition complete-LL terms (which term drives the loss).

A small-M CPU config for fast local iteration lives in `smoothing_gpu_config_cpu_small.json` (48 WC2026 teams); select it with `RBSQMC_CONFIG=...`.

---

## 6 Known Issues (why EM is not converging)

The A100 run (`outputs_gpu_active/`) shows three failure signals: the log marginal likelihood is flat/declining across epochs (`[-50513, -50515, -50716, ...]`), the parameters barely move from their initial values, and the M-step loss resets to ~1.6 at the start of every epoch (a signature of Monte Carlo noise dominating the Q-function estimate). Root causes, in rough order of impact:

1. **`N_TRAJECTORIES = 2` is far too low.** The MCEM Q-function estimate is an average over only 2 smoothed trajectories, so it is dominated by Monte Carlo noise. This is the single biggest lever. Raising it to 8–16 (or more) is the first fix.
2. **`gamma_Q` is dead weight.** It is estimated in the M-step and appears in the parameter set, but it is **never used**: both the transition log-likelihood (`_complete_log_likelihood`) and the filter's covariance trajectory (`compute_gamma_trajectory`) use `gamma_0`, not `gamma_Q`. The M-step is wasting gradient effort on a parameter that has no effect on the objective's data terms.
3. **Loss-scaling imbalance.** The transition term is scaled by $2M(T-1) \approx 2.3\text{M}$ dimensions while the observation term is scaled by $2T \approx 10\text{K}$. The M-step therefore effectively ignores `alpha`/`beta` (the observation parameters), which is why they barely move.
4. **`learning_rate = 0.001` + global-norm clip too small.** Combined with the noise, the M-step makes negligible progress per epoch.
5. **`N = 50` particles for $M = 228$ teams is too low.** The filter resamples unconditionally every time step (systematic resampling resets weights to uniform), so degeneracy is *per-step*, not cumulative. But a low per-step ESS still means the observation potential concentrates mass onto few particles.
6. **Bootstrap proposal.** Using the transition as the proposal ignores the observation, causing per-step weight degeneracy when the observation is informative (the proposal rarely lands near the observed score).
7. **ESS diagnostics were missing.** The filter did not report ESS, so the per-step degeneracy was invisible. Now tracked via `_ess_from_log_weights` in `E_step` (see Section 5.3).

These are addressed by the improvement plan (Phase 2: diagnostics; Phase 3: fix `gamma_Q`, raise `N_TRAJECTORIES`, rebalance loss, tune config; Phase 4: small-M CPU iteration then promote to A100).

### 6.1 Post-convergence issues (L4 run, `outputs_gpu_l4/`)

The L4 run (N=300, n_trajectories=12, `scale` free) converged — log marginal `[-5315, ..., -5303]`, ESS healthy (~220/300), prediction mean log-likelihood −3.41 — but the **rankings were still noise** (Egypt, US, Saudi Arabia, Haiti top; Portugal, Belgium, Switzerland bottom). Two root causes, both now addressed:

1. **`kappa` was too large (0.58).** The OU half-life is $t_{1/2} = \ln 2 / \kappa \approx 1.2$ days, so a team's strength reverted to the mean almost immediately between matches (median gap 1 day, mean 5.3 days). This destroys persistence and makes rankings noise. **Fix:** clamp $\kappa$ to `_KAPPA_MAX = 0.1` (half-life ≥ 7 days) in `_constrain`.
2. **State variance collapsed to ~0.2.** The stationary state variance is $\Gamma_0[i,i] \cdot B[k,k]$. With no prior, EM drove $\Gamma_0 \to 0$, so attack/defence states hovered near the mean with almost no spread. **Fix:** add a quadratic shrinkage prior on $\Gamma_0$ toward the initial regional-correlation prior (`gamma_0_prior = 0.1`), keeping the state variance at a meaningful scale.

> **Note on $\mu_0$.** $\mu_0$ is unidentifiable from the likelihood: team strengths only appear in *differences* ($x_i^{\text{att}} - x_j^{\text{def}}$), so shifting all teams' strengths by a constant leaves the goal rates unchanged. It is therefore held fixed at zero.
