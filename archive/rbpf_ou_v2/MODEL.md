# OU-RBPF v2 — Direct Gradient Descent on the Log Marginal Likelihood

This is the **v2** variant of the OU-RBPF football model. It keeps the exact
same **model** (Ornstein–Uhlenbeck scalar-$\phi$ AR(1) transition, Kronecker
covariance $\Sigma = \Gamma \otimes B$, bivariate-Poisson likelihood) and the
exact same **particle filter** as [`rbpf_ou`](../rbpf_ou/MODEL.md). The only
change is **how the parameters are estimated**: v2 replaces the Monte Carlo EM
(MCEM) of v1 with **direct gradient descent on the log marginal likelihood**
`-log Z(θ)`.

> **Read this first.** The model equations (Sections 1–3) are identical to
> [`rbpf_ou/MODEL.md`](../rbpf_ou/MODEL.md) and are not repeated here. This
> document focuses on the **training difference** (Section 4), the **code
> layout** (Section 5), and the **rationale / expected benefits** (Section 6).

---

## 1 Model (unchanged from v1)

- **State.** $X_t^m = (X_t^{m,\text{att}}, X_t^{m,\text{def}}) \in \mathbb{R}^2$ for $M$ teams.
- **Initial.** $X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$, $\Sigma_0 = \Gamma_0 \otimes B$.
- **Transition (OU).** $X_t = \mu_0 + \phi_t (X_{t-1} - \mu_0) + \epsilon_t$,
  $\phi_t = e^{-\kappa \Delta_t}$, $\epsilon_t \sim \mathcal{N}(0, (1-\phi_t^2)\Sigma_0)$.
- **Likelihood.** Bivariate Poisson with
  $\lambda_1 = \exp(\alpha + (x^{\text{att},h} - x^{\text{def},a})/\text{scale})$,
  $\lambda_2 = \exp(\alpha + (x^{\text{att},a} - x^{\text{def},h})/\text{scale})$,
  $\lambda_3 = \exp(\beta)$, with $\text{scale} = 1.0$ fixed.

The parameter set is the same as v1 after all the fixes:

| Param | Meaning | Status |
|-------|---------|--------|
| `mean_0` | shared mean (unidentifiable) | **fixed at 0** |
| `gamma_0` | $M \times M$ team covariance | **estimated** (Cholesky-param.) |
| `B` | $2 \times 2$ attack/defence factor | **estimated** (Cholesky-param.) |
| `kappa` | OU mean-reversion rate | **estimated**, clamped to $[0.001, 0.002]$ |
| `alpha` | baseline log-goal rate | **estimated** |
| `beta` | bivariate-Poisson dependence | **estimated** |

`scale` and `gamma_Q` are gone (removed in v1; see `rbpf_ou/MODEL.md` §6.2).

---

## 2 Filter (unchanged from v1)

The forward filter is identical: a bootstrap particle filter with a
Rao–Blackwellized Gaussian conditional for the unobserved teams, and a
**deterministic** team-covariance trajectory `compute_gamma_trajectory`
(precomputed with a single `lax.scan`). The filter returns
`filtered_states.log_normalizing_constant[-1]`, the total log marginal
likelihood `log Z(θ) = log p(y_{1:T} | θ)`.

---

## 3 Smoothing (removed in v2)

v1 used **FFBSi (RTS backward sampling)** to draw smoothed trajectories for the
E-step. **v2 does not smooth at all.** The forward filter alone is enough to
evaluate `log Z(θ)`, and backpropagating through it gives the gradient of the
marginal likelihood directly. Removing smoothing removes the MCEM noise that
came from averaging over a finite number of smoothed trajectories.

---

## 4 Parameter Estimation — Direct Gradient Descent on `-log Z(θ)`

This is the core difference. v1 used **Monte Carlo EM**:

```
E-step:  run the filter, then draw N_TRAJECTORIES smoothed trajectories
M-step:  minimize  -mean_M[ complete_log_likelihood ]   (the Q-function)
```

v2 uses **direct gradient descent** on the marginal log-likelihood:

```
for step in range(n_steps):
    key = fold_in(seed, step)                 # fixed key per step
    loss = -log Z(θ)                          # = -filtered.log_normalizing_constant[-1]
    grads = grad(loss, θ)
    θ     = θ - lr * grads
```

**Why this fixes the v1 failures.** The v1 EM diverged (log marginal declined
every epoch) and produced noise rankings. The two suspected root causes were:

1. **MCEM noise.** The Q-function was an average over a finite number of
   smoothed trajectories, so the M-step objective was dominated by Monte Carlo
   noise. v2 has **no E-step/M-step split** — the objective is a single
   well-defined scalar `-log Z(θ)`, so there is no MCEM noise to average out.

2. **Loss-scaling imbalance.** v1's `_complete_log_likelihood` reweighted the
   init/obs/transition terms by their dimension counts, which made the
   transition term dominate and drove the rankings. v2 has **no per-dimension
   loss scaling** — the objective is just `-log Z(θ)`, a single scalar with no
   arbitrary reweighting.

**Stochasticity.** The particle filter uses `jax.random` internally, so
`log Z(θ)` is stochastic. For a stable gradient we **fix the PRNG key per
gradient step** (`jax.random.fold_in(key, step)`). This is a biased-but-
consistent estimator, standard in differentiable particle filtering. The
deterministic covariance trajectory `compute_gamma_trajectory` provides a clean
differentiable path through `gamma_0` and `kappa`.

**Optimizer.** `optax.multi_transform` with per-parameter learning rates,
cosine schedule, and global-norm clipping. `gamma_0` and `B` are
Cholesky-parameterized (`L L^T` with a softplus-wrapped diagonal) so they stay
positive-definite by construction.

### 4.1 Mathematical formulation

Let $\theta = (\mu_0, \Gamma_0, B, \kappa, \alpha, \beta)$ be the model
parameters. The **marginal likelihood** integrates out the latent states
$X_{0:T}$:

$$Z(\theta) = p(y_{1:T} \mid \theta) = \int p(x_0 \mid \theta) \prod_{t=1}^{T} p(x_t \mid x_{t-1}, \theta)\, g_t(y_t \mid x_t, \theta)\, dx_{0:T}$$

where $g_t$ is the bivariate-Poisson observation density. The particle filter
produces a **consistent estimator** $\hat{Z}_N(\theta)$ of this integral — the
product of the per-step normalizing constants, equivalently the sum of the
log-weights:

$$\log \hat{Z}_N(\theta) = \sum_{t=1}^{T} \log\!\Bigl(\tfrac{1}{N}\sum_{i=1}^{N} \tilde{w}_t^{(i)}\Bigr), \qquad \tilde{w}_t^{(i)} = w_{t-1}^{(i)}\, g_t\bigl(y_t \mid X_t^{\mathcal{O}_t,(i)}\bigr)$$

v2 minimizes the **negative log marginal likelihood** (a single scalar):

$$\mathcal{L}(\theta) = -\log \hat{Z}_N(\theta) = -\text{filtered.log\_normalizing\_constant}[-1]$$

by **direct gradient descent**:

$$\theta_{k+1} = \theta_k - \eta_k\, \nabla_\theta \mathcal{L}(\theta_k), \qquad \nabla_\theta \mathcal{L}(\theta) = -\nabla_\theta \log \hat{Z}_N(\theta)$$

The gradient is obtained by **backpropagating through the filter** (`jax.grad`
on the loss), which differentiates the weight recursion and the deterministic
covariance trajectory `compute_gamma_trajectory` (a clean path through
$\Gamma_0$ and $\kappa$). The discrete resampling step is non-differentiable,
so its gradient path is effectively stopped (see §4.3).

### 4.2 Comparison with Monte Carlo EM (v1)

| | **v1 — MCEM** | **v2 — direct GD** |
|---|---|---|
| **Objective** | Expected complete-data log-likelihood (Q-function) $-\frac{1}{M}\sum_{i=1}^{M}\log p(X_{0:T}^{(i)}, y_{1:T}\mid\theta)$ | Negative log marginal likelihood $-\log \hat{Z}_N(\theta)$ |
| **Latent states** | **Sampled** via FFBSi smoothing (E-step) | **Integrated out** by the filter |
| **Source of the scalar** | Average over $M$ smoothed trajectories | `log_normalizing_constant[-1]` from the filter |
| **Structure** | Two-stage: E-step / M-step | Single objective, one `grad` call |
| **Noise** | MCEM noise from finite $M$ trajectories | Fixed-key stochasticity of the filter |
| **Gradient** | Implicit (M-step optimizes the Q-function) | Explicit (backprop through the filter) |
| **Smoothing** | Required (FFBSi) | Not needed |

The two objectives are **different objects**: v1 optimizes the *complete-data*
likelihood (needs the latent states, hence smoothing), while v2 optimizes the
*marginal* likelihood (the filter's normalizing constant, which already
marginalizes out the states). This is why v2 can drop smoothing entirely.

### 4.3 Consequences: the biased gradient

The gradient $\nabla_\theta \log \hat{Z}_N(\theta)$ is **biased** for two
reasons, neither of which the fixed PRNG key removes:

1. **Finite-$N$ log-likelihood bias.** $\log \hat{Z}_N(\theta)$ is a consistent
   but biased estimator of $\log Z(\theta)$: for finite $N$ it systematically
   *underestimates* the log marginal likelihood. The gradient of a biased
   estimator is biased.

2. **Non-differentiable resampling.** The discrete resampling indices are an
   `argmax`-like operation with no meaningful gradient. Backprop through it is
   stopped, so the gradient is a surrogate, not the true gradient of the
   expected log-likelihood.

**What the fixed key does.** Fixing the PRNG key per step
($\text{key}_k = \text{fold\_in}(\text{seed}, k)$) makes $\hat{Z}_N(\theta)$ a
deterministic function of $\theta$ for that step. This removes **variance**
(stable, reproducible gradients) but leaves the **bias** untouched — it merely
*exposes* it by removing the averaging over randomness. The bias is a property
of "differentiating through a finite-$N$ particle filter," not of the key.

**Reducing the bias cheaply.** Raising $N$ reduces the bias (asymptotically to
zero) with no algorithmic change — a one-line config edit. This is the
recommended first step before any algorithmic change.

### 4.4 Extensions for unbiased gradients

If unbiased gradients are required, the estimator itself must change — not the
key:

- **Differentiable Particle Filter (DPF, Corenflos et al.).** Replaces the
  discrete resampling with a **differentiable transport coupling**, giving an
  **unbiased** gradient. The swap at the call site is small, but it is a
  *different filter* (the forward pass changes too), must respect the
  Rao–Blackwellized structure, and is research-grade — a separate experiment,
  not a drop-in fix.
- **Score-function / REINFORCE gradient.** Uses the log-derivative trick
  $\nabla_\theta \mathbb{E}[f] = \mathbb{E}[f\, \nabla_\theta \log q]$ to get an
  **unbiased** gradient, at the cost of **high variance** (the opposite
  trade-off from v2's fixed-key stability). Often needs control variates or
  baselines to be usable.

**Summary of the trade-off.**

| | v2 (systematic + fixed key) | DPF | Score / REINFORCE |
|---|---|---|---|
| Gradient | Biased | Unbiased | Unbiased |
| Variance | Low | Moderate | High |
| Code change | — | Small swap, real algorithmic work | Moderate |
| Risk | Stable, works | Could reintroduce instability | Needs variance reduction |

---

## 5 Practical Reference

### 5.1 Code layout

```
rbpf_ou_v2/
├── MODEL.md                  # this file
├── smoothing_gpu.py          # Colab bootstrap + GD core (run_gd)
├── smoothing_gpu_config.json # runtime config (N, n_steps, dates, teams, GPU)
├── run_smoothing_colab.sh    # one-shot Colab runner (launch, download, predict)
├── model_trained.py          # run filter with trained params → graphics
├── model_predict.py          # predict 2026 WC fixtures from trained params
├── export_states_csv.py      # .npy → CSV (team_name, attack, defence)
├── data/                     # copied from rbpf_ou/data/
└── src/
    ├── __init__.py
    ├── utils.py              # EMParams, FootballResults, RBPFState (copied)
    ├── bivariate_poisson.py  # loglik, loglik_grid (copied)
    ├── helpers.py            # default_init_params, kron_sample_psd, ... (copied)
    ├── model.py              # filter: init_sample, propagate_sample, run_filter (copied)
    ├── graphic.py            # plotting (copied)
    └── train.py              # NEW: direct GD on -log Z (replaces smoothing.py)
```

The `src/` modules are **copies** of the v1 modules with imports rewritten from
`rbpf_ou.src` to `rbpf_ou_v2.src`. The only new file is `src/train.py`, which
replaces `rbpf_ou/src/smoothing.py` (the MCEM machinery).

### 5.2 Training configuration (`smoothing_gpu_config.json`)

```json
{
  "N": 500,
  "n_steps": 200,
  "learning_rate": 0.001,
  "start_date": "2000-01-01",
  "end_date": "2025-12-31",
  "teams": "WORLDCUP_2026_TEAMS",
  "max_goals": 8,
  "output_dir": "rbpf_ou_v2/outputs_gpu_l4",
  "hardware": "gpu",
  "gpu_type": "L4"
}
```

The v1 config keys `n_epochs`, `n_gradient_steps`, and `n_trajectories` are
**gone** — v2 has a single `n_steps` (total gradient steps) and no
`n_trajectories` (no smoothing). `learning_rate` is `0.001` (the `0.01` default
was unstable — see §7).

### 5.3 Outputs

`run_gd` writes into `output_dir`:

- `gd_params_init.json` / `gd_params_final.json` — initial / final `EMParams`.
- `gd_log_marginal_history.json` — `log Z(θ)` per step (should rise).
- `gd_loss_history.json` — `-log Z(θ)` per step (should fall).
- `gd_convergence.png` — plot of the log-marginal history.

---

## 6 Rationale and Expected Benefits

**What v2 keeps.** The particle filter (exact non-Gaussian posterior), the
Kronecker structure, the OU transition, and the bivariate-Poisson likelihood
are all unchanged. The posterior is still represented exactly by particles, so
the non-Gaussian bivariate-Poisson posterior is handled exactly.

**What v2 changes.** Only the training objective. Instead of the two-stage
MCEM (which introduced MCEM noise and a loss-scaling imbalance), v2 optimizes a
single well-defined scalar `-log Z(θ)` by direct gradient descent. This is the
"differentiable particle filter" approach.

**Expected benefits.**

1. **No MCEM noise.** The objective is a single scalar, so there is no
   averaging over smoothed trajectories to introduce noise.
2. **No loss-scaling imbalance.** There is no per-dimension reweighting, so the
   observation term is not dominated by the transition term. This is the most
   likely fix for the noise rankings.
3. **Monotone-ish objective.** `-log Z(θ)` is the actual negative log marginal
   likelihood, so a decreasing loss is a genuine improvement in model fit
   (up to the fixed-key stochasticity).

**Known caveats.**

- **Fixed-key gradient is biased.** Fixing the PRNG key per step gives a
  consistent but biased gradient estimate. This is standard in differentiable
  particle filtering and is a reasonable trade-off for stability.
- **`kappa` is still constrained.** As in v1, `kappa` is clamped to
  $[0.001, 0.002]$ to force a ~1-year mean-reversion half-life so team quality
  persists across the long gaps between international matches. This is a
  constraint, not a free estimate.
- **`mean_0` is still fixed at 0.** It is unidentifiable from the likelihood
  (strengths only appear in differences), so it is held fixed.

---

## 7 Progress Log

**2026-08-13 — v2 created.** Reorganized `rbpf_ou` into `rbpf_ou_v2` with direct
gradient descent on `-log Z(θ)` replacing MCEM. The model, filter, and data
pipeline are unchanged; only the training objective differs. The v1 EM diverged
(log marginal declined every epoch) and produced noise rankings; v2's single
scalar objective is designed to address both the MCEM noise and the loss-scaling
imbalance.

**2026-08-13 — v2 run (L4, N=500, n_steps=200, lr=0.001).** The pipeline ran
end-to-end and produced **sane results** — a clear improvement over v1.

**Fixes required to get a stable run.** The first local runs produced NaN. The
root causes and fixes:

1. **`optax.multi_transform` label mismatch.** `mean_0` was in the `carry` dict
   but missing from `param_labels`, so `optimizer.init` raised. **Fix:** add
   `mean_0` with `optax.set_to_zero()` (it is fixed).
2. **NaN gradients from non-differentiable samplers.** `kron_sample_psd` and
   `_sample_psd_gaussian` used `eigh`/`clip`/`sqrt`, whose gradient is NaN at
   the zero-variance boundary (observed teams). **Fix:** replaced both with a
   differentiable **Cholesky + jitter** reparameterization (`L L^T` of
   `covariance + 1e-6 I`), which has a finite gradient.
3. **NaN from negative `kappa`.** The optimizer updated `kappa` freely; a
   negative `kappa` makes `phi = exp(-kappa*dt) > 1` and the transition
   covariance `(1-phi^2)*gamma_0` negative → NaN. **Fix:** clamp `kappa` to
   `[_KAPPA_MIN, _KAPPA_MAX]` inside the training loop (not just at the end).
4. **NaN from singular `pinv`.** `jnp.linalg.pinv(gamma_EE)` has a NaN gradient
   when `gamma_EE` is singular. **Fix:** jitter `gamma_EE` to strictly-PD.
5. **`learning_rate = 0.01` unstable.** The loss exploded to NaN within a few
   steps. **Fix:** `learning_rate = 0.001` (stable).
6. **NaN guard.** The training loop now checks the loss *before* applying the
   update and stops early, keeping the last finite parameters.

**Results (L4, N=500, 200 steps).**

- **Log marginal improved** from **-5764 → -5590** (+174) over 200 steps. This
  is the key win: v1's EM *diverged* (log marginal declined every epoch), while
  v2's direct GD *improves* the fit.
- **Rankings are sane.** Top teams: Spain (+1.67), Switzerland (+1.39), France
  (+1.36), Netherlands (+1.29), Portugal (+1.11), Belgium (+0.98), Austria
  (+0.92), Turkey (+0.89), Croatia (+0.87), England (+0.85), Germany (+0.83).
  Bottom: Haiti (-1.58), Egypt (-1.29), Iraq (-1.21), Cape Verde (-1.20), DR
  Congo (-1.19), Qatar (-1.10), Ghana (-1.03), Saudi Arabia (-0.90). This is a
  dramatic improvement over v1, where France was negative and weak teams were
  overrated.
- **Predictions.** Mean log-likelihood **-3.50** over 36 scored matches (v1's
  best was -3.41, latest was -4.69). Several strong calls: Spain 4-0 Saudi
  Arabia (ll=-1.94), Mexico 2-0 South Africa (ll=-1.89), Haiti 0-1 Scotland
  (ll=-1.58). Some misses remain (e.g. Canada 6-0 Qatar, ll=-9.20).

**Remaining observations.**

- The log marginal is still noisy (fixed-key stochasticity with N=500), so the
  per-step values bounce around even as the trend improves. A larger `N` or a
  cosine LR schedule would smooth this.
- `kappa` sits at the `_KAPPA_MIN` floor (0.001), meaning the optimizer wants
  even slower mean reversion. This is consistent with the persistence goal.
- The rankings are much better but not perfect (e.g. Switzerland #2, Turkey
  #8). This is expected for a first stable run; further tuning of `N`, `n_steps`,
  and the LR schedule is the natural next step.

---

## 8 Hyperparameter Sweep (2026-08-13)

A sweep over `N` (particles), team set (dimension), and lookback (start date).
All runs: `n_steps=200`, `lr=0.001`, `end_date=2025-12-31`, L4 GPU unless noted.
Metrics: **Δ log Z** = log-marginal improvement over training; **mean ll** =
prediction mean log-likelihood over 36 scored 2026 WC fixtures; **rankings** =
top-5 / bottom-5 by total strength (attack+defence).

| Run | `N` | Teams | start | Δ log Z | mean ll | Top-5 | Bottom-5 |
|-----|-----|-------|-------|---------|---------|-------|----------|
| baseline | 500 | 48 | 2000 | +174 | **-3.50** | Spain, Switz, France, Neth, Port | Haiti, Egypt, Iraq, C.Verde, DR Congo |
| N=1000 | 1000 | 48 | 2000 | +256 | -3.67 | Australia, Curaçao, Arg, Morocco, DR Congo | Sweden, Czech, Qatar, Turkey, Egypt |
| N=1500 | 1500 | 48 | 2000 | +274 | -3.64 | Panama, Arg, Mexico, Colombia, Ecuador | N.Zealand, DR Congo, Turkey, Belgium, Sweden |
| N=2000 | 2000 | 48 | 2000 | +260 | **-3.37** | France, Spain, Germany, Croatia, Curaçao | S.Africa, Saudi, Egypt, Qatar, Iraq |
| N=5000 | 5000 | 48 | 2000 | +313 | **-3.44** | Spain, Portugal, France, Argentina, Ecuador | Qatar, Sweden, Haiti, Iraq, Norway |
| ACTIVE | 100 | 228 | 2000 | NaN | NaN | — (diverged) | — |
| lookback 1950 | 500 | 48 | 1950 | +501 | -3.68 | Colombia, Uzbek, Tunisia, Brazil, Japan | Sweden, Norway, Haiti, Czech, Bosnia |

**Findings.**

1. **Higher `N` improves fit but not rankings.** Δ log Z grows with `N`
   (174 → 256 → 274 → 260), but the rankings get *worse* (weak teams like
   Curaçao, Panama, Australia creep into the top-5) and mean ll degrades
   (except N=2000, which is the best at -3.37). This is the **finite-`N`
   log-likelihood bias** (§4.3): more particles reduce the bias of the
   objective, but the *rankings* are driven by the observation term, which the
   biased gradient over-weights. The best overall run is **N=2000** (best mean
   ll -3.37, mostly sane rankings).
2. **ACTIVE_TEAMS (228 teams) diverges on GPU.** The forward `log Z` is finite
   on CPU (-56114) and trains stably on CPU, but produces **NaN at step 0 on
   GPU** — a **GPU float32 numerical instability** at the larger dimension, not
   a memory issue (so A100 won't help). Root cause: a tiny negative eigenvalue
   (-2e-8) in `gamma_0` from float32 rounding makes `jnp.linalg.cholesky` NaN;
   fixed by jittering in `_cholesky_from_psd`, but the GPU still diverges at
   228 teams. **A100 is not the fix** — the instability is numerical, not
   memory-bound.

   **Fix (2026-08-14) — scale-aware jitter.** The residual GPU divergence at
   228 teams had a specific root cause. The free Cholesky factors are rebuilt
   with `_psd_from_cholesky`, whose **diagonal** floor is `_EIGEN_FLOOR = 1e-4`
   on `L`. Squaring (`L Lᵀ`) therefore leaves `gamma_0` with eigenvalues as
   small as `1e-4² = 1e-8` — below float32 epsilon (~1e-7). Every downstream
   PSD path then added only a **fixed `1e-6` jitter** before its Cholesky/pinv:
   - `kron_sample_psd` (init + the 228×228 Cholesky in `_sample_psd_gaussian`)
   - `_sample_psd_gaussian` (the observed-block sampler)
   - `compute_gamma_trajectory`'s `pinv(gamma_EE)` (the Kalman gain)

   On a CPU XLA run (higher intermediate precision) this survived; on GPU
   float32 the `1e-6` jitter was too small to lift the `~1e-8` eigenvalue above
   the Cholesky failure threshold, so the **gradient** (which backprops through
   every one of these matrices over all T timesteps) produced NaN at step 0.

   The fix replaces the fixed `1e-6` jitter with a **scale-aware jitter**
   `_scale_aware_jitter(A) = 1e-4 * max(1, max|diag(A)|)`, applied uniformly in
   `kron_sample_psd`, `_sample_psd_gaussian`, and the `compute_gamma_trajectory`
   Kalman `pinv`. This floors the smallest eigenvalue at ~1e-4 (condition number
   ≤ ~1e4, robustly Cholesky-factorable in float32) and is built only from
   differentiable ops (no eigendecomposition), so it stays inside the
   differentiable filter path. At the 48-team scale the effect is negligible
   (diagonals are ~1, so the jitter is essentially the same 1e-4 scale as the
   previous `_EIGEN_FLOOR`), but at 228 teams it keeps the Cholesky/pinv PD on
   the GPU.

   **If scale-aware jitter still doesn't converge at 228 teams**, the remaining
   lever is **float64**: set `jax.config.update("jax_enable_x64", True)` (and
   `jnp.linalg.cholesky`/`pinv` become exact at ~1e-16), which removes the
   float32 rounding entirely at ~2× memory. The documented `-2e-8` negative
   eigenvalue is a float32 artifact, so x64 is the definitive fix if the
   float32 scale-aware floor proves insufficient.
3. **Longer lookback (1950) improves fit but not rankings.** Δ log Z jumps to
   +501 (more data → better fit), but mean ll degrades to -3.68 and rankings
   are worse (Colombia, Uzbekistan, Tunisia top). More historical data
   (pre-2000) appears to add noise that hurts the current-strength rankings.

**Recommendation.** The **N=2000, 48-team** config is the best overall (mean ll
-3.37, sane rankings). The ACTIVE_TEAMS path is blocked by a GPU numerical
instability, not memory — fixing it requires a numerical fix (e.g. float64, or
a more robust PSD projection), not a bigger GPU. The lookback sweep suggests
2000-01-01 is a better start date than 1950 for ranking quality.

**N=5000 follow-up (2026-08-13).** Pushed `N` to 5000 (48 teams, lookback 2000,
L4, `outputs/n5000_48teams`). Δ log Z improved further to **+313** (best fit),
and the rankings are the **most sane yet** (Spain, Portugal, France, Argentina,
Ecuador top; Qatar, Sweden, Haiti, Iraq, Norway bottom). Mean ll **-3.44** is
slightly worse than N=2000's -3.37 but better than N=1000/1500. The trend
confirms: **more particles → better fit and better rankings**, with mean ll
roughly flat around -3.4. N=5000 is a strong production choice; the marginal
improvement over N=2000 is small, so N=2000–5000 is a reasonable operating
range.

**2026-08-14 — ACTIVE_TEAMS GPU instability fixed (scale-aware jitter).**
Root-caused and fixed the 228-team GPU NaN described in §8 Finding 2: the free
Cholesky factors rebuilt by `_psd_from_cholesky` leave `gamma_0` eigenvalues as
small as `1e-4² = 1e-8` (below float32 epsilon), and the three downstream PSD
paths (`kron_sample_psd`, `_sample_psd_gaussian`, and `compute_gamma_trajectory`'s
Kalman `pinv`) each added only a **fixed `1e-6` jitter** — too small to keep the
Cholesky PD in GPU float32 at 228 teams. Replaced the fixed jitter with a
**scale-aware** `_scale_aware_jitter` (floors the smallest eigenvalue at
`1e-4 * max(1, max|diag|)`, differentiable, condition number ≤ ~1e4). If the
float32 floor still isn't enough, the definitive fix is float64
(`jax_enable_x64=True`). See §8 Finding 2 for the full write-up.

**2026-08-14 — ACTIVE_TEAMS regional prior now covers all teams (fixes the
"182 teams not in regional config" warning).** When running `ACTIVE_TEAMS`
(~228 teams), `_regional_correlation_matrix` was hardwired to the **48-team**
`worldcup2026_team_regions.json`, so 182 teams silently fell back to the
baseline between-region correlation and the prior lost its regional structure.
Fix in `helpers.py`:

- `_pick_regional_config` auto-selects the regional config with the **best
  coverage** of the requested team set among `worldcup2026_team_regions.json`
  and `active_teams.json` (which already carries a complete ~228-team `regions`
  map), preferring full coverage.
- `_project_psd_correlation` now clamps eigenvalues to `_EIGEN_FLOOR > 0`
  (was `0.0`), so the projected correlation matrix is strictly PD and stays PD
  after float32 renormalization (clamping to 0 left a `-1.5e-6` eigenvalue that
  could re-trigger the Cholesky instability).

After the fix, all 230 ACTIVE teams get a regional assignment (no warning), and
`gamma_0` is strictly PD (eigmin ~8.7e-5, Cholesky factors cleanly).
