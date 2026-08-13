# RW-RBPF v2 — Random-Walk Model with Direct Gradient Descent on the Log Marginal Likelihood

This is the **v2** variant of the random-walk RBPF football model. It keeps the
exact same **model** (pure random-walk transition, Kronecker covariance
$\Sigma = \Gamma \otimes B$, bivariate-Poisson likelihood) and the exact same
**particle filter** as [`rbpf_rw`](../rbpf_rw/MODEL.md). The only change is
**how the parameters are estimated**: v2 replaces the Monte Carlo EM (MCEM) of
`rbpf_rw` with **direct gradient descent on the log marginal likelihood**
`-log Z(θ)`.

> **Read this first.** The model equations (Sections 1–3) are identical to
> [`rbpf_rw/MODEL.md`](../rbpf_rw/MODEL.md) and are not repeated here. This
> document focuses on the **training difference** (Section 4), the **code
> layout** (Section 5), and the **rationale / expected benefits** (Section 6).

---

## 1 Model (unchanged from `rbpf_rw`)

- **State.** $X_t^m = (X_t^{m,\text{att}}, X_t^{m,\text{def}}) \in \mathbb{R}^2$ for $M$ teams.
- **Initial.** $X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$, $\Sigma_0 = \Gamma_0 \otimes B$.
- **Transition (random walk).** $X_t = X_{t-1} + \varepsilon_t$,
  $\varepsilon_t \sim \mathcal{N}(0, \Delta_t \cdot Q)$, $Q = \Gamma_Q \otimes B$.
  There is **no mean reversion** (no `kappa`).
- **Likelihood.** Bivariate Poisson with
  $\lambda_1 = \exp(\alpha + (x^{\text{att},h} - x^{\text{def},a})/\text{scale})$,
  $\lambda_2 = \exp(\alpha + (x^{\text{att},a} - x^{\text{def},h})/\text{scale})$,
  $\lambda_3 = \exp(\beta)$, with $\text{scale} = 1.0$ fixed.

The parameter set is the same as `rbpf_rw`:

| Param | Meaning | Status |
|-------|---------|--------|
| `mean_0` | shared mean (unidentifiable) | **fixed at 0** |
| `gamma_0` | $M \times M$ team factor of initial covariance | **estimated** (Cholesky-param.) |
| `gamma_Q` | $M \times M$ team factor of transition covariance | **estimated** (Cholesky-param.) |
| `B` | $2 \times 2$ attack/defence factor | **estimated** (Cholesky-param.) |
| `alpha` | baseline log-goal rate | **estimated** |
| `beta` | bivariate-Poisson dependence | **estimated** |

---

## 2 Filter (unchanged from `rbpf_rw`)

The forward filter is identical: a bootstrap particle filter with a
Rao–Blackwellized Gaussian conditional for the unobserved teams, and a
**deterministic** team-covariance trajectory `compute_gamma_trajectory`
(precomputed with a single `lax.scan`). The random-walk prediction is

$$\Gamma_{t \mid t-1} = \Gamma_{t-1 \mid t-1} + \Delta_t \cdot \Gamma_Q$$

The filter returns `filtered_states.log_normalizing_constant[-1]`, the total
log marginal likelihood `log Z(θ) = log p(y_{1:T} | θ)`.

---

## 3 Smoothing (removed in v2)

`rbpf_rw` used **FFBSi (RTS backward sampling)** to draw smoothed trajectories
for the E-step. **v2 does not smooth at all.** The forward filter alone is
enough to evaluate `log Z(θ)`, and backpropagating through it gives the
gradient of the marginal likelihood directly. Removing smoothing removes the
MCEM noise that came from averaging over a finite number of smoothed
trajectories.

---

## 4 Parameter Estimation — Direct Gradient Descent on `-log Z(θ)`

This is the core difference. `rbpf_rw` used **Monte Carlo EM**:

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

**Why this fixes the `rbpf_rw` failures.** The MCEM in `rbpf_rw` had the same
two structural problems as the OU MCEM: (1) **MCEM noise** from averaging over a
finite number of smoothed trajectories, and (2) **loss-scaling imbalance** from
reweighting the init/obs/transition terms by their dimension counts. v2 has
**no E-step/M-step split** and **no per-dimension loss scaling** — the objective
is a single well-defined scalar `-log Z(θ)`.

**Stochasticity.** The particle filter uses `jax.random` internally, so
`log Z(θ)` is stochastic. For a stable gradient we **fix the PRNG key per
gradient step** (`jax.random.fold_in(key, step)`). This is a biased-but-
consistent estimator, standard in differentiable particle filtering. The
deterministic covariance trajectory `compute_gamma_trajectory` provides a clean
differentiable path through `gamma_0` and `gamma_Q`.

**Optimizer.** `optax.multi_transform` with per-parameter learning rates,
global-norm clipping, and a NaN early-stop. `gamma_0`, `gamma_Q`, and `B` are
Cholesky-parameterized (`L L^T` with a softplus-wrapped diagonal) so they stay
positive-definite by construction.

**Differentiable samplers.** The `rbpf_rw` model used an eigendecomposition-based
`_sample_psd_gaussian`, which has a **NaN gradient** at the zero-variance
boundary (observed teams). v2 replaces it (and `init_sample`) with a
**differentiable Cholesky + jitter** reparameterization so gradients flow through
`gamma_0`/`gamma_Q`/`B`. `init_sample` also now draws each particle from the
prior (dispersion), matching the OU v2 fix.

---

## 5 Practical Reference

### 5.1 Code layout

```
rbpf_rw_v2/
├── MODEL.md                  # this file
├── smoothing_gpu.py          # Colab bootstrap + GD core (run_gd)
├── smoothing_gpu_config.json # runtime config (N, n_steps, dates, teams, GPU)
├── run_smoothing_colab.sh    # one-shot Colab runner (launch, download, predict)
├── model_trained.py          # run filter with trained params → graphics
├── model_predict.py          # predict 2026 WC fixtures from trained params
├── export_states_csv.py      # .npy → CSV (team_name, attack, defence)
├── data/                     # copied from rbpf_rw/data/ (+ fixtures)
└── src/
    ├── __init__.py
    ├── utils.py              # EMParams with gamma_Q (copied from rbpf_rw)
    ├── bivariate_poisson.py  # loglik, loglik_grid (copied from rbpf_rw)
    ├── helpers.py            # default_init_params (gamma_Q), kron_sample_psd
    ├── model.py              # RW filter + differentiable Cholesky sampler
    ├── graphic.py            # plotting (copied from rbpf_ou_v2, has GD plots)
    └── train.py              # NEW: direct GD on -log Z (adapts OU v2 train.py)
```

The `src/` modules are **copies** of the `rbpf_rw` modules with imports
rewritten from `rbpf_rw.src` to `rbpf_rw_v2.src`, plus the differentiable
sampler and `kron_sample_psd` additions. The only new file is `src/train.py`,
which replaces `rbpf_rw/src/smoothing.py` (the MCEM machinery).

### 5.2 Training configuration (`smoothing_gpu_config.json`)

```json
{
  "N": 5000,
  "n_steps": 200,
  "learning_rate": 0.001,
  "start_date": "2000-01-01",
  "end_date": "2025-12-31",
  "teams": "ACTIVE_TEAMS",
  "max_goals": 8,
  "output_dir": "rbpf_rw_v2/outputs/n5000_active",
  "hardware": "gpu",
  "gpu_type": "L4"
}
```

The `rbpf_rw` config keys `n_epochs`, `n_gradient_steps`, and `n_trajectories`
are **gone** — v2 has a single `n_steps` (total gradient steps) and no
`n_trajectories` (no smoothing). `learning_rate` is `0.001`.

### 5.3 Outputs

`run_gd` writes into `output_dir`:

- `gd_params_init.json` / `gd_params_final.json` — initial / final `EMParams`.
- `gd_log_marginal_history.json` — `log Z(θ)` per step (should rise).
- `gd_loss_history.json` — `-log Z(θ)` per step (should fall).
- `gd_convergence.png` — plot of the log-marginal history.

---

## 6 Rationale and Expected Benefits

**What v2 keeps.** The particle filter (exact non-Gaussian posterior), the
Kronecker structure, the random-walk transition, and the bivariate-Poisson
likelihood are all unchanged. The posterior is still represented exactly by
particles, so the non-Gaussian bivariate-Poisson posterior is handled exactly.

**What v2 changes.** Only the training objective. Instead of the two-stage MCEM
(which introduced MCEM noise and a loss-scaling imbalance), v2 optimizes a
single well-defined scalar `-log Z(θ)` by direct gradient descent. This is the
"differentiable particle filter" approach.

**Expected benefits.**

1. **No MCEM noise.** The objective is a single scalar, so there is no
   averaging over smoothed trajectories to introduce noise.
2. **No loss-scaling imbalance.** There is no per-dimension reweighting, so the
   observation term is not dominated by the transition term.
3. **Monotone-ish objective.** `-log Z(θ)` is the actual negative log marginal
   likelihood, so a decreasing loss is a genuine improvement in model fit
   (up to the fixed-key stochasticity).

**Known caveats.**

- **Fixed-key gradient is biased.** Fixing the PRNG key per step gives a
  consistent but biased gradient estimate. This is standard in differentiable
  particle filtering and is a reasonable trade-off for stability.
- **`gamma_Q` is floored.** The transition covariance factor `gamma_Q` is
  floored at `_GAMMA_Q_FLOOR = 1e-6` so the random-walk transition stays
  non-degenerate (analogous to the `_KAPPA_MIN` floor in OU v2).
- **`mean_0` is still fixed at 0.** It is unidentifiable from the likelihood
  (strengths only appear in differences), so it is held fixed.

---

## 7 Progress Log

**2026-08-13 — v2 created.** Replicated the OU v2 direct-GD approach for the
random-walk model. The model, filter, and data pipeline are unchanged from
`rbpf_rw`; only the training objective differs (direct GD on `-log Z(θ)`
replacing MCEM). The differentiable Cholesky samplers and `init_sample`
dispersion were ported from OU v2 so gradients flow through
`gamma_0`/`gamma_Q`/`B`. **Status: local CPU sanity check passed** (no NaN,
params move); **not yet run on GPU.** The first test is N=5000, ACTIVE_TEAMS
(228 teams) on L4 — note that ACTIVE_TEAMS diverged on GPU for OU v2 due to a
float32 numerical instability, so the RW model may hit the same issue.
