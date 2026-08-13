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
  "learning_rate": 0.01,
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
`n_trajectories` (no smoothing).

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
imbalance. **Status: not yet run.** The first step is to run
`./run_smoothing_colab.sh` and check that `gd_log_marginal_history.json` rises
(and `gd_loss_history.json` falls) across steps, and that the rankings are sane.
