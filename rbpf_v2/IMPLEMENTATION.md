# RBPF v2: Model and Implementation Specification

## 1. Purpose and scope

This document is the implementation specification for `rbpf_v2`. The model,
data representation, likelihood, parameterization, Monte Carlo EM loop, and
diagnostics are copied from `rbpf`. The smoothing implementation is deliberately
corrected: `rbpf_v2` must use an RB-aware Gaussian-mixture backward simulator,
not independent full-state materialization followed by point-particle FFBS.

The implementation has three stages:

1. A Rao--Blackwellized particle filter (RBPF) approximates each filtering
   distribution by a weighted mixture of Gaussians.
2. An RB-aware forward-filtering/backward-simulation (FFBS) smoother draws
   complete, temporally coherent latent paths from that mixture.
3. Monte Carlo expectation-maximization (MCEM) updates the model parameters
   from the sampled paths.

The key invariant is:

> A forward particle stores the mean of a Gaussian component. It is not a
> complete draw of the football state.

At filter state $t$, component $i$ means

\[
X_t\mid I_t=i,y_{1:t}
\sim \mathcal N(m_t^{(i)},P_t),
\qquad P_t=\Gamma_{t\mid t}\otimes B.
\]

All algorithms and data structures must preserve this interpretation.

## 2. Notation and tensor shapes

Let:

- $D$: number of observed days;
- $N$: number of forward filtering components;
- $S$: number of backward-sampled trajectories;
- $M$: number of teams;
- $K=2$: number of traits per team (attack and defence);
- $L$: maximum number of matches on one day;
- $d=MK=2M$: flattened latent dimension.

The principal arrays are:

| Quantity | Shape | Meaning |
|---|---:|---|
| `mean_0` | `(M, 2)` | Long-run OU mean |
| `gamma_0` | `(M, M)` | Stationary covariance between teams |
| `B` | `(2, 2)` | Attack/defence covariance factor |
| `particles.x` | `(D+1, N, M, 2)` | Forward Gaussian-component means |
| `log_weights` | `(D+1, N)` | Forward component log weights |
| `gamma` | `(D, M, M)` | Filtered team covariance after each day |
| `gamma_pred` | `(D, M, M)` | Predicted team covariance before each day |
| `gamma_observed` | `(D, L, 2, 2)` | Match-team marginal covariance |
| `kalman_gain` | `(D, L, M, 2)` | Conditional update gain for each match |
| smoothed paths | `(D+1, S, M, 2)` | Complete latent trajectories |

The Kronecker convention is

\[
\operatorname{Cov}(\operatorname{vec}_C X)=\Gamma\otimes B,
\]

where rows of $X\in\mathbb R^{M\times2}$ are teams and columns are traits.
Consequently, `gamma_*` always acts on the team axis and `B` acts on the trait
axis.

### 2.1 Timeline contract

There are $D+1$ latent states for $D$ observed days:

```text
latent/filter state          0       1       ...       D
observed day/transition              0       ...       D-1
filtered team covariance    gamma_0  gamma[0] ...      gamma[D-1]
prediction into next state           gamma_pred[0] ... gamma_pred[D-1]
```

For backward transition $d\rightarrow d+1$, use exactly:

```text
gamma_filtered = concat([gamma_0, gamma])[d]
gamma_pred_next = gamma_pred[d]
phi_next = exp(-kappa * (timestamp[d] - timestamp_prev[d]))
```

Do not shift `gamma_pred` by an additional element. The terminal state's
`gamma_pred_next` and `phi_next` fields are unused dummy values.

## 3. Data preparation

Input matches are sorted and grouped by calendar date. Every date is one latent
transition and one observation batch. Match arrays are padded to $L$, with a
Boolean `match_mask` distinguishing real rows from padding.

Required preprocessing rules:

1. Team names are mapped to stable, sorted integer IDs in `[0, M)`.
2. Scores above `max_goals` are excluded because the bivariate-Poisson finite
   sum is compiled with that static upper bound.
3. Each team may appear in at most one match on a given day. This makes the
   sequential same-day conditioning blocks disjoint.
4. Every elapsed time
   `dt[d] = timestamp[d] - timestamp_prev[d]` must be strictly positive.
5. Matches on the same date share one latent state; they must not be represented
   as separate zero-time transitions.

The data containers copied from `rbpf` are conceptually:

```text
Matches:
    home_id, away_id, home_score, away_score       # each (D, L)

FootballResults:
    date, timestamp, timestamp_prev                 # each (D,)
    matches: Matches
    match_mask                                      # (D, L)

RBPFFootballResults extends FootballResults with:
    gamma, gamma_pred, gamma_observed, kalman_gain
```

## 4. Probabilistic model

### 4.1 Latent state

For team $m$,

\[
X_t^m=(A_t^m,D_t^m),
\]

where $A_t^m$ is attack strength and $D_t^m$ is defence strength. Stacking
all teams gives $X_t\in\mathbb R^{M\times2}$.

The initial distribution is

\[
X_0\sim\mathcal N(\mu_0,\Sigma_0),
\qquad \Sigma_0=\Gamma_0\otimes B.
\]

Although the current `rbpf` test initialization stores `mean_0`
deterministically, the v2 probabilistic model and terminal/backward sampler must
treat state zero as the Gaussian prior above.

### 4.2 OU dynamics

For elapsed time (dt_t>0), define

\[
\phi_t=\exp(-\kappa dt_t).
\]

The transition is

\[
X_{t+1}=\mu_0+\phi_t(X_t-\mu_0)+\varepsilon_t,
\]

with

\[
\varepsilon_t\sim\mathcal N(0,Q_t),
\qquad
Q_t=(1-\phi_t^2)(\Gamma_0\otimes B).
\]

This construction makes $\Gamma_0\otimes B$ the stationary covariance. The
complete-state transition mean and team covariance are

\[
f_t(x)=\mu_0+\phi_t(x-\mu_0),
\qquad
\Gamma_{Q,t}=(1-\phi_t^2)\Gamma_0.
\]

### 4.3 Bivariate-Poisson score likelihood

For a match between home team $h$ and away team $a$, let

\[
\begin{aligned}
\log\lambda_1 &= \alpha+A_t^h-D_t^a,\\
\log\lambda_2 &= \alpha+A_t^a-D_t^h,\\
\log\lambda_3 &= \beta.
\end{aligned}
\]

Equivalently, the goals have the representation

\[
Y_h=U+W,\qquad Y_a=V+W,
\]

where

\[
U\sim\operatorname{Pois}(\lambda_1),\quad
V\sim\operatorname{Pois}(\lambda_2),\quad
W\sim\operatorname{Pois}(\lambda_3)
\]

independently. Therefore

\[
\begin{aligned}
p(Y_h=y_h,Y_a=y_a)
={}&e^{-(\lambda_1+\lambda_2+\lambda_3)}
\frac{\lambda_1^{y_h}}{y_h!}
\frac{\lambda_2^{y_a}}{y_a!}\\
&\times
\sum_{k=0}^{\min(y_h,y_a)}
\binom{y_h}{k}\binom{y_a}{k}k!
\left(\frac{\lambda_3}{\lambda_1\lambda_2}\right)^k.
\end{aligned}
\]

Implement the finite sum in log space using `gammaln` and `logsumexp`. Padded
matches contribute exactly zero to the day's log likelihood. The daily
potential is

\[
G_t(y_t\mid X_t)=
\prod_{\ell:\,\text{mask}_{t\ell}}
p(y_{t\ell}\mid X_t^{h_{t\ell}},X_t^{a_{t\ell}},\alpha,\beta).
\]

## 5. Deterministic covariance recursion

The RBPF covariance path depends on the parameters and match schedule, but not
on sampled particle values. Compute it once per filter run.

For day $t$, first predict:

\[
\Gamma_{t+1\mid t}
=\phi_t^2\Gamma_{t\mid t}
+(1-\phi_t^2)\Gamma_0.
\]

Set `gamma_current = gamma_pred[t]`. For each valid match with team indices
$O=(h,a)$, compute

\[
\Gamma_{OO}=\Gamma_{\text{current}}[O,O],
\qquad
K=\Gamma_{\text{current}}[:,O]\Gamma_{OO}^{-1},
\]

then condition on the sampled match-team values:

\[
\Gamma_{\text{new}}
=\Gamma_{\text{current}}
-K\Gamma_{\text{current}}[O,:].
\]

The rows and columns in $O$ are then set exactly to zero: those coordinates
have been explicitly sampled and have no residual conditional variance inside
the component. Continue sequentially for every match that day.

Save:

```text
gamma_pred[t]          = covariance before any match on day t
gamma_observed[t, l]   = Gamma_OO immediately before match l
kalman_gain[t, l]      = K immediately before match l
gamma[t]               = covariance after all matches on day t
```

Use a Cholesky solve for positive-definite `gamma_observed` blocks. Symmetrize
every covariance after subtraction to suppress floating-point asymmetry.

### Covariance-recursion pseudocode

```text
function COMPUTE_COVARIANCE_PATH(data, gamma_0, kappa):
    gamma_previous = gamma_0

    for day t in 0 .. D-1:
        dt = timestamp[t] - timestamp_prev[t]
        require dt > 0
        phi = exp(-kappa * dt)

        gamma_pred[t] = SYMMETRIZE(
            phi^2 * gamma_previous + (1 - phi^2) * gamma_0
        )
        gamma_current = gamma_pred[t]

        for match l in 0 .. L-1:
            if not match_mask[t, l]:
                gamma_observed[t, l] = zeros(2, 2)
                kalman_gain[t, l] = zeros(M, 2)
                continue

            O = [home_id[t, l], away_id[t, l]]
            gamma_OO = gamma_current[O, O]
            K = SOLVE_RIGHT(gamma_current[:, O], gamma_OO)

            gamma_observed[t, l] = gamma_OO
            kalman_gain[t, l] = K
            gamma_current = SYMMETRIZE(
                gamma_current - K @ gamma_current[O, :]
            )
            zero rows and columns O in gamma_current

        gamma[t] = gamma_current
        gamma_previous = gamma_current

    return gamma, gamma_pred, gamma_observed, kalman_gain
```

## 6. Forward Rao--Blackwellized particle filter

### 6.1 Filtering representation

The filter approximation is

\[
\widehat p_N(X_t\mid y_{1:t})
=\sum_{i=1}^{N}w_t^{(i)}
\mathcal N(X_t;m_t^{(i)},P_t),
\qquad
P_t=\Gamma_{t\mid t}\otimes B.
\]

All components share $P_t$ because they share the same observation schedule.
They differ in their means and weights.

### 6.2 One component propagation

Given component mean $m_t^{(i)}$, predict the next mean:

\[
\bar m_{t+1}^{(i)}
=\mu_0+\phi_t(m_t^{(i)}-\mu_0).
\]

For each valid match on day $t$, with $O=(h,a)$, sample only the four
attack/defence values entering the likelihood:

\[
X_O^{(i)}\sim
\mathcal N\left(\bar m_O^{(i)},\Gamma_{OO}\otimes B\right).
\]

Update the mean of every team by Gaussian conditioning:

\[
m^{(i)}\leftarrow
m^{(i)}+K(X_O^{(i)}-m_O^{(i)}),
\]

then explicitly set `m[O] = X_O`. Repeating this over the day's disjoint
matches leaves a complete component mean whose played-team blocks are sampled
and whose unplayed-team blocks are conditional means.

The component's incremental log weight is the sum of the score log likelihoods:

\[
\log\widetilde w_{t+1}^{(i)}
=\log w_t^{(i)}+
\sum_{\ell:\,\text{valid}}
\log p(y_{t\ell}\mid X_{t+1}^{h_{t\ell},(i)},
X_{t+1}^{a_{t\ell},(i)}).
\]

If systematic resampling is performed every day, resample component means and
all associated particle-tree fields together. Preserve the pre-resampling
normalized weights needed by the smoothing interface according to the SMC
library's `FilterStates` contract.

### 6.3 Full RBPF pseudocode

```text
function RUN_RBPF(key, data, params, N):
    validate data and parameter shapes
    require all dt > 0

    gamma, gamma_pred, gamma_observed, kalman_gain =
        COMPUTE_COVARIANCE_PATH(data, params.gamma_0, params.kappa)

    # State zero represents the Gaussian prior mixture component.
    for particle i in 0 .. N-1:
        means[0, i] = params.mean_0
        log_weights[0, i] = -log(N)

    for day t in 0 .. D-1:
        phi = exp(-params.kappa * dt[t])

        # Resample according to the chosen SMC schedule. With daily
        # systematic resampling, obtain N parent component means here.
        parent_indices = SYSTEMATIC_RESAMPLE(log_weights[t], N)

        for particle i in 0 .. N-1:
            key_i = SPLIT_KEY(key)
            m = params.mean_0 + phi * (
                means[t, parent_indices[i]] - params.mean_0
            )

            for match l in 0 .. L-1:
                if not data.match_mask[t, l]:
                    continue

                O = [home_id[t, l], away_id[t, l]]
                covariance_O = KRON(gamma_observed[t, l], params.B)
                sampled_O = GAUSSIAN_SAMPLE(key_i, m[O], covariance_O)
                m = m + kalman_gain[t, l] @ (sampled_O - m[O])
                m[O] = sampled_O

            means[t+1, i] = m
            incremental_log_weight[i] = DAILY_SCORE_LOGLIKELIHOOD(
                data[t], m, params.alpha, params.beta
            )

        log_weights[t+1] = LOG_NORMALIZE(incremental_log_weight)
        update accumulated log normalizing constant
        save parent indices and all filter metadata

    return FilterStates(
        means, log_weights, parent_indices, log_normalizing_constant
    ), augmented_data
```

The exact resample-before-propagate versus propagate-before-resample placement
must follow Cuthbert's filter contract. It does not change the representation
invariant or the weight equation.

## 7. RB-aware full-state backward simulation

### 7.1 Why point FFBS must not be used

Ordinary point-particle FFBS assumes

\[
p(X_t\mid y_{1:t})\approx
\sum_iw_t^{(i)}\delta_{x_t^{(i)}}(X_t).
\]

That assumption is false here: `particles.x[t, i]` is $m_t^{(i)}$, the mean
of a Gaussian component with residual covariance $P_t$.

Consequently, v2 must not:

- pass the forward means directly to a point-particle backward sampler;
- return a selected component mean as a full latent state;
- independently materialize one full state for every time/component pair before
  smoothing;
- compute component selection with the narrow complete-state noise $Q_t$.

Independent materialization produces unrelated high-dimensional clouds at
adjacent times. Point FFBS then tries to connect them through $Q_t$, causing
weight collapse and duplicated backward indices.

### 7.2 Terminal draw

At state $D$, sample a mixture component using the final filter weights:

\[
I_D\sim\operatorname{Categorical}(w_D^{(1:N)}).
\]

Then materialize the complete terminal state:

\[
X_D^*\sim
\mathcal N\left(m_D^{(I_D)},
\Gamma_{D\mid D}\otimes B\right),
\qquad
\Gamma_{D\mid D}=\gamma[D-1].
\]

Select the $S$ terminal components either with systematic resampling (matching
the current Cuthbert integration) or independent categorical draws. Conditional
on those indices, use independent Gaussian noise for the $S$ full-state draws.

### 7.3 Backward component selection

Assume a complete $x_{t+1}^*$ has already been sampled. For every forward
component $i$ at state $t$, compute its predicted mean:

\[
a_{t+1}^{(i)}
=\mu_0+\phi_t(m_t^{(i)}-\mu_0).
\]

The forward component contains uncertainty $P_t$, so after integrating over
that uncertainty its next-state prediction has covariance

\[
R_{t+1}=\phi_t^2P_t+Q_t
=\Gamma_{t+1\mid t}\otimes B.
\]

The categorical backward weights are

\[
\widetilde w_t^{(i)}
\propto
w_t^{(i)}
\mathcal N(x_{t+1}^*;a_{t+1}^{(i)},R_{t+1}).
\]

In log space:

\[
\ell_i=\log w_t^{(i)}
-\frac12\left[
d\log(2\pi)+\log|R_{t+1}|
+r_i^\mathsf TR_{t+1}^{-1}r_i
\right],
\]

where $r_i=\operatorname{vec}(x_{t+1}^*-a_{t+1}^{(i)})$. Normalize with
`logsumexp` or pass the unnormalized logits directly to a categorical sampler.

### 7.4 Conditional previous-state draw

After sampling $I_t$, condition its Gaussian component on $x_{t+1}^*$.
Define

\[
J_t=P_t\phi_tR_{t+1}^{-1}.
\]

Then

\[
X_t\mid x_{t+1}^*,I_t,y_{1:t}
\sim\mathcal N(h_t^{(I_t)},C_t),
\]

with

\[
h_t^{(i)}=m_t^{(i)}
+J_t(x_{t+1}^*-a_{t+1}^{(i)}),
\]

\[
C_t=P_t-J_tR_{t+1}J_t^\mathsf T.
\]

The sampled $X_t^*$, not merely the conditional mean $h_t^{(I_t)}$, is the
fixed endpoint used at the next backward iteration.

### 7.5 Kronecker implementation

Because

\[
P_t=\Gamma_{t\mid t}\otimes B,
\qquad
R_{t+1}=\Gamma_{t+1\mid t}\otimes B,
\]

the full $2M\times2M$ matrices need not be formed. Compute

\[
J_{\Gamma,t}
=\phi_t\Gamma_{t\mid t}\Gamma_{t+1\mid t}^{-1},
\qquad
J_t=J_{\Gamma,t}\otimes I_2,
\]

and

\[
\Gamma_{C,t}
=\Gamma_{t\mid t}
-J_{\Gamma,t}\Gamma_{t+1\mid t}J_{\Gamma,t}^\mathsf T,
\qquad
C_t=\Gamma_{C,t}\otimes B.
\]

For an error matrix $E\in\mathbb R^{M\times2}$, use

\[
\operatorname{vec}(E)^\mathsf T
(\Gamma\otimes B)^{-1}\operatorname{vec}(E)
=\operatorname{tr}(E^\mathsf T\Gamma^{-1}EB^{-\mathsf T}),
\]

and

\[
\log|\Gamma\otimes B|
=2\log|\Gamma|+M\log|B|.
\]

Factor `gamma_pred_next` once per time step and solve against all particle
residuals as multiple right-hand sides. Do not compute a dense inverse.

### 7.6 Backward-simulation pseudocode

```text
function RB_BACKWARD_SIMULATION(
    key, filter_states, augmented_data, params, S
):
    means = filter_states.particles.x
    weights = filter_states.log_weights

    gamma_filtered = CONCAT([params.gamma_0], augmented_data.gamma)

    # Terminal mixture selection and full-state materialization.
    for trajectory s in 0 .. S-1:
        I[D, s] = CATEGORICAL(weights[D])
        X[D, s] = SAMPLE_KRON_GAUSSIAN(
            mean=means[D, I[D, s]],
            gamma=gamma_filtered[D],
            B=params.B
        )

    for t in D-1 down to 0:
        dt = timestamp[t] - timestamp_prev[t]
        phi = exp(-params.kappa * dt)
        gamma_t = gamma_filtered[t]
        gamma_next_pred = augmented_data.gamma_pred[t]

        # Shared across components and smoother trajectories at this t.
        factor = CHOLESKY(gamma_next_pred)
        J_gamma = phi * SOLVE_RIGHT(gamma_t, gamma_next_pred)
        gamma_cond = SYMMETRIZE(
            gamma_t - J_gamma @ gamma_next_pred @ TRANSPOSE(J_gamma)
        )

        for component i in 0 .. N-1:
            predicted_mean[i] = params.mean_0 + phi * (
                means[t, i] - params.mean_0
            )

        for trajectory s in 0 .. S-1:
            x_next = X[t+1, s]

            for component i in 0 .. N-1:
                residual = x_next - predicted_mean[i]
                log_compatibility[i] = KRON_GAUSSIAN_LOGPDF(
                    residual,
                    gamma=gamma_next_pred,
                    B=params.B,
                    cached_factor=factor
                )
                backward_logits[i] = (
                    weights[t, i] + log_compatibility[i]
                )

            I[t, s] = CATEGORICAL(backward_logits)
            i = I[t, s]
            conditional_mean = means[t, i] + J_gamma @ (
                x_next - predicted_mean[i]
            )
            X[t, s] = SAMPLE_KRON_GAUSSIAN(
                mean=conditional_mean,
                gamma=gamma_cond,
                B=params.B
            )

    return SmoothedStates(x=X, component_indices=I)
```

Forward ancestor indices are not used to choose $I_t$. Backward simulation
makes a fresh selection from all $N$ filtering components at each time.

## 8. Cuthbert integration contract

Cuthbert's backward callback does not directly receive time-specific filtered
and predicted covariances. Create a smoother-only particle tree:

```text
RBSmootherParticle:
    x                 # forward component mean; backward complete draw
    gamma_filtered    # Gamma_(t|t)
    gamma_pred_next   # Gamma_(t+1|t)
    phi_next          # OU coefficient for t -> t+1
```

Broadcast the three metadata fields over the component axis because Cuthbert
indexes every leaf in a particle tree. Construct the timeline as follows:

```text
filtered_gamma = concat([gamma_0], gamma)           # length D+1
pred_with_dummy = concat([gamma_pred, gamma_pred[-1:]])
phi_with_dummy = concat([phi, phi[-1:]])
```

The custom terminal resampler must:

1. select terminal component indices using the final logits;
2. select all corresponding metadata leaves;
3. replace each selected `.x` mean with a Gaussian full-state draw using its
   `gamma_filtered` and $B$.

The custom backward callback must:

1. ignore Cuthbert's point-state `log_density` argument;
2. ignore `x1_ancestor_indices` for component selection;
3. calculate marginalized component weights using `gamma_pred_next`;
4. draw the previous full state from the conditional Gaussian;
5. return the selected component metadata with `.x` replaced by that draw.

The ordinary complete-state transition density remains necessary for the MCEM
objective and diagnostics. It is simply not the correct density for backward
component selection.

## 9. Gaussian and Kronecker numerical primitives

Implement the following reusable primitives:

### `kron_logdet(A, B)`

For $A\in\mathbb R^{M\times M}$ and $B\in\mathbb R^{K\times K}$,

\[
\log|A\otimes B|=K\log|A|+M\log|B|.
\]

Require positive determinant signs where a proper Gaussian density is needed.

### `kron_quad(A, B, residuals)`

Evaluate all quadratic forms without constructing `kron(A, B)`. Factor $A$
once and solve all particles/paths and trait columns as multiple right-hand
sides. Since $B$ is $2\times2$, its solve is cheap.

### `sample_kron_psd(key, mean, gamma, B)`

If $L_\Gamma L_\Gamma^\mathsf T=\Gamma$ and
$L_B L_B^\mathsf T=B$, sample $Z$ with IID standard-normal entries and
return

\[
X=\text{mean}+L_\Gamma ZL_B^\mathsf T.
\]

`gamma_filtered` and `gamma_cond` can be positive semidefinite because sampled
coordinates have exactly zero conditional variance. Use an eigendecomposition
for their square roots, clip only roundoff-scale negative eigenvalues, and fail
on materially negative eigenvalues. `gamma_pred` must be positive definite and
should use Cholesky factorization.

## 10. Monte Carlo EM

Let

\[
\Theta=(\Gamma_0,B,\kappa,\alpha,\beta),
\]

with $\mu_0$ fixed. Each EM epoch consists of an E-step under
$\Theta^{(k)}$ and an M-step over fixed paths.

### 10.1 E-step

1. Run the forward RBPF with $\Theta^{(k)}$.
2. Run the RB-aware backward simulator to obtain $S$ complete paths
   $X_{0:D}^{*(s)}$.
3. Stop gradients through all filtering and smoothing draws.
4. Save the filter log normalizing constant as a marginal-likelihood
   diagnostic; it is not the MCEM objective.

No separately materialized forward cloud exists in v2.

### 10.2 Complete-data density

For one path,

\[
\log p_\Theta(X_{0:D},y_{1:D})
=\log p_\Theta(X_0)
+\sum_{t=0}^{D-1}\log p_\Theta(X_{t+1}\mid X_t)
+\sum_{t=0}^{D-1}\log p_\Theta(y_{t+1}\mid X_{t+1}).
\]

The initial term is

\[
\log p_\Theta(X_0)
=-\frac12\left[
d\log(2\pi)+\log|\Gamma_0\otimes B|
+r_0^\mathsf T(\Gamma_0\otimes B)^{-1}r_0
\right],
\]

where $r_0=\operatorname{vec}(X_0-\mu_0)$.

For transition $t\rightarrow t+1$, set

\[
r_t=\operatorname{vec}\left[
X_{t+1}-\mu_0-\phi_t(X_t-\mu_0)
\right]
\]

and use covariance

\[
Q_t=(1-\phi_t^2)(\Gamma_0\otimes B).
\]

This is where the complete-state $Q_t$, rather than the smoothing selection
covariance $R_{t+1}$, is correct.

The Monte Carlo auxiliary objective is

\[
\widehat Q(\Theta\mid\Theta^{(k)})
=\frac1S\sum_{s=1}^{S}
\log p_\Theta(X_{0:D}^{*(s)},y_{1:D}).
\]

### 10.3 Parameter constraints and identification

Optimize unconstrained raw parameters and decode them as follows:

- `gamma_0 = L L.T`, where `L` is lower triangular and its diagonal is
  `softplus(raw_diag) + 1e-4`;
- `kappa = softplus(kappa_raw) + 1e-6`;
- $B$ is diagonal and identified by $\det B=1$:

  \[
  B=\operatorname{diag}(e^r,e^{-r}),
  \qquad r=5\tanh(r_{\text{raw}});
  \]

- `alpha` and `beta` are unconstrained;
- `mean_0` is fixed at its initial value.

The determinant constraint prevents the scale non-identifiability

\[
(c\Gamma_0)\otimes(B/c)=\Gamma_0\otimes B.
\]

### 10.4 Covariance prior and M-step

To discourage covariance collapse, retain the inverse-Wishart kernel on
$\Gamma_0$:

\[
\log p(\Gamma_0)
=-\frac12(\nu+M+1)\log|\Gamma_0|
-\frac12\operatorname{tr}(S_\Gamma\Gamma_0^{-1}).
\]

Following `rbpf`, initialize

```text
nu = M + 10
S_gamma = (nu + M + 1) * initial_gamma_0
```

The loss minimized by Adam is

\[
\mathcal L(\Theta)
=-\left[
\widehat Q(\Theta\mid\Theta^{(k)})+
\log p(\Gamma_0)
\right].
\]

Run `n_gradient_steps` against the same stopped-gradient paths. Accept the
candidate only if its loss is finite and no larger than the starting loss
(within numerical tolerance); otherwise restore both raw parameters and Adam
state.

### 10.5 Full MCEM pseudocode

```text
function RUN_MCEM(data, initial_params, config, key):
    require all dt > 0
    fixed_mean_0 = STOP_GRADIENT(initial_params.mean_0)
    raw_params = ENCODE(initial_params)
    optimizer_state = ADAM_INIT(raw_params)
    construct inverse-Wishart prior from initial gamma_0

    for epoch in 0 .. n_epochs-1:
        params = DECODE(raw_params, fixed_mean_0)

        filter_states, augmented_data = RUN_RBPF(
            SPLIT_KEY(key), data, params, n_filter_particles
        )
        smoothed_states = RB_BACKWARD_SIMULATION(
            SPLIT_KEY(key), filter_states, augmented_data,
            params, n_smoother_particles
        )
        paths = STOP_GRADIENT(TRANSPOSE_TO_PATH_MAJOR(smoothed_states.x))

        starting_raw = raw_params
        starting_optimizer_state = optimizer_state
        starting_loss = MCEM_LOSS(raw_params, paths)

        repeat n_gradient_steps times:
            loss, gradient = VALUE_AND_GRAD(MCEM_LOSS)(raw_params, paths)
            updates, optimizer_state = ADAM_UPDATE(
                gradient, optimizer_state, raw_params
            )
            raw_params = APPLY_UPDATES(raw_params, updates)

        candidate_loss = MCEM_LOSS(raw_params, paths)
        if not finite(candidate_loss) or candidate_loss > starting_loss + tol:
            raw_params = starting_raw
            optimizer_state = starting_optimizer_state

        save parameter, likelihood, density, and smoother diagnostics

    final_params = DECODE(raw_params, fixed_mean_0)
    run one final RBPF for the final marginal-likelihood estimate
    return final parameters, histories, diagnostics, and final filter state
```

## 11. Diagnostics and acceptance criteria

The v2 implementation is not complete until it passes all of the following.

### 11.1 Structural validation

- `particles.x.shape == (D+1, N, M, 2)`.
- Smoothed paths have shape `(D+1, S, M, 2)`.
- Every `dt` is strictly positive.
- Every `gamma_pred[t]` is positive definite.
- Filtered and conditional covariances are symmetric positive semidefinite.
- The covariance timeline matches the state/transition contract in Section 2.1.
- No team appears twice on one day.

### 11.2 Dense Gaussian reference test

For a small $M$, explicitly construct

```text
P = kron(gamma_filtered, B)
R = phi**2 * P + (1 - phi**2) * kron(gamma_0, B)
J = P * phi @ inverse(R)
C = P - J @ R @ J.T
```

and verify that the Kronecker implementation agrees for:

- every component log weight;
- $J_t$;
- every conditional mean $h_t^{(i)}$;
- conditional covariance $C_t$;
- random-draw empirical means and covariances.

### 11.3 Linear-Gaussian end-to-end reference

On a small linear-Gaussian model, compare sampled smoothing statistics with a
Kalman/RTS reference:

- (E[X_t\mid y_{1:D}]);
- `Cov(X_t | y)`;
- (E[X_tX_{t+1}^\mathsf T\mid y]).

Agreement should improve with the number of filtering components and smoother
paths.

### 11.4 Particle diagnostics

Track at every time:

- effective sample size of forward weights;
- number of unique backward component indices;
- entropy or maximum probability of backward logits;
- initial and transition Mahalanobis statistics;
- finite values of all likelihood and prior terms.

For complete OU transitions, the mean Mahalanobis quadratic divided by latent
dimension should be of order one. It need not equal one exactly after smoothing,
but a large persistent value or a value that does not improve with $N$ signals
an incorrect backward kernel or timeline.

## 12. Result evaluation and model-risk checks

Evaluation is a required stage of every training run, not an optional plotting
step. It must answer three separate questions:

1. Is the numerical implementation internally correct?
2. Did MCEM produce a non-degenerate latent model?
3. Does the fitted model predict held-out football results better than the
   initial model and simple baselines?

A higher complete-data objective alone is not evidence that all three are true.

### 12.1 Known failure modes from the review

The evaluation report must explicitly check the following issues.

| Failure mode | Symptom | Required check |
|---|---|---|
| Forward means treated as complete states | Initial/full-state variation is far too small | Verify the particle representation and test Gaussian draws against component covariance |
| Point FFBS applied to RB components | Transition residuals are implausibly large | Confirm component selection uses $R_{t+1}$, followed by a draw from $\mathcal N(h_t,C_t)$ |
| Independent full-state materialization | Backward weights collapse in the high-dimensional state | Record backward ESS, entropy, maximum probability, and unique selected indices |
| Wrong covariance in backward weights | Selection uses $Q_t$ instead of $\phi_t^2P_t+Q_t$ | Dense small-model comparison of every backward logit |
| Residual covariance discarded | Selected means are returned without Gaussian uncertainty | Compare empirical conditional draws with $h_t$ and $C_t$ |
| Forward genealogy traced as smoothing | Early paths share very few ancestors | Verify fresh backward selection from all components and plot diversity by time |
| Covariance timeline shifted | Wrong `gamma_pred` is paired with a transition | Assert the $(D+1)$-state/$D$-transition contract before JIT |
| Transition term dominates MCEM | Total objective improves mainly through the $2M$-dimensional transition density | Report initial, transition, observation, and prior terms separately and per unit |
| Transition covariance collapses | `logdet(Q)` becomes very negative while transition log density becomes artificially large | Track eigenvalues, log determinant, trace, effective rank, condition number, and prior contribution |
| Invalid or nearly singular covariance | NaNs, unstable solves, or clipped substantive negative eigenvalues | Cholesky-check every predicted covariance and report PSD margins for filtered/conditional covariances |
| Insufficient path dependence | One-time summaries look plausible but OU parameters are wrong | Evaluate lag-one moments and transition residuals from complete paths |
| Monte Carlo instability | Results change materially with seed, $N$, or $S$ | Repeat fits/evaluations and report between-run variation |

The review reports a transition Mahalanobis ratio near 27 for the incorrect
materialized point-cloud smoother. The corrected implementation should produce
a ratio of order one and remain stable or improve as particle count increases.
This is a diagnostic target, not a universal hypothesis-test cutoff, because
smoothing conditions paths on observations.

### 12.2 Objective decomposition

For every epoch, record these terms before and after the M-step:

```text
mean initial log density
mean summed transition log density
mean summed observation log density
inverse-Wishart log prior
total log posterior objective
particle-filter log marginal-likelihood estimate
```

Also record normalized quantities so dimensionality does not obscure the
comparison:

```text
initial log density / latent dimension
transition log density / (D * latent dimension)
observation log density / number of valid matches
prior log density / number of covariance parameters
```

Decompose the transition term into its Gaussian normalization and quadratic
parts:

\[
\sum_t\log p(X_{t+1}\mid X_t)
=\underbrace{\sum_t
\left[-\frac d2\log(2\pi)-\frac12\log|Q_t|\right]}_{
\text{normalization}}
+
\underbrace{-\frac12\sum_t r_t^\mathsf TQ_t^{-1}r_t}_{
\text{quadratic penalty}}.
\]

This decomposition is essential for detecting transition-loss pathology. A
transition term that increases because residuals shrink is potentially useful;
a transition term that increases because $\log|Q_t|\rightarrow-\infty$ while
the covariance eigenvalues collapse is degenerate overfitting.

A positive Gaussian log density is not by itself an error: continuous
densities can exceed one. The warning is the joint pattern of a rapidly growing
normalization term, vanishing covariance scale, and poor or unstable predictive
performance.

Flag a run for review when any of the following occurs:

- the transition term improves while the observation term worsens materially;
- the normalization term grows rapidly but the quadratic penalty does not show
  a commensurate improvement;
- the smallest eigenvalue, trace, or effective rank of $\Gamma_0$ trends toward
  zero;
- the condition number grows without stabilizing;
- representative $Q_t$ eigenvalues approach numerical precision;
- `kappa` or the bounded attack/defence log-variance ratio remains at its
  parameterization boundary;
- the prior term, rather than the likelihood, supplies most of the final
  objective change.

These are warnings rather than universal numeric thresholds. Store the raw
series so thresholds can be selected for the dataset and floating-point dtype.

### 12.3 Smoother-quality evaluation

For paths $X_{0:D}^{*(s)}$, define complete-state OU residuals

\[
r_t^{(s)}=X_{t+1}^{*(s)}-\mu_0
-\phi_t(X_t^{*(s)}-\mu_0)
\]

and Mahalanobis quadratics

\[
q_t^{(s)}=
\operatorname{vec}(r_t^{(s)})^\mathsf T
Q_t^{-1}
\operatorname{vec}(r_t^{(s)}).
\]

Report

\[
\rho_{\text{transition}}
=\frac{1}{DSd}\sum_{t,s}q_t^{(s)},
\]

plus its median, 5th percentile, 95th percentile, and time series. Track the
analogous initial-state Mahalanobis ratio.

For each backward time and trajectory, calculate normalized component
probabilities $\pi_{t,s}^{(i)}$ and report:

\[
\operatorname{ESS}_{t,s}
=\frac{1}{\sum_i(\pi_{t,s}^{(i)})^2},
\qquad
H_{t,s}=-\sum_i\pi_{t,s}^{(i)}\log\pi_{t,s}^{(i)}.
\]

Save the minimum, mean, and quantiles of ESS and entropy, the maximum component
probability, and the number of unique selected component indices at every time.
Persistent ESS near one, maximum probability near one, or a single unique index
signals backward degeneracy.

The evaluation must also calculate smoothed:

- means and marginal variances by team, trait, and day;
- lag-one covariance $E[X_tX_{t+1}^\mathsf T\mid y]$;
- attack/defence credible intervals;
- path-to-path variability rather than only mean trajectories.

### 12.4 Parameter and covariance evaluation

For the initial parameters and every accepted epoch, save:

```text
gamma_0 minimum/maximum eigenvalue
gamma_0 trace, log determinant, and determinant sign
gamma_0 condition number and effective rank
minimum/median/maximum diagonal variance
B eigenvalues and log determinant
kappa and OU half-life log(2) / kappa
alpha, beta, and lambda_3 = exp(beta)
minimum/maximum eigenvalues of Q(dt) for representative dt values
```

Plot these series across parameter index $\theta_0,\ldots,\theta_K$. The
determinant-one constraint on $B$ must hold numerically. Every covariance used
as a proper Gaussian density must have positive determinant sign and finite log
determinant.

### 12.5 Held-out predictive evaluation

Use a chronological split; never randomly mix future matches into the training
set. Recommended evaluation is rolling-origin:

1. Fit parameters on dates through cutoff $c$.
2. Carry the filtered distribution forward without conditioning on future
   scores.
3. For each next match day, calculate its particle-mixture predictive score
   distribution.
4. Score the realized results, then optionally update the filter before moving
   to the following day.
5. Repeat for several cutoffs or folds.

For each held-out match, evaluate the entire score grid up to `max_goals`, not
only the most likely score. At minimum report:

- mean negative log predictive density of the observed exact score;
- home/draw/away Brier score and calibration by probability bin;
- ranked probability score for home and away goal marginals;
- mean absolute error of expected home and away goals;
- coverage and width of predictive goal intervals;
- observed versus predicted rates of home wins, draws, away wins, total goals,
  both-teams-to-score, and high-scoring tails.

Compare against:

- the untrained initial parameter model;
- a constant-rate independent-Poisson baseline;
- if available, the previous production/trained model.

A fitted model is accepted for predictive use only if it improves proper
held-out scoring rules without severe calibration deterioration. In-sample
complete-data likelihood is not a substitute for this test.

### 12.6 Monte Carlo convergence and reproducibility

For a representative training/evaluation window, run a small convergence grid:

```text
N in {N_base, 2 * N_base, 4 * N_base}
S in {S_base, 2 * S_base}
at least 3 independent PRNG seeds per selected configuration
```

Compare:

- filter log marginal likelihood;
- final parameters and OU half-life;
- transition Mahalanobis ratio;
- backward ESS and unique-index curves;
- smoothed means, variances, and lag-one covariances;
- held-out predictive scores.

Report Monte Carlo means and standard deviations. Increasing $N$ and $S$
should stabilize these quantities. If parameter movement across seeds exceeds
the claimed fitted effect, treat the fit as unresolved.

### 12.7 Evaluation pseudocode

```text
function EVALUATE_RUN(training_result, train_data, holdout_data, config, key):
    params = training_result.final_params
    paths = training_result.final_smoothed_paths

    structural = VALIDATE_TIMELINE_SHAPES_AND_COVARIANCES(
        training_result, train_data
    )
    objective = DECOMPOSE_OBJECTIVE_HISTORY(
        training_result.mstep_history,
        per_dimension=True,
        per_transition=True,
        per_match=True
    )
    covariance = SUMMARIZE_PARAMETER_PATH(
        training_result.params_history,
        representative_dt=config.representative_dt
    )
    smoother = EVALUATE_SMOOTHED_PATHS(
        paths,
        train_data,
        params,
        backward_probabilities=training_result.backward_probabilities,
        component_indices=training_result.backward_component_indices
    )
    predictive = ROLLING_ORIGIN_PREDICTIVE_EVALUATION(
        params, holdout_data, config.max_goals, SPLIT_KEY(key)
    )
    baselines = EVALUATE_BASELINES(holdout_data, config.max_goals)

    warnings = APPLY_MODEL_RISK_CHECKS(
        structural, objective, covariance, smoother, predictive, baselines
    )
    WRITE_JSON_METRICS_AND_PLOTS(
        structural, objective, covariance, smoother,
        predictive, baselines, warnings
    )

    return EvaluationReport(
        structural=structural,
        objective=objective,
        covariance=covariance,
        smoother=smoother,
        predictive=predictive,
        baselines=baselines,
        warnings=warnings,
        passed=(no hard structural failure and predictive checks pass)
    )
```

### 12.8 Required evaluation artifacts

Each run must write machine-readable JSON/Parquet metrics and the following
plots:

```text
evaluation_summary.json
objective_terms_by_epoch.png
transition_normalization_vs_quadratic.png
covariance_eigenvalues_and_condition.png
ou_half_life_and_parameters.png
transition_mahalanobis_by_time.png
backward_ess_entropy_and_unique_indices.png
smoothed_team_trajectories_with_intervals.png
heldout_log_score_by_date.png
result_calibration.png
goal_marginal_calibration.png
baseline_comparison.json
```

The summary must distinguish hard failures from warnings. Hard failures include
invalid timelines, non-finite values, non-positive predicted covariances, shape
mismatches, and failure of the dense Gaussian reference tests. Transition-term
dominance, low backward diversity, covariance contraction, and weak predictive
performance are model-risk warnings that block promotion until investigated.

## 13. Test strategy and mandatory final checks

Tests are part of the implementation, not follow-up work. Every executable
script must have at least one test that invokes its public entry point with a
small deterministic configuration. The final check must run the tests and a
real CPU smoke pipeline successfully from a clean process.

### 13.1 Test levels

Use four levels:

1. **Unit tests** validate individual likelihood, covariance, Kronecker,
   parameter-transform, and evaluation functions.
2. **Reference tests** compare optimized RB calculations with explicit dense
   Gaussian calculations and, where applicable, an RTS smoother.
3. **Integration tests** run the filter, smoother, M-step, and evaluation
   together on a small synthetic dataset.
4. **Script smoke tests** invoke each CLI entry point as a subprocess and
   verify its exit status and output artifacts.

All stochastic tests must use fixed PRNG seeds. Statistical tests must use
enough draws for a meaningful tolerance without making the default suite slow.
Mark larger convergence tests as `slow` so they can run in the final gate and
CI while remaining optional during rapid local development.

### 13.2 Unit-test matrix

#### Data preparation

Test that:

- matches are sorted and grouped into one state per date;
- padding values never contribute to the likelihood;
- team IDs are stable and contiguous;
- a team playing twice on one day raises `ValueError`;
- non-positive `dt` raises before JIT compilation;
- chronological train/holdout splitting never leaks a future match into
  training data;
- scores outside `max_goals` follow the documented filtering policy.

#### Bivariate-Poisson likelihood

For small goal counts, compare `loglik` and the complete score grid with a
direct probability-space implementation. Test that:

- the grid probabilities are finite, non-negative after exponentiation, and
  sum to the represented truncated mass;
- the observed-score lookup agrees with the corresponding grid cell;
- swapping home and away teams and scores produces the corresponding swapped
  likelihood;
- padded matches add zero log likelihood;
- gradients with respect to `alpha`, `beta`, and team strengths are finite;
- `max_goals < min(home_score, away_score)` returns the documented failure
  value rather than silently truncating the finite sum.

#### Parameter encoding and covariance constraints

Test round trips through `encode_EM_params` and `decode_EM_params`. Assert:

- `gamma_0` is symmetric positive definite;
- `kappa > 0`;
- `B` is positive definite, diagonal, and `det(B) == 1` within tolerance;
- the Kronecker covariance is invariant to the scale-identification transform;
- the inverse-Wishart kernel agrees with a direct dense computation;
- all decoded values and gradients remain finite near permitted boundaries.

#### Kronecker primitives

For random small positive-definite matrices, compare:

```text
kron_logdet(A, B)       versus slogdet(kron(A, B))
kron_quad(A, B, V)      versus dense quadratic forms
sample_kron_psd         versus target empirical mean/covariance
J_gamma and gamma_cond  versus dense RTS gain/covariance
```

Include rank-deficient positive-semidefinite filtered covariances. Confirm that
roundoff-scale negative eigenvalues are clipped and materially negative
eigenvalues raise an error.

#### Deterministic covariance recursion

For a small schedule, compare every saved `gamma_pred`, `gamma_observed`,
`kalman_gain`, and `gamma` with a direct NumPy implementation. Explicitly test:

- state zero uses `gamma_0`;
- transition `d -> d+1` uses `gamma_pred[d]`;
- no extra timeline shift is present;
- played-team rows and columns are zero after conditioning;
- multiple disjoint matches on one day are processed sequentially;
- all saved matrices have the required symmetry and definiteness properties.

### 13.3 Forward-filter tests

Use a two-to-four-team synthetic dataset. Test that:

- output shapes are exactly `(D+1, N, M, 2)` and `(D+1, N)`;
- every forward `.x` is documented and treated as a component mean;
- the shared covariance path has the correct shapes and timeline;
- normalized particle weights sum to one at every state;
- the accumulated log normalizing constant is finite;
- systematic resampling returns valid indices and preserves all particle-tree
  leaves consistently;
- two runs with the same key are identical;
- changing the key changes sampled components without changing the deterministic
  covariance path;
- JIT and eager execution agree within dtype-appropriate tolerance.

On a case with no informative score likelihood, compare the empirical filtered
mixture mean and covariance with the analytically propagated Gaussian. The
mixture covariance must include both between-component variation and the
within-component residual covariance:

\[
\operatorname{Cov}(X_t)
=\sum_iw_i\left[P_t+(m_i-\bar m)(m_i-\bar m)^\mathsf T\right].
\]

### 13.4 Backward-smoother tests

These tests are mandatory because the smoother is the primary correction in
v2.

#### Dense one-step reference

For each component in a small model:

1. Form full dense $P_t$, $Q_t$, and $R_{t+1}$.
2. Compute dense backward logits, $J_t$, $h_t^{(i)}$, and $C_t$.
3. Compare them with the Kronecker callback outputs.

The test must fail if the callback substitutes $Q_t$ for $R_{t+1}$.

#### Conditional-draw moments

Hold one component and future state fixed, draw many previous states, and
verify that their empirical mean and covariance agree with $h_t^{(i)}$ and
$C_t$ within Monte Carlo tolerance. Separately verify that the terminal
resampler draws the covariance $\Gamma_{D\mid D}\otimes B$, rather than
returning component means.

#### Component-selection frequencies

For fixed backward logits, draw many indices and compare empirical frequencies
with softmax probabilities. Include a case where the forward-largest component
is not the most compatible with the selected future endpoint.

#### Particle-tree contract

Verify that Cuthbert indexes every metadata leaf with the same selected
component, while `.x` is replaced by the conditional full-state draw. Confirm
that neither `log_density` nor forward ancestor indices affect the custom
backward choice.

#### Linear-Gaussian end-to-end reference

Run many smoother paths for a small linear-Gaussian model and compare sampled:

- smoothed means;
- marginal covariances;
- lag-one covariances;

with an exact Kalman/RTS reference. The discrepancy must decrease with larger
$N$ and $S$, within Monte Carlo error.

#### Regression test for the reviewed failure

Construct or retain a fixed fixture that reproduces the old materialized-cloud
failure. Assert that the v2 smoother:

- has an order-one transition Mahalanobis ratio rather than the historical
  value near 27;
- does not persistently select only one backward component;
- produces nonzero residual uncertainty in coordinates not sampled by the
  forward filter;
- is stable when particle count is increased.

Do not require the ratio to equal exactly one, since the paths are conditioned
on observations.

### 13.5 MCEM tests

Use a tiny dataset, few particles, and one or two gradient steps. Test that:

- the E-step returns complete paths with the documented timeline;
- gradients do not flow through sampled paths;
- initial, transition, observation, and prior terms are finite;
- the transition decomposition sums back to the transition log density;
- the M-step accepted candidate does not worsen the fixed-path objective beyond
  tolerance;
- a non-finite or worse candidate restores both parameters and optimizer state;
- covariance constraints remain valid after an accepted update;
- one EM epoch and the final filter complete successfully;
- saved parameter and diagnostic histories have consistent epoch indices.

Include a covariance-collapse regression test: create a candidate with very
small transition covariance and confirm that diagnostics flag the shrinking
eigenvalues/log determinant and that the prior prevents this candidate from
being silently interpreted as a healthy improvement.

### 13.6 Evaluation-function tests

Build controlled arrays for which the expected metrics are known. Test:

- transition and initial Mahalanobis ratios;
- backward ESS, entropy, maximum probability, and unique-index counts;
- objective normalization per dimension, transition, and match;
- covariance eigenvalue, condition-number, effective-rank, and OU half-life
  calculations;
- score-grid normalization and exact-score log predictive density;
- home/draw/away probabilities and Brier score;
- marginal ranked probability score;
- predictive interval coverage and width;
- warning versus hard-failure classification;
- comparison and promotion logic against baselines.

The rolling-origin test must assert that prediction for day $t$ is computed
before conditioning on day $t$'s score.

### 13.7 Script and artifact smoke tests

Provide small CLI scripts with explicit configuration flags:

```text
scripts/train.py
scripts/evaluate.py
scripts/smoke_test.py
```

Every script must expose a callable `main(argv=None) -> int` so it can be tested
without shell-specific behavior. Test both direct function invocation and one
subprocess invocation per script.

The smoke configuration should use:

```text
CPU backend
2-4 teams
2-4 observed days
8-16 filter particles
8-16 smoother paths
1 EM epoch
1-2 M-step gradient updates
fixed PRNG seed
temporary output directory
```

The script test must assert:

- exit code is zero;
- no traceback, NaN, or infinity appears in captured output or saved metrics;
- the final filter and RB-aware smoother both ran;
- the evaluation stage ran after training;
- all required JSON/plot artifacts exist and are nonempty;
- JSON artifacts parse successfully and contain only serializable finite
  numeric values;
- `evaluation_summary.json` records the configuration, seed, pass/fail state,
  and any model-risk warnings.

Model-risk warnings such as weak predictive skill may be permitted in a smoke
test. Structural or numerical hard failures must make the script return a
nonzero exit code.

### 13.8 Test layout

Use the following test structure:

```text
rbpf_v2/
    tests/
        conftest.py
        fixtures/
            small_matches.csv
        unit/
            test_bivariate_poisson.py
            test_data.py
            test_parameters.py
            test_kron.py
            test_covariance_recursion.py
            test_filter.py
            test_evaluation.py
        reference/
            test_backward_dense.py
            test_conditional_draws.py
            test_rts_reference.py
            test_review_regressions.py
        integration/
            test_e_step.py
            test_mcem.py
            test_rolling_evaluation.py
            test_artifacts.py
        scripts/
            test_train_script.py
            test_evaluate_script.py
            test_smoke_script.py
```

`conftest.py` should enable JAX's CPU backend before importing model modules and
provide deterministic small-model fixtures. Tests must write only to pytest's
`tmp_path`.

### 13.9 Mandatory final execution gate

From the repository root, the implementation is complete only when all of the
following commands exit with status zero:

```bash
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v2/tests/unit -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v2/tests/reference -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v2/tests/integration -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v2/tests/scripts -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v2/tests -m slow
RBPF_V2_SMOKE_DIR="$(mktemp -d /tmp/rbpf_v2_smoke.XXXXXX)"
RBSQMC_PLATFORM=cpu python -m rbpf_v2.scripts.smoke_test \
    --seed 0 \
    --n-particles 16 \
    --n-smoother-paths 16 \
    --n-epochs 1 \
    --n-gradient-steps 2 \
    --output-dir "$RBPF_V2_SMOKE_DIR"
```

The smoke script must cleanly create its target directory or require an empty
one; it must never overwrite a non-smoke training run. CI should execute the
same gate with a CI-owned temporary directory.

After those commands, perform these programmatic final assertions:

```text
all tests passed
smoke process exit code == 0
no non-finite values in evaluation_summary.json
all hard structural checks passed
dense Gaussian and RTS reference tests passed
timeline is D+1 states for D transitions
terminal and backward states are full Gaussian draws
backward component weights used R, not Q
transition Mahalanobis diagnostic is finite and not in the known failure regime
required training and evaluation artifacts exist
```

Do not report the implementation as successful merely because it imports,
compiles, or completes one optimizer step. Success means the mathematical
reference tests, full automated suite, and end-to-end smoke command all pass.

## 14. Proposed `rbpf_v2` module layout

`rbpf_v2` should mirror the existing package while separating model operations
from smoothing operations clearly:

```text
rbpf_v2/
    __init__.py
    IMPLEMENTATION.md
    data/
        results.parquet
        active_teams.json
        worldcup2026.json
        worldcup2026_team_regions.json
    src/
        utils.py                # NamedTuple/PyTree data contracts
        data.py                 # loading, filtering, grouping, validation
        bivariate_poisson.py    # score log likelihood
        helpers.py              # initialization and parameter transforms
        model.py                # covariance recursion and forward RBPF
        smoothing.py            # RB-aware FFBS, MCEM, diagnostics
        evaluation.py           # holdout scoring and model-risk report
        graphic.py              # plots and reporting
        model_trained.py        # trained-model entry point
    scripts/
        __init__.py
        train.py                # MCEM training CLI
        evaluate.py             # saved-model chronological evaluation CLI
        smoke_test.py           # tiny end-to-end final-check pipeline
    tests/                      # test tree from Section 13.8
```

The files may initially be copied from `rbpf`, but `smoothing.py` must make
these changes:

1. Remove `materialize_rb_filter` from the E-step.
2. Add `RBSmootherParticle` and attach covariance/timing metadata.
3. Add a terminal mixture resampler that draws complete terminal states.
4. Add the marginalized RB backward callback described in Section 7.
5. Use that callback in `build_smoother` instead of
   `exact_sampling.simulate`.
6. Update diagnostics and metadata so they no longer claim that a precomputed
   full-state cloud exists.
7. Call `evaluation.py` after the final filter and save the artifacts required
   by Section 12.8 alongside the training outputs.

## 15. End-to-end algorithm summary

```text
PREPROCESS MATCHES
    group by day, pad match batches, validate schedule and positive dt

FOR EACH MCEM EPOCH
    COMPUTE SHARED COVARIANCE PATH
        OU predict Gamma
        sequentially condition Gamma on teams sampled that day

    RUN FORWARD RBPF
        predict each component mean
        sample only teams entering the day's likelihood
        condition all other team means analytically
        score observed results
        normalize and resample
        save Gaussian-component means, covariances, and weights

    RUN RB-AWARE BACKWARD SIMULATION
        choose terminal components and draw full terminal states
        move backward through time
        score every component using marginalized prediction covariance R
        choose a component
        Gaussian-condition it on the selected future state
        draw the complete previous state

    RUN M-STEP
        freeze the sampled paths
        evaluate initial + OU transition + score log densities
        add inverse-Wishart covariance prior
        optimize constrained parameterization with Adam
        accept only a finite, non-worsening candidate

RUN FINAL FILTER

EVALUATE RESULTS
    validate shapes, timeline, and covariance definiteness
    decompose initial, transition, observation, and prior terms
    diagnose transition covariance collapse and Mahalanobis residuals
    measure backward ESS, entropy, and component diversity
    evaluate parameter stability across particles, paths, and seeds
    score chronological holdout matches against baseline models
    write metrics, plots, warnings, and a promotion decision

SAVE MODEL, TRAINING OUTPUTS, AND EVALUATION REPORT
```

This produces complete paths suitable for the existing path-based MCEM
objective while preserving the variance reduction obtained by
Rao--Blackwellization in the forward filter.
