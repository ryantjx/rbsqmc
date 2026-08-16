# RBPF Smoothing Review and Recommendations

## How to read this document

The document is ordered from newest to oldest. Recommendations in a newer
dated section supersede conflicting recommendations in older sections.
Historical sections are retained because they explain why each design changed.

1. **2026-08-16:** Latest transition diagnostics and the current
   Rao--Blackwellized backward-kernel recommendation.
2. **2026-08-15 23:45:** Audit of the `150826 2345` output snapshot.
3. **2026-08-15, earlier:** Intermediate fixed-cloud point-particle FFBSi
   design, now superseded for the 96-dimensional model.
4. **2026-08-15, time unavailable:** Earlier review of the analytic RB
   smoother and its then-unresolved timeline inconsistencies.

## 2026-08-16 — Current recommendation: Rao--Blackwellized backward simulation

> **Update (2026-08-16):** This section supersedes the earlier recommendation
> to materialize an independent full-state cloud at every time and run ordinary
> point-particle FFBSi. That construction is marginally valid as the particle
> count tends to infinity, but the latest diagnostics show severe finite-sample
> degeneracy in the 96-dimensional state.

### Current implementation status as of 2026-08-16

- The daily timeline is aligned and all recorded transition intervals are
  positive.
- The E-step now materializes full states, so RB component means are no longer
  passed directly to the M-step as complete states.
- Runtime diagnostics and focused regression tests cover initial
  materialization, covariance health, timeline alignment, transition
  statistics, parameter serialization, and smoother diversity.
- The independent fixed-cloud construction remains in `smoothing.py` and is
  the current bottleneck: it produces transition Mahalanobis ratios near 27 and
  severe backward-index duplication.
- The custom RB terminal sampler and RB backward sampling function described
  below have not yet been implemented.

### Why the fixed full-state cloud is still failing

The latest run has fixed the original representation error at time zero:

```text
materialized initial Mahalanobis ratio       approximately 1
timeline alignment                           correct
minimum eigenvalue of Gamma_0                0.058
condition number of Gamma_0                  13.8
```

The remaining transition diagnostics are not consistent with plausible OU
paths:

```text
transition normalization sum                 +160,126
transition quadratic penalty                 -1,411,918
transition log-density                       -1,251,792
mean transition Mahalanobis statistic          2,631.7
latent dimension                                     96
mean Mahalanobis ratio                            27.4
```

The sign of the transition density is not the diagnostic: a continuous
Gaussian log-density can be positive. The problem is that the transition
quadratic energy is approximately 27 times the latent dimension. The smoother
also uses only about 5.8 distinct previous particles on average and sometimes
only one.

`materialize_rb_filter` currently draws each time/component independently from
its filtered Gaussian marginal. Ordinary FFBSi then tries to connect two
independent 96-dimensional clouds using the narrow daily OU covariance. With
100 candidates, particularly for the 470 transitions having `dt == 1`, it is
very unlikely that any previous draw is close to a future draw in all 96
coordinates. More particles can reduce the problem, but this approach has the
usual exponential high-dimensional matching problem.

The next E-step should therefore retain the Gaussian components during backward
selection and sample a continuous state from the exact conditional Gaussian
after selecting a component. This is the Rao--Blackwellized backward kernel.

### Exact backward sampling versus RB backward sampling

The word `exact` in Cuthbert's `exact_sampling_simulate` means exact sampling
from the supplied **finite discrete cloud**. It does not mean exact sampling
from the continuous Gaussian mixture represented by the RBPF.

| | Exact point-particle backward sampling | RB backward sampling |
|---|---|---|
| Forward candidate | One complete stored state | A Gaussian component: mean plus covariance |
| Backward choice | Select one stored point | Select one Gaussian component |
| Returned state | The selected stored point | A new conditional draw inside the selected component |
| Uses residual `gamma` | No | Yes |

For an ordinary particle filter:

```text
stored particle = complete state
select particle index
return that stored state
```

For this RBPF:

```text
stored particle = Gaussian-component mean
select component index using the sampled future state
adjust that component using the future state
sample a complete previous state from the adjusted Gaussian
```

The RB algorithm still selects an ancestor **component**. The extra Gaussian
draw is required because selecting a component does not identify a point inside
that component. It resolves the uncertainty deliberately retained in
`Gamma_{t|t} kron B`.

Here “ancestor” means the index selected by the backward probability over all
filter components at time `t`. It need not be the forward resampling ancestor
stored by the particle filter. Simply tracing stored forward ancestors is a
different, more path-degenerate smoothing method.

Returning the selected component mean discards this uncertainty. Independently
materializing one point from every component before smoothing retains the
uncertainty marginally, but chooses those points before seeing the future state.
That is the source of the high-dimensional point-matching problem.

The RB kernel instead samples in the useful order:

```text
sampled future state X[t+1]
            ↓
select a previous Gaussian component
            ↓
condition that component on X[t+1]
            ↓
sample X[t] from the conditional mean and covariance
```

### Objects represented by the forward filter

At filter-state index `t`, particle `i` represents

\[
X_t\mid I_t=i,y_{1:t}
\sim
\mathcal N\left(m_t^{(i)},P_t\right),
\qquad
P_t=\Gamma_{t\mid t}\otimes B.
\]

The stored `particles.x[t, i]` is the component mean `m_t^(i)`, not a complete
state. The covariance `Gamma_{t|t}` is shared by the components because its
updates depend on the match schedule, not on scores or sampled latent values.

For observed day `d`, the current timeline is

```text
previous filter state               d
current filter state                d + 1
filtered previous covariance        [Gamma_0, gamma[:-1]][d]
predicted current covariance         gamma_pred[d]
filtered current covariance          gamma[d]
transition inputs                    model_inputs[d]
```

Thus the backward transition from state `d + 1` to state `d` must pair
`Gamma_filtered[d]` with `gamma_pred[d]`. It must not shift `gamma_pred` by one
again.

### Backward component weights

Suppose a complete future state `x_next` has already been sampled at state
`t + 1`. For each filtered component `i` at state `t`, define

\[
\phi_t=\exp(-\kappa\,dt_t),
\]

\[
a_{t+1}^{(i)}
=
\mu_0+\phi_t\left(m_t^{(i)}-\mu_0\right),
\]

and

\[
R_{t+1}
=
\phi_t^2P_t+Q_t
=
\Gamma_{t+1\mid t}\otimes B,
\qquad
Q_t=(1-\phi_t^2)(\Gamma_0\otimes B).
\]

The discrete component probabilities are

\[
\log \widetilde w_t^{(i)}
=
\log w_t^{(i)}
+
\log\mathcal N\left(
x_{t+1}^*;
a_{t+1}^{(i)},
R_{t+1}
\right).
\]

Normalize these log weights and sample one component index `I_t`. This is the
only discrete FFBSi-style choice. Do not evaluate the density at the stored
component mean as though it were a complete previous state. The observation
likelihood at `t + 1` is constant across candidate previous components once
`x_next` is fixed, so it is not required in these weights.

### Conditional Gaussian draw after selecting the component

After selecting `I_t = i`, sample the complete previous state from

\[
X_t\mid X_{t+1}=x_{t+1}^*,I_t=i,y_{1:t}
\sim
\mathcal N(h_t^{(i)},C_t),
\]

where

\[
J_t=P_t\phi_tR_{t+1}^{-1},
\]

\[
h_t^{(i)}
=
m_t^{(i)}
+J_t\left(x_{t+1}^*-a_{t+1}^{(i)}\right),
\]

and

\[
C_t=P_t-J_tR_{t+1}J_t^\top.
\]

The common Kronecker factor makes this cheaper than a dense `96 x 96` solve:

\[
J_t
=
\left(
\phi_t\Gamma_{t\mid t}
\Gamma_{t+1\mid t}^{-1}
\right)\otimes I_2,
\]

\[
C_t
=
\left[
\Gamma_{t\mid t}
-
\phi_t^2\Gamma_{t\mid t}
\Gamma_{t+1\mid t}^{-1}
\Gamma_{t\mid t}
\right]\otimes B.
\]

Consequently, the implementation only needs team-space solves with
`gamma_pred[d]`; `B` cancels from the smoother gain and remains in the
conditional draw covariance. Symmetrize the conditional team covariance before
factorization.

### Complete backward algorithm

1. At terminal state `T`, sample a component `I_T` using the terminal filtering
   weights.
2. Draw
   `X_T ~ Normal(m_T[I_T], Gamma_filtered[T] kron B)`.
3. For `t = T-1, ..., 0`, compute the marginalized component weights above
   using the already sampled `X_{t+1}`.
4. Sample `I_t` from those weights.
5. Draw `X_t` from the RTS conditional associated with `I_t` and `X_{t+1}`.
6. Return the complete continuous path `X_0:T` to the M-step, along with the
   selected component indices for diagnostics.

This procedure avoids matching two independent full-state clouds. It selects
among the lower-level RB mixture components and analytically accounts for the
Gaussian uncertainty that each component represents.

### Reference algorithm for the current daily timeline

The following is a direct reference implementation of one RB backward
trajectory. It is intentionally written as a reverse scan so the state and
covariance alignment is explicit. It assumes the existing `_kron_quad`,
`_kron_logdet`, and `_psd_sqrt` helpers.

```python
def _sample_kron_psd(key, mean, gamma, B):
    """Draw an array with covariance gamma kron B."""
    noise = jax.random.normal(key, mean.shape)
    L_gamma = _psd_sqrt(gamma)
    L_B = _psd_sqrt(B)
    return mean + L_gamma @ noise @ L_B.T


def sample_one_rb_backward_trajectory(
    key,
    filtered_states,
    rb_inputs,
    params,
):
    """Sample X[0:D+1] and its selected RB component indices.

    State 0 is the timestamp-zero prior state. State d+1 is the
    posterior state after observed day d.
    """
    component_means = jax.lax.stop_gradient(
        filtered_states.particles.x
    )                                      # (D+1, N, teams, 2)
    component_log_weights = jax.lax.stop_gradient(
        filtered_states.log_weights
    )                                      # (D+1, N)

    # Covariance belonging to every filter-state index.
    filtered_gamma = jnp.concatenate(
        [params.gamma_0[None], rb_inputs.gamma], axis=0
    )                                      # (D+1, teams, teams)

    dt = rb_inputs.timestamp - rb_inputs.timestamp_prev  # (D,)
    n_states, n_components = component_means.shape[:2]
    dimension = params.mean_0.size

    if n_states != dt.size + 1:
        raise ValueError("Expected D+1 filter states for D observed days")
    if filtered_gamma.shape[0] != n_states:
        raise ValueError("Filtered covariance and state timelines differ")

    # 1. Select a terminal component and draw a full terminal state.
    key, terminal_index_key, terminal_state_key = jax.random.split(key, 3)
    terminal_index = jax.random.categorical(
        terminal_index_key,
        component_log_weights[-1],
    )
    terminal_state = _sample_kron_psd(
        terminal_state_key,
        component_means[-1, terminal_index],
        filtered_gamma[-1],
        params.B,
    )

    def backward_step(carry, values):
        x_next, key = carry
        (
            means_t,
            log_weights_t,
            gamma_t,
            gamma_pred_next,
            dt_t,
        ) = values

        key, component_key, state_key = jax.random.split(key, 3)
        phi_t = jnp.exp(-params.kappa * dt_t)

        # 2. Predict every RB component from t to t+1.
        predicted_means = (
            params.mean_0[None]
            + phi_t * (means_t - params.mean_0[None])
        )                                  # (N, teams, 2)

        # 3. Integrate over each component when calculating how likely it
        #    is to produce the already sampled future state.
        residuals = (
            x_next[None] - predicted_means
        ).reshape(n_components, -1)
        quadratic = _kron_quad(
            gamma_pred_next,
            params.B,
            residuals,
        )
        log_determinant = _kron_logdet(
            gamma_pred_next,
            params.B,
        )
        log_predictive_density = (
            -0.5 * dimension * jnp.log(2.0 * jnp.pi)
            -0.5 * log_determinant
            -0.5 * quadratic
        )

        # 4. Select the previous RB component.
        backward_logits = log_weights_t + log_predictive_density
        component_index = jax.random.categorical(
            component_key,
            backward_logits,
        )

        # 5. Condition that component on x_next.
        # J_gamma = phi * gamma_t @ inv(gamma_pred_next), computed by solve.
        J_gamma = phi_t * jnp.linalg.solve(
            gamma_pred_next,
            gamma_t.T,
        ).T
        future_error = x_next - predicted_means[component_index]
        conditional_mean = (
            means_t[component_index] + J_gamma @ future_error
        )
        conditional_gamma = (
            gamma_t
            - J_gamma @ gamma_pred_next @ J_gamma.T
        )
        conditional_gamma = 0.5 * (
            conditional_gamma + conditional_gamma.T
        )

        # 6. Resolve the continuous uncertainty inside the selected
        #    component. This is the step absent from exact point FFBSi.
        x_t = _sample_kron_psd(
            state_key,
            conditional_mean,
            conditional_gamma,
            params.B,
        )
        return (x_t, key), (x_t, component_index)

    # Transition d connects state d to state d+1, so use all D predictions
    # without an additional [1:] shift.
    scan_values = (
        component_means[:-1],              # states 0 ... D-1
        component_log_weights[:-1],
        filtered_gamma[:-1],               # gamma_0, gamma[0:D-1]
        rb_inputs.gamma_pred,               # arrival covariance for 1 ... D
        dt,
    )
    (_, _), (previous_states, previous_indices) = jax.lax.scan(
        backward_step,
        init=(terminal_state, key),
        xs=scan_values,
        reverse=True,
    )

    states = jnp.concatenate(
        [previous_states, terminal_state[None]], axis=0
    )
    component_indices = jnp.concatenate(
        [previous_indices, terminal_index[None]], axis=0
    )
    return states, component_indices


def sample_rb_backward_trajectories(
    key,
    filtered_states,
    rb_inputs,
    params,
    n_trajectories,
):
    keys = jax.random.split(key, n_trajectories)
    return jax.vmap(
        lambda trajectory_key: sample_one_rb_backward_trajectory(
            trajectory_key,
            filtered_states,
            rb_inputs,
            params,
        )
    )(keys)
```

This code samples from the same mathematical backward kernel as the archived
`smoother_rts`, but uses the current timeline convention:

```text
filter states             0, 1, ..., D
observed-day transitions  0, 1, ..., D-1
filtered covariance       [gamma_0, gamma[0], ..., gamma[D-1]]
prediction for transition d                    gamma_pred[d]
```

Before JIT compilation, validate that every `dt` is positive and every
`gamma_pred[d]` is positive definite. Do not silently project
`gamma_pred[d]` for the backward weights and then use a different unprojected
matrix for the conditional draw. Filtered and conditional covariances may be
positive semidefinite; their square-root helper should clip only numerical
roundoff and report materially negative eigenvalues.

### How this fits Cuthbert `build_smoother`

`build_smoother` can remain the outer backward-scan machinery, but both of its
sampling hooks must obey the RB representation:

- The terminal `resampling_fn` must first select terminal component indices and
  then draw full terminal states from their component Gaussians. Returning the
  selected component means is not sufficient.
- Replace `exact_sampling_simulate` with a custom
  `rb_backward_sampling_fn`. For each supplied full `x1`, it computes the
  marginalized weights over `x0_all` component means, selects an index, draws
  the conditional full `x0`, and returns both that draw and the selected index.

Conceptually its contract is:

```text
rb_backward_sampling_fn(
    key,
    previous_component_means,
    future_full_states,
    previous_log_weights,
    previous_filtered_gamma,
    current_predicted_gamma,
    phi,
    B,
) -> previous_full_states, selected_component_indices
```

Cuthbert's hook does not directly pass the per-time covariance metadata to the
backward function. Use a smoother-only particle structure containing `x`, the
shared filtered covariance, the predicted covariance for arrival at that
state, and `phi`/`dt`; alternatively implement the short backward scan outside
Cuthbert. Do not obtain the missing metadata from global mutable state or infer
the time index from array values.

The existing `exact_sampling_simulate` cannot implement this kernel: by design,
it returns `x0_all[index]`. It never makes the conditional Gaussian draw.

### Numerical requirements and regression tests

- For every positive `dt`, `gamma_pred[d]` should be positive definite. Prefer
  a Cholesky solve and report a failure rather than silently using `pinv`.
- Filtered and conditional covariances may be positive semidefinite because
  observed team blocks have been conditioned exactly. Use an eigenvalue square
  root that clips only roundoff-sized negative values.
- Test the component weights, conditional mean, and conditional covariance
  against a direct dense Gaussian-conditioning calculation in a small model.
- Compare smoothing mean, covariance, and lag-one covariance against an exact
  Kalman/RTS reference.
- Retain the transition Mahalanobis diagnostics. The ratio need not equal one
  exactly after conditioning on observations, but it should be order one and
  stable as the number of particles increases, rather than remaining near 27.
- Continue monitoring component-index diversity. Some duplication is normal;
  persistent collapse to one component across many dates is not.

## 2026-08-15 23:45 — Audit snapshot `150826 2345`

This audit reviews the artifacts in `rbpf/outputs/smoothing` written at
2026-08-15 23:39, together with the model and source tree visible at the
requested 23:45 snapshot. The source file `rbpf/src/smoothing.py` was modified
at 23:41, two minutes after the artifacts were written, so exact source/artifact
provenance is not guaranteed. The output itself nevertheless contains a strong
diagnostic signature that identifies which state representation was used.

> **Historical scope:** This section describes that timestamped run. Its
> recommendation to restore full-state materialization was implemented later.
> The 2026-08-16 diagnostics then exposed the separate high-dimensional
> fixed-cloud problem discussed in the current section above.

### Executive conclusion

The positive transition log-density is not caused by a sign error. A positive
Gaussian log-density is expected here: each daily transition is a density in
96 dimensions, most time gaps are short, and the OU innovation covariance is
small in the chosen latent-state units.

There is, however, a critical E-step representation error in the audited run:
Rao--Blackwellized component means were treated as complete latent states.
The source at the audited snapshot showed `materialize_rb_filter` commented out
and passed `filtered_states` directly to Cuthbert FFBSi. The M-step then
evaluated a full-state OU density on conditional means whose residual Gaussian
uncertainty had been omitted. This biased the covariance update and helped
explain the rapidly increasing condition number of `Gamma_0` and declining
observed-data likelihood.

The inverse-Wishart prior is active and penalizes departure from the initial
covariance, but a prior cannot repair an incorrect E-step distribution. Restore
full-state materialization before increasing the prior strength.

### Run configuration and data scale

The run used the configuration present in `main` at the audited snapshot:

| Quantity | Value |
|---|---:|
| Date range | 2000-01-01 to 2026-01-01 |
| Teams | 48 |
| Latent dimension per state | `48 * 2 = 96` |
| Matches | 1,790 |
| Unique observed days / transitions | 1,073 |
| Filter particles | 10 |
| Smoother trajectories | 10 |
| EM epochs | 5 |
| Adam steps per M-step | 10 |
| Learning rate | `1e-3` |

The time differences are all positive. Their minimum is 1 day, median is 2
days, mean is 8.85 days, and maximum is 260 days. There are 470 one-day gaps
and 597 gaps of at most two days. Thus many transitions have a small OU
innovation multiplier.

### Output review

#### Observed-data likelihood

`em_log_marginal_history.json` contains:

```text
theta_0  -5440.9263
theta_1  -5440.5483
theta_2  -5462.7739
theta_3  -5479.8853
theta_4  -5491.4463
```

There is a negligible improvement of `0.38` followed by three substantial
decreases. From `theta_0` to `theta_4`, the estimate falls by `50.52` log units.
With only 10 filter particles and independent random keys, an individual
estimate is noisy, but a decline of this size and direction is not evidence
that EM is working well.

There are six parameter entries (`theta_0` through `theta_5`) but only five
likelihood entries. The final filtering run evaluates `theta_5`, but its value
is printed and then discarded. Therefore the saved likelihood curve is not a
complete evaluation of `em_final_params.json`.

#### M-step terms

All five candidate M-steps reduced the fixed-path negative MAP objective, so
the gradient sign and within-step acceptance rule were consistent in the
audited source. Across the accepted candidates:

```text
transition log-density: 119927.34 -> 124995.87
observation log-density:  -5142.15 ->  -5143.16
prior kernel:              3226.69 ->   2460.34
```

The observation term does not show a sustained improvement. The transition
term gains about `5068.5`, while the inverse-Wishart kernel loses about `766.3`.
The prior is therefore opposing the covariance movement, but the complete-data
transition term dominates it.

The absolute sign of `prior_log_density` has no interpretation because the
inverse-Wishart normalizing constant is omitted. Only changes in the kernel are
meaningful for optimization.

#### `Gamma_0` trajectory

The covariance is not collapsing uniformly toward zero. Its trace increases,
while one or more eigen-directions shrink sharply:

| Parameter entry | Min eigenvalue | Max eigenvalue | Condition number | Trace | Log determinant |
|---:|---:|---:|---:|---:|---:|
| `theta_0` | 0.0800 | 1.3814 | 17.3 | 7.680 | -108.851 |
| `theta_1` | 0.0610 | 1.4029 | 23.0 | 7.820 | -108.888 |
| `theta_2` | 0.0488 | 1.4288 | 29.3 | 8.004 | -108.953 |
| `theta_3` | 0.0369 | 1.4663 | 39.8 | 8.267 | -109.079 |
| `theta_4` | 0.0246 | 1.5009 | 60.9 | 8.614 | -109.324 |
| `theta_5` | 0.0143 | 1.5379 | 107.8 | 9.095 | -109.689 |

The median marginal variance rises from `0.1600` to `0.1928`, and the trace
rises by about 18%. The correct description is therefore **growing anisotropy
and loss of rank**, not scalar covariance collapse. The minimum eigenvalue
falls by a factor of 5.6 and the condition number grows by a factor of 6.2.

The smallest ordinary Cholesky diagonal remains approximately `0.279` in the
final matrix, far above `GAMMA_CHOL_FLOOR = 1e-4`. This demonstrates why a floor
on Cholesky diagonals does not bound the minimum eigenvalue of a correlated
matrix. Off-diagonal combinations can still create a nearly singular
eigen-direction.

Other parameter movements are moderate:

```text
kappa: 0.010000 -> 0.010296     (OU half-life 69.3 -> 67.3 days)
alpha: 0.200000 -> 0.156750
beta: -4.000000 -> -3.950920
B: diag(1, 1) -> diag(1.03633, 0.96495)
det(B): approximately 1 throughout
```

`B` is not near its transform boundary and the determinant-one identification
is working.

### Why the transition density is so positive

For a transition residual `r_t`, define

```text
Q_t = (1 - phi_t^2) * (Gamma_0 kron B)
q_t = r_t.T @ inv(Q_t) @ r_t
```

Then the transition log-density decomposes as

```text
ell_t = [-0.5*d*log(2*pi) - 0.5*logdet(Q_t)] - 0.5*q_t,
```

with `d = 96`. For the final accepted M-step, the exact aggregate decomposition
from the saved parameters and timestamps is:

```text
sum Gaussian normalization terms     172835.13
sum quadratic penalties              -47839.26
reported transition log-density       124995.87
```

For genuine draws from a 96-dimensional transition, `E[q_t] = 96`. Across
1,073 transitions, the expected quadratic penalty is therefore

```text
-0.5 * 1073 * 96 = -51504.
```

The corresponding expected transition log-density under the final candidate
is about `121331.13`, which is already strongly positive. The reported value is
only about `3664.7` higher than that reference. Per transition, the reported
density is `116.5`, versus an under-model expectation of `113.1`.

This positivity arises because:

1. Gaussian values are continuous **densities**, not probability masses, and
   densities can exceed one.
2. The density is evaluated in 96 dimensions.
3. At `kappa ~= 0.01`, a one-day interval has
   `1 - exp(-2*kappa*dt) ~= 0.0198`; the initial median per-coordinate
   innovation variance is only about `0.00317` for a one-day gap.
4. Almost half of all intervals are one day.

It is therefore incorrect to diagnose the transition solely from it being
positive or much larger than the observation log-probability. The observation
term is a discrete log probability and is necessarily non-positive; the
transition is a unit-dependent high-dimensional log density. Do not divide or
arbitrarily reweight these terms merely to make their plotted magnitudes
similar, because that would no longer be EM for the stated model.

The transition increase in this run is also not mainly caused by the
determinant normalization exploding. From the first to the last accepted
candidate, the summed normalization changes only from `172898.47` to
`172835.13` (a decrease of about `63.3`). The approximately `5068.5` transition
gain comes almost entirely from reducing the quadratic penalty: the inferred
average Mahalanobis statistic falls from `98.7` to `89.2` per transition. The
optimizer is reshaping `Gamma_0` around the supplied paths, particularly along
low-residual directions.

### Confirmed representation error in the audited output

An RBPF particle in this implementation is not a complete state. It represents

```text
X_t | component i, y_1:t
    ~ Normal(m_t[i], gamma_t kron B),
```

where `particles.x` stores `m_t[i]`. `propagate_sample` samples the teams that
play and analytically conditions the remaining teams, so unplayed entries are
conditional means with residual covariance stored separately in `gamma_t`.

The audited E-step passed `filtered_states` directly to `cuthbert.smoother`;
the `materialize_rb_filter` call was commented out. Cuthbert's
`exact_sampling.simulate` treats every stored particle as a complete point and
returns a selected stored point. It does not know about `gamma_t`. Consequently
the smoother and M-step treat `m_t[i]` as `X_t[i]` and discard the analytical
uncertainty.

The output proves this happened for the timestamp-zero state. For each accepted
candidate, the logged initial density equals its Gaussian normalization term to
floating-point precision:

```text
epoch 0: normalization 20.670079, logged 20.670080, inferred quadratic ~= 0
epoch 4: normalization 21.471244, logged 21.471252, inferred quadratic ~= 0
```

Thus every smoothed `x_0` is exactly `mean_0`, which is the deterministic RB
component mean returned by `init_sample`. A full initial-state draw would have
an expected Mahalanobis statistic of 96 and an expected initial log-density
equal to the normalization term minus 48. At the initial parameters that is
approximately `20.63 - 48 = -27.37`, not `+20.63`.

This is the strongest direct evidence of the bug. The transition Mahalanobis
statistics are closer to 96 because played teams are genuinely sampled during
forward propagation, but the residual uncertainty of unplayed teams is still
missing. The covariance M-step is therefore fitted to a hybrid of sampled
values and conditional means rather than complete latent trajectories.

### Review of the rest of the model

#### Components that are internally consistent

- Daily grouping and `timestamp_prev` produce strictly positive time gaps for
  this run; no `dt = 0` transition is present.
- `compute_gamma_trajectory` performs one OU covariance prediction per day and
  sequentially conditions on disjoint same-day matches.
- `propagate_sample` starts from the OU-predicted conditional mean, samples the
  currently observed teams from `gamma_observed kron B`, and updates all
  conditional means with the matching Kalman gain.
- The filter observation potential is valid because teams appearing in that
  day's likelihood have been explicitly sampled.
- The row-major Kronecker convention in `_kron_quad`, `_kron_logdet`, and
  `L_gamma @ noise @ L_B.T` is consistent with `Gamma_0 kron B`.
- `log_transition_density` implements the stationary OU transition covariance
  correctly for **full** states.
- The M-step minimizes the negative complete-data MAP objective, carries both
  raw parameters and Adam state forward, and rejects a final candidate that
  fails to improve the fixed-path objective.
- The inverse-Wishart kernel on `Gamma_0` is correct up to a constant, and its
  hyperparameters remain fixed over EM epochs.

#### Remaining model/code concerns

1. **Wrong E-step input to FFBSi (critical).** The materialization function
   exists but is bypassed. This must be fixed before interpreting covariance
   learning or tuning the prior.
2. **Only 10 particles and 10 trajectories (high).** This is suitable for a
   smoke test, not for learning a full 48-by-48 covariance matrix. It increases
   Monte Carlo error and path duplication.
3. **Full `Gamma_0` is highly flexible (high).** It has 1,176 unique covariance
   entries. Even with many time points, weakly observed eigen-directions can be
   fitted to a small, correlated set of smoothed trajectories. Consider a
   structured covariance after correctness is restored: regional block plus
   diagonal, low-rank plus diagonal, sparse precision, or shrinkage toward the
   regional initial matrix.
4. **The prior does not guarantee an eigenvalue floor (medium).** The current
   inverse-Wishart prior discourages singularity but can be overwhelmed by
   1,073 transition terms. `dof = dimension + 10` is a modeling choice, not a
   theorem. Tune it only after the E-step is fixed. A stronger prior should not
   be used to conceal missing latent variance.
5. **`kappa` and covariance scale are weakly coupled (medium).** For small daily
   gaps, `1 - exp(-2*kappa*dt) ~= 2*kappa*dt`, so the innovation covariance is
   approximately `2*kappa*dt*Gamma_0 kron B`. Short-gap data primarily identify
   this product. The initial distribution and long gaps add information, but
   diagnostics should report both `kappa` and representative eigenvalues of
   `Q_t`, not only `Gamma_0`.
6. **Initial and final likelihood history is incomplete (medium).** Save the
   final `theta_5` likelihood, align likelihood labels with parameter indices,
   and store run configuration and random seed with every artifact.
7. **Fresh-key likelihood comparisons are noisy (medium).** Evaluate each
   parameter entry with several independent filter runs and report mean and
   standard error. Common random numbers can be an additional diagnostic but
   should not replace independent replication.
8. **`pinv` can hide covariance failures (medium).** `gamma_OO` should be
   positive definite before a valid match. Prefer a Cholesky solve and explicitly
   fail or log when the minimum eigenvalue is below tolerance.
9. **`_psd_sqrt` silently clips all negative eigenvalues (medium).** Clip only
   roundoff-sized negatives and raise on materially negative values; otherwise
   an invalid filtering covariance can be silently converted into a different
   PSD covariance.
10. **No RBPF/EM regression diagnostics (high at the audited snapshot).** The
    model, materialization, timeline alignment, and transition statistics were
    not covered. This was addressed on 2026-08-16 by runtime diagnostics and
    focused synthetic regression tests; full end-to-end parameter recovery on
    simulated match data remains a recommended follow-up.

#### `MODEL.md` inconsistencies to correct

- The opening calls the dynamics a random walk, while the implementation is a
  stationary OU process.
- The first definition reverses the Kronecker factor dimensions. The code uses
  `Gamma_0` as the `M x M` team covariance and `B` as the `2 x 2`
  attack/defence covariance.
- The transition equation has `X_{t-1}` on the left; it should have `X_t`.
- The smoothing section mixes an analytic Rao--Blackwellized RTS-mixture
  smoother with Cuthbert's point-particle FFBSi. Cuthbert `exact_sampling`
  samples exactly from the supplied **discrete** cloud; it does not integrate
  or draw from the Gaussian covariance attached conceptually to an RB
  component.
- The statement that differing transition and observation dimensions should be
  corrected by loss scaling is misleading. Their unweighted sum is the stated
  joint log-density. Structural regularization is preferable to arbitrary term
  weights.
- The particle likelihood estimator `Z_hat` is unbiased under the standard
  particle-filter assumptions; `log(Z_hat)` is biased by Jensen's inequality.
  The current text incorrectly states that `E[Z_hat] != Z` as the generic
  particle-filter result.

### Prioritized next steps

#### P0: restore a correct E-step before another substantive run

Use `materialize_rb_filter` once after filtering and pass the resulting fixed
point cloud to Cuthbert:

```python
materialized_states = materialize_rb_filter(
    materialize_key,
    filtered_states,
    model_inputs_rbpf.gamma,
    params.gamma_0,
    params.B,
)

smoothed_states = cuthbert.smoother(
    smoother_obj,
    materialized_states,
    model_inputs_rbpf,
    parallel=False,
    key=smoother_key,
)
```

Do not materialize inside the backward potential and do not draw a new value
after Cuthbert selects an index. The cloud must remain fixed for the whole
FFBSi pass. Keep `build_smoother` and `exact_sampling_simulate`.

The observation part of `joint_log_potential` is constant across candidate
previous states once the future full state is fixed, so it cancels in backward
weight normalization. It may be retained because Cuthbert calls this a joint
potential, or removed for efficiency after a test confirms identical sampled
index probabilities. It is not responsible for the positive transition term.

#### P0: do not strengthen the prior yet

Keep the current prior for the first corrected comparison. A stronger prior
would make it harder to distinguish a fixed E-step from prior-induced
stability. After materialization is restored, compare:

1. current prior (`dof = M + 10`);
2. no prior on a small synthetic recovery test;
3. stronger shrinkage only if eigenvalues still drift under correctly sampled
   paths.

#### P1: add transition and representation diagnostics

For every E/M epoch, save:

```text
initial normalization, quadratic statistic, and log density
transition normalization sum
transition quadratic sum and q_t / 96 distribution
min/max eigenvalue, condition number, trace, logdet of Gamma_0
representative min/max eigenvalues of Q_t for dt = 1, 2, 7, 30
unique smoother paths or unique particle indices per time
filter likelihood mean and standard error over multiple seeds
```

The most direct regression check is the initial-state statistic. With full
materialization and parameters equal to the sampling parameters:

```text
mean Mahalanobis(x_0 - mean_0) approximately 96
mean initial log-density approximately normalization - 48
```

It must not remain approximately zero / equal to the normalization term.

#### P1: rerun in stages

1. Use a short date range and 10 particles to verify shapes and the initial
   Mahalanobis test.
2. Use a synthetic dataset generated from known parameters and verify recovery
   without covariance rank loss.
3. Increase to at least 50--100 filter particles and smoother trajectories for
   the real-data diagnostic run.
4. Run several seeds before concluding that the observed likelihood moves in a
   particular direction.

#### P1: add focused tests

- `encode_EM_params` followed by `decode_EM_params` preserves the identified
  covariance `Gamma_0 kron B`.
- Materialized `x_0` has empirical mean `mean_0` and covariance
  `Gamma_0 kron B`.
- Materialized posterior particles preserve played-team values when their
  residual covariance rows/columns are zero.
- The average transition Mahalanobis statistic for simulated full OU states is
  close to the latent dimension.
- Passing RB means directly produces the expected failing initial-Mahalanobis
  test, preventing this regression.
- Padding contributes zero likelihood and does not change covariance or state.
- All valid daily `dt` values are strictly positive.
- Same-day duplicate teams raise before JAX conversion.
- The final saved likelihood corresponds to `em_final_params.json`.

### Acceptance criteria for claiming the smoother/EM works

Do not use a positive transition log-density as a failure criterion. Instead,
require all of the following:

1. The M-step receives complete continuous states, but backward component
   selection uses the RB means and covariances rather than treating those means
   as complete states. The terminal Mahalanobis diagnostic should be
   statistically consistent with dimension 96.
2. No materially negative covariance eigenvalues are silently clipped.
3. `Gamma_0` eigenvalues stabilize rather than showing a persistent monotone
   loss of rank across epochs.
4. The repeated-seed mean observed-data likelihood does not show the sustained
   deterioration present in this audit.
5. Results are stable when particle and smoother counts are increased.
6. A synthetic recovery test estimates known parameters within Monte Carlo
   uncertainty.

Until these conditions hold, the large positive transition term should be
understood as mostly a normal high-dimensional density value combined with a
separate, confirmed state-representation bias—not as proof that the transition
formula has the wrong sign.

## 2026-08-15 — Intermediate fixed-cloud FFBSi design (superseded)

> **Historical scope:** The exact time of this design note was not recorded. It
> predates the 2026-08-16 diagnostic run. The point-particle construction below
> fixed the earlier “RB means as states” error, but was subsequently superseded
> because independent 96-dimensional clouds produced transition degeneracy.

<details>
<summary>Show the intermediate point-particle design review</summary>

In this collapsed section, “current” and “now” refer to the repository state
reviewed on 2026-08-15, not the latest repository state.

### Status at the time of this review

The revised daily forward RBPF is now conceptually consistent:

- there is one OU prediction per day;
- matches within a day are processed sequentially;
- each match uses the covariance and conditional mean left by earlier matches;
- `gamma_observed[d, j]` is the pre-match `2 x 2` team covariance;
- `kalman_gain[d, j]` is the corresponding `(num_teams, 2)` gain;
- the per-day particle scan starts at the OU-predicted mean;
- padding is skipped before covariance updates or random sampling;
- filter particles represent Gaussian-component conditional means, not complete
  latent states.

The main unresolved issue is now the boundary between these RB filter
components and the standard FFBSi implementation.

### How Cuthbert FFBSi actually samples

`cuthbertlib.smc.smoothing.exact_sampling.simulate` does not draw a new
continuous value from a weighted Gaussian mixture. For a selected future state
`x1`, it computes

\[
\log \widetilde w_t^{(i)}
=
\log w_t^{(i)}+\log p(x_{t+1}\mid x_t^{(i)}),
\]

samples an index, and returns the stored value `x0_all[index]`. The word
`exact` means exact sampling from the supplied *discrete particle
approximation*. This agrees with the intended standard FFBSi algorithm and
should continue to be used through `build_smoother`.

There is nevertheless a representation problem if the current filter output is
passed directly to it. An RB particle is a Gaussian component

\[
X_t\mid I_t=i,y_{0:t}
\sim
\mathcal N\left(m_t^{(i)},\Gamma_{t\mid t}\otimes B\right),
\]

while `exact_sampling.simulate` assumes each stored `x0_all[i]` is already a
complete state. Passing `m_t^(i)` directly would recreate the original error:
conditional means would be treated as complete latent states in both the
backward transition density and the M-step.

### Recommended `smoothing.py` construction

#### 1. Keep the forward RBPF output unchanged

Do not turn the forward filter particles into full-state particles. The forward
filter should retain its Rao--Blackwellized representation because
`propagate_sample` and the analytical covariance trajectory depend on that
contract.

#### 2. Build a fixed full-state particle cloud for FFBSi

Before calling `build_smoother`, make a separate copy of the filtering
trajectory and materialize one complete state from every Gaussian component:

\[
\widetilde X_t^{(i)}
=m_t^{(i)}+L_{\Gamma_t}Z_t^{(i)}L_B^\top,
\qquad Z_t^{(i)}[a,b]\overset{iid}{\sim}\mathcal N(0,1).
\]

Here `L_Gamma @ L_Gamma.T = Gamma_t` and `L_B @ L_B.T = B`. This matrix
construction has covariance `Gamma_t (x) B` for the row-major `(team, trait)`
layout used in the code.

With the current filter timeline, `D` daily inputs produce `D+1` filter states:

```text
filter state 0     prior component, covariance gamma_0
filter state d+1   component after day d, covariance gamma[d]
```

The materialization covariance sequence must therefore be

```text
[gamma_0, gamma[0], gamma[1], ..., gamma[D-1]]
```

and have the same leading length as `filtered_states.particles.x`.

Filtered `gamma[d]` is intentionally positive semidefinite and rank deficient
because teams sampled that day have zero residual covariance. Use a PSD square
root, such as an eigendecomposition that clips only small negative roundoff
eigenvalues to zero. Do not make this covariance artificially full rank with
jitter.

Replace only `particles.x` in a copy of the filter-state trajectory with these
fixed draws. Preserve the original filtering log weights. The original RB means
should remain available separately for diagnostics.

This materialization belongs to the **E-step** and happens once per EM epoch,
before FFBSi. It does not happen inside the backward-weight calculation or
after selecting a previous index. Holding this cloud fixed is what lets
Cuthbert perform ordinary point-particle FFBSi.

```python
def _psd_sqrt(A):
    values, vectors = jnp.linalg.eigh(0.5 * (A + A.T))
    # Only roundoff-sized negative values should reach this point.
    values = jnp.maximum(values, 0.0)
    return (vectors * jnp.sqrt(values)[None, :]) @ vectors.T


def materialize_rb_filter(key, filtered, gamma, gamma_0, B):
    means = filtered.particles.x                 # (D+1, N, teams, 2)
    gammas = jnp.concatenate([gamma_0[None], gamma], axis=0)
    L_gamma = jax.vmap(_psd_sqrt)(gammas)
    L_B = _psd_sqrt(B)
    keys = jax.random.split(key, means.shape[:2])

    def draw_time(keys_t, means_t, L_t):
        return jax.vmap(
            lambda k, m: m + L_t @ jax.random.normal(k, m.shape) @ L_B.T
        )(keys_t, means_t)

    full_x = jax.vmap(draw_time)(keys, means, L_gamma)
    particles = filtered.particles._replace(x=full_x)
    return filtered._replace(particles=particles)
```

One draw per component gives a standard Monte Carlo particle approximation to
the RB Gaussian mixture. It introduces more variance than an analytic RB
smoother. If needed, use more filter particles or materialize several children
per component with weights `w_i / L`.

#### 3. Use the full-state OU transition density

Once the stored candidates are fixed complete states, the backward density is
the ordinary OU transition

\[
X_{d+1}\mid X_d
\sim
\mathcal N\left(
\mu_0+\phi_d(X_d-\mu_0),
(1-\phi_d^2)(\Gamma_0\otimes B)
\right).
\]

Thus Cuthbert's backward candidate weights are

\[
\log w_d^{(i)}
+\log p(\widetilde X_{d+1}^*\mid\widetilde X_d^{(i)}).
\]

`kalman_gain` and `gamma_observed` are not part of this backward density; they
were required to construct the forward filtering mixture. `gamma_pred` is also
not required by standard full-state FFBSi, although retaining it is useful for
covariance diagnostics or a future analytic RB smoother.

The next day's observation likelihood is constant across candidate previous
states after the future state is fixed. Including it in Cuthbert's joint
`log_potential` therefore cancels when the backward weights are normalized. A
transition-only log density is simpler and avoids repeating that calculation.

```python
def full_state_ou_transition_log_density(x_prev, x_next, inputs, *, params):
    dt = inputs.timestamp - inputs.timestamp_prev  # required: dt > 0
    phi = jnp.exp(-params.kappa * dt)
    scale = 1.0 - phi**2
    predicted = params.mean_0 + phi * (x_prev.x - params.mean_0)
    residual = (x_next.x - predicted).reshape(-1)
    dimension = residual.size
    A = scale * params.gamma_0

    return (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * _kron_logdet(A, params.B)
        -0.5 * _kron_quad(A, params.B, residual[None])[0]
    )
```

#### 4. Continue to call `build_smoother`

This is the **E-step** structure. Split independent keys for filtering,
materialization, and smoothing:

```python
def E_step(params, model_inputs, n_particles, n_smoother_particles, key):
    filter_key, materialize_key, smoother_key = jax.random.split(key, 3)
    filtered, rb_inputs = run_filter(
        filter_key, model_inputs, params, n_particles, MAX_GOALS
    )
    point_filter = materialize_rb_filter(
        materialize_key, filtered, rb_inputs.gamma, params.gamma_0, params.B
    )

    smoother = build_smoother(
        log_potential=partial(
            full_state_ou_transition_log_density, params=params
        ),
        backward_sampling_fn=exact_sampling_simulate,
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
        n_smoother_particles=n_smoother_particles,
    )
    smoothed = cuthbert.smoother(
        smoother, point_filter, rb_inputs, False, smoother_key
    )
    return smoothed, filtered, rb_inputs
```

Cuthbert will select a terminal state from the fixed terminal cloud and then,
at each earlier time, select and return one fixed previous full-state candidate.
It will not draw another Gaussian value during the backward pass. These paths
can then be treated as complete latent trajectories by the M-step.

#### 5. Do not combine incompatible smoother variants

There are two different valid constructions:

1. **Standard FFBSi:** materialize fixed full-state candidates first, then
   select previous candidates with `exact_sampling_simulate`. This is
   asymptotically valid but is no longer the recommended design for this
   96-dimensional model.
2. **Analytic RB backward simulation:** retain Gaussian components, integrate
   them when computing backward component weights, and draw from an RTS
   conditional after selecting a component. This needs a custom backward
   sampler and became the selected design on 2026-08-16.

Using analytic mixture weights but returning the selected component mean would
not be correct for either construction.

### M-step code sketch

The M-step receives the full-state paths selected in the E-step. With the
separate timestamp-zero prior, `path[0]` is the prior state and day `d` is
`path[d+1]`.

```python
def _matrix_normal_logpdf(x, mean, A, B):
    residual = (x - mean).reshape(-1)
    dimension = residual.size
    return (
        -0.5 * dimension * jnp.log(2.0 * jnp.pi)
        -0.5 * _kron_logdet(A, B)
        -0.5 * _kron_quad(A, B, residual[None])[0]
    )


def daily_observation_loglik(x, inputs, alpha, beta, max_goals):
    matches = inputs.matches

    def one_match(_, match):
        h, a, yh, ya, valid = match
        value = jax.lax.cond(
            valid,
            lambda _: loglik(
                jnp.array([yh, ya]), x[h], x[a],
                alpha=alpha, beta=beta, max_goals=max_goals, scale=1.0,
            ),
            lambda _: jnp.array(0.0),
            operand=None,
        )
        return None, value

    _, values = jax.lax.scan(
        one_match,
        None,
        (matches.home_id, matches.away_id,
         matches.home_score, matches.away_score, inputs.match_mask),
    )
    return values.sum()


def complete_log_joint(params, path, inputs, max_goals):
    initial = _matrix_normal_logpdf(
        path[0], params.mean_0, params.gamma_0, params.B
    )

    def day_term(x_prev, x_day, day_inputs):
        transition = full_state_ou_transition_log_density(
            RBPFState(x_prev), RBPFState(x_day), day_inputs, params=params
        )
        observation = daily_observation_loglik(
            x_day, day_inputs, params.alpha, params.beta, max_goals
        )
        return transition + observation

    daily = jax.vmap(day_term)(path[:-1], path[1:], inputs)
    return initial + jnp.sum(daily)


def M_step(smoothed, inputs, raw, opt_state, optimizer, n_steps):
    paths = jax.lax.stop_gradient(
        smoothed.particles.x.transpose(1, 0, 2, 3)
    )

    def loss(raw):
        params = decode_params(raw)  # positive-definite constrained parameters
        q = jax.vmap(
            lambda path: complete_log_joint(params, path, inputs, MAX_GOALS)
        )(paths).mean()
        return -(q + log_parameter_prior(params))

    for _ in range(n_steps):
        value, grads = jax.value_and_grad(loss)(raw)
        updates, opt_state = optimizer.update(grads, opt_state, raw)
        raw = optax.apply_updates(raw, updates)

    return decode_params(raw), raw, opt_state, value
```

`daily_observation_loglik` is the existing masked sum of match likelihoods for
one day. `log_parameter_prior` can return zero for MLE. Average once across
smoothing paths; do not separately rescale the initial, transition, and
observation terms.

### Remaining unresolved items at the time

#### A. First-state and first-observation semantics — resolved by convention

Retain the separate prior state and define it to be the latent state at
timestamp `0`. Every daily row is a subsequent transition and observation. For
`D` observed days, the filter and smoother therefore contain `D+1` states:

```text
filter state 0     prior state at timestamp 0
filter state d+1   state after the transition to and observations on day d
```

The first daily timestamp must be strictly greater than zero, and all later
daily deltas must also be positive. For the current World Cup subset beginning
at `2000-01-01`, the first observed timestamp is `23` and the minimum daily
delta is `1`, so this assumption holds.

The first daily row must continue to be passed through the ordinary filter
step so that its observations affect the particle weights and marginal
likelihood. The M-step then contains one prior density, `D` positive-time OU
transition densities, and `D` daily observation terms. Do not replace a future
zero-time transition with `max(1 - phi**2, 1e-6)`; reject data that violates the
positive-time convention instead.

#### B. Same-team, same-day validation — resolved

`data.py` now concatenates all `(date, home_team)` and `(date, away_team)`
appearances and rejects any duplicated `(date, team)` pair before team IDs and
daily JAX arrays are constructed. This catches home/home, home/away, away/away,
and self-match conflicts. Team ID zero remains safe because `match_mask`, not
the numeric ID, determines whether a slot is valid.

#### C. `smoothing.py` was incomplete

`M_step._loss_fn` currently has no body, so the module is not executable. The
M-step and `run_EM` should be completed only after the smoother returns fixed
full-state trajectories rather than RB means.

#### D. Complete-data likelihood

For `S` fixed full-state smoothing paths, use

\[
\frac1S\sum_{s=1}^S\left[
\log p_\theta(X_0^{(s)})
+\sum_d\log p_\theta(X_{d+1}^{(s)}\mid X_d^{(s)})
+\sum_d\sum_{j\in\mathcal M_d}
  \log p_\theta(y_{d,j}\mid X_{d+1}^{(s)})
\right].
\]

Use one common average over smoothing paths. Do not normalize the initial,
transition, and observation pieces by different dimensions. Apply
`stop_gradient` to the E-step trajectories: the M-step changes the parameters
inside the density but does not differentiate through FFBSi selections or
materialization draws.

#### E. Kronecker scale identifiability

Both factors cannot be freely scaled because

\[
(c\Gamma_0)\otimes(B/c)=\Gamma_0\otimes B.
\]

Independent covariance priors may encourage a scale but do not make the
likelihood identifiable. Fix one factor's scale, for example with
`trace(B) = 2`, `det(B) = 1`, or one fixed diagonal entry, while retaining a
positive-definite parameterization. Only tune covariance priors after the
E-step representation and M-step likelihood are correct.

#### F. Numerical covariance handling

- Continue symmetrizing OU predictions and Schur complements.
- Prefer a stable solve or Cholesky solve for a valid positive-definite
  `gamma_OO`; a pseudoinverse can hide a repeated-team or covariance bug.
- Distinguish small negative roundoff eigenvalues from real degeneracy.
- Do not add jitter to an intentionally singular filtered covariance and then
  treat the modified matrix as the model covariance.

#### G. Padding and stale call sites

- `_log_potential` still calculates a padded likelihood before discarding it
  with `jnp.where`; `jax.lax.cond` is safer if padded values can create invalid
  values or gradients.
- `model_trained.py` still uses the old covariance-trajectory returns and old
  helper/filter signatures.
- The unused `rbpf_ou_v2` `propagate_sample` import, unreachable legacy code,
  and incorrect `M`/`N` shape comments remain cleanup items.

#### H. Optimizer lifecycle

Do not initialize Adam inside every gradient step. Give the optimizer state one
explicit owner. If generalized EM intentionally carries momentum across outer
iterations, initialize it in `run_EM` and pass it into and out of each M-step.
If every M-step is treated as a separate optimization problem for a newly
sampled Monte Carlo objective, reset Adam once at the beginning of that M-step.

One gradient step is permissible only as a generalized-EM update when it
improves the current Monte Carlo `Q` objective. Prefer a small minimum number of
steps or an objective-improvement stopping rule, and rerun the filter/smoother
only after that M-step has finished.

### Tests still required at the time

1. Reject home/home, home/away, and away/away repeat appearances on one day;
   allow disjoint same-day matches and real team ID zero.
2. Compare a two-match covariance scan with manual sequential Schur updates.
3. Verify padding does not change covariance, random keys, or particle states.
4. Compare RB mean/covariance propagation with direct conditioning of a small
   multivariate Gaussian.
5. Check that materialized states empirically have mean `m_t^(i)` and covariance
   `gamma[t] (x) B`, while already sampled team entries remain unchanged.
6. Verify every FFBSi state at time `t` equals one of the fixed materialized
   candidates at that time. This confirms that no backward Gaussian draw occurs.
7. Compare FFBSi means, variances, and lag-one moments with an exact small
   linear-Gaussian Kalman/RTS reference, allowing finite-particle error.
8. Test the selected timeline convention, first-observation inclusion, model
   input alignment, and positive `dt` wherever a Gaussian transition density is
   evaluated.
9. Compare the vectorized M-step likelihood with a direct loop on a tiny fixed
   full-state trajectory.
10. Monitor covariance eigenvalues, Kronecker scale normalization, finite
    gradients, and improvement of the current `Q` objective before accepting an
    M-step.

### `MODEL.md` notation identified for later clarification

The stochastic model can remain unchanged, but the document should eventually
state consistently that `Gamma_0` is the `num_teams x num_teams` covariance and
`B` is the `2 x 2` attack/defence covariance; put `X_t` on the left of the OU
transition; include `kappa` in the estimated parameter vector when optimized;
and distinguish standard point-particle FFBSi from analytic RB backward
simulation.

</details>

## 2026-08-15, time unavailable — Earlier analytic RB smoother review

> **Historical scope:** This review predates the daily-timeline and
> initialization fixes. Its Gaussian backward-kernel derivation is consistent
> with the current 2026-08-16 recommendation, while its implementation-status
> findings describe an older archived implementation. The referenced
> `archive/rbpf_ou/src/smoothing.py` is empty in the 2026-08-16 workspace, so
> the prose below is the retained record of that earlier code review.

<details>
<summary>Show the earlier analytic RB smoother review</summary>

In this collapsed section, “current” refers to the older archived
implementation reviewed on 2026-08-15.

### Executive summary

The Rao--Blackwellized backward simulator in
[`archive/rbpf_ou/src/smoothing.py`](archive/rbpf_ou/src/smoothing.py) has the
correct core Gaussian construction for transitions with `dt > 0`:

1. It selects a terminal mixture component using the filtering weights.
2. It samples a full terminal latent state from that component's Gaussian.
3. It evaluates backward component weights using the full future state and the
   component prediction distribution.
4. It samples the previous full state from the corresponding RTS conditional.

The implementation is not yet correct end-to-end because the time-zero
particles and the covariance trajectory use incompatible representations. The
first observation is also omitted by the filter, and zero-time transitions are
handled with an approximate full-rank density. These issues should be fixed
before adjusting covariance priors, Adam settings, or the number of M-step
gradient updates.

### Required representation contract

At every filtering time `t`, particle `i` should represent one Gaussian
component:

\[
X_t \mid I_t=i,y_{0:t}
\sim
\mathcal N\left(m_t^{(i)},P_t\right),
\qquad
P_t=\Gamma_{t\mid t}\otimes B.
\]

The corresponding code contract should be:

- `filtered_states.particles.x[t, i] = m_t^{(i)}`: the component mean, not a
  complete latent-state draw.
- `gamma_t[t] = Gamma_{t|t}`: the residual filtered team covariance shared by
  all particles.
- `gamma_pred_t[t] = Gamma_{t|t-1}`: the predicted team covariance before the
  observed block at time `t` is sampled.
- `kalman_gain_t[t]`: the gain mapping the sampled home/away block into the
  full component mean.

The current propagation code follows this contract after initialization: it
samples only the home/away block and stores conditional means for the remaining
teams. The current `init_sample` does not: it samples the complete state while
`compute_gamma_trajectory` simultaneously assigns it a nonzero residual
covariance.

### What was already correct in the archived smoother

#### Terminal draw

The terminal step samples

\[
I_T\sim w_T,
\qquad
X_T^*\sim
\mathcal N\left(m_T^{(I_T)},\Gamma_{T\mid T}\otimes B\right).
\]

This correctly materializes a full latent state instead of treating the RBPF
mean as a full draw.

#### Backward component weights

For `dt > 0`, the component weights are correctly based on

\[
w_{t\mid t+1}^{(i)}
\propto
w_t^{(i)}
\mathcal N\left(
X_{t+1}^*;
m_{t+1\mid t}^{(i)},
P_{t+1\mid t}
\right),
\]

where

\[
m_{t+1\mid t}^{(i)}
=
\mu_0+\phi_{t+1}(m_t^{(i)}-\mu_0)
\]

and

\[
P_{t+1\mid t}
=
\Gamma_{t+1\mid t}\otimes B.
\]

#### RTS conditional draw

The archived implementation includes the required OU factor in the smoother
gain:

\[
J_t=P_t\phi_{t+1}P_{t+1\mid t}^{-1}.
\]

With the shared Kronecker factor `B`, this becomes

\[
J_t=
\left(
\phi_{t+1}\Gamma_{t\mid t}
\Gamma_{t+1\mid t}^{-1}
\right)\otimes I_2.
\]

The sampled conditional is therefore

\[
X_t^*\mid X_{t+1}^*,I_t
\sim
\mathcal N\left(
m_t^{(I_t)}+J_t(X_{t+1}^*-m_{t+1\mid t}^{(I_t)}),
P_t-J_tP_{t+1\mid t}J_t^\top
\right).
\]

The time alignment in the backward pass--`gamma_t[:-1]` paired with
`gamma_pred_t[1:]`--is also correct once initialization is aligned.

### Problems identified at the time

#### 1. Time-zero particles and covariance disagree

`init_sample` currently draws

\[
X_0^{(i)}\sim\mathcal N(\mu_0,\Gamma_0\otimes B),
\]

which is a complete latent-state draw. Conditional on that particle, its
remaining covariance should be zero.

At the same time, `compute_gamma_trajectory` starts from `gamma_0`, conditions
on the first match's observed teams, and records a nonzero
`gamma_t[0]`. The smoother then treats the full initial draw as a Gaussian
component mean with this additional covariance. At the next prediction, that
analytical uncertainty is propagated on top of uncertainty already represented
by the full draw.

This double-counts uncertainty and violates the Gaussian-mixture contract.

#### 2. The first observation is not used by the E-step

`run_filter` calls `init_prepare` with row zero and filters only rows `1:`.
The initial Cuthbert weights are zero, so

\[
p(y_0\mid X_0^{\mathcal O_0})
\]

does not enter the filter or its marginal likelihood. However, the M-step's
complete-data likelihood includes every observation, including `y_0`.

The E-step therefore does not condition on all the data used by the M-step,
which breaks the EM construction.

#### 3. Same-day transitions use an approximate density

When `dt == 0`, the model implies

\[
\phi=1,\qquad Q=0,\qquad X_{t+1}=X_t.
\]

The archived smoother correctly sets `X_t_star = X_next_star`, but it first
computes component weights after projecting the singular prediction covariance
onto a full-rank matrix with an eigenvalue floor. This assigns positive density
outside the true support and can select an incompatible component.

#### 4. The M-step loss was not the likelihood in `MODEL.md`

The archived loss divides the three complete-data terms by different factors:

- the initial density by `2M`;
- the observation sum by `2T`;
- the transition sum by `2M(T-1)`.

Adding these separately normalized terms produces a weighted pseudo-objective,
not the complete-data log-likelihood

\[
\log p(X_0)
+\sum_t\log p(X_t\mid X_{t-1})
+\sum_t\log p(y_t\mid X_t)
\]

specified in [`rbpf/MODEL.md`](rbpf/MODEL.md). Multiplying the entire objective
by one common constant would preserve EM; scaling its components differently
does not.

### Recommended implementation sequence at the time

#### Step 1: Make time-zero initialization Rao--Blackwellized

Use the prior as the time-zero prediction:

\[
\Gamma_{0\mid-1}=\Gamma_0.
\]

For each initial particle, sample only the first observed home/away block:

\[
X_0^{\mathcal O_0,(i)}
\sim
\mathcal N\left(
\mu_0^{\mathcal O_0},
\Gamma_0^{\mathcal O_0\mathcal O_0}\otimes B
\right).
\]

Then store the conditional full-state mean

\[
m_0^{(i)}
=
\mu_0+K_0
\left(X_0^{\mathcal O_0,(i)}-\mu_0^{\mathcal O_0}\right),
\]

with

\[
K_0=
\Gamma_0[:,\mathcal O_0]
\left(\Gamma_0[\mathcal O_0,\mathcal O_0]\right)^{-1}.
\]

The residual component covariance is

\[
\Gamma_{0\mid0}
=
\Gamma_0-K_0\Gamma_0[\mathcal O_0,:].
\]

This makes the initial particle representation identical to every later RBPF
particle representation.

#### Step 2: Weight the initial particles with the first score

After constructing the initial components, set

\[
\log w_0^{(i)}
=
\log p\left(y_0\mid X_0^{\mathcal O_0,(i)}\right).
\]

This can be done by explicitly replacing the weights and normalizing-constant
increment returned by `init_prepare`, or by representing the prior as a
separate state and passing the first match through an ordinary filter step.
The latter often gives the cleanest time indexing.

#### Step 3: Make the covariance initialization explicit

Do not depend on `timestamp[0] - timestamp_prev[0] == 0` to create the initial
update. Construct it explicitly:

1. `gamma_pred_t[0] = gamma_0`.
2. Compute `kalman_gain_t[0]` from the first observed block.
3. Compute `gamma_t[0] = Gamma_{0|0}`.
4. Scan only over model-input rows `1:`.

For each subsequent row,

\[
\Gamma_{t\mid t-1}
=
\phi_t^2\Gamma_{t-1\mid t-1}
+(1-\phi_t^2)\Gamma_0,
\]

followed by conditioning on the observed block to obtain
`Gamma_{t|t}`.

#### Step 4: Resolve zero-time observations

Preferred approach: group all matches sharing a date into one latent-state
time point and apply their likelihood updates sequentially without inserting
state transitions.

If matches remain as separate rows, the backward sampler must evaluate the
singular Gaussian on its actual support or trace a compatible ancestor for
`dt == 0`. It should not make the covariance full rank solely to obtain a
density.

#### Step 5: Validate the smoother before running EM

Use a small linear-Gaussian system where an exact Kalman/RTS smoother is
available. Compare the empirical mean, covariance, and lag-one covariance of
many RB backward trajectories with the exact smoothing moments.

Also verify the following repository-specific invariants:

- Materializing `X_t ~ N(m_t, gamma_t (x) B)` from the filtering mixture
  reproduces the intended filtered covariance.
- `X_t == X_{t-1}` for every `dt == 0` smoothed transition.
- The first score changes the initial weights and the log marginal likelihood.
- No observation row is silently dropped.
- Filter particles, covariance arrays, and model inputs all have identical
  temporal length and semantics.

For full-state OU draws, define

\[
r_t=X_t-\mu_0-\phi_t(X_{t-1}-\mu_0).
\]

For positive-time transitions, the diagnostic

\[
r_t^\top Q_t^{-1}r_t
\]

should average approximately `2M`, the state dimension. A much larger value is
evidence that the smoothing trajectories and the transition covariance still
do not describe the same random variable.

#### Step 6: Restore the model's complete-data objective

Once the E-step is correct, use the unweighted sum of initial, transition, and
observation log densities. If optimization remains poorly conditioned, address
that through:

- parameter-specific learning rates or preconditioning;
- Cholesky/softplus parameterizations;
- a fixed, correctly centered covariance prior;
- analytic or Rao--Blackwellized sufficient statistics;
- more smoothing trajectories;
- convergence checks within each M-step.

Do not use term-specific normalization as a substitute for the probabilistic
objective.

#### Step 7: Revisit regularization and M-step optimization

Only after the filtering and smoothing distributions pass the validation
checks should the covariance prior and Adam lifecycle be tuned. Otherwise,
stronger priors or more gradient steps merely optimize an inconsistent
transition objective more aggressively.

### Suggested acceptance criteria at the time

The first correction phase is complete when all of the following hold:

- Every filter particle is documented and tested as a conditional mean with
  covariance `gamma_t[t] (x) B`.
- Time-zero particles and `gamma_t[0]` use the same conditioning operation.
- `y_0` contributes to both the filtering distribution and log marginal
  likelihood.
- Positive-time RB backward moments agree with an exact small-model RTS
  reference.
- Same-day backward paths respect `X_t = X_{t-1}` without artificial jitter.
- The standardized OU residual diagnostic is near `2M`.
- The M-step evaluates the same complete-data likelihood stated in
  `MODEL.md`.

At that point, transition-score behavior can be interpreted as parameter
learning rather than as a state-representation artifact.

### Minor `MODEL.md` clarifications identified at the time

The core model can remain unchanged, but the document contains several
notation issues worth correcting separately:

- The transition equation should have `X_t` on the left-hand side, not
  `X_{t-1}`.
- The factors should consistently be described as `Gamma_0` of shape `M x M`
  and `B` of shape `2 x 2`.
- The bootstrap proposal mean should be the OU prediction
  `mu_0 + phi_t (X_{t-1} - mu_0)`, not simply `X_{t-1}`.
- The RTS gain should be
  `P_t phi_{t+1} P_{t+1|t}^{-1}`.
- If `kappa` is optimized, it should be included in the parameter vector
  `Theta`.

These clarifications do not change the intended stochastic model; they make
the written specification match the implementation target.

</details>
