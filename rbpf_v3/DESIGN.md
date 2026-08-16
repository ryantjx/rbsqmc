# RBPF v3 implementation and design

## Delivered system

`rbpf_v3` is self-contained. The bivariate-Poisson likelihood, parameter
transforms, forward filter, plotting helpers, persistence helpers, and bundled
data originate from `rbpf`, with the package prefix changed. The v3 data loader
also applies the same-day conflict safeguard described below. The two backward
smoothers deliberately do not share executable smoothing or MCEM code.

Public entry points:

- `rbpf_v3.src.smoothing`: Cuthbert-integrated smoother and MCEM.
- `rbpf_v3.src.smoothing_noncuthbert`: direct JAX reverse-scan smoother and MCEM.
- `python -u -m rbpf_v3.scripts.run_smoothing`: standalone Cuthbert run.
- `python -u -m rbpf_v3.scripts.run_smoothing_noncuthbert`: standalone direct run.
- `python -m rbpf_v3.scripts.compare_performance`: compare two compatible timing summaries.
- `rbpf_v3/run_smoothing_colab.sh`: Cuthbert GPU deployment and artifact retrieval.

Both smoother modules independently define `BackwardDiagnostics`,
`SmoothedStates`, `MCEMConfig`, Gaussian primitives, the E-step, the
complete-data objective, diagnostics, and `run_mcem`. Tests compare their
signatures and result fields to prevent interface drift.

## Data and tensor flow

Before tensor construction, the v3 loader removes every fixture for which
either team appears in more than one match on the same recorded date. This is
an intentional v3 divergence from the copied baseline. The covariance update
assumes a team is observed at most once per day: conditioning the same team
twice makes the second observed covariance singular. In the cached historical
data, reversed Morocco--Senegal fixtures on 1975-03-22 exposed this case and
caused all subsequent Gaussian particle draws to become non-finite. Removing
all rows participating in a same-day team conflict avoids choosing an
arbitrary fixture as authoritative and restores the filter's covariance
precondition. For the 1950--2025 World Cup-team slice, this removes the two
conflicting rows and one now-empty date, leaving 2,541 observed dates before
the one-date holdout.

For `D` observed days, `N` filter components, `S` smoothed paths, `M` teams,
and two traits:

```text
FootballResults
  -> run_filter
  -> means                     (D+1, N, M, 2)
     log weights               (D+1, N)
     filtered gamma            (D, M, M)
     predicted gamma           (D, M, M)
  -> backend smoother
  -> complete paths            (D+1, S, M, 2)
     component indices         (D+1, S)
     time diagnostics          (D+1,)
     optional probabilities    (D+1, S, N)
  -> stopped-gradient MCEM objective
  -> accepted or rolled-back parameter update
  -> evaluation and artifacts
```

Transition `t -> t+1` uses `gamma_filtered[t]`, `gamma_pred[t]`, and
`exp(-kappa * dt[t])`. Backward component selection integrates the filtering
component uncertainty with the predicted covariance. Complete-state process
covariance is used only in the complete-data transition density.

The Kronecker convention is `Gamma (teams) x B (traits)`. Production code
does not form a dense `2M x 2M` covariance or inverse. Cholesky triangular
solves evaluate batched quadratics, and symmetric eigendecompositions provide
PSD square roots for conditional draws.

## Cuthbert backend choices

The adapter passed to Cuthbert contains only component means and scalar time
indices. Covariance timelines remain closure data with shape `O(D*M*M)` and
are dynamically indexed by the backward callback; they are never broadcast
over particles.

Terminal component selection uses Cuthbert's systematic resampler. The
terminal means are replaced by batched Gaussian draws. The custom backward
callback explicitly discards Cuthbert's point-state density closure and
forward ancestor indices, evaluates all `S*N` Rao--Blackwellized logits, draws
new component indices, and replaces selected means with conditional Gaussian
draws. Cuthbert's `parallel=False` backward driver controls the reverse
timeline, after which Cuthbert-only state is discarded.

## Direct JAX backend choices

The direct module imports neither Cuthbert nor the Cuthbert smoother. Its
forward-filter import is local to `E_step`, so the smoothing implementation can
be imported and tested with Cuthbert blocked.

An eager wrapper validates shapes, finite means, positive elapsed times, and
positive-definite predicted covariances. It then calls one named module-level
jitted function. Terminal selection/drawing is batched, and the complete
backward dependency is represented by `jax.lax.scan(reverse=True, unroll=1)`.
The carry is only the `S` future states. This keeps shapes fixed and avoids a
day-unrolled Python/XLA graph.

## MCEM, diagnostics, and failure handling

Each backend independently scans days for transition and masked observation
terms, stops gradients through sampled paths, and uses module-level compiled
objective/value-and-gradient callables. The transform keeps `gamma_0`
positive definite, `kappa` positive, and `det(B)=1`. The unconstrained
stationary mean `mean_0` is optimized in the same Adam parameter tree, using
its contributions to the initial-state and OU transition densities. An M-step
is accepted only when its fixed-path objective is finite and non-worsening;
rejection restores the transformed parameters, `mean_0`, and Adam state.

The score likelihood has a location symmetry between the global attack and
defence offsets in `mean_0` and `alpha`. The requested implementation leaves
`mean_0` unconstrained, so raw column means are gauge-dependent; team-centred
contrasts are the identifiable part. Runs should monitor column-mean drift,
and longer optimization may benefit from an explicit centring constraint or
proper mean prior if a unique parameterization is required.

Host logging synchronizes filter, smoother, and reported objective stages
before printing elapsed times. Logs are flushed and mirrored to
`RBSQMC_PROGRESS_LOG`. Exceptions are logged as `ERR` and re-raised. Full
backward probabilities are disabled by default; ESS, entropy, maximum
probability, and selected-component diversity remain available without their
persistent `O(D*S*N)` storage.

Lag-one diagnostics retain the elementwise team/trait moment `(D, M, 2)`.
They intentionally avoid serializing a full cross-team outer moment
`(D, M, 2, M, 2)` for every epoch, which is not required by the MCEM objective
and would dominate artifact size.

Every standalone run writes initial/final parameters, training arrays,
training/evaluation summaries, performance metadata, baseline comparison,
progress log, and ten diagnostic plots. JSON writing and deployment validation
reject non-finite values and evaluation hard failures.

After MCEM, both backends make an explicit final filter pass at the optimized
parameters. The runners persist its component means, weights, genealogy,
normalizing constants, covariance trajectories, gains, and a compact summary
under `outputs/.../optimal_filter`. The copied plotting implementation also
writes optimal-filter strength, timeline, correlation, and log-normalizer
figures there.

## Performance comparison

Timing separates the filter and backward sampler inside each E-step, the
M-step, training, evaluation, and total runtime. The first epoch includes
compilation; the final E-step uses the same compiled shapes and is the warmed
comparison. Runs are comparable only when `D`, `N`, `S`, `M`, seed, and device
match.

Initial local CPU smoke measurement (`D=3`, `N=6`, `S=5`, `M=2`, seed 42):

| Backend | Warm final backward | Training | Total |
|---|---:|---:|---:|
| Cuthbert | 0.321 s | 4.674 s | 6.488 s |
| Direct JAX | 0.303 s | 4.085 s | 5.951 s |

On this deliberately tiny workload, direct JAX was approximately 1.06x faster
for the warmed backward stage and 1.09x faster end to end. These numbers are a
smoke comparison, not a large-workload conclusion; accelerator results and
the required size grid should be read from the emitted
`performance_summary.json` files. GPU acceptance uses the Cuthbert backend.

The accepted L4 Cuthbert deployment (`D=644`, `N=50`, `S=50`, `M=48`) recorded:

| Measurement | Seconds |
|---|---:|
| First filter / backward (includes compilation) | 4.660 / 9.783 |
| Warm final filter / backward | 1.592 / 2.618 |
| Warm M-step (20 updates, final epoch) | 10.071 |
| Training | 121.927 |
| Evaluation | 8.847 |
| Total Python runner | 142.686 |

The optimized final filter had shape `(645, 50, 48, 2)`, completed without a
hard evaluation failure, and produced final log normalizer `-3284.489`.

The subsequent A100 Cuthbert deployment used the conflict-cleaned 1950--2025
timeline with `D=2540`, `N=250`, `S=250`, and optimized `mean_0`. It completed
all five accepted epochs and the final filter without non-finite values:

| Measurement | Result |
|---|---:|
| First filter / backward | 9.519 s / 17.521 s |
| Warm final filter / backward | 6.483 s / 10.813 s |
| Warm final-epoch M-step | 47.890 s |
| Training / evaluation / total | 401.353 s / 10.407 s / 457.500 s |
| Initial / final log normalizer | -10686.762 / -10550.878 |
| Final transition Mahalanobis ratio | 0.99933 |
| Final `gamma_0` minimum eigenvalue / condition | 0.07535 / 16.72 |
| Final `mean_0` range / L2 norm | [-0.09135, 0.09804] / 0.59485 |

In ten common-random-number filter evaluations at `N=250`, the final
parameters improved mean log normalizer by 137.79 over initialization (95%
paired interval [134.29, 141.28]). Setting final `mean_0` back to zero removed
125.95 log units of fit (95% paired interval [121.68, 130.21]); replacing it
with only its two column means produced a similar 124.96-unit loss over five
seeds. Thus most of the training-fit gain is associated with learned
team-specific stationary means, rather than only their global offsets.

This run provides evidence of stable optimization and calibrated transition
residuals, not broad predictive superiority: its holdout contains only one
match, whose negative log predictive density (2.249) is worse than the
constant-Poisson baseline (2.099). Backward component probabilities also
remain sharply concentrated (median ESS 1.045 of 250), despite using an
average of 64.35 distinct selected components per time.

## Verification and reproducibility

```bash
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/unit -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/reference -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/integration -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/scripts -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests -m slow

python -u -m rbpf_v3.scripts.run_smoothing --synthetic
python -u -m rbpf_v3.scripts.run_smoothing_noncuthbert --synthetic
python -m rbpf_v3.scripts.compare_performance \
  rbpf_v3/outputs/smoothing_cuthbert/performance_summary.json \
  rbpf_v3/outputs/smoothing_noncuthbert/performance_summary.json

uv run bash rbpf_v3/run_smoothing_colab.sh
```

The principal limitation is that Cuthbert remains a dependency of the copied
forward filter even though the direct smoothing module itself is Cuthbert-free.
Monte Carlo outputs are distributionally equivalent rather than pathwise
identical because the backends own independent resampling/control flow.

## Completed deployment acceptance

On 2026-08-16, commit `91918a9` passed the primary command:

```bash
uv run bash rbpf_v3/run_smoothing_colab.sh
```

The run provisioned an L4, verified `CudaDevice(id=0)`, completed five accepted
MCEM epochs, ran the final optimized-parameter filter, generated and downloaded
all required artifacts, passed strict finite/evaluation validation, exited
successfully, and left no active Colab sessions. The validated local output is
`rbpf_v3/outputs/smoothing`.

On 2026-08-17, the conflict-cleaned `D=2540`, `N=S=250` A100 run also completed
five accepted epochs, the optimized-parameter filter, and the runner's internal
finite/evaluation checks. Its measurements and remaining statistical caveats
are recorded in the performance section above. The local download omits
`training_arrays.npz` and `optimal_filter/filter_states.npz`; consequently it
does not satisfy the stricter `validate_outputs.py` artifact-completeness check
until those large state artifacts are retrieved or the deployment contract is
explicitly revised.
