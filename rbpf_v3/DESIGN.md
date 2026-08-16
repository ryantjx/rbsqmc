# RBPF v3 implementation and design

## Delivered system

`rbpf_v3` is self-contained. The data loader, bivariate-Poisson likelihood,
parameter transforms, forward filter, plotting helpers, persistence helpers,
and bundled data are wholesale copies of `rbpf`, with only the package prefix
changed. The two backward smoothers deliberately do not share executable
smoothing or MCEM code.

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
positive definite, `kappa` positive, `det(B)=1`, and `mean_0` fixed. An M-step
is accepted only when its fixed-path objective is finite and non-worsening;
rejection restores both parameters and Adam state.

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
