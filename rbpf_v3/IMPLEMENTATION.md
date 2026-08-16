# RBPF v3: Dual-Smoother Implementation Specification

## Implementation groups

| Group | Deliverable | Core rule |
|---|---|---|
| A | Wholesale `rbpf` baseline copy | Copy the complete existing data/model/filter implementation into `rbpf_v3` before changing smoothing |
| B | Independent backend contracts | Matching APIs without shared executable smoothing code |
| C | `smoothing.py` | Standalone Cuthbert-integrated RB backward sampler and MCEM |
| D | `smoothing_noncuthbert.py` | Standalone direct JAX reverse-scan sampler and MCEM |
| E | JAX performance | Fixed shapes, staged scans, batched solves, stable JIT cache |
| F | Progress and diagnostics | Timestamped host logs around synchronized stages |
| G | Tests and acceptance | Dense references, backend equivalence, scaling gates |
| H | Colab GPU deployment | Reproducible launch, streamed logs, artifact retrieval |

### Primary success metric

The implementation is successful only when this command runs from the local
`uv` environment and exits with status zero:

```bash
uv run bash rbpf_v3/run_smoothing_colab.sh
```

Equivalent execution after `uv sync` is:

```bash
source .venv/bin/activate
./rbpf_v3/run_smoothing_colab.sh
```

The command must complete a real Colab GPU training/evaluation run, stream
progress locally, download and validate all required artifacts, and stop the
Colab session. Dry runs, mocked deployment tests, imports, CPU smoke tests, or
successful compilation do not satisfy the primary success metric.

## A. Wholesale copy of the existing `rbpf` baseline

Complete Group A before implementing or modifying either smoother. Copy the
existing data, model, forward-filter, utility, parameter, and plotting
implementation from `./rbpf` into `./rbpf_v3`. The v3 implementation must be
self-contained: runtime code under `rbpf_v3` must not import Group A symbols
from `rbpf`.

### A.1 Required copy manifest

Copy these files wholesale:

```text
rbpf/src/bivariate_poisson.py  -> rbpf_v3/src/bivariate_poisson.py
rbpf/src/data.py               -> rbpf_v3/src/data.py
rbpf/src/graphic.py            -> rbpf_v3/src/graphic.py
rbpf/src/helpers.py            -> rbpf_v3/src/helpers.py
rbpf/src/model.py              -> rbpf_v3/src/model.py
rbpf/src/model_trained.py      -> rbpf_v3/src/model_trained.py
rbpf/src/utils.py              -> rbpf_v3/src/utils.py
rbpf/data/                     -> rbpf_v3/data/
```

Copy the full contents of `rbpf/data`, including the local results cache and
all team metadata JSON files. Do not copy generated outputs, `__pycache__`,
documentation, review files, archives, or either existing smoothing module;
they are not part of the Group A runtime baseline. Groups B-D define the v3
smoothing modules separately.

After copying, mechanically change only internal package imports from
`rbpf.src...` to `rbpf_v3.src...`. This namespace rewrite is the only permitted
Group A source change. Do not refactor, optimize, rename, clean up, or otherwise
reinterpret the copied implementation.

### A.2 Preserved behavior and contracts

The copied modules must preserve the existing:

- date grouping, padding, masks, team IDs, and bundled data behavior;
- bivariate-Poisson likelihood;
- covariance recursion and Rao-Blackwellized propagation;
- Cuthbert particle filter and systematic resampling;
- parameter initialization, encoding, and decoding;
- plotting and result-persistence helpers;
- `FilterStates`, `EMParams`, `FootballResults`, and `RBPFFootballResults`
  contracts;
- public call signatures, defaults, dtypes, array shapes, and random-key
  behavior.

The copied `rbpf_v3/src/model.py::run_filter` is the only forward filter used
by Groups B-D. Do not create or retain a second v3 forward-filter
implementation.

All later v3 code imports the local copies:

```python
from rbpf_v3.src.bivariate_poisson import loglik
from rbpf_v3.src.data import get_results
from rbpf_v3.src.helpers import (
    default_init_params,
    decode_EM_params,
    encode_EM_params,
    log_inverse_wishart_kernel,
)
from rbpf_v3.src.model import run_filter
from rbpf_v3.src.utils import EMParams, RBPFFootballResults
```

### A.3 Copy verification gate

Before beginning Group B:

1. Verify every source and data path in A.1 exists under `rbpf_v3`.
2. Verify `rg "from rbpf\\.src|import rbpf\\.src" rbpf_v3/src` returns no
   matches.
3. Compare every copied source file with its `rbpf` origin after normalizing
   only the package prefix; any other diff fails the gate.
4. Run the same fixed-key, fixed-parameter filter call through `rbpf` and
   `rbpf_v3`, then assert identical tree structure, shapes, dtypes, and
   numerically equal array leaves.
5. Confirm both local-cache data loading and plotting-helper imports work from
   `rbpf_v3` without importing `rbpf`.

No smoothing work may compensate for, fork, or silently change Group A
behavior. If the baseline must change, change `rbpf` first and repeat the
wholesale copy and verification gate.

### A.4 Filter representation

```text
filtered_states.particles.x    (D+1, N, M, 2)
filtered_states.log_weights    (D+1, N)
rb_inputs.gamma                (D, M, M)
rb_inputs.gamma_pred           (D, M, M)
```

Each forward particle is a Gaussian-component mean:

\[
X_t\mid I_t=i,y_{1:t}
\sim\mathcal N(m_t^{(i)},\Gamma_{t\mid t}\otimes B).
\]

It is not a complete state draw.

### A.5 Timeline

```text
state                    0          1        ...       D
transition/day                      0        ...       D-1
filtered covariance      gamma_0    gamma[0] ...       gamma[D-1]
predicted covariance                pred[0]  ...       pred[D-1]
```

```python
gamma_filtered = jnp.concatenate([params.gamma_0[None], rb_inputs.gamma])
dt = rb_inputs.timestamp - rb_inputs.timestamp_prev
phi = jnp.exp(-params.kappa * dt)
```

Transition `t -> t+1` uses exactly:

```python
gamma_t = gamma_filtered[t]
gamma_pred_next = rb_inputs.gamma_pred[t]
phi_t = phi[t]
```

No additional shift is allowed.

## B. Independent backend contracts

There is no `smoothing_common.py`. The two smoother modules are deliberately
standalone so their compilation, execution, and memory performance can be
measured independently. They must not import smoothing code, result types,
Gaussian primitives, diagnostics, objectives, or MCEM functions from one
another.

Each module independently implements Gaussian calculations, backward
selection, conditional draws, result types, complete-data densities,
diagnostics, and its MCEM driver. Interface-parity tests, rather than shared
declarations, keep their public signatures and result schemas aligned.

### B.1 Result contract

```python
class BackwardDiagnostics(NamedTuple):
    ess_by_time: jax.Array              # (D+1,)
    entropy_by_time: jax.Array          # (D+1,)
    max_probability_by_time: jax.Array  # (D+1,)
    unique_indices_by_time: jax.Array   # (D+1,)
    probabilities: jax.Array | None     # optional (D+1, S, N)


class SmoothedStates(NamedTuple):
    x: jax.Array                        # (D+1, S, M, 2)
    component_indices: jax.Array        # (D+1, S)
    diagnostics: BackwardDiagnostics
```

Full probabilities are disabled by default to avoid `O(D*S*N)` persistent
storage.

### B.2 Backward kernel

Terminal selection and draw:

\[
I_D\sim\operatorname{Categorical}(w_D),\qquad
X_D^*\sim\mathcal N(m_D^{(I_D)},\Gamma_{D\mid D}\otimes B).
\]

For a future endpoint `x_next`, component prediction is:

\[
a_{t+1}^{(i)}=\mu_0+\phi_t(m_t^{(i)}-\mu_0).
\]

Backward selection integrates component uncertainty:

\[
\ell_i=\log w_t^{(i)}+
\log\mathcal N(x_{t+1}^*;a_{t+1}^{(i)},
\Gamma_{t+1\mid t}\otimes B).
\]

After selecting `I_t`:

\[
J_{\Gamma,t}=\phi_t\Gamma_{t\mid t}
\Gamma_{t+1\mid t}^{-1},
\]

\[
\Gamma_{C,t}=\operatorname{sym}\left(
\Gamma_{t\mid t}-J_{\Gamma,t}\Gamma_{t+1\mid t}J_{\Gamma,t}^T
\right),
\]

\[
h_t=m_t^{(I_t)}+J_{\Gamma,t}(x_{t+1}^*-a_{t+1}^{(I_t)}),
\]

\[
X_t^*\sim\mathcal N(h_t,\Gamma_{C,t}\otimes B).
\]

The draw `X_t*`, not `h_t`, is the next backward endpoint.

Never:

- use forward genealogy for `I_t`;
- use complete-state `Q_t` for component selection;
- return a selected component mean as a complete state;
- materialize an independent full state for every forward time/component.

### B.3 Gaussian primitives

Implement:

```python
kron_logdet(gamma, B)
kron_quad_batched(gamma_chol, B_chol, residuals)
psd_sqrt(matrix, tolerance)
sample_kron_psd_batched(key, means, gamma_sqrt, B_sqrt)
backward_shared_terms(gamma_t, gamma_pred_next, phi_t)
batched_backward_step(...)
```

Rules:

- never form a production `2M x 2M` Kronecker matrix;
- never compute a dense inverse;
- Cholesky-factor positive-definite `gamma_pred_next` once per time;
- eigendecompose PSD `gamma_t`/`gamma_cond` once per time;
- factor constant `B` once per smoother call;
- flatten batch/trait axes into multiple right-hand sides;
- evaluate all `S*N` logits in one operation;
- use `jax.scipy.special.logsumexp` or `jax.nn.softmax` in log space.

For residuals `(S, N, M, 2)`, return quadratics `(S, N)` without a Python
loop over `S` or `N`.

### B.4 Complete-data objective

Use complete-state transition covariance only here:

```python
phi_t = jnp.exp(-params.kappa * dt_t)
residual = x_next - params.mean_0 - phi_t * (x_prev - params.mean_0)
Q_gamma_t = (1.0 - phi_t**2) * params.gamma_0
```

The objective is the path mean of:

```text
initial density
+ summed OU transition densities
+ summed masked observation likelihoods
+ inverse-Wishart gamma_0 prior
```

Transition and observation accumulation must use `jax.lax.scan`; paths use
`jax.vmap` or a memory-bounded `jax.lax.map`. No Python loop may scale with
`D`, `S`, or `N`.

Stop gradients through all E-step paths:

```python
paths = jax.lax.stop_gradient(smoothed.x)
```

### B.5 Independent MCEM implementations

```python
run_mcem(key, model_inputs, initial_params, config, *, e_step_fn)
```

Both backend modules independently implement this function with the same
arguments and result schema.

Retain:

- Cholesky/softplus `gamma_0` transform;
- positive `kappa` transform;
- diagonal determinant-one `B`;
- unconstrained optimization of `mean_0` alongside the transformed parameters;
- inverse-Wishart kernel;
- finite, non-worsening M-step acceptance;
- restoration of parameters and optimizer state after rejection.

## C. Standalone Cuthbert backend: `smoothing.py`

This is the default backend used by `scripts/run_smoothing.py`.

### C.1 Compact adapter

Cuthbert indexes every particle-tree leaf. Do not broadcast covariance arrays
to `(D+1, N, M, M)`.

```python
class RBSmootherParticle(NamedTuple):
    x: jax.Array
    time_index: jax.Array
```

Store only component means and a broadcast scalar time index `(D+1, N)`.
Close the callback over shared arrays:

```text
gamma_filtered    (D+1, M, M)
gamma_pred        (D, M, M)
phi               (D,)
```

`make_rb_smoother_filter_states(...)` must preserve weights and normalizing
constants, stop gradients through means, and use `0..N-1` as smoother-facing
component labels instead of forward ancestors.

### C.2 Terminal resampling

Implement Cuthbert's exact signature:

```python
rb_terminal_resampling(
    key, logits, positions, n, *, gamma_terminal, B_sqrt
)
```

Use Cuthbert systematic resampling, retain compact metadata, draw all `S`
terminal states in one batch, and replace selected `.x` means.

### C.3 Backward callback

```python
rb_backward_sampling_fn(
    key,
    x0_all,
    x1_all,
    log_weight_x0_all,
    log_density,
    x1_ancestor_indices,
    *,
    mean_0,
    B,
    gamma_filtered,
    gamma_pred,
    phi,
)
```

It must begin with:

```python
del log_density, x1_ancestor_indices
```

Use `t = x0_all.time_index[0]`, dynamically index shared covariance arrays,
call the module-local batched backward step, index the compact particle tree using
the chosen components, replace `.x` with conditional draws, and return
`(particles, component_indices)`.

### C.4 Builder and E-step

```python
build_smoother(params, rb_inputs, n_smoother_particles, max_goals)
```

must return `cuthbert.inference.Smoother` using:

```python
cuthbert.smc.backward_sampler.build_smoother(
    log_potential=complete_state_joint_log_potential,
    backward_sampling_fn=partial(rb_backward_sampling_fn, ...),
    resampling_fn=partial(rb_terminal_resampling, ...),
    n_smoother_particles=n_smoother_particles,
)
```

The complete-state potential satisfies Cuthbert's object contract; the custom
callback must not evaluate the point-state `log_density` closure.

The E-step runs the existing filter, adapts its result, runs
`cuthbert.smoother(..., parallel=False)`, converts to its matching `SmoothedStates`,
and immediately drops Cuthbert-only state.

## D. Standalone direct backend: `smoothing_noncuthbert.py`

This module must not import Cuthbert or `smoothing.py` at module import time.
Because the reused
forward filter depends on Cuthbert, import `run_filter` lazily inside `E_step`.
The direct smoother itself must operate on an already-produced filter result
without Cuthbert.

### D.1 Reverse scan

Draw terminal states in one batch. Run the entire backward pass as one reverse
scan:

```python
_, outputs = jax.lax.scan(
    backward_body,
    terminal_states,
    scan_inputs,
    reverse=True,
    unroll=1,
)
paths = jnp.concatenate([outputs.x_previous, terminal_states[None]], axis=0)
```

Scan inputs:

```text
means[:-1]             (D, N, M, 2)
log_weights[:-1]       (D, N)
gamma_filtered[:-1]    (D, M, M)
gamma_pred             (D, M, M)
phi                    (D,)
time_keys              (D, 2)
```

The carry and output shapes/dtypes are fixed. Timeline tests must prove
`outputs[t]` is state `t`.

Expose an eager validator and a compiled core:

```python
def rb_backward_simulation(...):
    validate_inputs(...)
    return _rb_backward_simulation_jit(...)
```

## E. JAX performance plan

The target environment currently uses JAX `0.11.0`. Use stable public JAX
APIs only.

### E.1 Compilation boundaries

Create named, module-level functions and compile them once. Do not create
jitted lambdas or partials inside epoch/gradient loops.

| Function | Transformation | Static values |
|---|---|---|
| Direct full smoother | `jax.jit` around reverse `lax.scan` | `S`, diagnostics mode |
| Backward step | staged by scan/Cuthbert; optional standalone `jax.jit` for tests | none unless output shape changes |
| Complete-data terms | `jax.jit` + `lax.scan` + path batching | shape/config only |
| M-step | `jax.jit(jax.value_and_grad(loss))` | none beyond fixed config |
| Candidate evaluation | `jax.jit(loss)` | none beyond fixed config |

Static arguments trigger separate compilations. Mark only booleans/counts that
change shapes or control structure. Keep arrays, parameters, dates, and keys
dynamic.

Warm each compiled function once before timing. Reuse the same callable object
and stable shapes/dtypes so JAX's in-memory compilation cache is effective.

### E.2 Structured control flow

- Use `lax.scan` for sequential time dependencies; Python loops inside `jit`
  are unrolled and create large XLA programs.
- Use `vmap` for `S*N` work when memory permits.
- Use `lax.map(..., batch_size=path_batch_size)` when full `vmap` exceeds GPU
  memory. Make batch size a small static configuration set.
- Keep padded match shapes fixed across days.
- Use `lax.cond`/`jnp.where` for traced conditions; never call Python `bool`
  on a tracer.

Start with `scan(unroll=1)`. Benchmark small unroll factors only after the
rolled version passes; never fully unroll `D`.

### E.3 Randomness

Split named stage keys on the host. Inside scans use pre-split time keys or
`jax.random.fold_in(base_key, time_index)`. Split path keys in one batched
operation. Both backends must be deterministic for a fixed seed/backend.

### E.4 Memory

Persistent storage should be limited to:

```text
forward means         O((D+1)*N*M*2)
paths                 O((D+1)*S*M*2)
component indices     O((D+1)*S)
shared covariance     O(D*M*M)
```

Do not store full probabilities unless requested. Do not allocate Cuthbert
covariance leaves `(D+1, N, M, M)`.

Use `jax.checkpoint`/`jax.remat` only around differentiated scan bodies when a
memory profile demonstrates that saved reverse-mode intermediates cause OOM;
it trades memory for recomputation and should not wrap the stopped-gradient
smoother.

Use `donate_argnames` only for buffers that are dead after a call. Never donate
fixed paths reused across Adam steps, current parameters needed for rollback,
or diagnostic arrays.

### E.5 Measurement and profiling

JAX dispatch is asynchronous. Accurate timing requires synchronization:

```python
result = compiled_fn(...)
jax.block_until_ready(result)
```

Use synchronization at stage boundaries, not inside scans.

For investigation, support:

- `jax.profiler.trace(...)` for accelerator traces;
- `jax.profiler.save_device_memory_profile(...)` after synchronization;
- `jax_explain_cache_misses` during development;
- optional persistent compilation cache configuration in the runner.

Do not enable profiling in normal training.

## F. Progress and diagnostics

### F.1 Host logger

All local and Colab scripts use one flush-safe logger:

```python
from datetime import datetime

def progress(message: str, *, stream="OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    print(f"[{outer}] {stream}: [{inner}] {message}", flush=True)
```

Required output sequence:

```text
[2026-08-16 20:37:38] OUT: [20:37:38] loading data and initial parameters...
[2026-08-16 20:37:38] OUT: [20:37:38] backend=cuthbert device=GpuDevice(...)
[2026-08-16 20:37:38] OUT: [20:37:38] EM loop starting
[2026-08-16 20:37:39] OUT: [20:37:39] epoch 0/5: running E-step (filter + backward simulation)...
[2026-08-16 20:38:12] OUT: [20:38:12] epoch 0/5: E-step complete in 33.1s; logZ=-123.45
[2026-08-16 20:38:12] OUT: [20:38:12] epoch 0/5: running M-step (20 updates)...
[2026-08-16 20:38:18] OUT: [20:38:18] epoch 0/5: M-step complete in 5.8s; accepted=True
[2026-08-16 20:38:18] OUT: [20:38:18] epoch 1/5: running E-step (filter + backward simulation)...
```

Also log:

- data dimensions `D, N, S, M`;
- first-call compilation warnings;
- every epoch boundary;
- M-step progress every configurable `log_every_gradient_steps`;
- accepted/rejected parameters and objective change;
- final filter, evaluation, and artifact paths;
- caught exceptions with `ERR` before re-raising.

### F.2 Synchronization contract

Host progress surrounds asynchronous stages:

```python
progress("epoch 0/5: running E-step (filter + backward simulation)...")
smoothed, filtered, inputs = e_step_fn(...)
jax.block_until_ready(smoothed.x)
progress("epoch 0/5: E-step complete ...")
```

Likewise synchronize the loss used for reported M-step timing.

Do not use `jax.debug.print` for routine epoch progress or inside every scan
iteration. It adds callbacks and can perturb/reorder compiled execution.
Allow opt-in `jax.debug.print(..., ordered=True)` only for small debugging runs;
call `jax.effects_barrier()` before exit when compiled prints are enabled.

### F.3 Required diagnostics

Both backends report the same schema:

- filter log-normalizing constant;
- backward ESS, entropy, max probability, and unique indices by time;
- initial/transition Mahalanobis ratios and quantiles;
- initial, transition, observation, prior, and total objective terms;
- transition normalization and quadratic penalty;
- covariance eigenvalues, trace, log determinant, condition number, and rank;
- `B` determinant/eigenvalues, `kappa`, and OU half-life;
- finite-value, PSD/PD, shape, and timeline checks;
- backend, seed, device, compile time, and execution time.

Diagnostics use scans/batching and do not force full probability storage.

## G. Scripts, tests, and acceptance

### G.1 Scripts

```text
scripts/run_smoothing.py              Cuthbert backend
scripts/run_smoothing_noncuthbert.py  direct backend
```

Both expose identical flags, including:

```text
--initial-params
--n-particles
--n-smoother-paths
--n-epochs
--n-gradient-steps
--path-batch-size
--log-every-gradient-steps
--return-backward-probabilities
--profile
--output-dir
```

Save `em_initial_params.json` before the first filter. Record backend and timing
metadata in JSON. The Colab launcher calls the Cuthbert script by default and
streams stdout without buffering.

The scripts are standalone and default to separate output directories:

```text
rbpf_v3/outputs/smoothing_cuthbert
rbpf_v3/outputs/smoothing_noncuthbert
```

Each writes `performance_summary.json` with synchronized filter, backward,
objective, M-step, evaluation, and total timings. Performance comparisons use
the same data, parameters, seed, shapes, device, and warm-up policy and report
both backward-only and end-to-end timings.

### G.2 Mathematical tests

Against dense `P = kron(gamma_t, B)` and
`R = kron(gamma_pred_next, B)`, test:

- backward logits;
- gain and conditional covariance;
- conditional means;
- terminal and conditional draw moments;
- failure when `Q_t` is substituted for `R`.

### G.3 Backend tests

Cuthbert:

- builder returns `cuthbert.inference.Smoother`;
- adapter contains only `x` and time indices;
- point density and forward ancestors do not affect fixed-key output;
- no covariance metadata is broadcast over particles.

Non-Cuthbert:

- module imports with Cuthbert blocked;
- reverse pass uses `lax.scan`;
- eager validation and compiled core agree;
- smoother itself calls no Cuthbert function.

Using the same saved filter result, compare backend smoothed means, variances,
lag-one moments, selection frequencies, Mahalanobis values, ESS, and entropy
within Monte Carlo tolerance.

### G.4 Performance gates

For fixed `M`, benchmark after warm-up:

```text
D in {32, 128, 512}
N in {8, 32}
S in {8, 32}
```

Runtime should scale approximately linearly in `D` and `S*N`. Tests must also
assert:

- no Python loop scales with `D`, `S`, or `N` in compiled paths;
- no `(D+1, N, M, M)` covariance metadata exists;
- default mode omits full probabilities;
- no recompilation occurs across epochs with unchanged shapes/dtypes;
- `D=512` compiles without a day-unrolled graph;
- a GPU smoke run completes without host/device OOM.

### G.5 Prerequisite test gate

```bash
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/unit -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/reference -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/integration -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests/scripts -m "not slow"
RBSQMC_PLATFORM=cpu python -m pytest -q rbpf_v3/tests -m slow
```

Run identical smoke configurations for both backends. These tests are required
before the deployment gate, but do not by themselves complete the
implementation. Correctness prerequisites are:

- reuse of the existing filter;
- dense-reference and backend-equivalence tests passing;
- backward selection using `R`, not `Q`;
- terminal/previous states being Gaussian draws;
- no day-unrolled Python computation;
- stable JIT cache across epochs;
- finite order-one transition Mahalanobis diagnostics;
- timestamped progress visible throughout local and Colab runs;
- initial/final parameters, metrics, timings, and artifacts saved.

## H. Development of `run_smoothing_colab.sh`

Develop `rbpf_v3/run_smoothing_colab.sh` as the supported orchestration layer
for running the Cuthbert smoother on a Colab GPU. The shell script owns local
validation, session lifecycle, downloads, and cleanup. Remote Python owns VM
bootstrap, CUDA verification, and execution of `scripts/run_smoothing.py`.

### H.1 Components and responsibilities

```text
run_smoothing_colab.sh
    local entry point, Colab session, downloads, validation, cleanup

smoothing_gpu_config.json
    single source of runtime and deployment configuration

scripts/run_smoothing_gpu.py
    remote bootstrap, environment setup, streaming child process

scripts/run_smoothing.py
    data loading, initial parameters, Cuthbert MCEM, evaluation, artifacts
```

The shell script must remain a thin orchestrator. Model logic and JAX imports
do not belong in Bash.

### H.2 Shell functions

Implement the launcher with small, testable functions:

```bash
read_config KEY
require_command NAME
download_required FILE
download_optional FILE
validate_local_outputs
cleanup
main
```

Functionality:

- `read_config` reads JSON through Python and fails on missing required keys.
- `require_command` checks `colab` and `python3` before creating a session.
- `download_required` fails the run when an artifact is unavailable or empty.
- `download_optional` records a warning without hiding required failures.
- `validate_local_outputs` parses JSON, rejects non-finite values, and checks
  the evaluation hard-failure field.
- `cleanup` stops only the configured session and is safe when launch failed.
- `main` performs validation, launch, download, validation, and status output.

Use `set -euo pipefail`. Resolve all paths from `BASH_SOURCE` so the launcher
works from any current directory. Do not use `eval`, source JSON as shell code,
or run destructive Git commands.

### H.3 Configuration functionality

`smoothing_gpu_config.json` defines:

```text
start_date, end_date
n_particles, n_smoother_paths
n_epochs, n_gradient_steps, learning_rate
max_goals, holdout_days, seed
initial_params
output_dir
gpu_type, colab_timeout
repo_url
```

Validate numeric ranges before launch. `output_dir` must resolve to
`rbpf_v3/outputs/smoothing` for the standard deployment.

Allow deployment overrides without editing JSON:

```bash
GPU_TYPE=L4
COLAB_TIMEOUT=7200
SESSION=rbsqmc-rbpf-v3-smoothing
```

Training hyperparameters remain config-driven so the local launcher and remote
runner cannot silently disagree.

### H.4 Session lifecycle

`main` executes:

```text
validate commands and files
read and validate config
install EXIT trap
launch colab run --gpu ... --keep --timeout ... --session ...
stream remote output until the child exits
confirm session state
download required and optional artifacts
validate downloaded outputs
print final status
stop session through cleanup trap
```

Keep the session alive through downloads. Preserve a nonzero exit code from
Colab, training, download, or validation even if cleanup succeeds.

### H.5 Remote bootstrap functionality

`scripts/run_smoothing_gpu.py` must avoid JAX and project imports until the
accelerator environment is configured:

```text
clone repository or git pull --ff-only
load committed smoothing_gpu_config.json
install missing dependencies
set RBSQMC_PLATFORM=cuda
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_ALLOCATOR=platform
set MPLCONFIGDIR to a writable directory
verify jax.default_backend() == "gpu"
print jax.devices()
run python -u -m rbpf_v3.scripts.run_smoothing
```

Use `sys.executable` and `/content/rbsqmc` as the working directory. Abort
before loading data when CUDA is unavailable.

### H.6 Initial-parameter functionality

When `initial_params` is empty, use the standard regional initialization. When
set, forward it as:

```text
--initial-params PATH
```

The path must exist remotely. Validate team count, array shapes, positive
definiteness, and positive `kappa`. Save the exact starting value before the
filter runs:

```text
rbpf_v3/outputs/smoothing/em_initial_params.json
```

This file must remain available after a failed or interrupted optimization.

### H.7 Progress functionality

The remote bootstrap runs training with a line-buffered merged stream:

```python
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
```

Forward each child line immediately and set `RBSQMC_PROGRESS_LOG` to:

```text
/content/rbsqmc/rbpf_v3/outputs/smoothing/progress.log
```

The user must see live lines such as:

```text
[2026-08-16 20:37:38] OUT: [20:37:38] EM loop starting
[2026-08-16 20:37:39] OUT: [20:37:39] epoch 0/5: running E-step (filter + backward simulation)...
```

Do not buffer output until completion. If training fails, print the exit code,
attempt to retrieve `progress.log`, preserve the failure status, and clean up.

### H.8 Artifact functionality

Required training artifacts:

```text
progress.log
em_initial_params.json
em_final_params.json
training_arrays.npz
training_summary.json
performance_summary.json
evaluation_summary.json
baseline_comparison.json
optimal_filter/filter_states.npz
optimal_filter/optimal_filter_summary.json
```

Required evaluation plots:

```text
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
optimal_filter/top_strengths.png
optimal_filter/timeseries_states.png
optimal_filter/correlation_matrix.png
optimal_filter/log_normalizing_constant.png
```

Download into the local `rbpf_v3/outputs/smoothing` directory. Required files
must exist and be nonempty. Parse all JSON, reject non-finite numeric values,
and fail when `evaluation_summary.json` reports a hard failure.

### H.9 Failure and recovery functionality

Handle these cases explicitly:

- missing `colab`, Python, config, or bootstrap script;
- invalid config or initial parameters;
- unavailable requested GPU;
- clone, pull, or dependency-install failure;
- nonzero training exit;
- timeout or interrupted session;
- missing/empty artifacts;
- invalid JSON or failed evaluation.

On any failure, emit a timestamped `ERR` line, retrieve `progress.log` when the
session exists, stop the session, and return nonzero. Never convert a failed
training run into success because cleanup or optional plotting succeeded.

### H.10 Development sequence

Implement in this order:

1. config reader, validation, and `--dry-run` command rendering;
2. remote bootstrap without training;
3. CUDA assertion and environment ordering;
4. unbuffered training subprocess and live progress forwarding;
5. initial-parameter forwarding and early persistence;
6. required/optional artifact download helpers;
7. local JSON and evaluation validation;
8. failure-path progress retrieval and cleanup;
9. mocked deployment tests;
10. one real Colab GPU acceptance run.

### H.11 Deployment tests

Test:

- `bash -n rbpf_v3/run_smoothing_colab.sh`;
- invocation from outside the repository directory;
- missing commands/files and invalid config;
- environment overrides and dry-run rendering;
- `--initial-params` forwarding;
- environment setup before JAX import;
- GPU assertion failure;
- line-by-line stdout/stderr forwarding;
- child exit-code propagation;
- mocked launch, download, validation, and cleanup order;
- required versus optional artifact behavior;
- progress-log recovery after simulated failure.

## I. Implementation documentation

Create `rbpf_v3/DESIGN.md` after implementation. It records the delivered
files and entry points, tensor/data flow, the separate design choices in
`smoothing.py` and `smoothing_noncuthbert.py`, numerical and JIT decisions,
randomness and memory behavior, interface-parity enforcement, failure modes,
correctness evidence, and reproducible CPU/GPU performance measurements.

The real GPU gate must show at least one completed E-step and M-step in live
output, download and validate all required artifacts, and confirm the Colab
session is stopped.

### H.12 Primary acceptance command

Run from the repository root using the local `uv` environment:

```bash
uv run bash rbpf_v3/run_smoothing_colab.sh
```

Success requires all of the following in the same invocation:

1. the local launcher finds `colab` and reads the committed config;
2. the requested Colab GPU starts and JAX reports the GPU backend;
3. `run_smoothing.py` loads or creates valid initial parameters;
4. the filter, backward smoother, every configured EM epoch, final filter, and
   evaluation complete;
5. progress remains visible locally throughout the run;
6. the launcher exits zero;
7. every required artifact is downloaded into
   `rbpf_v3/outputs/smoothing` and is nonempty;
8. JSON validation passes with finite values and no hard evaluation failure;
9. `em_initial_params.json` and `em_final_params.json` are present;
10. the configured Colab session is stopped.

Capture the launcher exit code and deployment summary in the final
implementation report. No other test result may be used to claim completion
when this command has not passed.

## JAX references

- [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
- [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [`jax.lax.map`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)
- [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)
- [Asynchronous dispatch and JIT timing](https://docs.jax.dev/en/latest/jit-compilation.html)
- [Compiled printing and callbacks](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html)
- [Slow tracing and compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
- [Device-memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
