# Review of Chapter 2 and the Colab SQMC comparison





### 3 Perform a review based on the changes that have been made

## Fixed-work SQMC comparison

**Decision: the current results are sufficient; no new CPU/GPU experiment is needed for the fixed-work presentation.** Run [`07092026_1002`](sqmc/comparison/outputs/07092026_1002/) already evaluates the same SQMC implementation on both backends for **100 observations**, five dimensions and five particle counts. Its 50 timing rows form **25 matched CPU/GPU pairs**. The fixed-budget table is a subsequent selection from these measurements, rather than the only comparison the experiment performed.

This plan concerns the current SQMC CPU/GPU experiment. Historical SMC-baseline proposals elsewhere in this review remain historical context; adding a baseline is outside this plan. The items below are proposed implementation tasks, not completed code changes.

**1 (High): define fixed work and reuse the measured results**

Hold the observation sequence, dimension `d`, particle count `N`, filtering horizon `T=100`, model, float64 precision and scrambling policy fixed between CPU and GPU. Match randomisation keys within each accuracy replicate. Vary `N` across the measured grid to assess scaling and approximation error. Filtering steps consume observations: increasing `T` changes the inference problem. Repeating whole trajectories estimates timing variability and randomisation error; it does not improve a single trajectory's estimator unless a separate averaging procedure is defined.

The existing files provide:

| Quantity | Existing source | What can be deduced |
|---|---|---|
| Timing and fixed-work speedup | `sqmc/results.json` / `.csv` | Median CPU time divided by median GPU time for each matched `(d,N)`; seven raw timings and their quartiles are retained. |
| Accuracy at matched work | `sqmc/results.json`, `accuracy_records.json` | Held-out RMSE and its existing bootstrap interval on each backend; numerical differences between matched configurations. |
| Accuracy versus runtime | `sqmc/results.json` | Each backend's measured time/error curve, without choosing a runtime budget. |
| Agreement of filtering estimates | `sqmc/*_validation.npz`, `reference_d*.npz`, `accuracy_records.json` | Direct checks of matched estimates, observations, replicate keys and errors against the Kalman reference. |
| Reproducibility | Root/stage configuration, metadata and manifests | Run/source identifiers, hardware, precision, effective parameters and input integrity. |

For example, the saved `N=2048`, `T=100` measurements give:

| Dimension | CPU time (ms) | GPU time (ms) | CPU/GPU speedup | Validation RMSE on both backends, rounded |
|---:|---:|---:|---:|---:|
| 2 | 454.65 | 71.26 | 6.38× | 0.02418 |
| 5 | 314.56 | 47.59 | 6.61× | 0.10314 |
| 10 | 330.97 | 40.66 | 8.14× | 0.29819 |
| 30 | 366.39 | 36.38 | 10.07× | 1.03129 |
| 60 | 464.88 | 39.20 | 11.86× | 1.79198 |

Across all 25 matched configurations, the largest absolute CPU/GPU difference in validation RMSE is approximately `2.22e-16`. The supported conclusion is **the same measured filtering accuracy in less steady-state runtime**. The different error values across dimensions should not be interpreted as a benefit caused by changing hardware.

- [ ] Reuse the existing stage/artifact validators before analysis; require complete grids and matching source/configuration provenance.
- [ ] Pair records explicitly by `(dimension, n)` within this single run; reject duplicates or missing backends rather than pairing by row position.
- [ ] Verify the fixed horizon, precision, model and randomisation settings from configuration/metadata; match accuracy records by `(dimension, n, phase, dataset, replicate)` and check their keys.
- [ ] Preserve the original seven timed repetitions, two warm-ups and sixteen validation replicates. Do not launch filters or invent additional measurements during analysis.

**2 (High): add an offline fixed-work summary without changing the budget schema**

Implement a proposed `sqmc/comparison/analyze_fixed_work.py` entry point that reads saved artifacts and performs no JAX execution, device allocation or Colab calls. For each pair calculate:

```text
speedup = median(cpu_samples_seconds) / median(gpu_samples_seconds)
time_reduction_percent = 100 * (1 - gpu_median_seconds / cpu_median_seconds)
validation_rmse_absolute_difference = abs(cpu_validation_rmse - gpu_validation_rmse)
```

Use the ratio of separately measured medians, not the median of arbitrarily paired CPU/GPU timing ratios. Preserve negative time reductions when CPU is faster. Validate positive finite timing values; do not discard inconvenient configurations.

- [ ] Write `fixed_work_results.json` and `.csv` with `dimension`, `n`, `n_steps`, CPU/GPU medians and quartiles, speedup, time reduction, each backend's validation RMSE/interval, and the RMSE difference.
- [ ] Store raw-run identity, source commit, input hashes, analysis-script hash and analysis settings alongside the derived outputs. Distinguish benchmark source provenance from the later analysis version.
- [ ] Use a fresh, separate analysis directory with `logs.txt` and completion/failure status; reject an existing destination. Leave the original results, logs and manifests unchanged.
- [ ] Retain the original SQMC `cpu_gpu_comparison.json` as the **fixed-budget** summary. Do not silently replace its existing particle-selection/error-ratio schema with runtime speedups.
- [ ] Initially report the seven timings and their IQRs as descriptive timing variability. An IQR or a ratio of quartile endpoints is not a confidence interval for speedup. If adding bootstrap speedup intervals, document the conditional resampling assumptions; matching replicate keys does not make the sequential CPU/GPU timing observations paired measurements of the same hardware conditions.

Proposed command, **to become available after implementation**:

```bash
.venv/bin/python -m sqmc.comparison.analyze_fixed_work \
  --run-dir sqmc/comparison/outputs/07092026_1002 \
  --output-dir sqmc/comparison/outputs/07092026_1002_fixed_work
```

**3 (High): make acceleration and accuracy-versus-time the primary figures**

The current benchmark already saves `sqmc/runtime.png` and `sqmc/accuracy_vs_runtime.png`. The dissertation renderer also already produces a faceted runtime-versus-particle-count figure. Reuse these measurements and extend the rendering, rather than designing another timing experiment.

- [ ] Add **speedup versus particle count**, one panel per dimension: `N` on the horizontal axis, CPU/GPU median-runtime ratio on the vertical axis, and a horizontal reference at one. Display CPU-favouring cases as well as GPU-favouring cases.
- [ ] Add a faceted **validation RMSE versus measured runtime** figure: runtime on the horizontal axis, held-out error on the vertical axis, consistent CPU/GPU colours, and particle counts identified at the measured points. At matched `N`, GPU points should appear further left when faster, at essentially the same error.
- [ ] Use the retained validation bootstrap intervals as vertical uncertainty bars and, if shown, timing IQRs as separately labelled horizontal variability bars. The error intervals remain conditional on one dataset per dimension and are not uncertainty intervals for runtime or hardware equivalence.
- [ ] Retain runtime versus `N` as supporting detail. Use the paired summary table or an optional RMSE-versus-`N` plot to demonstrate accuracy agreement; do not separate overlapping CPU/GPU accuracy curves by changing the data.
- [ ] Connect measured points only as visual guides, keeping particle-count ordering explicit. Do not smooth or extrapolate an unmeasured convergence curve or claim an optimal time to an accuracy threshold.
- [ ] Export legible PDF/PNG figures and a LaTeX table from the same derived records. Extend [the dissertation renderer](dissertation/drafts/figures/make_comparison_figures.py) to consume the summary, with an explicit input run and a fresh export destination.

**4 (Medium): update section 2.3 and future benchmark exports**

Lead the SQMC results with a fixed-work question: **how much faster does the GPU execute the same filter, and is filtering accuracy preserved?** Follow with the accuracy-versus-time curves. Keep the existing fixed-budget results as a secondary application of that acceleration: a faster backend can process more particles under a runtime cap.

- [ ] Update [section 2.3](dissertation/drafts/chapters/2_high_performance_sqmc.tex) to describe `T=100` fixed steps and matched `N`; preserve all existing commented content.
- [ ] Report both the small-count CPU advantages and the `6.38–11.86×` GPU speedups at `N=2048`. Generate displayed numbers from the derived table rather than manually transcribing them.
- [ ] State that matched accuracy is preserved to observed numerical precision; it is not an assertion of bitwise equality, a formal equivalence test, or an SQMC-versus-SMC result.
- [ ] Explain that first-call compilation/execution and host–device transfers are outside the steady-state speedup. Preserve the one-dataset and packed-Hilbert-resolution qualifications; a low runtime at high dimension does not establish low filtering error.
- [ ] Document the analysis command and the distinction between fixed-work runtime ratios and fixed-budget error ratios in [COMPARISON.md](sqmc/comparison/COMPARISON.md).
- [ ] For future runs, factor the summary arithmetic into a pure helper shared by the offline analyzer and `benchmark_sqmc.py`, and emit a separately named `fixed_work_comparison.json`. Preserve the existing budget outputs and shared algorithm calls.
- [ ] Update future artifact contracts with an explicit schema/version policy. Older complete runs must remain usable through offline derivation even though they lack the newly named summary file. Post-processing the saved run does not require provisioning or rerunning Colab.

**5 (Medium): validate the presentation and define when new experiments are needed**

- [ ] Add focused analysis tests for correct pair matching, missing/duplicate pairs, the ratio-of-medians definition, invalid timings, CPU-favouring speedups and input-provenance mismatches. Use synthetic records, not fixed performance thresholds.
- [ ] Derive all 25 fixed-work pairs from `07092026_1002`; reproduce the example table above at the stated rounding precision and check the reported accuracy differences against saved records/arrays.
- [ ] Confirm that offline analysis leaves every source artifact's checksum unchanged and creates no GPU session or filter execution.
- [ ] Inspect the generated panels for clipped legends/ticks and accurate labels, then compile the dissertation in an isolated build directory. Preserve the working index and verify existing commented text is unchanged.

Additional runs would be needed to study **different horizons, larger particle counts, batched independent filters, another device, transfer-inclusive timings, or broader uncertainty across datasets and sessions**. In particular, seven timing repetitions on one session cannot establish stable performance across sessions, and one dataset per dimension cannot establish general statistical performance. The saved first-call timing combines tracing, compilation and execution; isolated compilation cost cannot be recovered from that scalar alone.

**Recommended completion order:** validate the existing run → derive the fixed-work summary offline → generate speedup and accuracy-versus-time figures/table → revise section 2.3 and its documentation → optionally automate the same exports in future benchmark runs. **A new full A100 run is not a prerequisite for the requested fixed-work comparison.**

## 07092026

**Shared-implementation review.** This section assesses [qmc.py](sqmc/qmc/qmc.py), [hilbert_sort.py](sqmc/hilbert_sort/hilbert_sort.py), and [sqmc.py](sqmc/sqmc/sqmc.py), separately from the comparison runners. Local CPU diagnostics reproduced the issues below. The existing QMC, Hilbert, and SQMC test suites reported **90 passed**, with six SciPy warnings about Sobol sample counts; those passing tests do not cover these failure cases. These are findings and proposed fixes, not implemented changes.

Each topic gives the observed problem, recommended code changes, and acceptance checks. Code blocks are implementation sketches for the named functions; proposed arguments and state types still need integration and testing. Keep `sample()` and `scramble=True` as the public sampling and scrambling controls.

**1 (High): lax.scan advances only to 8**

- [x] Completed

**Location:** `qmc.py`, `Halton.sample` and `Sobol.sample`. A three-step compiled scan requesting eight points each time returned the same batch three times; the Python counter reached eight instead of 24. `_num_generated` changes during tracing rather than on every runtime iteration.

**Recommended code changes:** Extend `sample()` with an optional explicit-state path while retaining ordinary `sample(n)` behavior for existing eager callers. Use the proposed explicit-state branch below for compiled calls.

Represent the explicit state as a JAX-compatible tree containing the current index and the randomization state needed by that generator. Keep dimensions, output dtype, and batch size static. Both paths must use the same internal point-generation routines in `qmc.py`; the explicit-state path must not read or update `_num_generated` or another captured Python attribute. The eager wrapper can update its counter after calling the shared calculation.

Carry the returned sampling state through the SQMC trajectory alongside the particle state. Integrate that state through a compatible filter-state extension or adapter after checking Cuthbert's interface; do not assume its existing particle-state fields already contain a QMC counter. Keep sequence continuation distinct from drawing a freshly scrambled point set: continuation increments the index, whereas a fresh randomization initializes the aligned index again with a new scramble.



```python
# Proposed explicit-state branch of the existing sample() method.
# _sample_from_state is a shared internal calculation, not a new algorithm.
def sample(self, n, *, state=None):
    if state is not None:
        return self._sample_from_state(n, state)  # points, next_state
    points, next_state = self._sample_from_state(n, self._eager_state())
    self._num_generated = int(next_state.index) - self.start_index
    return points

# Only the pure branch belongs inside a compiled scan.
def step(qmc_state, _):
    points, next_state = engine.sample(8, state=qmc_state)
    return next_state, points

final_state, batches = jax.lax.scan(step, initial_state, None, length=3)
```

The internal helpers and state representation above are proposed additions. For Sobol, an explicit `start_index` is supplied by topic 2; the helper must retain each generator's configured scrambling state and validate bounds.

**Acceptance:** under both eager execution and `lax.scan`, three continuation calls of eight points must match one 24-point call and finish at the same index. Repeating a compiled call with the same explicit input state must reproduce its result without modifying the engine. Fresh-randomization mode must instead satisfy the key checks in item 6.

**2 (High): dropping the first Sobol point breaks balanced-net sampling**

- [x] Completed

**Location:** `qmc.py`, `Sobol.sample`. The expression `_num_generated + 1` skips index zero. In one dimension, indices `0…3` give `0, 0.5, 0.75, 0.25`, one point per quarter; indices `1…4` give `0.5, 0.75, 0.25, 0.375`, leaving the first quarter empty and the second with two points. The code comment cites Owen as justification, but the paper warns that dropping the first point can destroy net structure and worsen accuracy. With LMS+shift the first point is generally the digital shift, not the origin. [Owen, On dropping the first Sobol' point](https://arxiv.org/abs/2008.08051). This is high priority where the experiment claims balanced nets; an explicitly described ordinary-sequence timing benchmark has a different requirement.

**Recommended code changes:** Add an explicit start-index policy, consistent with Halton's existing constructor option. Preserve the current default for compatibility initially, but configure the SQMC experiment to start at zero. For a balanced block, require a power-of-two count `N` and a start divisible by `N`. For independently scrambled point sets, restart at zero for every new randomization. Do not impose these restrictions on every general sequence request; validate them where the benchmark claims balanced nets.

Keep all coordinates of each point together when sorting the resampling coordinate. Handle zero-valued propagation coordinates at the inverse-CDF boundary rather than dropping the entire point. Document the selected indexing policy in the run configuration and revise the misleading origin-dropping comment.



```python
# Proposed Sobol constructor argument; preserve default behavior initially.
# def __init__(..., start_index=1):
self.start_index = start_index

# In sample(), replace the unconditional +1:
first_index = self.start_index + self._num_generated

# Where a benchmark requests a balanced block (static values):
if n <= 0 or n & (n - 1):
    raise ValueError("Balanced Sobol blocks require a power-of-two count.")
if first_index % n:
    raise ValueError("Balanced Sobol blocks require an aligned start.")
```

Use `start_index=0` for the balanced-net experiment. Replace the existing comment with:

```python
# Keep aligned power-of-two blocks for balanced-net experiments.
# Dropping index zero can destroy balance (Owen, 2020).
# With scrambling, the first point is generally not the origin.
# Handle inverse-CDF endpoints without dropping an entire point.
```

**Acceptance:** compare aligned unscrambled points with an independent reference, verify the first-coordinate strata, and check chunked continuation. Test that a benchmark request for an unaligned balanced block is rejected or explicitly labeled as ordinary sequence sampling.

**3 (Medium): Halton accepts negative and overflowing indices**

- [x] Completed

**Location:** `qmc.py`, Halton constructor and `sample`. `start_index=2**32` reproduced the sequence beginning because indices are converted to `uint32`; `start_index=-1` was also accepted. The digit-precision limit alone does not prevent integer wraparound.

**Recommended code changes:** Validate `start_index` as a nonnegative integer before converting it to a device dtype. With the current `uint32` implementation, define the maximum exclusive index as the minimum of `2**32` and the supported digit range of every active coordinate. Validate `start + n <= limit` using arithmetic that cannot wrap. This permits a final block ending exactly at the limit while rejecting the next point.

For the proposed compiled state, keep the counter in a representation capable of holding the exclusive end value, or carry an explicit exhausted flag. Validate before narrowing to `uint32`; checking after conversion cannot detect the original overflow. Propagate invalid dynamic requests through an explicit checked error path rather than silently wrapping them. Wider point indices should be a separate, tested extension of all digit kernels.



```python
# Host-side validation before creating uint32 point indices.
if isinstance(start_index, bool) or not isinstance(start_index, int):
    raise TypeError("start_index must be a Python integer.")
if start_index < 0:
    raise ValueError("start_index must be nonnegative.")

limit = min(2**32, *(base**digits for base, digits in
                     zip(self._bases, self._digits_per_dim)))
first_index = self.start_index + self._num_generated
if first_index < 0 or n > limit or first_index > limit - n:
    raise ValueError("Requested Halton block exceeds the index range.")
```

Retain the existing positive-integer validation for `n`. Dynamic explicit-state indices need an equivalent checked runtime path using a wide counter; this host-side snippet alone does not validate traced values.

**Acceptance:** reject negative/noninteger starts, `start_index=2**32`, and blocks crossing the limit. Exercise the last valid indices using tiny batches, checking them against independently calculated radical inverses without allocating the whole sequence.

**4 (Medium): float32 Sobol points can round to 1.0**

- [x] Completed

**Location:** `qmc.py`, `_sobol_sample_batched` output conversion. At valid index 715827882, the first coordinate was `0.9999999990686774` in float64 and `1.0` in float32. The diagnostic set the counter to that index without allocating the prefix. A normal inverse CDF at one is infinite.

**Recommended code changes:** For Sobol output conversion, preserve the generator's `[0,1)` contract: after conversion to the requested floating dtype, cap any rounded-one value at `nextafter(1,0)` in that dtype. Keep exact zero valid at the generator level. Document that float32 cannot represent every distinct 30-bit Sobol coordinate, so clipping prevents an invalid endpoint but does not restore lost precision; use float64 for the principal comparison.

Before a normal inverse CDF, apply an explicit open-interval policy to initialization and propagation coordinates in both SMC and SQMC. Use the concrete 30-bit boundary helper below. Save the bounds in metadata and describe the resulting finite-precision approximation. Resampling coordinates need the separate cumulative-probability convention in item 7.



```python
# At the end of _sobol_sample_batched:
points = integer_points.astype(dtype) * scale
zero = jnp.asarray(0, dtype=dtype)
one = jnp.asarray(1, dtype=dtype)
return jnp.minimum(points, jnp.nextafter(one, zero))

# Shared normal-quantile boundary helper for the 30-bit experiment:
def normal_coordinates(u):
    zero = jnp.asarray(0, dtype=u.dtype)
    one = jnp.asarray(1, dtype=u.dtype)
    lower = jnp.asarray(2.0**-31, dtype=u.dtype)
    upper = jnp.minimum(jnp.asarray(1.0 - 2.0**-31, dtype=u.dtype),
                        jnp.nextafter(one, zero))
    return jax.scipy.special.ndtri(jnp.clip(u, lower, upper))
```

**Acceptance:** reproduce the reported high-index float32 case, check that generated values stay below one, and verify finite normal quantiles for exact zero and rounded-one inputs in both supported dtypes. Confirm that interior values remain unchanged by the endpoint policy.

**5 (High): filter_prepare consumes a discarded QMC batch**

- [x] Completed

**Location:** `sqmc.py`, `filter_prepare`. Eight-particle initialization advanced the counter to eight; preparation advanced it to 16; one combine step advanced it to 24. Preparation uses the generated batch only to discover its shape.

**Recommended code changes:** Derive a particle shape specification by applying `jax.eval_shape` to the deterministic `init_transform` with an abstract uniform-coordinate input of the required shape. Then add the leading particle dimension to every output leaf. Alternatively, retain that specification from initialization. Build preparation placeholders from the specification rather than calling `init_sample`.

Do not simply wrap the current stateful `init_sample` in `jax.eval_shape`: tracing it can still mutate the Python counter. Keep `filter_prepare` independent of all sampling and scrambling operations, and preserve the shared filter's expected array-tree structure.



```python
# Use the deterministic transform, not init_sample, for shape inference.
uniform_spec = jax.ShapeDtypeStruct((qmc.d - 1,), qmc.dtype)
particle_spec = jax.eval_shape(
    lambda u: init_transform(u, model_inputs), uniform_spec
)
particles = jax.tree.map(
    lambda spec: jnp.empty((n_filter_particles,) + spec.shape, spec.dtype),
    particle_spec,
)
```

Pass the transform/specification into `filter_prepare` from `build_filter`; its present arguments do not include `init_transform` or `qmc`. Cache static specifications where possible.

**Acceptance:** preparation leaves both eager and explicit sampling states unchanged, including when traced. Initialization consumes one batch; each actual update consumes one further batch in continuation mode. Check single-array and multi-leaf transform shapes.

**6 (High): trajectory keys do not control QMC randomization**

- [x] Completed

**Location:** `sqmc.py`, `build_filter` and `filter_combine`. Sampling uses the captured engine, without using filter-state keys to select the scramble. Supplying new trajectory keys therefore does not create independent QMC randomizations.

**Recommended code changes:** Split each trajectory key into separate initialization and observation streams. For each independently randomized SQMC point set, use a fresh key with the existing `Sobol(..., scramble=True, key=...)` scrambling logic, starting from original direction numbers. Factor that logic into a shared pure internal calculation if needed so it can populate the explicit sampling state without mutating a captured engine. A newly supplied key must control the scramble actually consumed by `sample()`.

Use the SQMC filter state's key according to Cuthbert's key-consumption convention, with a distinct next key carried forward. Document the key schedule and verify initialization does not reuse a key consumed by an update. With `scramble=False`, retain deterministic behavior and do not describe different trajectory keys as independent QMC randomizations. Include fresh scrambling inside the timed full trajectory when that is the algorithm being compared.



```python
# A fresh local engine uses the existing constructor's scrambling logic.
# start_index=0 is the proposed constructor option from topic 2.
def sample_for_key(key):
    engine = Sobol(d=state_dimension + 1, scramble=True,
                   key=key, dtype=jnp.float64, start_index=0)
    return engine.sample(n_filter_particles)

init_key, updates_key = jax.random.split(trajectory_key)
init_uniforms = sample_for_key(init_key)
# Split update keys separately; consume a distinct key for each observation.
update_keys = jax.random.split(updates_key, n_steps)
```

This is a concrete option for independent per-step scrambling: the engine is local to the call, so its Python counter is never shared between scan iterations. JAX traces the construction and executes its array scrambling operations with the dynamic key. Validate that behavior under JIT. Sequence continuation still needs topic 1's explicit-state path. Integrate initialization and update sampling into the shared filter rather than implementing a benchmark-local filter.

**Acceptance:** identical keys reproduce particles and likelihoods; different trajectory keys produce different point sets under JIT; every observation uses its designated randomization. Compare the actual point sets consumed by SQMC with direct shared-QMC calls for the same state/key, and record those keys for accuracy replicates.

**7 (Medium): inverse-CDF resampling can select a zero-weight ancestor**

- [x] Completed

**Location:** `sqmc.py`, `resample_from_uniform`. Logits `[-inf,0]` represent weights `[0,1]`. For uniform zero, the current left-sided search selected ancestor zero, despite its zero mass.

**Recommended code changes:** Adopt half-open cumulative intervals: for `u` in `[0,1)`, select the first cumulative probability strictly greater than `u`. In the current implementation, this means a right-sided `searchsorted`, with the cumulative weights normalized by their final value so their upper endpoint is exactly one. Normalizing the entire cumulative array also keeps a trailing zero-weight plateau at one; merely forcing its final element to one can introduce an artificial interval.

Validate that weights have a finite positive total and uniforms are finite and in the documented range. Negative-infinite logits are valid zero weights when at least one weight is positive; all-negative-infinite or NaN logits require an explicit failure path. If tolerating a rounded input of exactly one, map it to `nextafter(1,0)` explicitly. Do not conceal invalid inputs by clipping every returned index into bounds.



```python
# Numerical core, after checked validation of logits and uniforms.
weights = jax.nn.softmax(logits)
cdf = jnp.cumsum(weights)
cdf = cdf / cdf[-1]
one = jnp.asarray(1, dtype=sorted_uniforms.dtype)
zero = jnp.asarray(0, dtype=sorted_uniforms.dtype)
# Explicit policy: tolerate rounded 1, but reject values outside [0, 1].
u = jnp.minimum(sorted_uniforms, jnp.nextafter(one, zero))
idx = jnp.searchsorted(cdf, u, side="right", method="sort")
return idx, jnp.zeros_like(sorted_uniforms)
```

Validate nonempty one-dimensional weights, finite uniforms, and valid nonzero total mass before this calculation. Use a checked wrapper/runtime error mechanism compatible with JIT; the snippet is not a complete replacement for validation.

**Acceptance:** `[0,1]` with `u=0` selects ancestor one; `[0.5,0,0.5]` with `u=0.5` selects ancestor two. Also test leading/trailing zero weights, uniforms near one, invalid totals, and eager/JIT agreement. These checks supplement ordinary random-input tests because finite QMC values can land exactly on boundaries.

**8 (Medium): Hilbert standardization overflows on large finite inputs**

- [x] Completed

**Location:** `hilbert_sort.py`, mean/standard-deviation calculation. Points `[[3,1],[-2,5],[0,-4],[1,0]]` sorted as `[2,3,0,1]`; scaling by `1e200` produced `[0,1,2,3]` because variance overflow collapsed the normalized coordinates. All input values remained finite.

**Recommended code changes:** For each finite nonempty column, compute `scale = max(abs(x))`, replacing zero scale with one. Divide the column by that scale before calculating its mean and standard deviation. Perform the existing centering, variance normalization, logistic transform, and quantization on these scaled values. Scaling first bounds the values used in the reductions and prevents the reported square-overflow case; constant columns should still map to the interval center.

Retain the documented power-of-two grid and stable tie ordering. Specify how nonfinite inputs are handled at the public boundary, since the sorter currently documents finite inputs. Do not change quantization simply to force agreement with an independently configured reference.



```python
# Replace direct standardization of work; retain empty-input handling above.
work = x.astype(_FLOAT_DTYPE)
scale = jnp.max(jnp.abs(work), axis=0, keepdims=True)
scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
scaled = work / scale
means = jnp.mean(scaled, axis=0, keepdims=True)
std = jnp.std(scaled, axis=0, keepdims=True)
safe_std = jnp.where(std > 0, std, jnp.ones_like(std))
standardized = (scaled - means) / safe_std
unit_coordinates = invlogit(standardized)
# Continue with the existing quantization, Hilbert indices, and stable sort.
```

**Acceptance:** the reported four-point example and its `1e200` rescaling must produce the same ordering. Add very small finite magnitudes, all-zero/constant columns, supported dimensions, and permutation checks. For broader rescaling tests, compare quantized keys as well as permutations so genuine rounding at cell boundaries is not mistaken for a traversal defect.

**9 (Medium): the SQMC example omits the first observation**

- [x] Completed

**Location:** `sqmc.py`, `main`, and `test_sqmc.py`, `AnalyticalKalmanTest`. Prior initialization sets zero weights, but the update loop starts at `t=1`. The test compares with `observations[1:]`, so it accepts omission of the first likelihood contribution.

**Recommended code changes:** Initialize the prior before observing data, then execute the resample/propagate/weight update for every observation, including `observations[0]`. Generate example data from the same prior and transition-before-observation convention. Make the example's `log_potential` return a scalar per particle, and report a final mean weighted by the final normalized weights.

Change the Kalman comparison from `observations[1:]` to the full observation array with the same initial-time convention. Add a one-observation case so omitting the first update cannot pass unnoticed. Keep exact reference checks separate from particle-approximation tolerances.



```python
# Prior initialization followed by every observation update.
init_key, steps_key = jax.random.split(trajectory_key)
step_keys = jax.random.split(steps_key, n_steps + 1)
state = filter_.init_prepare({"y": observations[0]}, key=init_key)
state = state._replace(key=step_keys[0])
for observation, next_key in zip(observations, step_keys[1:]):
    prepared = filter_.filter_prepare({"y": observation}, key=next_key)
    state = filter_.filter_combine(state, prepared)

weights = jax.nn.softmax(state.log_weights)
final_mean = jnp.tensordot(weights, state.particles, axes=(0, 0))
# In the reference test, use observations, not observations[1:].
expected = _scalar_kalman_loglikelihood(observations, sigma_x, sigma_y)
```

This example assumes a single-array particle state and a scalar per-particle log potential. Apply the sampling/key fixes above before assessing independent randomized trajectories.

**Acceptance:** the one-step reference for `y=(1,-1)` has mean `(5/9,-5/9)`, marginal variance `5/9`, and log-likelihood approximately `-3.0932517270701183`. Test that changing the first observation changes the filter result and that the full sequence contributes to the likelihood. Assess particle errors with documented tolerances and independent randomizations after item 6 is implemented.

**Hilbert design qualifications.** The power-of-two grid and reduced coordinate resolution at high dimensions are documented design choices. They do not establish a bit-level traversal defect. Keep the grid explicit in comparisons and evaluate changes to resolution separately from the overflow fix.

**Recommended order:** topics 1, 5, and 6 establish sampling state and randomization; topics 2, 3, 4, and 7 repair alignment and boundaries; topics 8 and 9 address Hilbert robustness and the example. Run focused regression checks before using the corrected implementations for performance claims.

## Changes made (07092026)

All nine shared-implementation topics are now implemented and marked `[x] Completed` above. The full `sqmc/tests/` suite passes **154 tests** (91 QMC, 25 SQMC, 38 Hilbert). The changes below are the concrete implementations of the review's recommendations.

### `sqmc/qmc/qmc.py`

- **Topic 1 — explicit sampling state.** Added `QMCState` (a `NamedTuple` with a `uint32` `next_index` field, defined before the `QMC` base class so the abstract `sample` signature can reference it). `sample(n, *, state=None)` now returns `(points, next_state)` when `state` is given and does not mutate the engine, so it is safe to trace inside `jax.lax.scan`; the eager `sample(n)` path is unchanged. Added `_eager_state()` and `_sample_from_state()` (the pure shared computation, using the offset form `jnp.arange(n) + first_index` so the batch size stays static while the start index can be a traced value). The field is named `next_index` (not `index`, which collides with `tuple.index`).
- **Topic 2 — Sobol start-index policy.** `Sobol.__init__` now takes `start_index: int = 1` (default preserves origin-dropping for backward compatibility), validated as a nonnegative integer. `_eager_state()` uses `start_index + _num_generated`. Added `_validate_balanced_block(n)` requiring a power-of-two count and an aligned start (only where a benchmark claims balanced nets). Revised the origin-dropping comment.
- **Topic 3 — Halton index validation.** `Halton.__init__` validates `start_index` is a nonnegative integer and rejects `start_index >= _index_limit()`. Added `_index_limit()` = `min(2**32, base**digits over all dims)`. The eager `sample()` path validates `first_index < 0 or n > limit or first_index > limit - n` using wrap-safe arithmetic, so a final block may end exactly at the limit but the next point is rejected.
- **Topic 4 — float32 Sobol rounding.** `_sobol_sample_batched` caps output at `nextafter(1, 0)` in the requested dtype (`jnp.minimum(points, jnp.nextafter(one, zero))`), preserving the `[0, 1)` contract. Added a module-level `normal_coordinates(u)` helper applying the 30-bit open-interval policy (`clip` to `[2^-31, 1-2^-31]` and `nextafter(1,0)`) before `ndtri`, for SMC/SQMC initialization and propagation coordinates.
- Added `scramble` and `dtype` class attributes to the `QMC` base class so consumers can rely on them.

### `sqmc/hilbert_sort/hilbert_sort.py`

- **Topic 8 — standardization overflow.** `hilbert_sort` now scales each column by `scale = max(abs(x))` (zero scale replaced with one) before computing the mean and standard deviation, preventing square-overflow on large finite inputs (e.g. `1e200`). Constant columns still map to the interval centre. The power-of-two grid, quantization, and stable tie ordering are unchanged.

### `sqmc/sqmc/sqmc.py`

- **Topic 5 — `filter_prepare` no longer consumes a QMC batch.** `filter_prepare` now derives the per-particle shape via `jax.eval_shape` on the deterministic `init_transform` with a `ShapeDtypeStruct((qmc.d - 1,), qmc.dtype)` input, then broadcasts to `(N, ...)`. `build_filter` passes `init_transform` and `qmc` into `filter_prepare`. Initialization consumes one batch; each actual update consumes one further batch; preparation consumes nothing.
- **Topic 6 — trajectory keys control QMC randomization.** Added `_sample_points(qmc, key, n)`: when `qmc.scramble` is True it constructs a fresh `Sobol(scramble=True, key=key, start_index=0)` per call (independent randomization); otherwise it uses the captured engine's deterministic sequence. `filter_combine` scrambles with `state_1.key` and carries `state_2.key` forward. Verified key-controlled scrambling works under `jax.jit` and `jax.lax.scan`.
- **Topic 7 — half-open resampling intervals.** `resample_from_uniform` now uses a right-sided `searchsorted` on `cdf / cdf[-1]` (so the upper endpoint is exactly one), maps a rounded input of exactly one to `nextafter(1, 0)`, and squeezes a trailing `(N, 1)` logits dimension. It validates one-dimensional nonempty inputs.
- **Topic 9 — first observation included.** `main` samples `x_0 ~ N(0, 1)`, splits the trajectory key into separate initialization and observation streams, and updates on every observation including the first. It reports a final mean weighted by the final normalized weights.

### Tests

- `sqmc/tests/test_qmc.py`: added explicit-state scan/reproducibility/counter tests (topics 1), start-index and balanced-block tests (topic 2), Halton index-limit tests (topic 3), and float32 rounding / `normal_coordinates` tests (topic 4).
- `sqmc/tests/test_hilbert_sort.py`: added large/small finite magnitude, all-zero column, and mixed-magnitude tests (topic 8).
- `sqmc/tests/test_sqmc.py`: added resampling-boundary tests (topic 7), `filter_prepare`-no-consume tests (topic 5), key-controlled-randomization tests (topic 6), and first-observation tests (topic 9).

### Environment

- `requirements.txt` had a malformed trailing `chex` line (no version); fixed to `chex==0.1.92`.
- `cuthbertlib` (installed 0.0.15) is not yet pinned in `requirements.txt`; the review's Stage 3 plan calls for a dedicated comparison dependency lock including it.

<!-- [## Changes

The following improvements are needed in the current [Chapter 2 source](dissertation/drafts/chapters/2_high_performance_sqmc.tex) and the benchmark pipeline launched by [run_comparison_colab.sh](sqmc/comparison/scripts/run_comparison_colab.sh). Each item follows from the findings in the Original review below. These are recommendations for future implementation.

| Priority | Current issue and review finding | Required improvement |
|---|---|---|
| High | **Model mismatch — finding 1.** Data generation starts at zero; particle filters weight prior particles at the first observation; Kalman applies a transition first. | Use the same initial distribution and observation-time convention in data generation, both filters, Kalman, and the chapter. Validate the one-step posterior before regenerating accuracy results. |
| High | **SMC key reuse — finding 2.** Initialization and subsequent filtering reuse random keys. | Separate initialization, resampling, propagation, and carried keys. Execute the bootstrap filter through Cuthbert's SMC implementation and verify that random draws do not reuse keys. |
| High | **QMC implementation and description disagree — finding 3.** The paired runner constructs its own digital shift instead of using the shared class's LMS+shift; index-1 blocks lose net balance. | Construct the generator from `qmc.py` with `scramble=True` and use its `sample()` API. Execute SQMC through `sqmc.py` and retain `hilbert_sort.py` for particle ordering. Resolve sequence advancement under JAX, aligned point sets, and endpoint handling in the shared implementation; document the actual randomization in the chapter. |
| High | **Repeated accuracy values — finding 4.** Repetitions use the same captured seed, and three particle counts do not bracket common error targets. | Use independent trajectory randomizations, retain replicate seeds and errors, and report uncertainty. Expand the particle-count grid and compare measured runtime at a common accuracy target before drawing time-to-accuracy conclusions. |
| High | **Timing claim exceeds implementation — finding 5.** Only the scan is compiled; initialization and host conversion remain outside, and raw timing samples are discarded. | Compile the complete numerical trajectory, return device arrays, and synchronize outputs. Distinguish device-resident and host-to-host timing, save individual measurements, and describe compilation/setup costs accurately in the chapter. |
| High | **Configuration is not forwarded — finding 7.** The remote child reads cloned defaults instead of submitted settings; `qmc.scramble=false` is ignored; the suite's dry-run flag still executes benchmarks. | Persist and forward the effective configuration through every runner, including environment overrides and scrambling settings. Validate the fields actually used and make dry runs stop before any benchmark execution. |
| High | **Incomplete log and failure recovery — finding 8.** The launcher does not automatically retain `logs.txt`; output is buffered, and downloads occur only after the whole suite succeeds. | Stream stdout/stderr to `logs.txt` in each run and benchmark folder. Download completed outputs and logs on failure before stopping the session, preserve the original failure status, and report download/cleanup failures. |
| Medium | **Benchmark outputs are difficult to isolate — finding 8.** Per-dimension charts overwrite one another and metadata retains only the final component commands. | Split QMC, Hilbert sort, and SQMC into independently selectable stages under `benchmark_comparison.py`. Give each stage and dimension its own output folder and retain every executed command and its status. |
| High | **Source and hardware are not reproducible — finding 8.** The runner clones a moving branch, dependencies are unpinned, hardware is recorded as `cuda:0`, and run/session identifiers can collide. | Record an immutable source commit, dependency versions, effective configuration, and actual GPU model/memory. Use unique run/session identifiers and stop only the session created by that invocation. Include required aggregation dependencies such as `pandas`. |
| Medium | **Chapter artifacts are not fully generated — finding 8.** The accuracy figure is missing from generation/download; CSV structure and panel order differ from the saved chapter artifacts. | Generate every referenced table and figure from one validated run, preserve dimension-specific plots, and use a common artifact manifest for generation and download. Match captions, panel order, column labels, and displayed values to those artifacts. |
| High | **Likelihood validation is absent — finding 9.** The benchmark stores particle log-likelihoods but does not calculate an exact Kalman log-likelihood. | Add the innovation-density reference and independent-replicate likelihood-error summaries after fixing the model. State what was checked and distinguish likelihood unbiasedness from log-likelihood bias. |
| Medium | **Several chapter claims exceed the evidence — findings 6 and 9.** Runtime ranges, crossover descriptions, hardware explanations, variance rates, theorem scope, and the Sobol dimension limit need correction. | Recompute ranges and sampled crossover brackets from saved data. Distinguish same-code GPU/CPU speedups from comparisons against different CPU implementations. Qualify causal explanations until profiling/ablations support them. Correct the QMC estimator, paired-coordinate permutation, theorem assumptions, citations, and dimension limits. |

The empirical-design and results block is currently commented out in the chapter source. When bringing it into the rendered chapter, regenerate its artifacts and apply the corrections above first.

**Completion order:** fix model, shared implementation use, and random streams; establish truthful timing and configuration forwarding; add separate outputs, persistent logs, and reproducible execution; run independent accuracy/runtime experiments; then regenerate the chapter and revise its claims. The implementation plan below gives the associated acceptance checks.

### Task lists

The lists follow the twelve Changes rows above. All tasks are pending; mark a task complete only when its implementation or validation evidence is available.

#### 1. Align the model and reference — High, finding 1

- [ ] Define the common convention as `X_0 ~ N(0,I)` with transitions before observations `Y_1,...,Y_T` in the chapter and benchmark.
- [ ] Update `generate_observations` to sample the initial state from that prior.
- [ ] Make both particle filters resample, propagate, and weight at every observation, including the first.
- [ ] Check that Kalman uses the same prior, process variance, observation variance, and time indexing.
- [ ] Validate the one-step posterior against an independent analytical calculation and check multi-step output alignment.

#### 2. Repair the SMC key schedule — High, finding 2

- [ ] Route the paired bootstrap benchmark through Cuthbert's SMC filter, using the existing shared wrapper where appropriate.
- [ ] Separate initialization keys from filtering keys and follow Cuthbert's key-consumption convention.
- [ ] Ensure each resampling and propagation operation receives an unused key; carry an unused key to the next step.
- [ ] Test fixed-key reproducibility, different-key sensitivity, and key reuse with JAX key checking where supported.

#### 3. Use shared QMC and SQMC implementations — High, finding 3

- [ ] Replace benchmark-local direction loading and digital shifting with construction of the shared `Sobol` class using `scramble=True`.
- [ ] Use `sample()` for QMC sampling and execute the paired SQMC benchmark through `sqmc.py`, including its shared `hilbert_sort.py` calls.
- [ ] Specify how sequence position and randomization keys advance at runtime inside JAX; keep sampling logic in `qmc.py` and preserve ordinary `sample(n)` usage.
- [ ] Implement and validate aligned power-of-two sampling and finite normal-quantile endpoints without discarding the first point of a net.
- [ ] Avoid consuming QMC points solely to infer particle shapes in filter preparation.
- [ ] Test shared-code delegation, point-set balance, eager/compiled agreement, and independent trajectory randomizations; then update the chapter's algorithm description.

#### 4. Measure independent accuracy and time to accuracy — High, finding 4

- [ ] Give each method, dataset, and accuracy replicate a distinct recorded random stream, while using common observations for paired comparisons.
- [ ] Save replicate identifiers, keys, squared errors, and likelihood estimates; calculate aggregate RMSE by averaging squared errors before taking the square root.
- [ ] Add uncertainty estimates that respect the dataset/replicate structure.
- [ ] Run a pilot with a wider power-of-two particle-count grid and use it to choose target errors and a publication grid.
- [ ] Compare measured runtime at common attainable error targets; label targets outside either method's grid as unresolved.

#### 5. Define and validate timing boundaries — High, finding 5

- [ ] Compile initialization and all numerical filtering steps as one trajectory returning JAX arrays.
- [ ] Keep NumPy conversion, Python scalar conversion, serialization, and plotting outside that compiled function.
- [ ] Measure device-resident execution with pre-placed inputs and explicit output synchronization; measure host-to-host execution separately.
- [ ] Record setup, compilation/first execution, warmups, and individual steady-state timings with clear definitions.
- [ ] Alternate or randomize method timing order and verify that new trajectory keys reuse the compiled executable.
- [ ] Update the chapter's timing description to match the measured boundaries.

#### 6. Forward and validate configuration — High, finding 7

- [ ] Add launcher support for a selected configuration file and resolve CLI/environment overrides into one effective configuration.
- [ ] Save the submitted configuration remotely and pass its absolute path to `benchmark_comparison.py` via `--config`.
- [ ] Forward every relevant child setting, including `scramble=false`, dimensions, particle counts, and repetition counts.
- [ ] Validate the actual dimension lists, backend availability, and numerical counts used by the children.
- [ ] Make every dry-run entry point print its effective commands without running benchmarks or creating run artifacts.
- [ ] Use mocked subprocess tests to verify that a custom horizon and disabled scrambling reach the final child commands.

#### 7. Preserve logs and partial results — High, finding 8

- [ ] Create `logs.txt` in the local run directory before provisioning and in each benchmark/dimension output folder before execution.
- [ ] Stream stdout and stderr to both the console and the corresponding log, including setup errors and tracebacks.
- [ ] Save completed configuration results incrementally and record run/stage completion status.
- [ ] Attempt to download available results and nested logs after a failed stage, before session cleanup.
- [ ] Preserve the original exit status and report download or cleanup failures explicitly.
- [ ] Test successful runs and deliberate late-stage failures with a fake Colab CLI; verify that earlier outputs and logs survive.

#### 8. Separate benchmark stages and outputs — Medium, finding 8

- [ ] Extract independently runnable QMC, Hilbert-sort, and paired SQMC/SMC stages behind `benchmark_comparison.py`.
- [ ] Add stage selection so any one benchmark or combination can run with the same configuration mechanism.
- [ ] Store outputs under distinct `qmc/`, `hilbert/`, and `sqmc/` folders, with dimension subfolders where relevant.
- [ ] Merge component CSVs without overwriting individual charts and record every executed command and its status.
- [ ] Verify independent stage selection, output isolation, and aggregate plot generation with a small smoke configuration.

#### 9. Record reproducible source, environment, and hardware — High, finding 8

- [ ] Select an immutable repository commit for each run, verify the remote checkout, and record source/configuration hashes.
- [ ] Pin a tested dependency environment, including `pandas`, Cuthbert, and Cuthbertlib; record the installed package versions.
- [ ] Record JAX/JAXlib, CUDA/runtime/driver versions, precision and PRNG settings, and the physical GPU model and memory.
- [ ] Use unique run and session identifiers; reject output-directory collisions and stop only the session owned by the invocation.
- [ ] Validate ownership and collision behavior locally, then verify provenance and cleanup in a small pinned GPU run.

#### 10. Reproduce all chapter artifacts — Medium, finding 8

- [ ] Inventory every empirical table and figure referenced by the chapter and define one required-artifact manifest.
- [ ] Generate the missing accuracy-versus-particle-count figure from validated replicate records.
- [ ] Generate tables and figures from the same versioned results, with consistent CSV fields and explicit Sobol/Halton panel order.
- [ ] Preserve dimension-specific outputs and use the manifest to validate and download all required artifacts.
- [ ] Check table rounding, SQMC/SMC column labels, axes, captions, and plotted values against the saved records.
- [ ] Integrate the validated empirical section into the chapter, compile it, and inspect the rendered table and figure pages.

#### 11. Validate the likelihood estimate — High, finding 9

- [ ] Add Kalman innovation log densities and total exact log-likelihood using the model convention from task list 1.
- [ ] Check the reference against a one-step analytical density and an independent multi-step joint Gaussian calculation.
- [ ] Save particle-minus-reference log-likelihood errors for independent replicates and summarize their bias, RMSE, and uncertainty.
- [ ] If assessing likelihood unbiasedness numerically, use stable likelihood-ratio calculations and report Monte Carlo uncertainty.
- [ ] Describe the actual validation in the chapter and distinguish likelihood unbiasedness from bias in log-likelihood and normalized filtering means.

#### 12. Correct chapter claims and theory — Medium, findings 6 and 9

- [ ] Recompute runtime ranges and sampled crossover brackets from the supporting CSVs, reporting the grid's limits.
- [ ] Separate same-implementation GPU/CPU comparisons from speedups against SciPy or other CPU reference implementations.
- [ ] Remove or qualify component-cost and Hilbert-resolution explanations unless profiling, occupancy diagnostics, or ordering ablations support them.
- [ ] Correct the QMC estimator and apply the same permutation to resampling and propagation coordinates in the SQMC pseudocode.
- [ ] State the assumptions behind variance, convergence, and time-uniform results; correct the cited DOI and Sobol/table dimension limits.
- [ ] Rewrite statistical-efficiency conclusions using independent experiments and measured time-to-target comparisons, retaining unresolved outcomes where evidence is insufficient.

## Original review

Reviewed on 6 September 2026. Scope: [the chapter](dissertation/drafts/chapters/2_high_performance_sqmc.tex), [the requested launcher](sqmc/comparison/scripts/run_comparison_colab.sh), its configuration and Python runners, the underlying kernels, and the saved numerical artifacts. Local HEAD and remote `main` observed during review were `322f1ed9db2b79ae8db65adddf4aa04cd0c212d1`. The findings below assess the reviewed implementation and its saved results.

**Overall assessment: substantial revision is needed before treating this as a validated performance comparison.** The chapter accurately transcribes its main results, and the separation of kernel benchmarks from the paired GPU experiment is useful. However, the filters do not implement the same initial-time model as the Kalman reference; the SMC baseline reuses random keys; the SQMC randomization differs from the description; and the timing boundary differs from the claimed fully compiled trajectory. These affect scientific interpretation even when the script completes successfully.

**Evidence and execution.** The chapter's 15 table rows match all 75 reported numeric entries, at their printed precision, in [run `05092026_1409`](sqmc/comparison/scripts/outputs/05092026_1409/sqmc_smc_gpu_results.json). Its runtime, accuracy, QMC-speedup, and Hilbert-speedup figures are byte-identical to the corresponding files in that run. Two earlier runs, `05092026_1317` and `05092026_1359`, have the same accuracy values and different timings. The loose `sqmc/comparison/logs.txt` ends after the Hilbert benchmarks and is not evidence of a completed paired experiment.

Shell syntax checking and the requested launcher's `--dry-run` passed. A fresh execution using `SESSION=comparison_astra_review bash sqmc/comparison/scripts/run_comparison_colab.sh`, retaining the configured A100 and benchmark grid, **completed successfully with exit code 0**. It downloaded all 11 required artifacts into [run `06092026_0858`](sqmc/comparison/scripts/outputs/06092026_0858/) and stopped its session; a subsequent `colab sessions` check reported no active sessions. Local diagnostic checks used JAX 0.11.0 on CPU; those checks establish correctness properties, not GPU performance. Both the saved chapter run and the fresh run record JAX 0.11.1, SciPy 1.16.3, NumPy 2.1.3, Python 3.13.15, and `cuda:0`. The [fresh execution log](sqmc/comparison/scripts/outputs/06092026_0858/review_launcher.log), [diagnostic source](sqmc/comparison/scripts/outputs/06092026_0858/review_diagnostics.py), and [diagnostic output](sqmc/comparison/scripts/outputs/06092026_0858/review_diagnostics.log) are retained with the downloaded artifacts. Run the diagnostic source from the repository root with `.venv/bin/python` to repeat those checks.

**Fresh-run performance.** The launcher took approximately **8 minutes 51 seconds**, including provisioning, dependencies, benchmarks, downloads, and shutdown. The remote benchmark stages took about 4 minutes 35 seconds for QMC, 1 minute 17 seconds for Hilbert, and 2 minutes 19 seconds for the paired filters. These suite costs are distinct from the per-trajectory timings in the chapter. First filter calls, including compilation and execution, took 3.88–6.07 seconds for SQMC and 1.59–2.13 seconds for SMC per configuration; cold starts matter for workloads that frequently change shapes.

| Dimension | Fresh SQMC trajectory time (ms) | Fresh SMC trajectory time (ms) | Fresh runtime ratio |
|---:|---:|---:|---:|
| 2 | 68.74–72.05 | 38.88–41.42 | 1.700–1.779 |
| 5 | 53.31–58.61 | 36.31–39.90 | 1.431–1.469 |
| 10 | 45.70–53.59 | 38.71–40.78 | 1.135–1.314 |
| 30 | 42.76–49.17 | 38.14–40.24 | 1.081–1.289 |
| 60 | 41.00–47.05 | 38.34–44.38 | 1.060–1.186 |

These ranges span particle counts. All 15 fresh SQMC medians again exceed SMC's. The maximum absolute difference from the chapter-run accuracy values is only `2.22e-16`, and all eight within-configuration accuracy repetitions again coincide. This reproduces the fixed-stream numerical output, not its statistical reliability.

At the largest microbenchmark count, fresh Sobol speedups range from 3.28× to 52.54×, Halton from 103.46× to 106.59×, and Hilbert from 168.78× to 391.90× against their named CPU references. These support the broad large-batch findings. Exact crossovers are less stable: at `d=60,N=2048`, Sobol speedup changes from 1.07× in the chapter run to 0.99×, while at `d=2,N=32768` it changes from 0.99× to 1.27×. Report crossover brackets and uncertainty instead of treating those counts as hardware constants.

The CSVs also permit comparison of the JAX implementation on CPU versus GPU. At the largest counts, fresh Halton acceleration on that basis is 7.94–12.25× and Hilbert acceleration is 69.78–116.72×. Those are still substantial improvements, but they distinguish accelerator benefits from the larger differences against SciPy and the Numba reference.

The fresh run also confirms the artifact issues discussed below: its QMC CSV has 240 rows and quartiles, its speedup panels place Halton first, and the accuracy figure used in the chapter is absent. Operational completion therefore does not imply complete chapter reproduction or scientific validation.

**Measured results in the chapter's source run.** Here, `time ratio` and `error ratio` mean SQMC divided by SMC. Error is the reported normalized root mean square filtering-mean error, using the existing reference. Ranges span the three particle counts, not confidence intervals.

| State dimension | Runtime ratio | Error ratio | Supported descriptive conclusion |
|---:|---:|---:|---|
| 2 | 1.758–1.823 | 0.396–0.565 | Large error reduction for this stream, with a substantial runtime premium |
| 5 | 1.320–1.434 | 0.843–0.923 | Smaller error reduction, still slower |
| 10 | 1.242–1.285 | 0.871–1.148 | Mixed error results |
| 30 | 1.167–1.214 | 0.914–0.976 | Modest error reduction in all three observed configurations |
| 60 | 1.077–1.197 | 0.943–1.007 | Similar errors, with a runtime premium |

All 15 SQMC median runtimes exceed SMC's in this run. The same is true across the 45 configurations in the three saved runs. The eight accuracy entries for each method and configuration are identical, as are their reported log-likelihoods. These are repeated executions of a fixed stream, not independent statistical replications.

**1. High priority: the filters and Kalman reference target different distributions at the first observation.**

In [benchmark_sqmc_smc.py](sqmc/sqmc/benchmark_sqmc_smc.py), `make_model().kalman` (line 181) starts with prior variance 1 and adds process variance 0.25 before every observation, including the first. Its first predictive variance is therefore 1.25, consistent with the chapter's observation at time 1. Both `_make_sqmc_runner().run` (line 311) and `_make_smc_runner().run` (line 376) instead draw prior particles with variance 1 and immediately weight them by `observations[0]`, without the first transition. Their first predictive variance is 1.

For a one-step observation `y=(1,-1)`, the chapter/reference posterior mean is `(5/9,-5/9)` and variance is `5/9`; the model actually filtered has mean `(1/2,-1/2)` and variance `1/2`. A local SQMC check with 65,536 particles returned `(0.500011,-0.500004)`, confirming convergence toward the latter target. Its log-likelihood was `-3.031023`, close to the implemented-model exact value `-3.031024`, rather than the chapter-model value `-3.093252`.

The discrepancy is modest when averaged over 100 steps, but it means the error does not isolate particle approximation to the claimed model. At `d=2,N=512`, recomputing only the exact reference to match the existing filter changes SQMC error from `0.034738` to `0.033266` and SMC error from `0.082118` to `0.079979`. This check does not repair the experiment or quantify the effect of fixing the filters themselves.

There is a second model mismatch: `generate_observations` (line 207) initializes the latent state at zero instead of sampling `N(0,I_d)`. The common dataset remains usable for a conditional comparison, but it was not generated from the stated model. Choose one time convention, align data generation, both filters, and Kalman, then regenerate the results.

**2. High priority: SMC reuses keys within a trajectory, beyond the acknowledged fixed-seed replication.**

`_make_smc_runner` uses `random.split(PRNGKey(seed),n)` for initialization, then restarts the scan from the same `PRNGKey(seed)`. Inside `_smc_step`, it splits the carried key into `resample_key, prop_key`, uses `random.split(prop_key,n)` for propagation, and carries the already-used `prop_key` forward (line 374).

With the local default partitionable Threefry implementation, the diagnostic `array_equal(random.split(key,128)[:2], random.split(key,2))` returned `True`. Thus this pattern can literally reuse particle-generation keys for later resampling and propagation, introducing dependence beyond the intended particle-filter dependence. Fix initialization with separate initialization and scan keys, and split each scan key into a fresh carried key, a resampling key, and a propagation key. JAX explicitly requires fresh keys for independent draws; reusing keys reproduces random values. [JAX random-number documentation](https://docs.jax.dev/en/latest/jax.random.html).

The reported SMC results should therefore not be presented as a clean reference implementation until the key schedule is repaired and rerun. The size and direction of its effect on the comparison remain unmeasured.

**3. High priority: the paired SQMC runner does not use the described LMS+shift construction, and its blocks are misaligned.**

`_make_sqmc_runner` (line 255) loads raw direction integers and calls `_make_digital_shift`; it does not construct `Sobol(scramble=True)` or apply the left matrix scramble. The general `Sobol` class implements LMS+shift, but this specialized benchmark bypasses it. The chapter's empirical design must describe digital shifting alone unless the runner is changed.

The runner also uses `first_index = 1 + t*n` (line 270). For power-of-two `n`, these are non-overlapping but unaligned blocks. A direct check at `n=128`, with the actual digital shift, found zero empty first-coordinate strata for starts 0 and 128, but one empty stratum and one doubly occupied stratum for starts 1 and 129. Non-overlap alone does not preserve the desired net balance. The cited Owen paper warns that dropping the first Sobol point can damage net structure and convergence; it does not justify doing so as standard practice. [Owen, On dropping the first Sobol' point](https://arxiv.org/abs/2008.08051).

Use aligned blocks or independently randomized balanced nets at each time, with a documented finite-precision endpoint policy before applying the Gaussian inverse CDF. One fixed shift across all times also creates inter-time dependence: marginal randomization is not enough, by itself, to invoke the standard SQMC likelihood-unbiasedness theorem. Either satisfy the theorem's randomization assumptions or explicitly leave that property unclaimed for this implementation.

**4. High priority: the accuracy experiment cannot establish general statistical efficiency or time to accuracy.**

The loops at lines 561–568 do not use `rep`; both runners capture the same seed on every invocation. The chapter correctly acknowledges this, which should be retained. However, the seed is also unchanged across particle counts: the SQMC shift depends on dimension and seed, not on `N`. The existing three saved runs therefore repeat the same accuracy experiment rather than provide independent evidence.

Use independent replicate keys/randomizations passed into an already-compiled runner, retain a shared dataset for each paired comparison, and summarize mean squared error across replicates before taking a square root. Report uncertainty for method comparisons. Additional datasets are needed if the claim is intended to extend beyond this observed trajectory.

The final statement that no time-to-accuracy superiority was observed is stronger than the design supports. No target error, matched-error comparison, or efficiency frontier is computed. At `d=2,N=256`, SQMC has lower observed error than every tested SMC particle count, so the grid does not even bracket SMC's cost of reaching that accuracy.

As an exploratory calculation only, define `R = (time_SQMC/time_SMC) * (error_SQMC/error_SMC)^2`. At `d=2`, the three values are `0.561, 0.285, 0.315`, favoring SQMC on this observed error–cost product. This is not an estimated expected-MSE efficiency ratio, nor a measured time-to-target result: the errors come from one stream, timing is nearly flat in `N`, and the correctness issues above remain. It illustrates why slower execution at equal `N` cannot answer the statistical-computational trade-off on its own.

**5. High priority for the performance claim: the complete trajectory is not JIT-compiled as described.**

Both outer `run` functions are ordinary Python functions. Their scans are compiled by `lax.scan`, but initialization, first-observation weighting, concatenation, and host conversion sit outside a whole-function `jax.jit`. A local attempt to apply `jax.jit` to the returned SQMC runner raised `TracerArrayConversionError`, because it returns `np.asarray(means)` and a Python `float`. This is compatible with a compiled scan body, not with the chapter's claim of a fully JIT-compiled trajectory. [JAX scan documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).

The timed calls also convert NumPy observations to device arrays and materialize results on the host. Those conversions do synchronize execution, so the timing is not merely asynchronous enqueue time; however, it measures a host-to-host wrapper with dispatch and transfer overhead. It is not a device-resident execution time. The isolated microbenchmarks use different timing boundaries and should not be treated as additive measurements of the filter's internals.

`_time_filter` records a combined `first_execution_seconds`, median, and quartiles. It does not separately measure runner setup or compilation; setup happens before this timer, and individual timing samples are discarded. Correct the cold-start description. To substantiate accelerator performance, return JAX arrays from a compiled full trajectory, pre-place inputs, synchronize outputs explicitly, and report device-resident and host-to-host timings separately. Alternate or randomize method timing order to reduce order-dependent effects.

**6. Medium priority: several accelerator explanations exceed or contradict the evidence.**

The introductory assertion that QMC generation is compute-bound contradicts the empirical subsection's latency-bound account. The Sobol implementation uses a parallel prefix scan, which has dependencies even though it parallelizes; blanket statements about absence of synchronization do not describe every implementation detail. Halton's `_radical_inverse` and `_scrambled_radical_inverse` accumulate floating-point values during the digit loop, so the claim that both generators do all work in integers before a single conversion is also inaccurate.

The current QMC microbenchmark wraps the entire Halton sampling function in `@jax.jit`. Its Python loop over coordinates is traced into the compiled function; one host kernel dispatch per coordinate is not established by that source. Profiling is needed before assigning the observed plateau to that mechanism.

Similarly, decreasing SQMC/SMC runtime ratios do not demonstrate that propagation and weighting grow with dimension. At `N=128`, SMC runtime actually falls from `0.0395 s` at `d=2` to `0.0369 s` at `d=60`; SQMC falls from `0.0696 s` to `0.0442 s`. Hilbert traversal depth changes from 31 to 1 coordinate bits over those dimensions, so the work is not dimension-independent simply because final keys fit in one integer. The end-to-end benchmark includes no component profile or sort ablation. It also includes sorting the QMC first coordinate and sort-based `searchsorted` resampling; “the global sort” is not the only collective operation.

The 62-bit qualification is useful, but one bit per coordinate at `d=60` still distinguishes up to `2^60` joint cells. It discards within-coordinate resolution; it does not establish that almost all ordering information is lost. Nor do these measurements establish that the packed key causes the observed accuracy degradation. Measure key occupancy, ESS, ancestry, and compare higher-resolution and alternative orderings before making that causal claim. The ESS and ancestry helper functions currently exist but are not used to collect diagnostics.

**Kernel results recomputed from the chapter's source CSVs.** GPU speedups below use the named CPU reference as denominator, so they combine hardware and implementation differences. They are not pure same-code GPU/CPU speedups.

| Dimension | Sobol speedup at 131,072 | Halton speedup at 131,072 | Hilbert speedup at 100,000 |
|---:|---:|---:|---:|
| 2 | 2.88× | 104.64× | 191.45× |
| 5 | 4.72× | 105.42× | 188.83× |
| 10 | 6.46× | 108.36× | 166.33× |
| 30 | 18.35× | 102.50× | 191.84× |
| 60 | 51.82× | 108.32× | 373.21× |

The large-batch speedup endpoints in the chapter are broadly supported. The following details need correction:

- Across all dimensions and counts, saved Sobol GPU times span approximately **130–787 microseconds**, not 130–370. At `d=2,N=32768`, speedup is **0.99×**, so the first sampled winning count there is 131,072; the crossover is only bracketed by the grid. At `d=60`, the first sampled win is 2,048. GPU cost also grows with dimension at the largest count, from 371 to 787 microseconds.
- Saved Hilbert GPU times span approximately **0.137–1.006 ms**, not 0.15–0.55 ms. At `d=2` they grow from 0.549 ms at 100 to 1.006 ms at 100,000. The first sampled win is 300 for `d=2,5,10`, and already 100 for `d=30,60`. Speedup at 100,000 is not monotone in dimension.
- At the sampled QMC counts 128 and 512, neither saved GPU generator beats SciPy. The standalone Hilbert grid does not contain 128, 256, or 512, so statements about the exact paired-filter counts are interpolation, not direct measurements. End-to-end CPU filtering was never timed; the filter's own GPU/CPU crossover is unknown.
- The Hilbert reference uses `floor(2**(62/d))`, while the JAX version uses `2**floor(62/d)`. They follow the same broad Hilbert construction but need not produce the identical ordering at all dimensions. The benchmark checks output shape rather than equivalence or permutation correctness. Document the quantization difference and validate it independently.

**7. High priority for reproducibility: the launcher does not execute the submitted benchmark configuration.**

In [run_comparison_gpu.py](sqmc/comparison/scripts/run_comparison_gpu.py), `comparison_command` accepts `config` but builds only `python -m sqmc.comparison.benchmark_comparison --output-dir ...`. Consequently, the child reads the default config from the freshly cloned `main`, not the JSON sent by the local shell. A diagnostic changing `sqmc_smc.n_steps` from 100 to 7 produced an identical child command. The current local and remote defaults can happen to match; this is nevertheless a real failure when configurations diverge. The metadata can then describe submitted settings that were never used.

Persist the resolved submitted config remotely, pass its path via `--config`, and record both submitted and effective settings. Also forward `qmc.scramble`: `run_qmc` currently omits the corresponding CLI flag, so a configured `false` is ignored. Validate the dimension lists actually used, not just the fallback scalar dimensions. The unified suite's `--dry-run` is parsed but never consulted in `main`; invoking that advertised flag executes benchmarks. The outer launcher's dry run is safe, but exercises only its own validation and command construction.

**8. Medium priority: output and provenance handling do not reproduce the full chapter.**

The launcher has useful operational features: strict shell error handling, sparse checkout, GPU availability checking, nonempty local artifact checks, and an exit trap to stop its session. Nonetheless:

- It clones a moving branch and records no resolved commit, source hashes, dependency lock, `jaxlib`/CUDA versions, or physical GPU model/VRAM. `gpu_name: "cuda:0"` establishes a device identifier, not the claimed A100 40 GB. Capture actual hardware and exact source provenance. An environment override of `GPU_TYPE` or `SESSION` also does not update the embedded JSON metadata.
- The paired benchmark generates only the runtime figure. The chapter's `sqmc_smc_gpu_accuracy_vs_n.png` is neither generated nor downloaded by this pipeline. The docstring additionally promises diversity, efficiency, and claims-evaluation files that the current `main` does not produce.
- The saved chapter QMC CSV has 180 rows and only median timings; the current writer emits quartiles and, for this GPU grid, 240 rows because the SciPy reference is repeated under both backend labels. The saved CSV lacks the GPU-labeled SciPy rows expected by the current speedup plotting function. Its provenance/transformation is therefore incomplete: directly rerunning the current plotting function on that saved file cannot reproduce its curves. The saved figure puts Sobol left and Halton right, whereas the current plotter alphabetically sorts sequence names and reverses that order. The current caption fits the saved figure but would not fit the newly generated order.
- Per-dimension QMC charts overwrite one another, leaving the last dimension at the top level; Hilbert also copies only the last dimension's chart there. The merged CSVs and dimension-speedup plots are the aggregate artifacts, not those single-dimension plots. Metadata records only the last QMC and Hilbert command.
- Results are downloaded only after the entire suite succeeds. A later failure triggers remote cleanup without salvaging completed earlier results. Subprocess output is captured until completion, limiting live progress visibility. Dependencies are unpinned, and `pandas` is required by aggregation but absent from the runner's required-package checks.
- Minute-resolution run IDs can collide, despite the comment that successive runs never overwrite. The default fixed session name is proactively stopped without checking ownership. Use unique session/run identifiers, save partial artifacts on failure, and report cleanup failures rather than suppressing them.

**9. Medium priority: log-likelihood validation and several theoretical statements need correction.**

The chapter says the log-likelihood is validated against Kalman. `kalman` returns only means and variances, and `_accuracy_metrics` merely stores the particle log-likelihood. There is no exact Kalman log-likelihood computation, discrepancy summary, or validation criterion. Add the innovation-density calculation and a stochastic accuracy check after aligning the initial-time convention. Do not equate likelihood unbiasedness with log-likelihood unbiasedness or unbiased normalized filtering means.

The theory should also be qualified independently of this finite experiment:

- The QMC introduction's displayed estimator omits `f` on its right-hand side and does not state the integral correctly. Replace it with `I = integral f(u) du`, approximated by `N^{-1} sum f(u^n)`.
- In the SQMC pseudocode, sorting the first coordinates must apply the same permutation to their associated propagation coordinates. Introduce a permutation `tau` and use `v_t^{tau(n)}`, as the executable runner does with `u[tau,1:]`; sorting only the resampling uniforms breaks their pairing with the rest of each QMC point.
- The claim of `O(N^-2 (log N)^(d-1))` variance for every square-integrable integrand and any net-preserving scrambling is too broad. Owen's general square-integrable scrambled-net result gives improvement over Monte Carlo, conventionally `o(N^-1)` variance; stronger explicit rates require additional assumptions. Neither “non-smooth” alone nor the name LMS+shift supplies all the needed conditions. The bibliography's DOI for `owen1997scrambling` is also incorrect; the publisher gives `10.1137/S0036142994277468`. [Owen, Monte Carlo Variance of Scrambled Net Quadrature](https://epubs.siam.org/doi/abs/10.1137/S0036142994277468).
- Gerber's cited time-uniform SQMC result is proved for the paper's particular toy filtering problem, with Kolmogorov distance. It is not a general guarantee for this random walk or arbitrary SQMC filters. The failure thresholds discussed for particle filters are also qualified, not literally every fixed threshold. State the setting and metric, and do not present a 100-observation experiment as validation of an infinite-horizon theorem. [Gerber, Safety of particle filters, version 5](https://arxiv.org/abs/2503.21334v5).
- “Under the same conditions as SMC” and “only a smaller constant” are overly compressed accounts of SQMC convergence. State the regularity and randomization assumptions of the relevant theorem; the foundational paper explicitly establishes rates smaller than the Monte Carlo rate in its setting. Three particle counts and one random stream cannot distinguish an asymptotic rate improvement from a constant improvement. [Gerber and Chopin, Sequential Quasi-Monte Carlo](https://arxiv.org/abs/1402.4039).
- The stated Sobol dimension limit of 1,111 does not describe the bundled direction table: the runner generates 21,201 rows. Distinguish the sequence/table capacity from this filter's 62-dimensional packed-Hilbert limit.

The table also needs separate `SQMC` and `SMC` labels over its two mean-error columns; `SQMC/SMC` currently appears over a column containing an SQMC error, with the other header blank. In the results discussion, “disappears at d ≥ 10” is too categorical: the `d=30` point errors favor SQMC at all three counts, although the reductions are modest and their statistical reliability is unknown.

**Recommended completion order.** First align the model and reference, fix the SMC key schedule, and define a balanced SQMC randomization with independent accuracy replicates. Next establish a truthful compiled timing boundary and correct configuration forwarding. Then rerun on a pinned source/environment, saving raw timings, exact Kalman log-likelihood comparisons, replicate seeds, and every chapter figure. Expand particle counts far enough to bracket common accuracy targets and report uncertainty on the trade-off. Finally regenerate the table and captions from those artifacts and revise the theoretical qualifications.

Until those changes are evaluated, the defensible conclusion is narrower: **the saved implementation is slower at matched particle counts, and its fixed-stream filtering-mean error is substantially lower at dimension 2; large standalone batches show substantial speedups against the named CPU references. General SQMC superiority, the cause of high-dimensional degradation, and time-to-accuracy superiority remain unresolved.**

**Implementation plan for the recommended completion order.**

The following is a proposed implementation specification for addressing the review findings. These steps and their acceptance criteria remain future work. Deliver it in the order shown, preserving the reviewed runs as historical evidence. The primary target is the specialized comparison benchmark. Changes to the general `cuthbert` filter interface should be a separate change unless required to share a corrected pure QMC helper.

| Stage | Main files to change | Completion criterion |
|---|---|---|
| 1. Correct the experiment | `sqmc/sqmc/benchmark_sqmc_smc.py`, `sqmc/qmc/qmc.py` | Model, random streams, point sets, and reference pass independent correctness checks |
| 2. Define timing and configuration | Paired benchmark, `sqmc/comparison/benchmark_comparison.py`, both comparison launchers | Whole trajectories compile; effective configuration reaches every child; dry runs have no execution side effects |
| 3. Make runs reproducible | Comparison launchers, configuration, new environment and artifact manifests | Exact source/environment are recorded; completed and partial artifacts survive download and cleanup |
| 4. Measure statistical efficiency | Paired benchmark, new analysis module, experiment configurations | Independent replicates and datasets support uncertainty estimates and a measured accuracy–runtime trade-off |
| 5. Rebuild the chapter | New artifact-generation module, chapter, bibliography | Every empirical table and figure is traceable to one validated run manifest |

**Required implementation reuse.** `benchmark_sqmc_smc.py` must construct its QMC generators through the public `Sobol` class in `sqmc/qmc/qmc.py` and call `hilbert_sort` from `sqmc/hilbert_sort/hilbert_sort.py`. The latter is already true in the reviewed code. The QMC path currently imports `Sobol` but bypasses its construction and scrambling by loading direction data and creating a shift locally; remove `_load_direction_integers` and `_make_digital_shift` from the benchmark when replacing that path. Direction-data loading, LMS construction, shifting, and Sobol sampling must have one implementation owned by `qmc.py`.

**Proposed sampling changes — documentation only, not implemented.** Keep `sample()` as the sampling entry point and `scramble=True` as the switch that enables the existing LMS+shift implementation. Keep QMC generation and scrambling in the shared class rather than duplicating them in the benchmark. Scrambling and sequence position are separate concerns: enabling the constructor scramble does not make the Python sequence counter advance inside JAX or make a fixed engine depend on each trajectory key.

The proposed completion work is:

1. Define an explicit JAX-compatible sampling state containing the sequence position and PRNG key. Specify a backward-compatible extension of `sample()` that takes and returns this state, preserving current `sample(n)` behavior for ordinary callers. Carry the state through initialization and `lax.scan`, rather than mutating a captured engine. This API is a proposal, not an implemented signature.
2. Reuse the existing `Sobol(..., scramble=True, key=...)` scrambling implementation for each independent randomization. Derive initialization and observation keys from each trajectory key. Generate new scrambling from original direction numbers, without repeatedly scrambling transformed directions. Keep all generation logic in `qmc.py`.
3. Specify aligned index-zero, power-of-two sampling for the experiment through that `sample()` extension. Retain the original index-1 behavior for existing calls until this explicit mode is implemented and tested. Do not describe current index-1 blocks as balanced nets. Keep the finite-precision endpoint policy explicit.
4. Change shared `sqmc.py` initialization and update calls to use the proposed explicit state only after the public `sample()` contract is implemented. Derive shapes in `filter_prepare` without consuming points. Continue using the shared Hilbert ordering and filter updates.
5. Require eager/compiled agreement, counter advancement at every runtime observation, fixed-key reproducibility, different trajectory randomizations for different keys, aligned point-set checks, and shared implementation delegation tests. Add regression tests for these requirements and require them to pass before claiming independent SQMC accuracy replicates or validated compiled timings.

The shared `sample()` contract must satisfy these requirements before the benchmark can claim correct sequence advancement and independent randomizations inside a compiled trajectory.

**Stage 1 — align the model, repair random streams, and validate accuracy.**

1. **Keep observations at times 1 through T.** In `generate_observations`, sample `x_0 ~ N(0,I_d)` using a dedicated initialization key, then generate T independent transitions and observation noises. In both particle runners, initialize particles from that same prior, set all log weights to zero and `log_z=0`, and scan over **all** observations. Every scan iteration resamples, propagates, and weights, including the first observation. Remove the special first-observation weighting and `mean0` concatenation. Return T posterior means aligned with `observations[0:T]`.
2. **Make the Kalman calculation an explicit reference result.** Return posterior means, marginal variances, per-step innovation log densities, and total log-likelihood. Before each update use `P_pred=P+0.25`, `S=P_pred+1`, and add `-0.5*sum(log(2*pi*S)+(y-mu_pred)^2/S)` to the likelihood. Keep the reference implementation independent of the particle update code so a shared bug cannot make validation pass.
3. **Pass randomness as an argument.** Replace captured runner seeds with a dynamic `trajectory_key`. Derive separate top-level streams for data, filter accuracy, timing, and bootstrap analysis. Record the master seed, stable integer stream identifiers, and dataset/replicate identifiers; avoid Python's process-dependent `hash()`. Within SMC, split once into initialization and scan keys, then split each scan key into `(next_key, resample_key, propagation_key)`. Carry only `next_key`. Initialization keys and propagation child keys are consumed once.
4. **Use an independent randomized net at initialization and at every observation.** Implement the proposed explicit-state `Sobol.sample()` path described above, with `scramble=True` controlling the existing shared scrambling implementation in `qmc.py`. The benchmark must call the public class API rather than those private helpers. Use separate step keys to create fresh unit lower-triangular LMS matrices and shifts. Initialization needs an N-by-d net; each filtering step needs an N-by-(d+1) net. Generate indices `0,...,N-1` for each independently randomized net, and require power-of-two N. This replaces the reviewed `1+t*N` construction; disjoint index blocks are unnecessary when each time uses an independent randomization. Apply the same `tau` permutation to resampling uniforms and propagation coordinates.
5. **Specify finite-precision behavior.** Retain 30-bit Sobol integers and float64 arithmetic initially. Before the Gaussian inverse CDF, use a shared endpoint helper that clips to `[2^(-31), 1-2^(-31)]` for both methods. Apply this to propagation/initialization coordinates, without dropping any point; retain ordinary inverse-CDF conventions for resampling. Record this policy and its finite-precision approximation in metadata. Assert `N <= 2^30` and the available direction-table dimension. Do not claim exact continuous-distribution unbiasedness merely because all outputs are finite.
6. **Store actual accuracy replications.** For each fixed dataset and `(d,N)`, invoke each compiled runner with independent replicate keys. Save the normalized squared error `E_r=(1/(T*d))*sum((mean_estimate-mean_exact)^2/P_exact)` and `log_z_estimate-log_z_exact` for every replicate. Report `sqrt(mean(E_r))`, rather than the mean of replicate square roots. Use the same observations for both methods and every N; give the two methods distinct random streams.

Add `sqmc/tests/test_comparison_model.py` for the benchmark-specific reference and runners, and extend `sqmc/tests/test_qmc.py` for pure randomized-net generation. These tests complement the existing `test_sqmc.py`, which exercises a different filter interface.

**Stage 1 acceptance checks:** exact one-step Kalman mean `(5/9,-5/9)`, variance `5/9`, and log-likelihood `-3.0932517270701183` for `y=(1,-1)`; a small multi-step Kalman likelihood checked against the independent joint Gaussian observation density with covariance `1 + 0.25*min(t,s) + indicator(t=s)` per coordinate; finite T-by-d particle outputs for T=1 and T>1; eager/compiled agreement; reproducibility for a fixed key and distinct generated streams for distinct keys. For powers of two, check one point per first-coordinate N-stratum, exact unscrambled agreement with SciPy at aligned indices, and LMS invariants without requiring different libraries to generate identical scrambles from the same seed. Use typed-key reuse checking where supported to catch the original SMC failure. Check exact small resampling examples with nonuniform weights and known permutations. Run a fixed-seed ensemble in low dimension against Kalman; inspect error and uncertainty as N grows instead of requiring every individual replicate to improve monotonically. Stochastic tolerances must be calibrated in the pilot and documented; finite-N particle likelihoods must not be asserted equal to Kalman to machine precision.

**Stage 2 — compile the full trajectory and make configuration authoritative.**

Both runner factories should close over static shapes and model constants and return a `jax.jit` function with this conceptual contract:

```python
run(observations_device, trajectory_key_device) -> (means_device, log_z_device)
```

All particle initialization, SMC random draws, SQMC LMS+shift generation, resampling, propagation, weighting, and scan operations belong inside this function. Keep host conversion, JSON serialization, and plotting outside. Generating fresh SQMC randomizations must remain inside the measured primary trajectory; a precomputed-net experiment can be a separately labeled ablation, with its setup cost and memory recorded.

Replace `_time_filter` with two explicitly named measurements. For `device_resident`, pre-place and synchronize observations and keys, then time execution through a synchronization of **all** returned arrays. For `host_to_host`, time input transfer, execution, and output materialization, using the same output payload for both methods. Report kernel/array setup, tracing/lowering, compilation, and first compiled execution separately; record the persistent compilation-cache policy. Reuse the compiled executable across dynamic keys and observation values of the same shape. Alternate method order using a timing-only key and save each timing sample with its order and replicate identifier. Keep diagnostic outputs such as full ancestry out of the primary timed return payload.

Add a shared configuration loader/validator, proposed as `sqmc/comparison/config.py`. Add shell `--config PATH` support and resolve CLI/environment overrides into one effective configuration before submission. The remote runner must write that configuration to `effective_config.json` and invoke the suite with `--config <absolute-path>`. The suite must pass every relevant field to its children, including `--scramble` or `--no-scramble`, accuracy repetitions, dimensions, and timing policy. Validate the actual dimension lists, integer counts, uniqueness, warmups, seeds, and power-of-two SQMC counts. Make backend selection effective in the paired benchmark: reject unavailable requested GPU execution, and identify CPU smoke runs accurately.

Implement dry-run command construction in all entry points before directory creation, dependency installation, cloning, device initialization, or benchmark execution. Defer heavy imports where necessary. Introduce a versioned result schema after the sampling work is complete so validated results cannot silently mix with the reviewed outputs or outputs produced with different model, sampling, or timing semantics.

**Stage 2 acceptance checks:** `.lower(...).compile()` succeeds for both complete runners; NumPy conversion is absent from compiled paths; repeated calls with new keys reuse the executable; the intended device is recorded. In `sqmc/tests/test_comparison_pipeline.py`, use subprocess/Colab stubs to show that a custom seven-step configuration and `scramble=false` reach the final child command. Assert that each dry run launches no child and writes no artifacts. Validate timing samples and summary arithmetic without hard-coded performance thresholds. These tests must not require a Colab session.

**Stage 3 — pin execution and preserve artifacts.**

Add `repo_commit` to the resolved config. For the remote runner, fetch and check out that immutable commit, verify `git rev-parse HEAD`, and record the hash of the locally submitted runner separately. Require a clean relevant source tree for a publication run and verify that the intended commit is available remotely; otherwise fail before provisioning rather than silently benchmarking remote `main`. Record hashes of the direction table, config, and relevant benchmark files.

Add a dedicated comparison dependency lock under `sqmc/comparison/scripts/config/`, including `pandas`, `cuthbert`, and `cuthbertlib`. Select exact versions after a compatibility smoke run. For the Colab CUDA/JAX stack, record and validate the installed versions against the supported environment; fail with a clear mismatch or deliberately rebuild that stack and rerun the smoke checks. Unconstrained upgrades should not occur during publication runs. Capture `jax`, `jaxlib`, CUDA plugin/runtime and driver versions, precision flags, JAX PRNG settings, full package freeze, actual GPU model/VRAM, CPU model/core count, and memory. Query physical GPU information using `nvidia-smi` rather than the `cuda:0` label.

Use a run identifier with UTC seconds and a random suffix, and derive a unique session name from it. Refuse to reuse an existing output directory or stop a session not created by this invocation. Stream subprocess logs to both console and disk. Give each dimension its own output directory. Write results atomically after each completed configuration and maintain `run_status.json` with `running`, `complete`, or `failed`, plus completed/missing configurations and any failure reason.

Replace independent required-output lists with one artifact manifest consumed by generation, validation, download, and chapter export. On successful execution, verify schemas, full requested grid coverage, finite metrics, sizes, and checksums. On failure, attempt to download completed artifacts and logs before stopping the owned session. Preserve the original nonzero exit status and report download/cleanup failures separately. Avoid overwriting metadata at successive orchestration layers; merge namespaced records for launcher, suite, and individual benchmarks.

**Stage 3 acceptance checks:** a deliberately failed final stage preserves earlier outputs locally and still attempts cleanup; a truncated artifact or mismatched checksum fails validation; an interrupted/resumed run cannot masquerade as complete; an existing unrelated session is untouched. Stub these failure paths locally. A small real GPU run must then record the actual hardware, resolved source commit, exact effective config, and a complete manifest, with successful session shutdown.

**Stage 4 — expand the experiment and measure time to accuracy.**

Keep experiment profiles as separate JSON files rather than repeatedly editing the publication configuration. Proposed starting values are:

| Profile | Dimensions | Particle counts | Horizon | Independent datasets × accuracy replicates | Warmups / timing repetitions |
|---|---|---|---:|---:|---:|
| CPU/GPU smoke | 2 | 128, 256 | 10 | 1 × 4 | 1 / 3 |
| Corrected baseline | 2, 5, 10, 30, 60 | 128, 256, 512 | 100 | 1 × 32 | 2 / 21 |
| Range-finding pilot | 2, 5, 10, 30, 60 | powers of two from 128 to 8,192 | 100 | 3 × 8 | 2 / 7 |
| Publication | 2, 5, 10, 30, 60 | pilot-selected power-of-two grid | 100 | 10 × 32 | 2 / 21 |

These counts are planned starting points, not guarantees of sufficient precision. Use the pilot to estimate runtime, memory, uncertainty, and feasible per-dimension job sizes. Freeze the final grid, target errors, and analysis before running fresh publication datasets/seeds; expand replication if the pilot indicates unacceptable interval width. Split long jobs by dimension/dataset under a shared experiment manifest, and merge only matching source, environment, and config versions. Do not let the current 3,600-second timeout determine which configurations survive into the analysis.

Add a pure analysis module, proposed as `sqmc/comparison/analyze_results.py`, that consumes saved data without running filters. Report conditional results for each dataset and the equal-dataset-weighted aggregate `sqrt(mean(E_dataset,replicate))`. Compute paired hierarchical bootstrap intervals by resampling datasets jointly across methods and particle counts, then replicate records within those datasets. Do not treat time steps or coordinates as independent replications. Keep timing uncertainty separate from accuracy uncertainty; resample the recorded timing blocks when estimating uncertainty in combined cost metrics.

Define a target-error list in config, initially `[0.05, 0.1, 0.2, 0.5, 1.0]`, and finalize it using pilot data only. For each dimension and target epsilon, select the lowest **measured** median trajectory time among grid points whose estimated aggregate error is at most epsilon. Record the selected N, error and interval, time and interval, and the SMC/SQMC time ratio. Also report whether the error interval's upper bound meets the target. Reapply selection within bootstrap samples to expose uncertainty from choosing N. If either method does not reach the target within the tested grid, mark the comparison unresolved; do not extrapolate from a presumed convergence rate. Distinguish a best observed grid result from a global optimal time-to-accuracy claim.

Extend the pilot grid geometrically when targets are not bracketed, subject to explicit time/memory limits; record those limits and unattained targets. Add 128, 256, and 512 directly to the Hilbert microbenchmark grid. Keep both JAX CPU/GPU ratios and ratios against SciPy/Numba. If retaining an end-to-end GPU/CPU crossover claim, actually execute the full-filter CPU comparison with matching semantics and timing boundaries; otherwise remove that claim.

Collect ESS, unique-parent fraction, and packed-Hilbert key occupancy in separate untimed diagnostic runs using the same dataset/replicate keys. Use array-valued, fixed-shape JAX calculations rather than dynamically sized `jnp.unique` outputs inside a scan. For causal high-dimensional claims, add a controlled ordering ablation with shared random inputs and a validated higher-resolution implementation. Until that separate experiment exists, limit the chapter to the observed association with dimension and the known bit-resolution limitation.

For likelihood validation, retain per-replicate log errors, bias and RMSE summaries, and per-step innovations. If assessing likelihood unbiasedness numerically, compute `mean(exp(log_z_estimate-log_z_exact))` stably and report its Monte Carlo uncertainty; high-dimensional weight collapse can make that diagnostic inconclusive. Neither a small log error nor an interval covering the exact value proves unbiasedness. The theorem discussion must remain separate from numerical checks.

**Stage 4 acceptance checks:** analysis of small synthetic records verifies square-before-average aggregation, paired dataset handling, reproducible bootstrap output, known Pareto dominance, unattained targets, and selection among nonmonotone runtimes. Every result has a dataset hash and replicate identifier. No completion criterion requires SQMC to win: successful evaluation can conclude that either method is preferable or that uncertainty/grid limits prevent a decision.

**Stage 5 — regenerate tables, figures, and prose from validated results.**

Add `sqmc/comparison/render_results.py` to produce the runtime and accuracy figures, error-versus-time figure, kernel speedup figures, and LaTeX result table from artifacts of the completed, validated schema only. Use an explicit sequence order `Sobol, Halton`, named SQMC and SMC error columns, and captions generated from the effective precision, hardware, timing policy, dataset count, and replicate count. Save all these outputs in the run directory before exporting them to `dissertation/drafts/figures` and a generated table file included by the chapter. Make the manifest require `sqmc_smc_gpu_accuracy_vs_n.png` and any other figure actually referenced by the chapter; remove obsolete output promises.

Generate the key reported quantities—ratios, error reductions, timing ranges, and sampled crossover brackets—from the same analysis records. Update the chapter's experimental design to describe initialization at time 0, observations at 1 through T, fresh LMS+shift per time and replicate, endpoint handling, the two timing boundaries, and actual uncertainty calculations. Apply the estimator, permutation, theorem-scope, dimension-limit, and bibliography corrections identified in this review. Describe component-cost explanations as hypotheses unless supported by profiling/ablations.

**Stage 5 acceptance checks:** regenerate every empirical figure and table from a single complete manifest; compare displayed numbers with analysis records at the stated rounding precision; verify that axis units, panel ordering, uncertainty bands, and captions agree. Compile the dissertation and inspect the resulting table/figure pages. Export only the validated publication run, retain its manifest beside the generated artifacts, and update this review with the repaired experiment's results and any remaining limitations.

**Execution checklist once the changes exist.** Run focused model/QMC tests and mocked pipeline tests first, then the existing SQMC/QMC/Hilbert suite for regressions. Exercise each dry-run entry point with a custom smoke config, execute a CPU smoke run, and then a pinned GPU smoke run. Proceed through corrected baseline, range-finding pilot, frozen publication runs, offline analysis, and chapter rendering. The new profile filenames, configuration fields, and analysis/render commands described here must be implemented before using them; the current launcher still has its original interface. A stage is complete only when its acceptance evidence is stored, not when its code has merely been edited. -->
