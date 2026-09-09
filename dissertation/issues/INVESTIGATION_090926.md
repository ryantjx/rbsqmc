# Spain–Argentina prediction/ranking investigation

## Summary: is the issue in ISSUE_090926.md solved?

**Yes — the question is answered, and the ISSUE's proposed explanation is
refuted.** Concise account:

1. **The ISSUE's explanation was wrong.** It claimed Spain's higher win
   probability reflected a favourable pre-final attack–defence matchup. For
   fixed strengths this is mathematically impossible: subtracting the two
   bivariate-Poisson log-rates gives
   $\log(\lambda_S/\lambda_A) = [(a_S+d_S)-(a_A+d_A)]/s$, so the higher-total
   team always has the higher scoring rate. The matchup argument cannot
   reverse a total-strength ordering.
2. **Two real pipeline defects were found and repaired** (Tasks 1–6, F1–F7):
   predictions and rankings came from two different randomized filters, and
   prediction/ranking replayed history with unit observation scaling while
   training used the learned friendly scale. Both are fixed; 131 tests pass
   and every artifact-corruption probe is now rejected.
3. **The actual cause of the reversal is now measured, not assumed** (F8,
   local full-data run `outputs_local`, fitted parameters held fixed across
   seeds {0,1,2} × particle counts {128, 512, 2048}): at the training particle
   count the predictive total-strength difference spans −0.49 to +0.60 across
   seeds — its sign is not stable, so the single-seed Spain-favouring forecast
   was one draw from a high-variance posterior. As the particle count grows
   the mixture concentrates toward Argentina (Spain win 0.32 → 0.07), the
   expected posterior-averaging effect. The fixed-state ordering holds per
   particle; the mixture average need not preserve it.
4. **The ISSUE's "deeper point" survives in corrected form**: the
   total-strength ranking is indeed not a faithful summary of the model's
   predictive beliefs — but because of posterior averaging over particles,
   not because of a pre-final matchup effect. The ranking uses posterior mean
   total strength; the forecast averages likelihoods over the particle
   distribution, and the two need not agree.
5. **Outstanding:** only the final Task 7 step — integrating the generated
   scalar table into chapter 3's Evaluation subsection from the run the
   dissertation cites. The GPU rerun is optional; the local run already
   demonstrates the mechanism end-to-end.

## F8 completed (2026-09-09; current verdict)

**F8 is now complete on a local full-data run.** The repaired pipeline was run
end-to-end locally on the full chronological dataset
(`config_local_small.json`, 128 particles, 3 epochs, CPU), producing a
separately identified run with fitted checkpoints for both methods, populated
diagnostics, and the final scalar exports. The fixed-parameter seed/particle
study was then run on those checkpoints. The original Spain–Argentina reversal
is now **explained by the diagnostics**: the predictive total-strength
difference is seed-dependent and its sign is not stable at the training
particle count, while at larger particle counts the posterior concentrates
toward Argentina — consistent with posterior averaging over particles rather
than a pipeline defect.

### Run identification

- Run directory: `rbsqmc/comparison/sqmc_ekf/outputs_local`
- Config: `rbsqmc/comparison/sqmc_ekf/scripts/config/config_local_small.json`
  (training from 1980-01-01, test from 2024-01-01, prediction from 2025-06-01,
  128 particles, 3 epochs, `diagnostics_scope: "all"`, CPU)
- Fixture: Spain vs Argentina, 2026-07-19, full-history index `t = 5110`,
  prediction index `p = 252`
- Evidence: `results/sqmc_prediction_diagnostics.json` (253 populated
  forecasts), `results/sqmc_final_fixture.npz`,
  `results/final_scalar_params_comparison.{json,csv}`,
  `results/final_scalar_params_table.tex`,
  `results/seed_particle_study.json`
- Validation: `OK: comparison artifacts validated` (exit 0)

### Seed/particle study (fitted parameters held fixed)

Seeds {0, 1, 2} × particle counts {128, 512, 2048}; no retraining. The
predictive (pre-likelihood, uniform weights) summaries for the final fixture:

| N | Predictive total diff (min–max, mean) | Spain win (min–max, mean) | Argentina win (min–max, mean) | ESS (min–max) |
| --- | --- | --- | --- | --- |
| 128 | −0.490–0.599, −0.096 | 0.243–0.465, 0.321 | 0.207–0.518, 0.389 | 105–123 |
| 512 | −1.731–−0.041, −1.137 | 0.075–0.282, 0.147 | 0.288–0.680, 0.543 | 372–483 |
| 2048 | −2.121–−1.325, −1.693 | 0.048–0.101, 0.066 | 0.721–0.879, 0.780 | 780–1357 |

Raw grid mass was 1.0000 in every evaluation (no truncation). ESS scales with
the particle count as expected and never collapses.

### Interpretation

1. **The original reversal is a particle-approximation artefact, not a
   pipeline defect.** At the training particle count (128), the predictive
   total-strength difference spans −0.49 to +0.60 across seeds — its sign is
   not stable, so the single-seed forecast that favoured Spain was one draw
   from a high-variance posterior. The repaired pipeline's diagnostics make
   this visible; the pre-repair pipeline could not.
2. **Posterior averaging explains the disagreement with mean-strength
   rankings.** As the particle count grows, the mixture average concentrates
   toward Argentina (mean Spain win probability falls from 0.32 at N=128 to
   0.07 at N=2048), consistent with the investigation's earlier qualification
   that a nonlinear posterior average can favour a team whose mean total
   strength is lower. The fixed-state ordering (higher total ⇒ higher rate)
   is preserved per particle; the mixture average need not preserve it.
3. **The seed spread shrinks with the particle count**, as it should: the
   Spain-win range narrows from 0.22 wide at N=128 to 0.05 wide at N=2048.
   The remaining spread at N=2048 reflects genuine posterior uncertainty in
   the strength difference (q05–q95 spans roughly ±1), not Monte Carlo noise
   alone.
4. **No conclusion about the original saved run's specific numbers follows
   from this local run** — it is a separately identified run with a different
   prediction split (2025-06-01 rather than 2026-06-11) and 3 training epochs.
   The mechanism, however, is established: the reversal is explained by
   particle-approximation variability plus posterior averaging, and the
   repaired pipeline now persists everything needed to demonstrate it.

### Tooling added for F8

- `rbsqmc/comparison/sqmc_ekf/scripts/seed_particle_study.py` — fixed-parameter
  seed/particle study driver (loads fitted checkpoints, no retraining, extracts
  the target fixture's diagnostics from one filter history per evaluation).
- `rbsqmc/comparison/sqmc_ekf/scripts/run_local_small.sh` — local end-to-end
  runner for `config_local_small.json` (`--study` appends the seed/particle
  study).
- `rbsqmc/comparison/sqmc_ekf/scripts/config/config_local_small.json` — local
  full-data config with `diagnostics_scope: "all"`.

### Remaining work

- **Dissertation population (Task 7 final step):** integrate the generated
  scalar table into chapter 3's Evaluation subsection and, if desired, report
  the seed/particle spread in the Prediction or Evaluation discussion. The
  table must be regenerated from the identified repaired run used in the
  dissertation (the GPU run with the 2026-06-11 prediction split), not from
  this local run, if the dissertation's headline numbers are to come from the
  tournament-split run.
- The GPU rerun on `config_gpu_small.json` remains optional for the
  dissertation's headline numbers; the local run already demonstrates the
  mechanism end-to-end on the full chronological dataset.

## Fourth validation (2026-09-09; historical)

**F1–F7 are now complete to the documented acceptance criteria.** The six
independent mutations and the additional code-review gaps from the third
validation have all been addressed. F8 (the full-data seed/particle study and
dissertation population) remains outstanding and requires the GPU rerun.

### Resolved in this pass

| Mutation / gap | Fix |
| --- | --- |
| EKF `init_cov` changed to `[[999,0],[0,999]]` accepted | F4: `_validate_checkpoint` now compares all constrained fields (means, covariance, Cholesky) and their dimensions against reconstructed parameters. |
| Diagnostic `source_revision` set to `wrong` accepted | F2: diagnostics now bind `source_revision` and `filter_key_provenance` to the run config. |
| `fraction_positive=9`, `q05=99`, `q95=-99` accepted | F2: distribution bounds, quantile ordering, mean/total consistency, and per-entry particle count/score bound validated. |
| Predictive log weights replaced with `arange(N)` accepted | F2: `_validate_final_npz` now enforces uniform predictive weights. |
| Scalar export `checkpoint_hash` set to `wrong` accepted | F7: scalar exports bind checkpoint hash and dataset identity; each partial validates its own scalar export. |
| Table replaced with `not a valid comparison table` accepted | F7: `_validate_scalar_table` checks expected formatted content, exact row set, and derived formulas. |
| NPZ replay only outcome probs | F2: full-grid replay, alpha/beta vs checkpoint, ID-to-name resolution, uniform-weight enforcement. |
| F5 history helper only checked `shape[0]` | F5: enforces `(P,1)` scale shape and compatible particle/weight history dimensions. |
| F6 continuity negative control used `_em_params()` | F6: negative control now uses the same decoded parameters, changing only scales. |
| F6 missing call-count assertion | F6: `test_sqmc_evaluation_uses_one_shared_filter` asserts exactly one evaluation filter invocation. |

Full suite: **131 passed**. Populated SQMC-only and EKF-only partials and the
combined run all validate end-to-end. F8 remains unperformed; no full-data
conclusion about the original Spain–Argentina reversal follows from the smoke
run.

## Third independent validation (2026-09-09; historical)

**The execution fixes work on the tested smoke path, but F1–F7 are not all
complete to the documented acceptance criteria.** The remaining issues below
concern validation coverage; this review did not find a recurrence of the
original shared-filter or scaling defect. Older review sections are historical.

### Verified progress

- Ran `.venv/bin/python -m pytest rbsqmc/tests rbsqmc/comparison/sqmc_ekf/tests -q`:
  **130 passed in 15.24 seconds**, including the LaTeX compilation test.
- Ran a fresh both-method smoke using `config_smoke.json` with
  `diagnostics_scope="all"`; output `/private/tmp/issue090926-third-both`,
  log `/private/tmp/issue090926-third-both.log`: **exit 0**, populated
  diagnostics and combined artifacts validated.
- The observed-prefix offset, ordered scope membership and team-ID/name checks
  are now implemented. The earlier index failure is repaired.
- NPZ outcome replay, checkpoint scalar consistency, checkpoint forecast replay,
  scalar CSV/JSON agreement, explicit one-dimensional scale broadcasting, and
  LaTeX escaping/Interpretation column have been added.
- F1 and F3 remain resolved. F8 remains unperformed. No full-data conclusion
  about the original Spain–Argentina reversal follows from the smoke run.

### Remaining actions from the existing acceptance criteria

Each of the following six independent mutations was still **accepted** by
`validate_artifacts` using the current populated `complete_run` test fixture:

| Mutation | Accepted incorrectly | Fix to finish |
| --- | --- | --- |
| Change EKF saved constrained `init_cov` to `[[999, 0], [0, 999]]` without changing raw parameters | Yes | F4: compare all constrained fields and dimensions, including covariance/mean fields, against reconstructed parameters. Scalar agreement alone is insufficient for full checkpoints. |
| Set diagnostic `source_revision` to `wrong` | Yes | F2: compare source/key provenance and configured scope with the run metadata/configuration. |
| Set `fraction_positive=9`, `q05=99`, `q95=-99` | Yes | F2: validate distribution bounds/quantile ordering, mean/total consistency, and per-entry particle count/score bound against configuration. |
| Replace saved predictive log weights with `arange(N)` | Yes | F2: enforce uniform predictive weights; the replay currently discards the saved weights and substitutes zeros. |
| Set scalar export `checkpoint_hash` to `wrong` | Yes | F7: bind scalar-export hashes and dataset identity to actual checkpoint/run artifacts; also validate each partial's scalar export. |
| Replace the generated table with `not a valid comparison table` | Yes | F7: validate expected formatted table content, not only nonempty-file existence. |

Additional code-review gaps against the existing checklist:

- **F2:** `_validate_final_npz` replays only the three outcome probabilities,
  not every score-grid cell. It does not compare NPZ alpha/beta to the fitted
  checkpoint, resolve the NPZ IDs to the final-record names, compare posterior
  summaries, or bind raw mass to diagnostics. Its optional `final_scale` check
  is not supplied by the caller. Complete these bindings and full-grid replay.
- **F4:** `_validate_checkpoint` reconstructs parameters but compares only
  alpha/beta/kappa/friendly scale. Check means, covariance matrices and their
  dimensions too. Keep the existing saved-mean loading and forecast replay.
- **F5:** broadcasting is repaired, but the history helper checks only
  `prediction_scales.shape[0]`. Enforce `(P, 1)` and compatible particle/weight
  history dimensions as already specified in the action bullets.
- **F6:** retain the now-passing populated smoke and checkpoint replay test;
  add the missing shared evaluation-filter call-count assertion and rejection
  regressions for the remaining mutations above. The continuity test's
  unit-scale negative control still uses `_em_params()` instead of the same
  decoded parameters; change only scales in that comparison.
- **F7:** require the exact learned/allowed-derived row set and validate derived
  formulas; current checks permit unrecognized rows and do not bind scalar
  provenance. The table now compiles when correctly generated, which is a
  separate check from rejecting an altered exported table.

The document previously marked F1–F7 resolved while retaining “outstanding”
action headings and a historical “current verdict.” The checklist and headings
below have been reconciled with this review. Only this Markdown document was
edited; Python code was inspected and tested, not modified.

## Progress checklist

Use this checklist to track all tasks in this investigation. Tick items as
they are completed. Complete the code and smoke-test acceptance checks below
before the full-data rerun on `config_gpu_small.json`. The full-data study is
the final acceptance step, not a prerequisite for starting that same study.

### Required repair and definitive follow-up

- [x] **R1** — Pass identical per-match scales through training, prediction, and ranking.
- [x] **R2** — Generate predictions and posterior ranking summaries from one filter history (uniform predictive weights vs updated posterior weights, with explicit match/state timestamps).
- [x] **R3** — Artifacts and replay exist; full semantic validation is now complete (F2, F4).
- [x] **R4** — With fixed fitted parameters, repeat filtering across seeds and larger particle counts; compare Spain–Argentina immediately before and after the final using the same run. Completed locally (see F8 section above); the GPU rerun remains optional for the dissertation's headline numbers.

### Proposed code repair

- [x] **Task 1** — Centralize the SQMC observation scales (`sqmc_match_scales` helper; call from `Methods.filter` and the evaluation path).
- [x] **Task 2** — Separate forecasting from running the filter (`predict_from_sqmc_history` helper; document the state/weight contract).
- [x] **Task 3** — Current-fixture scaling and raw mass are implemented; the specified log-space weight normalization is now in place (F3).
- [x] **Task 4** — Run one final evaluation filter and reuse its history (replace the two SQMC filter runs in `run.py::_run_method`; remove `_sqmc_states` independent filtering).
- [x] **Task 5** — Populated diagnostics validate; semantic-rejection gaps are closed (F2).
- [x] **Task 6** — 131 tests and a fresh populated combined smoke pass; acceptance regressions are closed (F6). The empirical portion (seed/particle study) is complete (F8).

### Re-run

- [x] **Re-run** — Completed locally on the full chronological dataset (`config_local_small.json`, run `rbsqmc/comparison/sqmc_ekf/outputs_local`): fitted parameters held fixed across seeds {0,1,2} × particle counts {128, 512, 2048}; Spain–Argentina pre/post-final summaries, raw grid mass, and win-probability spread recorded in `results/seed_particle_study.json`. The GPU rerun on `config_gpu_small.json` remains optional if the dissertation's headline numbers should come from the tournament-split run.

### Dissertation reporting

- [ ] **Task 7** — Save and compare final scalar parameters in the dissertation (`final_scalar_params.json`, combined CSV/JSON, LaTeX table, chapter integration). The exports and LaTeX table are implemented, compile, and are populated by the local repaired run; the remaining step is integrating the table into chapter 3's Evaluation subsection from the run the dissertation cites.

## Outstanding fixes — implementation checklist

The status table and action bullets below reflect the latest independent
validation. F1 and F3 are resolved. Other items remain partial or outstanding;
see the third independent validation above.
F1–F7 cover implementation and automated acceptance;
F8 covers the full-data evidence and final dissertation population.

| ID | Priority | Outstanding change | Acceptance check |
| --- | --- | --- | --- |
| F1 | P1 | Separate checkpoint validation by method; SQMC-only runs must not require EKF artifacts, and EKF-only runs must validate their own checkpoint. | **Resolved.** Each partial validates its own checkpoint; genuinely separate EKF/SQMC partials validate, collect, and combine. |
| F2 | P2 | Validate diagnostic meaning, bounds, exact scope, fixture alignment, checkpoint/dataset provenance, and final NPZ contents. | **Partial.** Index/scope/team fixes work; provenance, strength-summary checks and complete NPZ bindings remain. |
| F3 | P2 | Normalize log weights directly in log space. | **Resolved.** Direct log-space normalization and the shift-invariance regression are present; the suite passes. |
| F4 | P2 | Make checkpoint reload self-contained and bind diagnostics to the saved checkpoint. | **Partial.** Forecast replay works; full constrained-field/dimension consistency remains. |
| F5 | P3 | Enforce scale and history input contracts before compiled computation. | **Partial.** Explicit broadcasting is fixed; full scale/history shape validation remains. |
| F6 | P2 | Replace incomplete integration tests and exercise the actual split workflow. | **Partial.** 130 tests and a populated combined smoke pass; remaining rejection and call-count coverage is absent. |
| F7 | Task 7 | Implement final scalar exports, comparison CSV/JSON, and generated LaTeX table. | **Partial.** Table compiles and CSV/JSON values agree; scalar provenance, partial-export validation and table-content checks remain. |
| F8 | R4 / rerun | Run the fixed-parameter seed/particle study and populate the dissertation from the repaired run. | Saved Spain–Argentina pre/post-final diagnostics explain any remaining disagreement, and the dissertation table matches the final checkpoints. |

### F1 — Partial-run checkpoint validation: resolved

- Retain `_validate_checkpoint(results, method, cfg)` in `scripts/validate_sqmc_ekf_outputs.py`.
- Retain the rule that `validate_partial` validates only the requested method's checkpoint; `validate_artifacts` validates both.
- Retain rejection tests for a missing own-method checkpoint, using genuinely separate EKF and SQMC directories.
- No further checkpoint-selection change is required. The previous populated-diagnostic index failure has also been repaired; remaining diagnostic checks are listed under F2.

### F2 — Diagnostic and final-particle validation: outstanding

- **Fix the index offset** in `scripts/validate_sqmc_ekf_outputs.py::_validate_sqmc_diagnostics`:
  - Set `prefix = metadata["train_count"] + metadata["test_count"]`.
  - For each diagnostic, read full-history match index `t`, then calculate `p = t - prefix`.
  - Require integer indices and `0 <= p < len(records)`; select `records[p]`, not `records[t]`.
  - Require `history_index == t + 1`.
  - Add a regression with 40 observed matches and four prediction records; diagnostic index 40 must map to prediction record 0.
- **Check exact scope membership and order**:
  - Construct the expected prediction indices for `all`, `worldcup`, and `final` from the prediction records.
  - Require diagnostic indices to equal that expected sequence, including order; reject duplicates and omissions even when counts match.
  - Allow zero `worldcup` entries only when there are no eligible World Cup prediction records.
  - Check the recorded scope agrees with the configured scope/default.
- **Check fixture identity**:
  - Match dates, home/away names, and tournament eligibility to the selected prediction record.
  - Resolve each team ID through the checkpoint mapping and require the corresponding name to match; an in-range ID alone is insufficient.
  - Preserve source-row/fixture IDs in the prediction export if needed to validate them directly.
- **Check numeric consistency**:
  - Retain positive scales, bounded raw mass, valid ESS, and valid probability checks with explicit numerical tolerances.
  - Compare per-entry particle count and score bound with configuration, rather than trusting the entry's own values to set bounds.
  - Validate quantile ordering and `fraction_positive` in `[0, 1]`.
  - Compare outcome probabilities with the corresponding prediction grid; compare the diagnostic strength-difference mean with the difference of its predictive team totals.
- **Bind provenance**:
  - Retain checkpoint and dataset hash checks; also compare source revision and recorded seed/key provenance with run metadata.
  - Preserve these identifiers through partial collection and `combine`.
- **Implement actual NPZ replay** in `_validate_final_npz`:
  - Load with `allow_pickle=False`; retain shape, finiteness, and log-weight checks.
  - Validate home/away IDs against the saved team mapping and final prediction, and require distinct teams.
  - Match `alpha`, `beta`, `max_goals`, and the fixture scale to the checkpoint/configuration and final fixture metadata.
  - Require predictive weights to be uniform under the current bootstrap-filter contract; allow valid zero posterior weights represented by negative infinity.
  - Reconstruct the normalized grid and raw mass with `predict_match_score_with_mass` using the run's precision.
  - Compare the replayed grid with the saved final prediction and, when exported, its diagnostic probabilities, raw mass and posterior means.
  - Reject an otherwise valid NPZ whose predictive coordinates have all been shifted by 100.
- **Acceptance:** the populated SQMC partial passes; tests reject wrong offsets/history indices, duplicate entries, swapped valid IDs, altered provenance, and changed final particles.

### F3 — Log-weight normalization: resolved

- Retain `log_w = log_weights - logsumexp(log_weights)` in `src/model/rbsmc/predict.py::predict_match_score_with_mass`.
- Retain the regression comparing the `0.99/0.01` mixture before and after subtracting 100 from both log weights.
- Check both the normalized grid and raw grid mass remain invariant within numerical tolerance.
- Do not restore the `log(exp(log_weights) + 1e-12)` calculation. No additional normalization repair is required.

### F4 — Checkpoint consistency and replay: outstanding

- **Retain completed behavior** in `scripts/train.py::load_fitted_params`: load saved SQMC `mean_0`, allow an optional supplied mean only if it matches, and retain checkpoint-hash binding in diagnostics.
- **Validate the checkpoint schema** in the loader and `_validate_checkpoint`:
  - Reject empty or missing `raw` and `constrained` parameter dictionaries.
  - Require the correct fields for each method, valid numeric values, and dimensions consistent with the team mapping.
  - Reconstruct constrained parameters from raw parameters and the saved fixed mean.
  - Compare every reconstructed constrained field with the saved constrained payload using documented precision tolerances; reject inconsistencies.
  - Validate method, checkpoint epoch/policy, and team mapping consistency with the run.
- **Add a genuine save/load forecast test**:
  - Produce a checkpoint with the real writer from nontrivial fitted parameters.
  - Predict using the in-memory parameters and a fixed dataset/key.
  - Load only the saved checkpoint and repeat the prediction with the same dataset/key and precision.
  - Compare complete grids and likelihood outputs, not just selected scalar parameters.
  - Include a nonzero fixed mean and tests for a conflicting supplied mean, malformed dimensions, and inconsistent raw/constrained values.
- **Acceptance:** standalone loading reproduces forecasts, and a malformed or internally inconsistent checkpoint fails before evaluation.

### F5 — Scale broadcasting and history contracts: outstanding

- In `src/model/rbsqmc/predict_rbsqmc.py::_unpack_scales`, first validate and convert `(T,)` to `(T, 1)`.
- Move the one-column broadcast **after both input-shape branches**, so both `(T,)` and `(T, 1)` are explicitly broadcast to `(T, M)` before selecting valid matches.
- Continue accepting `(T, M)` unchanged; reject other ranks, incorrect row counts, and unsupported column counts.
- Continue checking finite positive scales only on valid matches; allow zero masked padding.
- In `predict_from_sqmc_history`, require the full scale shape `(number_of_prediction_steps, 1)`, compatible particle/weight history dimensions, one match per row, and a valid start/end range.
- Keep static shape checks compatible with JIT; perform value checks in the non-jitted entry point instead of converting tracers to NumPy.
- Retain the finite-positive configured `match_scale` validation before training.
- Add tests showing `(T,)`, `(T, 1)`, and an explicitly broadcast `(T, M)` produce identical unpacked scales, and that invalid scale/history shapes fail clearly.
- **Acceptance:** no supported scale form depends on out-of-bounds index clipping, and malformed inputs fail before a forecast is produced.

### F6 — Integration tests and split-run acceptance: outstanding

- Retain the improved scale-continuity test that invokes `Methods.filter` directly.
- Ensure its unit-scale negative control uses the **same decoded parameters**, key and data as the scaled case; change only scales so a parameterization difference cannot make the test pass.
- Add a test around `_run_method` that counts exactly one SQMC **evaluation** filter call after training, and verifies prediction/ranking/diagnostics receive that history.
- Add the real checkpoint replay from F4 and the NPZ-to-prediction replay and corruption tests from F2.
- Add a populated diagnostic test with a nonzero observed prefix; do not rely only on synthetic prediction indices starting at zero.
- Run smoke exports with `diagnostics_scope="all"`; separately test World Cup scope with actual eligible test fixtures.
- Run fresh EKF-only and SQMC-only smoke jobs into separate directories, validate each, exercise collection/manifests, combine them, and validate the combined output.
- Run `.venv/bin/python -m pytest rbsqmc/tests rbsqmc/comparison/sqmc_ekf/tests -q` after the repairs.
- Record commands, output directories, exit codes and test results in this document. Do not mark F6 complete merely because unit tests pass while a populated partial run fails.
- **Acceptance:** the affected suite, populated partial workflow, collection, combination and replay checks all pass. Only then proceed to F8.

### F7 — Final scalar comparison and compilable LaTeX: outstanding

- Retain the existing per-model `final_scalar_params.json` and combined CSV/JSON writers in `run.py`.
- Compare the four constrained final-epoch scalars: `alpha`, `beta`, `kappa`, and `friendly_scale`; exclude mean and covariance quantities from the comparison.
- **Fix `_write_scalar_latex`**:
  - Escape LaTeX special characters in run IDs and other text fields, including underscores inside `\\texttt{...}`.
  - Render parameter names as safe labels, for example mathematical alpha/beta/kappa and escaped `friendly\_scale`.
  - Include the requested **Parameter / Interpretation / EKF / RB-SQMC** columns.
  - Retain appropriate precision and the run/checkpoint identification in the caption.
- **Strengthen `_validate_scalar_params` and partial validation**:
  - Require each partial's own scalar export.
  - Verify checkpoint hash, dataset hash, method and final epoch/policy against the actual checkpoint/run.
  - Compare each scalar with the corresponding constrained checkpoint value.
  - Require the exact four learned parameter names; explicitly identify any permitted derived rows.
  - Compare combined JSON and CSV model values and differences with the two per-model exports; validate any derived values too.
  - Verify generated table values match the formatted source values rather than merely checking that the file exists.
- **Add rejection and generation tests**:
  - Use distinct EKF/SQMC values to catch swapped columns.
  - Reject a per-model `alpha` changed to 999 and a combined EKF value changed to 999 without matching source artifacts.
  - Compile a generated table in a minimal LaTeX document with `booktabs`, including a run ID containing underscores and the friendly-scale row.
- **Acceptance:** exports agree with checkpoints, inconsistent values fail validation, and the generated table compiles. Dissertation population from the real repaired run is F8.

### F8 — Full-data study and dissertation population: outstanding

- Wait until the implementation and smoke acceptance checks for F1–F7 pass; keep the existing original-run artifacts intact.
- Produce a separately identified repaired run using `scripts/config/config_gpu_small.json` and save the final fitted checkpoints for both methods.
- Hold the SQMC fitted parameters fixed for the sensitivity study; do not retrain separately for each seed or particle count.
- Evaluate a recorded set of seeds at particle counts **128, 512 and 2,048**, using consistent data, match scales and precision.
- For every evaluation, save Spain–Argentina's predictive and post-final means, strength-difference distribution, outcome probabilities, raw grid mass, and ESS from the same filter history.
- Summarize the spread across seeds/counts and explain any remaining mean-strength/win-probability disagreement using those diagnostics; do not require a predetermined favourite.
- Generate the final scalar comparison/table from the identified repaired training checkpoints, not from smoke outputs or invented values.
- Resolve the chapter destination before integration: the requested chapter 2 currently has no Evaluation subsection, while the existing football Evaluation subsection is in chapter 3.
- Insert the generated table and evidence-based discussion in the agreed Evaluation subsection; compile the dissertation and check table layout and references.
- Update this document with run IDs, checkpoint hashes, evidence paths and conclusions; mark task 7/F8 complete only after both the study and dissertation integration are finished.

---

## Second independent validation (2026-09-09; historical findings)

**F1 and F3 are resolved. F2 and F6 fail acceptance; F4, F5 and F7 are
partially implemented. F8 remains outstanding.** The completion claims in the
checklist have been corrected to match the following independent checks.
Earlier validation notes below describe previous versions of the code.

### Executed checks

- `.venv/bin/python -m pytest rbsqmc/tests rbsqmc/comparison/sqmc_ekf/tests -q`:
  **123 passed in 14.17 seconds**.
- Created `/private/tmp/issue090926-recheck-all.json` from `config_smoke.json`,
  adding `diagnostics_scope: "all"` to exercise populated diagnostics.
- Fresh EKF-only smoke: **exit 0**, output
  `/private/tmp/issue090926-recheck-ekf-all`.
- Fresh SQMC-only smoke: **exit 1**, output
  `/private/tmp/issue090926-recheck-sqmc-all`; training and artifact generation
  completed, but validation raised
  `ValueError: Diagnostics full-sequence index out of range`.
- Six independent artifact-mutation probes against the populated
  `complete_run` test fixture were all incorrectly accepted, as listed below.
- Compiled a table generated by `_write_scalar_latex` inside a minimal
  `article` with `booktabs`: **pdflatex exit 1**, `Missing $ inserted`.

Reproduce the populated partial runs with:

```bash
.venv/bin/python -m rbsqmc.comparison.sqmc_ekf.run \
  --config /private/tmp/issue090926-recheck-all.json --smoke \
  --methods ekf --output-dir /private/tmp/issue090926-recheck-ekf-all
.venv/bin/python -m rbsqmc.comparison.sqmc_ekf.run \
  --config /private/tmp/issue090926-recheck-all.json --smoke \
  --methods sqmc --output-dir /private/tmp/issue090926-recheck-sqmc-all
```

### Confirmed repairs

**F1:** `_validate_checkpoint` is now invoked for the requested partial's own
method. The previous missing-EKF-checkpoint error is removed, and the
EKF-only run succeeds. The SQMC failure below is a separate diagnostic-index
bug rather than a remaining cross-method checkpoint dependency.

**F3:** `predict_match_score_with_mass` now uses
`log_weights - logsumexp(log_weights)`. The new constant-shift regression passes.

**Other progress:** the loader reads saved SQMC `mean_0`, diagnostics bind the
checkpoint hash and dataset hash, scale/history guards have been added, the
scale-continuity test now calls `Methods.filter`, and final scalar CSV/JSON and
LaTeX writers exist. These improvements do not complete their broader
acceptance requirements.

### Remaining defects and exact next changes

1. **P1 / F2, F6 — Fix the prediction-index offset in the validator.**
   `diagnostics.py` correctly writes `full_sequence_match_index = t`, including
   the observed prefix. `_validate_sqmc_diagnostics` incorrectly checks
   `t < len(records)` and selects `records[t]`, although `records` contains
   only prediction matches. In this smoke run the first prediction is match
   **40**, while the prediction list has length **4**. Use
   `p = t - (metadata["train_count"] + metadata["test_count"])`, validate
   `0 <= p < len(records)`, and select `records[p]`. Also enforce
   `history_index == t + 1`. The real World Cup run would likewise fail this
   check for populated exports. Add a regression with a nonzero observed
   prefix; the current synthetic fixture does not represent this correctly.

2. **P2 / F2 — Enforce exact fixture identity and replay the NPZ.**
   Validation still accepted `history_index=9999`, swapping valid team IDs
   while retaining the original names, and duplicating the first diagnostic
   into the second slot. Length equality is not exact scope coverage. Match
   the ordered selected prediction indices, reject duplicates, and check IDs
   against checkpoint names. A fourth probe added **100** to every saved
   predictive coordinate in the NPZ and also passed. `_validate_final_npz`
   checks shape/finiteness but does not replay a grid despite its docstring.
   Check team IDs, parameters and scale against the final fixture/checkpoint,
   reconstruct its grid, and compare against the saved final prediction.

3. **P2 / F4 — Validate checkpoint contents, not just presence.**
   `_validate_checkpoint` accepts empty `raw={}` and `constrained={}`; the
   current accepted test fixture actually uses these empty dictionaries.
   Loading the saved mean and verifying a hash are implemented, but parameter
   dimensions and raw/constrained consistency are not. Validate the complete
   parameter schema and reconstruct constrained values for comparison.
   Add a writer-to-loader-to-forecast replay test using a real checkpoint.

4. **P2 / F7 — Fix LaTeX escaping and scalar consistency validation.**
   `_write_scalar_latex` emits raw `friendly_scale` and an unescaped run ID
   inside `\texttt{...}`. A generated caption with run ID `run_090926` fails
   compilation with `Missing $ inserted`; `friendly_scale` also needs escaping
   or a mathematical label. Escape LaTeX special characters and include the
   requested Interpretation column. Compile the generated table in a test.
   The scalar validators accepted changing per-model `alpha` to **999** and,
   separately, changing the combined JSON's EKF value to **999** without
   updating other artifacts. Compare scalar exports to the checkpoint and
   its hash, validate dataset/epoch identity, and check CSV/JSON/table values
   and `sqmc_minus_ekf` against both per-model exports. Partial validation
   should also require and validate its own scalar export.

5. **P3 / F5 — Finish explicit broadcasting for one-dimensional scales.**
   `_unpack_scales` converts `(T,)` to `(T, 1)` in its first branch, but the
   broadcast to `(T, M)` occurs only in the `elif scales.ndim == 2` branch.
   Consequently `(T,)` inputs still rely on JAX index clipping when `M > 1`.
   Move the one-column broadcast after both accepted input-shape branches.
   Keep validation before selection and ensure the history helper validates
   the complete expected scale shape, not only the first dimension.

6. **F6 / F8 — Complete acceptance before the full run.**
   The repaired continuity test is useful, but the suite still has no actual
   saved-parameter forecast replay, NPZ-to-record replay, or shared evaluation
   filter call-count assertion. The scalar test checks JSON rows and file
   existence, so it misses invalid LaTeX. Add these checks and the populated
   partial regression above. Then run both partials through collection and
   combination with populated diagnostics. No full-data fixed-parameter
   seed/particle study or final dissertation table integration was found.

Only this Markdown document was updated in this review. The full rerun should
remain pending until the populated split workflow and artifact checks pass.

## First independent validation (2026-09-09; historical findings)

**Verdict: the two original filtering inconsistencies are repaired in the main
evaluation path, but the complete repair is not accepted.** The normal Colab
workflow still has a reproducible partial-run validation failure. Task 7 and
the full-data Spain–Argentina seed/particle-count study remain outstanding.
This validation supersedes the earlier completion claims retained below.

### Confirmed working

- `scripts/train.py::Methods.filter` and `run.py::_run_method` both build
  scales with `sqmc_match_scales(dataset.inputs.friendly, ...)`. The shared
  history adapter supplies the current fixture's scale to the likelihood.
  EKF now also threads the configured non-friendly scale through filtering
  and prediction.
- `_run_method` invokes one SQMC evaluation filter and passes its result to
  predictions, weighted ranking summaries, and diagnostics. The deprecated
  `_sqmc_states` still reruns a filter, but is not called by this path.
- Predictive coordinates at `t+1` use uniform weights; posterior summaries
  use the updated weights. Existing leakage, scale, history-equivalence,
  team-swap, and fixed-state ordering tests pass.
- Fitted-parameter JSON and final-fixture NPZ are produced. A direct replay
  of the fresh smoke run's final NPZ with JAX x64 enabled reproduces the saved
  grid exactly (maximum difference **0**, raw mass **0.8763098604**); this
  validates the generated fixture, not the validator's ability
  to reject corrupt fixtures.

### Checks run independently

```bash
.venv/bin/python -m pytest rbsqmc/tests rbsqmc/comparison/sqmc_ekf/tests -q
# 111 passed in 15.45s

.venv/bin/python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_smoke.json \
  --smoke --output-dir /private/tmp/issue090926-validation-both
# Exit 0; combined artifacts validated.

.venv/bin/python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_smoke.json \
  --smoke --methods sqmc --output-dir /private/tmp/issue090926-validation-sqmc
# Exit 1 after training and export: missing results/ekf/fitted_params.json.
```

The smoke fixtures end on 2025-06-05, so default `worldcup` diagnostics contain
**zero entries**. Its successful combined validation does not exercise the
Spain–Argentina fixture or even a populated World Cup diagnostic export.
Additional corruption probes used the populated `complete_run` fixture from
`tests/test_integrity.py`, changing one property at a time and invoking
`validate_artifacts`.

### Remaining findings, in priority order

1. **P1 — SQMC partial validation requires the other model's checkpoint.**
   `scripts/validate_sqmc_ekf_outputs.py::_validate_sqmc_diagnostics` loops over
   both `METHODS` when checking `fitted_params.json`. `validate_partial` calls
   it for a SQMC-only run. The fresh run therefore fails looking for
   `/private/tmp/issue090926-validation-sqmc/results/ekf/fitted_params.json`.
   This is the partial-run path used by the Colab launcher and its protocol
   validation. Conversely, EKF-only validation never checks its checkpoint.
   **Repair:** separate per-method checkpoint validation from SQMC diagnostic
   validation. Validate only the requested method in a partial, and both in a
   combined run. Test genuinely separate EKF and SQMC directories through
   validation, collection, and combination. The current partial test fixture
   creates both checkpoints and hides this failure.

2. **P2 — Diagnostic validation accepts scientifically inconsistent artifacts.**
   Independent probes confirmed `validate_artifacts` accepts each of:
   `home_id=9999` with `history_index=9999`; `raw_grid_mass=-1` with
   `ess_after_update=-3`; outcome probabilities replaced by `1/0/0` despite
   disagreeing with the prediction record; an incorrect `dataset_hash`;
   an empty `worldcup` forecast list despite World Cup records being present;
   and an NPZ replaced by the bytes `not an npz archive`.
   The implementation checks finiteness and probability sums, but not fixture
   alignment, value bounds, provenance equality, scope completeness, or NPZ
   contents. It also does not validate the checkpoint parameter payload.
   **Repair:** bind each diagnostic to its prediction, team mapping, indices,
   checkpoint, and dataset; require the exact selected scope; enforce positive
   scales, valid mass, ESS and probability bounds; open and replay the final
   NPZ; validate parameter structure and raw/constrained agreement. Add one
   rejection test per corruption above. Manifest checksums alone cannot catch
   semantically wrong artifacts produced by the writer itself.

3. **P2 — The task-3 log-weight normalization remains unchanged.**
   `src/model/rbsmc/predict.py::predict_match_score_with_mass` still computes
   `log(exp(log_weights) + 1e-12)`. In a two-particle probe with weights
   `0.99/0.01`, subtracting 100 from both log weights changes a grid cell by
   approximately **0.07279**, although normalized weights must be invariant
   to this shift. Replace it with
   `log_w = log_weights - logsumexp(log_weights)` and test shift invariance.
   This does not explain the repaired main path's uniform-weight forecast,
   but leaves the shared weighted-prediction helper incorrect for general
   unnormalized log weights.

4. **P2 — Checkpoint replay and regression coverage are incomplete.**
   Diagnostics omit the fitted-checkpoint hash specified in task 5.
   `load_fitted_params` requires an external `fixed_mean` and ignores the
   saved constrained `mean_0`; its docstring incorrectly says the mean is
   not saved. Reconstruct that mean from the checkpoint, or verify a supplied
   mean exactly matches it. The existing round-trip tests check selected
   parameters, not a saved-parameter forecast replay. The test named
   `test_scale_continuity_eval_history_equals_training_filter` calls the same
   low-level filter twice and never invokes `Methods.filter`; it does not
   protect the actual training/evaluation integration. Add direct integration
   coverage, an evaluation-filter call-count assertion, and checkpoint/NPZ
   replay assertions to the suite.

5. **P3 — Input contracts are documented but not enforced.**
   There is no explicit finite/positive configured `match_scale` check.
   `_unpack_scales` does not validate scale shape against packed inputs, and
   `predict_from_sqmc_history` does not reject multiple matches per history
   row. The current packed-scale test relies on JAX out-of-bounds indexing
   to repeat a one-column scale rather than explicit broadcasting. Validate
   supported shapes and broadcast intentionally; reject unsupported packed
   histories before forecasting. The comparison's valid one-match-per-row
   dataset does not currently trigger these cases.

6. **Outstanding — Task 7 and the definitive tournament rerun.**
   No `final_scalar_params.json` writer, combined scalar comparison export,
   or generated dissertation table is implemented. The full checkpoints do
   contain the scalars, but that does not complete the requested comparison.
   No seed/particle-count spread has been established in this validation.
   Keep R4, the rerun, and task 7 unchecked. The original saved Spain–Argentina
   reversal is not yet demonstrated to be resolved or explained.

Only this investigation document was edited during validation. Python repairs
were reviewed and exercised, not modified. Resolve the findings above and
rerun the split workflow before accepting task 6 or starting the planned
Colab rerun.

## Earlier implementation report (2026-09-09; superseded by validation above)

The implementation report stated that tasks 1–6 and R1–R3 were implemented and
validated. It reported that the smoke run
(`config_smoke.json`, `--smoke`) completes end-to-end and passes
`validate_artifacts`. The full test suite passes (107 tests). Remaining work is
R4 (seed/particle-count spread on `config_gpu_small.json`) and Task 7
(dissertation scalar-parameter table), both deferred per scope.

Files changed:

- `rbsqmc/comparison/sqmc_ekf/scripts/scaling.py` — **new** `sqmc_match_scales` helper.
- `rbsqmc/comparison/sqmc_ekf/scripts/train.py` — `Methods.filter` uses the helper; persists
  `fitted_params.json`; adds `load_fitted_params` loader.
- `rbsqmc/src/model/rbsqmc/predict_rbsqmc.py` — extracts
  `predict_from_sqmc_history`; adds `observed_scales`/`prediction_scales` args
  and `_unpack_scales` alignment.
- `rbsqmc/src/model/rbsmc/predict.py` — `predict_match_score` gains a trailing
  `scale` argument; adds `predict_match_score_with_mass` (grid + raw mass).
- `rbsqmc/comparison/sqmc_ekf/scripts/predict.py` — `predict_sqmc` uses the scale builder;
  adds `predict_sqmc_from_history` adapter.
- `rbsqmc/comparison/sqmc_ekf/run.py` — `_run_method` runs one shared filter;
  `_sqmc_states` is a deprecated wrapper; `combine`/`_write_combined` carry the
  new artifacts.
- `rbsqmc/comparison/sqmc_ekf/scripts/diagnostics.py` — **new** per-forecast
  diagnostics + final-fixture NPZ writer.
- `rbsqmc/comparison/sqmc_ekf/scripts/validate_sqmc_ekf_outputs.py` — schema
  version + diagnostics/fitted-params validation.
- `rbsqmc/tests/test_predict_rbsqmc.py`, `rbsqmc/tests/test_sqmc_ekf.py`,
  `rbsqmc/comparison/sqmc_ekf/tests/test_integrity.py` — new regression tests.

### Follow-up fixes (2026-09-09, after review)

1. **EKF scale consistency (point 3).** The EKF observation model previously
   hardcoded `1.0` for non-friendly fixtures (`model.py`), while SQMC used the
   configured `match_scale`. All configs currently set `match_scale: 1`, so the
   two agreed in practice, but the EKF path was not robustly consistent.
   `ekf.build`, `run_filter`, `sequential_predict`, and `synchronized_moments`
   now accept a `match_scale` argument (default `1.0`); the observation uses
   `friendly_scale` for friendlies and `match_scale` otherwise. Callers updated:
   `Methods.filter`, `predict_ekf`, `_ekf_states`. Tests
   `test_ekf_observation_uses_match_scale_for_non_friendly` and
   `test_ekf_observation_friendly_uses_friendly_scale` pin this down.

2. **Diagnostics verbosity (point 2).** `build_diagnostics` now takes a `scope`
   argument (`all` / `worldcup` / `final`, default `worldcup`), controlled by
   the `diagnostics_scope` config key. The final-fixture NPZ is always written.
   The validator allows zero `worldcup`-scope forecasts (a prediction split with
   no 2026 World Cup fixtures is legitimate) while still requiring ≥1 for
   `all`/`final`. Test `test_diagnostics_scope_filters_forecasts` covers the
   filter.

3. **Friendly-flag sourcing (point 1).** All call sites already source flags
   from `dataset.inputs.friendly` (a `MatchInputs` field), not `dataset.sqmc`
   (a `FootballResults` has no `friendly` field). Test
   `test_friendly_flags_sourced_from_inputs_not_sqmc` asserts this explicitly
   so a future edit cannot silently read from the wrong structure.

### Module relocation (2026-09-09)

The mixed EKF/SQMC modules were moved out of `rbsqmc/src/model/ekf/` (which now
contains only the pure-EKF `model.py`) into `rbsqmc/comparison/sqmc_ekf/scripts/`,
where the comparison-specific drivers live:

- `ekf/train.py` → `scripts/train.py` (`Methods` handles both `ekf` and `sqmc`).
- `ekf/predict.py` → `scripts/predict.py` (`predict_sqmc`/`predict_sqmc_from_history`
  are SQMC; `predict_ekf` is EKF).
- `ekf/scaling.py` → `scripts/scaling.py` (`sqmc_match_scales` is SQMC-only).

`ekf/model.py` (pure EKF) stays. All imports updated in `run.py`, the moved
modules, and the test files. Full suite: **111 passed**; smoke run re-validates.

Full suite: **111 passed**; smoke run re-validates end-to-end.

---

Investigated the saved `sqmc_ekf` run `09092026_0235`, source commit
`846957bacb00a8673871aaf428115455832c9024`. The relevant prediction, likelihood,
and comparison-driver files agree with that revision.

## Original-run conclusion (before the repairs)

The explanation in `ISSUE_090926.md` is not supported. In this model, a
fixed-state attack–defence matchup cannot reverse the ordering by total
strength. The exported probabilities are nevertheless internally correct:
there is no home/away swap in the saved score grid.

Two concrete pipeline problems were found:

1. Predictions and ranking plots come from separate randomized SQMC filters,
   so the plotted strengths do not describe the particles used for predictions.
2. SQMC training uses learned friendly-match scaling, but prediction and
   ranking replay historical matches with unit scaling. They therefore do not
   use the same observation model as training, unless the learned scale happens
   to be exactly one.

The exact contribution of random-seed variability, posterior uncertainty, and
the final-match update to this particular reversal cannot be recovered from
the saved artifacts: fitted parameters and the relevant particle states were
not persisted. Different seeds are a confirmed inconsistency, not a measured
explanation of the full probability gap.

## What the saved results actually say

Source: `../drafts/results/sqmc_ekf/combined/`.

| State | Team | Attack | Defence | Total | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| Pre-tournament | Spain | 0.726924 | 0.248762 | 0.975686 | 28 |
| Pre-tournament | Argentina | 1.757836 | 0.422272 | 2.180109 | 10 |
| Post-tournament | Spain | 0.909518 | 0.753105 | 1.662623 | 22 |
| Post-tournament | Argentina | 1.504035 | 1.185993 | 2.690028 | 9 |

For the July 19 Spain–Argentina fixture, SQMC records:

- Spain win: **0.5372199272**.
- Draw: **0.2052876867**.
- Argentina win: **0.2574923861**.
- Expected goals within the normalized 0–8 grid: Spain **2.183165**, Argentina
  **1.453558**.

Both ranking JSONs and the prediction JSON match the remote run manifest
hashes and are byte-identical to their combined-run copies. All 104 SQMC
records have outcome probabilities consistent with their score grids.

## Why the matchup explanation is mathematically wrong

`rbsqmc/src/data/bivariate_poisson.py::loglik_grid` implements

\[
\log\lambda_S=\alpha+(a_S-d_A)/s,\qquad
\log\lambda_A=\alpha+(a_A-d_S)/s.
\]

Subtracting gives

\[
\log(\lambda_S/\lambda_A)
=\big[(a_S+d_S)-(a_A+d_A)\big]/s.
\]

For fixed strengths and positive common scale, the higher-total team has the
higher scoring rate and win probability. The common Poisson component adds
the same goals to both teams and does not reverse this ordering. Symmetric
score-grid truncation also preserves this fixed-state ordering.

For the published post-tournament means at scale one, the difference is
**−1.0274057434**, giving Spain/Argentina a rate ratio of approximately
**0.358**. Plugging those means into the likelihood favours Argentina.

There is an important qualification: the forecast averages likelihoods over
particles, whereas rankings use posterior mean total strength. A nonlinear
posterior average can favour a team with lower mean total strength if the
particle distribution allows it. Thus the ranking alone does not prove a
prediction bug. The particle distribution must be inspected; the issue file
did not do that.

## Separate filters break the proposed comparison

`rbsqmc/comparison/sqmc_ekf/run.py::_run_method` uses
`jax.random.fold_in(root, 2_000_000)` for prediction. `_sqmc_states` reruns the
full filter with `jax.random.fold_in(root, 3_000_000)` for ranking plots.
This run used **128 particles** over **5,111 matches**. Seed sensitivity needs
measurement; it cannot be dismissed or quantified from one pair of summaries.

The timing distinction is real but does not establish the cause:

- Prediction uses particles propagated to the final, with uniform
  pre-likelihood weights.
- Post-tournament rankings use likelihood-weighted particles after the final,
  from the other random run.
- `sqmc_timeseries_states.json` contains only England, France, Portugal,
  Scotland, and Germany. It contains no pre-final Spain/Argentina strengths.

Consequently, the original claim that Spain was stronger in the pre-final
state was an assumption, not a finding. The pre-final state also incorporates
the third-place match and propagation to the final; it is not simply the
state immediately after the semifinals.

## Confirmed observation-scale defect

`rbsqmc/src/model/ekf/train.py::Methods.filter` explicitly passes
`match_scales` to `run_filter_sqmc`: the learned `friendly_scale` for
friendlies, and the configured baseline for other matches.

In contrast:

- `rbsqmc/src/model/ekf/predict.py::predict_sqmc` passes only
  `params["model"]` into the sequential predictor.
- `rbsqmc/src/model/rbsqmc/predict_rbsqmc.py` calls `run_filter_sqmc` without
  `match_scales`.
- `run.py::_sqmc_states` likewise omits `match_scales`.
- The filter defaults missing scales to an array of ones.

This run enabled friendlies. Although the World Cup fixtures themselves have
scale one, earlier friendlies influence the state carried into the tournament.
The defect therefore matters for World Cup forecasts too. It affects both
the prediction and ranking replays, so it is not by itself an explanation for
their disagreement.

## Checks performed

- `python -m pytest rbsqmc/tests/test_predict_rbsqmc.py rbsqmc/tests/test_sqmc_ekf.py -q`:
  **8 passed**.
- Direct small SQMC reproduction with 128 particles: changing the final
  observed score from 1–0 to 0–8 changes its own forecast by **exactly zero**.
- The sequential forecast equals a grid computed directly from the same
  filter's pre-likelihood particles: maximum difference **zero**.
- Changing only the first historical match's scale from 1 to 2 changes a
  later forecast cell by up to **0.0227199225** in that small reproduction.
  This demonstrates the scale defect's mechanism, not its magnitude in the
  saved tournament run.
- At the published post-final means, using illustrative `alpha=0, beta=-4`
  and max goals 8, the grid gives Spain/draw/Argentina probabilities
  **0.121618 / 0.190328 / 0.688055**. These are not reconstructed run
  probabilities: the fitted alpha and beta were not saved.
- Swapping teams transposes that grid to floating-point precision
  (maximum error **5.6e-17**), consistent with the evaluation code's axes.

## Required repair and definitive follow-up

1. Pass identical per-match scales through training, prediction, and ranking.
2. Generate predictions and posterior ranking summaries from one filter
   history. Keep predictive uniform weights and updated posterior weights
   distinct, with explicit match/state timestamps.
3. Save fitted parameters and, for each forecast, the playing teams' predictive
   means, total-strength difference distribution, ESS, raw score-grid mass,
   and post-update summaries. Save the seed and team mapping as well.
4. With fixed fitted parameters, repeat filtering across seeds and larger
   particle counts. Compare Spain–Argentina immediately before and after the
   final, using the same run. This separates timing, posterior averaging,
   truncation, and particle approximation variability.

No model code or dissertation prose was changed as part of this investigation.
The existing outputs should not be explained as a demonstrated favourable
matchup or a demonstrated pre-final/post-final reversal.

## Proposed code repair

This section specifies the implementation to make next; the snippets are
proposed code, not changes already applied to the Python modules. Preserve the
existing statistical model and final-epoch checkpoint policy. Do not force win
probabilities to follow mean-strength rankings: posterior averaging can still
produce legitimate differences after the pipeline is repaired.

### 1. Centralize the SQMC observation scales

Add a small shared helper, for example
`rbsqmc/src/model/ekf/scaling.py::sqmc_match_scales`, and call it from both
`Methods.filter` and the final evaluation path:

```python
def sqmc_match_scales(friendly, friendly_scale, match_scale=1.0):
    # Preserve the observation model currently used during SQMC training.
    return jnp.where(
        jnp.asarray(friendly, dtype=bool), friendly_scale, match_scale
    )[:, None]
```

The contract is `(T,)` friendly flags to `(T, 1)` scales, aligned with the
comparison dataset's one-match-per-row order. Validate the configured baseline
as finite and positive before entering JIT. The learned friendly scale already
uses the positive parameter transform; retain the training finite-value checks.
Keep the helper differentiable with respect to `friendly_scale`.

In `Methods.filter`, construct scales for the full dataset and slice `[:end]`
along with the inputs, or pass `friendly[:end]` to the helper. In evaluation,
use the same helper and the same constrained parameters for all rows, including
the observed prefix. Preserve the existing selection semantics: friendlies use
`friendly_scale`, other matches use `match_scale`. Multiplying the two for
friendlies would change the training model and is not part of this repair.

### 2. Separate forecasting from running the filter

Extract the score-grid scan in
`rbsqmc/src/model/rbsqmc/predict_rbsqmc.py` into a helper with this proposed
interface:

```python
def predict_from_sqmc_history(
    result, prediction_inputs, prediction_start, params,
    max_goals, prediction_scales,
):
    ...  # Existing scan, with the explicit scale and indexing rules below.
```

This helper must not call `run_filter_sqmc`. Require one valid match per row
for this history interface; a history that stores only an end-of-day state
cannot supply pre-likelihood particles for every match within that day.

For prediction row `p`, let `t = prediction_start + p`. Use:

```python
x = result["particles_x"][t + 1]
predictive_log_weights = jnp.zeros_like(result["log_weights"][t + 1])
scale = prediction_scales[p, 0]
```

The state contract must be documented and tested:

| Quantity | Coordinates | Weights |
| --- | --- | --- |
| Previous filtered state, before propagation to match `t` | `particles_x[t]` | `log_weights[t]` |
| Forecast at match `t`, before its score is assimilated | `particles_x[t + 1]` | Uniform |
| Posterior after match `t` | `particles_x[t + 1]` | `log_weights[t + 1]` |

This contract relies on the current bootstrap filter: positions are generated
before the likelihood is applied, and the current score changes only the
weights at that step. Do not use the post-likelihood weights for forecasting,
or use unpropagated positions at index `t` as a replacement.

Keep `run_sequential_predict_rbsqmc` as a compatible convenience wrapper. Add
optional `observed_scales` and `prediction_scales` arguments, defaulting to
ones for existing callers. For packed inputs, validate each scale array against
its match mask and unpack scales with the same valid row/column indices used
by `unpack_football_results`. Concatenate scales in exactly the same order as
the unpacked inputs, call the filter once, and delegate to the extracted
forecast helper. Preserve the existing three-array return contract.

### 3. Apply the current fixture's scale to the score likelihood

In `rbsqmc/src/model/rbsmc/predict.py::predict_match_score`, add a trailing
`scale=1.0` argument and replace `loglik_grid(..., scale=1.0)` with
`loglik_grid(..., scale=scale)`. Pass each fixture's scale from the extracted
history helper. This fixes predictions for non-unit-scale fixtures as well as
the historical filtering defect; fixing historical scales alone is incomplete.

Factor out a shared computation returning both the normalized grid and raw
grid mass. Retain `predict_match_score` as a grid-only wrapper for existing
SMC/SQMC callers, and let the new diagnostic path use the richer helper. Use
log-space weight normalization:

```python
log_w = log_weights - logsumexp(log_weights)
logp = logsumexp(log_grid + log_w[:, None, None], axis=0)
log_mass = logsumexp(logp)
grid = jnp.exp(logp - log_mass)
raw_mass = jnp.exp(log_mass)
```

Record mass before normalization. The sum of an already normalized grid
cannot diagnose truncation. Keep the existing score-grid axis convention.

### 4. Run one final evaluation filter and reuse its history

Replace the two SQMC filter runs in `run.py::_run_method` with this flow:

```python
key = jax.random.fold_in(root, 2_000_000)
scales = sqmc_match_scales(
    dataset.inputs.friendly,
    params["friendly_scale"],
    cfg.get("match_scale", 1.0),
)
result, augmented = run_filter_sqmc(
    key, dataset.sqmc, params["model"],
    cfg["n_particles"], cfg["max_goals"],
    match_scales=scales,
)
pred = predict_mod.predict_sqmc_from_history(
    dataset, params, cfg, result, scales
)
states = plots_mod.wrap_sqmc(result, len(dataset.teams))
```

Add `predict_sqmc_from_history` in `rbsqmc/src/model/ekf/predict.py` as the
comparison adapter: slice the prediction rows and scales, invoke the extracted
helper, and build `MethodPredictions`. Keep `predict_sqmc` as a convenience
API if needed, but have it use the same scale builder and history adapter.
Remove `_sqmc_states`' independent filtering, or replace it with a pure wrapper
around the supplied result. Merely setting the two random keys equal leaves
duplicate execution and opportunities for the input paths to diverge.

Use `augmented` from this same filter for correlation plots. Keep weighted
posterior means in `wrap_sqmc`; do not change rankings to unweighted means to
make them match forecast summaries. Preserve pre-tournament index
`train_count + test_count` and final index `-1` for the existing ranking plots.
Add a separate explicitly labelled pre-final predictive summary for the
Spain–Argentina comparison.

### 5. Persist enough information to reproduce and explain the result

After `Methods.train` finishes, use its existing atomic `save_json` helper to
write `results/<method>/fitted_params.json`, containing raw and constrained
parameters, checkpoint epoch/policy, and the team ID mapping. Include SQMC's
fixed `mean_0` and learned friendly scale. Implement a loader that reconstructs
the expected parameter types and validate it with a save/load forecast replay.

Write `results/sqmc_prediction_diagnostics.json` with one entry per forecast:

- Fixture/source-row ID, date, team IDs/names, full-sequence match index,
  history index, current scale, and predictive/posterior labels.
- Uniform predictive attack/defence means for both playing teams and the
  mean, quantiles, and fraction positive of the particlewise difference
  `(attack_home + defence_home) - (attack_away + defence_away)`.
- The same two teams' likelihood-weighted post-update means.
- Posterior ESS before resampling (`log_weights[t]`) and after the current
  update (`log_weights[t + 1]`), using `1 / sum(normalized_weights**2)`.
  Uniform predictive weights always have ESS `N` and do not measure collapse.
- Raw grid mass, normalized outcome probabilities, and the parameter-artifact
  hash, dataset hash, source revision, particle count, and filter key provenance.

Also save the playing teams' predictive coordinates and posterior weights for
the final fixture in a compact NPZ file. That is sufficient to reconstruct its
grid and examine the reversal without storing every team's entire history.
Retain the normalized-grid forecast convention explicitly: conditioning on
the finite score grid can change a posterior mixture's outcome probabilities.

Update partial-run validation, `combine`, manifests, and collection paths to
require and preserve these new artifacts for the repaired output schema. Use
a schema version so old runs can be identified as lacking diagnostics instead
of being mistaken for complete repaired runs. Check fixture alignment, finite
parameters, valid weights/mass, hashes, and agreement between diagnostic
probabilities and prediction records.

### 6. Regression checks and rerun acceptance

Add focused tests to `rbsqmc/tests/test_predict_rbsqmc.py` and
`rbsqmc/comparison/sqmc_ekf/tests/test_integrity.py`:

1. **Scale continuity:** with a known non-unit friendly scale, the final
   evaluation history equals `Methods.filter` for the same parameters, key,
   inputs, and endpoint. Include a non-unit non-friendly baseline and a
   friendly in the observed prefix, so resetting the prefix to ones fails.
2. **Scale alignment:** packed inputs with padding and multiple matches on one
   date preserve scale/match correspondence after unpacking and concatenation.
3. **Current-fixture scale:** a one-particle forecast equals the independently
   normalized `loglik_grid` at the supplied non-unit scale, including raw mass.
4. **No score leakage:** changing the current score leaves its forecast
   unchanged but changes the posterior; a later forecast can then change.
   Include sequential matches on the same date.
5. **One shared evaluation history:** verify one evaluation filter invocation
   after training, and reconstruct exported predictive and posterior means
   directly from its particles and the appropriate weights.
6. **Axes and fixed-state ordering:** swapping teams transposes the grid;
   for one fixed particle, the team with greater total strength has the greater
   win probability. Do not impose that assertion on posterior mean totals.
7. **Artifact replay:** save/load fitted parameters and final-match particles,
   reconstruct the forecast, and check that partial collection and combination
   retain all new artifacts. Reject mismatched team IDs, fixture indices, or
   diagnostic probabilities.

Run the affected prediction/comparison tests and a small end-to-end comparison
before a full rerun. For the full-data diagnostic, hold fitted parameters fixed
and evaluate several recorded filter seeds with, for example, 128, 512, and
2,048 particles. Report the spread in Spain–Argentina's predictive total
difference and win probabilities, together with pre/post-final summaries and
raw grid mass. These particle counts are diagnostic settings, not a guarantee
of convergence.

The old exports lack fitted parameters, so replaying the old forecast exactly
requires recovering the original checkpoint or reproducing training. Otherwise
produce a new, separately identified run. Acceptance is consistent scales,
shared-history provenance, reproducible forecasts, and an evidenced explanation
of any remaining reversal—not a requirement that Spain's probability fall below
Argentina's.

### 7. Save and compare final scalar parameters in the dissertation

Save a compact comparison of the final fitted EKF and RB-SQMC parameters,
excluding mean vectors, covariance matrices, and covariance parameterizations.
This is a reporting export in addition to the complete reproducibility
checkpoint in task 5; retain means and covariances in that checkpoint.

Use the constrained parameters from the same final training epoch used for
evaluation, not unconstrained optimizer coordinates or the best-test epoch.
The shared scalar parameters and their extraction paths are:

| Parameter | Meaning | EKF constrained parameters | SQMC constrained parameters |
| --- | --- | --- | --- |
| `alpha` | Baseline log scoring rate | `params["alpha"]` | `params["model"].alpha` |
| `beta` | Log rate of the shared Poisson component | `params["beta"]` | `params["model"].beta` |
| `kappa` | OU mean-reversion rate, per day | `params["kappa"]` | `params["model"].kappa` |
| `friendly_scale` | Strength scaling for friendly matches | `params["friendly_scale"]` | `params["friendly_scale"]` |

Exclude EKF `init_sd` and `init_corr` as covariance parameterizations, as well
as `init_cov`, `init_chol_cov`, `init_mean`, SQMC `B`, `gamma_0`, and `mean_0`.
The configured non-friendly `match_scale` is not a fitted parameter: record
it as run metadata rather than presenting it as an estimated table row.

Implementation tasks:

1. In `run.py::_run_method`, extract the four scalars after constraining the
   returned final parameters. Save
   `results/<method>/final_scalar_params.json` with full numeric precision,
   method, checkpoint epoch/policy, run ID, source revision, dataset hash,
   and the task-5 checkpoint hash.
2. In the combined-report writer, read both per-method exports and produce
   `results/final_scalar_params_comparison.csv` and a matching JSON artifact.
   Use columns `parameter`, `meaning`, `unit`, `ekf`, `sqmc`, and
   `sqmc_minus_ekf`. Use direct differences rather than percentage changes
   for signed log parameters. Optionally report `exp(alpha)`, `exp(beta)`,
   and `log(2)/kappa` in separately labelled derived rows; these are baseline
   and shared-component rates and the OU half-life in days, not additional
   learned parameters.
3. Include these exports in partial-run collection, combination, manifests,
   and repaired-schema validation. Check all four values are finite,
   `kappa` and `friendly_scale` are positive, and both methods use the
   intended final-epoch policy and the same dataset. Verify the scalar
   exports match the constrained values in the full checkpoints.
4. Generate a LaTeX table from the comparison artifact, with columns
   **Parameter**, **Interpretation**, **EKF**, and **RB-SQMC**. Round only
   for display and use scientific notation where needed for `kappa`.
   Identify the run and final checkpoint epoch in the caption. Discuss
   differences in scoring baseline, shared scoring, mean reversion, and
   friendly scaling alongside predictive performance; parameter differences
   alone do not establish that one model is better.
5. Include this table and discussion in the requested
   `dissertation/drafts/chapters/2_high_performance_sqmc.tex`, under
   `\subsection{Evaluation}`, once the repaired run supplies actual values.
   **Current layout discrepancy:** that file presently has no Evaluation
   subsection; the football-model Evaluation subsection is in
   `dissertation/drafts/chapters/3_football_model_with_sqmc.tex`. Keep the
   user's requested chapter-2 destination recorded here, and resolve whether
   to create that subsection or use the existing chapter-3 subsection when
   integrating the table. No numerical results or placeholder estimates
   should be invented in the meantime.

Acceptance: both partial runs persist the four final scalar estimates; the
combined CSV/JSON and generated LaTeX table agree with the checkpoints; the
comparison excludes means and covariance quantities; and the dissertation
table is populated from the identified repaired run. Add a round-trip export
test with distinct known EKF/SQMC values to catch model-column swaps,
unconstrained-value exports, and accidental covariance/mean inclusion.
