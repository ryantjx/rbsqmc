# Review of `Performance comparison`

Review date: 2026-09-10  
Chapter: `dissertation/drafts/chapters/3_football_model_with_sqmc.tex`  
Section: `\section{Performance comparison}`

## Executive assessment

The section is not ready to freeze. Its overall narrative is reasonable, but the
tables, final-match discussion, ranking comparison, and several figure captions
are tied to the earlier dissertation SQMC run (`source_commit` `846957b`, 128
particles, 10 gradient replicas). The newer repaired run
`09092026_1431` uses `source_commit` `336842f`, 512 particles, 15 gradient
replicas, a shared evaluation filter, and consistent observation scaling. It
produces materially different SQMC predictions and rankings.

The EKF results are numerically unchanged between the two bundles, so the
discrepancies are SQMC-specific rather than a dataset or EKF regeneration
issue.

## Recommended revisions

1. Add and use an evaluation-only checkpoint replay. Save the selected
   best-test parameter checkpoint, then regenerate the full filter, forecasts,
   state trajectories, rankings, and figures from that checkpoint without
   retraining. The detailed procedure and public repository links are given
   below.
2. Choose the authoritative run. The repaired `09092026_1431` bundle should
   be preferred for the main comparison; preserve the dissertation bundle as
   an explicitly labelled historical run.
3. Regenerate all tables, prediction figures, ranking figures, and the scalar
   parameter table from that same run.
4. Correct the 104-fixture statement and the Brier equation.
5. Correct the correlation and timeseries figure descriptions.
6. Rewrite the Spain--Argentina paragraph using the repaired probabilities.
7. Separate predictive, posterior, and ranking estimands in the discussion.
8. Weaken claims that the comparison isolates only correlation or that the
   SQMC run is converged.
9. Verify the external FIFA-ranking source and date independently before
   retaining the FIFA comparison as evidence.

## Critical findings

### 1. The reported SQMC results are stale

The current section reports the dissertation bundle:

| Quantity | Section currently reports | Repaired 09/09 run |
|---|---:|---:|
| SQMC particles | 128 | 512 |
| SQMC gradient replicas | 10 | 15 |
| Final train log-likelihood | -14,991 | -14,649.7 |
| Final test log-likelihood | -1,005.2 | -1,005.4 |
| Best test log-likelihood | -970.9 at epoch 26 | -959.96 at epoch 49 |
| SQMC Brier score | 0.6521 | 0.6878 |
| SQMC Brier skill vs. uniform | 0.0218 | -0.0316 |
| SQMC mean predictive log-score | -3.523 | -3.664 |
| SQMC outcome accuracy | 0.5096 | 0.4712 |

The EKF values remain 0.5442 Brier, 0.1838 Brier skill, -3.131 mean
log-score, and 0.5288 outcome accuracy.

The newer run therefore strengthens the negative predictive result for SQMC,
but its ranking conclusions cannot be substituted into the existing prose
without updating the tables and figures together.

Evidence: [current performance metrics](../../rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/combined/results/performance_metrics.csv),
[dissertation performance metrics](../drafts/results/sqmc_ekf/combined/results/performance_metrics.csv),
[current configuration](../../rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/comparison_config.json),
and [dissertation configuration](../drafts/results/sqmc_ekf/comparison_config.json).

**Remedy.** Use `09092026_1431` as the authoritative comparison run. Use its
evaluation-only replay directory to regenerate every section table, figure,
caption, and result statement. Record the source commit, configuration,
dataset hash, particle count, replica count, and checkpoint policy alongside
each dissertation result. Change the training pipeline to save and evaluate
the best held-out-test checkpoint rather than only the final epoch.

### 2. The prediction-period count is wrong

The section says that the prediction period contains four World Cup fixtures.
The artifacts contain 104 World Cup fixtures and all headline metrics are
computed over 104 scored matches. Four is presumably a leftover description of
the knockout-stage subset or an earlier fixture file.

Replace “the four World Cup 2026 fixtures used for the headline scoring” with
“the 104 World Cup 2026 fixtures in the prediction period”. If the knockout
stage is discussed separately, state that it is a seven-match subset rather
than the headline evaluation set.

**Remedy.** Change the sentence, table captions, and any metric descriptions
to identify 104 scored World Cup fixtures as the headline evaluation set.
Label the seven knockout fixtures separately wherever they are used for
illustration or heatmaps. Cross-check the count against
`dataset_metadata.json` and the generated prediction records before compiling
the chapter.

### 3. The Brier-score equation has an indexing error

The text defines the score using

```tex
\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{N} (p_{ij}-o_{ij})^2
```

The inner sum is over the three outcome classes, not over the number of
matches. It should be, for example,

```tex
\operatorname{BS}
= \frac{1}{N}\sum_{i=1}^{N}\sum_{c\in\{H,D,A\}}
  (p_{ic}-o_{ic})^2.
```

The implementation and the reported uniform reference (2/3) use the
three-outcome definition, so this is a notation error rather than a scoring
implementation error.

**Remedy.** Replace the displayed equation with the corrected expression,
using an outcome index such as
`c\in\{H,D,A\}`. Leave the scoring implementation unchanged, but add a short
sentence that the sum is over the three outcomes and that the uniform
reference is `2/3` under this convention.

### 4. The correlation figure is misdescribed

The prose and caption say that Figure~`\ref{fig:topn-correlation}` shows the
correlation between top-(N) rankings before and after the World Cup. The
actual image is titled:

> Top 5 Positively / Top 5 Negatively Correlated Team Pairs (Final State)

It displays pairwise latent-state correlations such as `England - France` and
`Brazil - Haiti`; it does not display rank correlations as a function of
(N). The figure should either be replaced with a genuine rank-stability
plot, or the prose should describe the existing figure as final-state
posterior correlations between team-strength coordinates. The current label
“top-(N) rankings” is scientifically misleading.

Evidence: [figure](../drafts/results/sqmc_ekf/combined/images/sqmc_correlation_topn_bar.png).

**Remedy.** Retain the existing image and change the paragraph, caption, and
figure label to “final-state posterior correlations between team-strength
coordinates”. Remove every reference to rank correlations or top-
`N` ranking stability from this figure’s description.

### 5. The timeseries paragraph claims an uncertainty band that is not shown

The section says that RB-SQMC reports “the weighted particle mean with the
particle spread as an uncertainty band”. The supplied SQMC timeseries figure
shows lines only; there is no shaded particle-spread band. It also plots five
teams selected by final total strength, not one “representative team”.

Choose one of the following:

- add and validate a genuine uncertainty band, with its definition stated;
- change the prose to say that the figure shows weighted posterior means for
  the final top-five teams; or
- replace the figure with a clearly identified single-team plot.

Evidence: [SQMC timeseries figure](../drafts/results/sqmc_ekf/combined/images/sqmc_timeseries_states.png)
and [exported timeseries data](../drafts/results/sqmc_ekf/combined/images/json/sqmc_timeseries_states.json).

**Remedy.** Change the chapter text and caption to “weighted posterior means
for the final top-five teams”. Remove the claims about an uncertainty band and
a single representative team. Do not describe the existing line plot as an
uncertainty visualization.

### 6. The final Spain--Argentina paragraph is now obsolete

The EKF values are unchanged, but the SQMC final forecast changes sharply:

| Final-match quantity | Dissertation bundle | 09/09 run |
|---|---:|---:|
| Spain win | 0.5372 | 0.1463 |
| Draw | 0.2053 | 0.3842 |
| Argentina win | 0.2575 | 0.4695 |
| SQMC modal score | 2--1 | 0--0 |
| SQMC log probability of actual 1--0 | -2.734 | -2.300 |

The text currently says that SQMC assigns the highest probability to a Spain
win at 0.537. That statement is true only for the old bundle and must not be
retained if the repaired run is the dissertation result.

Evidence: [current SQMC predictions](../../rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/combined/results/sqmc_predictions.json)
and [old SQMC predictions](../drafts/results/sqmc_ekf/combined/results/sqmc_predictions.json).

**Remedy.** Remove the old claim that SQMC assigns Spain a 0.537 win
probability. Replace the paragraph with the probabilities and score
diagnostics from the authoritative run, explicitly identifying whether the
values are pre-match predictive probabilities or post-match posterior states.
Regenerate the final-match heatmap and cite the matching prediction record so
that the text, figure, and JSON artifact cannot drift apart.

### 7. The pre- and post-tournament ranking tables are stale

The current chapter’s SQMC pre-tournament top ten is:

> Brazil, England, Portugal, Belgium, France, Japan, Sweden, Colombia,
> Germany, Argentina.

The 09/09 run gives:

> Portugal, Belgium, England, France, Croatia, Japan, Czech Republic,
> Netherlands, Austria, Morocco.

The current chapter’s post-tournament SQMC top ten is:

> England, France, Portugal, Scotland, Germany, United States, Japan,
> Belgium, Argentina, Brazil.

The 09/09 run gives:

> Germany, Brazil, Sweden, Norway, England, Argentina, Australia, Ivory Coast,
> Canada, Netherlands.

Consequently, the FIFA comparison paragraph—which says that SQMC places
France second, Scotland fourth, the United States sixth, and Argentina ninth—
describes the old run, not the repaired run. It must be regenerated from one
identified run, with the run configuration and source commit stated in the
caption or surrounding text.

Evidence: [current pre-ranking](../../rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/sqmc/images/json/sqmc_pre_worldcup_rankings.json),
[current post-ranking](../../rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/sqmc/images/json/sqmc_post_worldcup_rankings.json),
[old pre-ranking](../drafts/results/sqmc_ekf/sqmc/images/json/sqmc_pre_worldcup_rankings.json),
and [old post-ranking](../drafts/results/sqmc_ekf/sqmc/images/json/sqmc_post_worldcup_rankings.json).

**Remedy.** Regenerate both ranking tables from the authoritative evaluation
directory and update the surrounding FIFA-comparison paragraph at the same
time. Include the run ID, source commit, checkpoint epoch/policy, and ranking
definition (`attack + defence`) in the table caption or immediately before
the tables. Do not combine the old ranking table with new prediction metrics;
all displayed results must come from the same fitted checkpoint and evaluation
filter.

## Interpretation of the ranking concern

The section should explicitly distinguish three quantities:

1. the pre-match predictive distribution;
2. the post-match particle posterior; and
3. the scalar ranking formed by attack plus defence.

RB-SQMC forecasts use particles after resampling by the previous weights, so
those propagated particles are equally weighted. The current score then
creates the updated particle weights. Ranking summaries use those updated
posterior weights. This is a valid particle-filter convention, but it means a
forecast probability and a post-tournament total-strength rank are not the
same estimand.

The deeper issue is that the old dissertation run used separate randomized
filters for predictions and rankings and omitted the learned friendly-match
scale when replaying the history for prediction/ranking. The repaired run uses
one shared evaluation filter and consistent scales. The old and new ranking
tables should therefore not be presented as if they were merely two methods
under identical SQMC conditions.

France’s new post-tournament movement is also interpretable without adding a
“progression bonus”. Spain and France both played eight matches. Spain went
7--0--1 with a 14--1 goal record; France went 6--0--2 with a 20--10 record,
including losses of 0--2 to Spain and 4--6 to England. In the repaired SQMC
summary, France falls from total strength 2.459 pre-tournament to 1.016
post-tournament, driven mainly by defence falling from 1.214 to -0.100. This
is qualitatively compatible with the final results, although the magnitude is
particle-sensitive and should not be treated as a stable football ranking.

The dissertation should not claim that deeper progression or more matches
must produce a higher latent-strength rank. The model updates strength from
score likelihoods conditional on opponents; it does not award tournament-depth
or appearance points.

**Remedy.** Define the ranking as the posterior mean of attack plus defence,
not as a points table or tournament-progression measure. Add a diagnostic table
for Spain and France showing their records, goals, pre/post latent components,
and rank changes. Report tournament progression separately as a descriptive
statistic and leave the SQMC ranking estimand unchanged.

## Claims that need softer wording

### “Any difference is attributable to the correlated RB construction”

This is too strong. Even with the repaired shared data pipeline, the methods
also differ in:

- Gaussian moment approximation versus particle approximation;
- stochastic versus deterministic likelihood-gradient estimates;
- particle count and RQMC replica count;
- numerical resampling error and posterior-summary conventions; and
- the documented propagation/time-index implementation.

The defensible claim is that both methods use the same frozen data split and
observation model, while comparing different approximations and numerical
representations of the filtering problem.

**Remedy.** Replace the stronger attribution with the defensible statement
above. Explicitly list the correlated state representation, Gaussian
approximation, particle approximation, optimization noise, particle count,
and RQMC replica count as possible sources of difference.

### “Higher log-likelihood is consistent with evaluating the non-Gaussian likelihood”

This should be framed as a possible explanation, not an inference from the
reported values. EKF logZ is a Gaussian-approximate normalising constant,
whereas SQMC logZ is a finite-particle likelihood estimate. Their raw values
are not a clean method-comparison score. The section already states this
caveat, but the subsequent interpretation still overstates it.

**Remedy.** Rewrite the claim as a qualified interpretation. Report
predictive metrics on their own scale. Do not use raw EKF and SQMC logZ values
as a direct method ranking; reserve logZ comparisons for a common
normalising-constant definition.

### “More particles or replicas would reduce the Monte Carlo error”

This is directionally reasonable but should not imply convergence has been
demonstrated. The newer 512-particle/15-replica run still changes rankings and
prediction metrics materially. A fixed-checkpoint seed/particle study is
needed before claiming numerical stability.

**Remedy.** Add a fixed-checkpoint sensitivity analysis over particle counts
and random seeds. Report the resulting variation in predictive metrics and
rankings. Describe the current results as a single finite-particle realization
and remove any claim of SQMC convergence until this analysis is reported.

## Recommended reproducibility improvement: evaluation-only checkpoint replay

The comparison should support a separate evaluation-only path after fitting.
Once an optimized parameter checkpoint has been saved, the full filter,
sequential prediction, state trajectories, and pre/post-World-Cup rankings
should be reproducible without running the optimizer again. This makes it
possible to compare alternative evaluation filters, inspect state changes, and
regenerate figures from exactly the same fitted parameters.

The scripts are maintained in the public
[`ryantjx/rbsqmc` repository](https://github.com/ryantjx/rbsqmc), including the
[comparison runner](https://github.com/ryantjx/rbsqmc/blob/main/rbsqmc/comparison/sqmc_ekf/run.py)
and the
[Colab/local launcher](https://github.com/ryantjx/rbsqmc/blob/main/rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh).
The evaluation command should take the saved `fitted_params.json` explicitly,
record its checkpoint hash, and write to a new output directory rather than
overwriting the training run. For example:

```bash
MPLCONFIGDIR=/private/tmp/sqmc_eval_mpl \
RBSQMC_PLATFORM=cpu \
JAX_ENABLE_X64=true \
MPLBACKEND=Agg \
python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/combined/results/comparison_config.json \
  --data rbsqmc/data/results.csv \
  --methods sqmc \
  --evaluate-params rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/sqmc/results/sqmc/fitted_params.json \
  --output-dir rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/local
```

This command does not train. It loads the saved SQMC parameters, filters the
complete chronological dataset, produces one-step-ahead prediction records,
and writes the state trajectory, ranking figures, prediction diagnostics, and
metrics below `outputs/09092026_1431/local/`. The output should also record the
parameter source, checkpoint epoch, source revision, dataset hash, particle
count, and filter-key provenance.

The distinction between the selected training checkpoint and the evaluation
run should be explicit. In the 09/09/2026 artifacts, the available
`fitted_params.json` is the final epoch-50 checkpoint, whereas the best held-out
test log-likelihood occurred at epoch 49. Future training runs should save the
best-test checkpoint (and its metadata) so that the evaluation-only command
can reproduce the intended optimized result without retraining.

## Files reviewed

- `dissertation/drafts/chapters/3_football_model_with_sqmc.tex`
- `dissertation/drafts/results/sqmc_ekf/combined/`
- `rbsqmc/comparison/sqmc_ekf/outputs/09092026_1431/`
- `dissertation/issues/INVESTIGATION_090926.md`
- `rbsqmc/comparison/sqmc_ekf/scripts/plots.py`
- `rbsqmc/comparison/sqmc_ekf/run.py`
- `rbsqmc/src/model/rbsqmc/predict_rbsqmc.py`
