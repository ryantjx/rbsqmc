# SQMC–EKF Comparison Report

Generated: 2026-09-09 17:58:13
Dataset rows: 49521 (train 4702, test 156, prediction count 253).
Config: {"training_start_date": "1980-01-01", "test_start_date": "2024-01-01", "prediction_start_date": "2025-06-01", "n_particles": 128, "max_goals": 8, "seed": 0, "n_epochs": 3, "learning_rate": 0.05, "n_reps": 2, "include_friendly": true, "teams": "worldcup2026", "match_scale": 1, "gauss_hermite_degree": 16, "gpu_type": "cpu", "diagnostics_scope": "all", "colab_timeout": 14400, "repo_url": "https://github.com/ryantjx/rbsqmc.git"}

## Training

| | EKF | SQMC |
|---|---|---|
| Final train logZ | -1.657e+04 | -1.554e+04 |
| Final test logZ | -567.2 | -546.2 |
| Best test logZ (epoch) | -567.2 (3) | -546.2 (3) |
| Compile (s) / execute (s) | 0.8673 / 1.174 | 2.072 / 8.096 |

> Note on logZ: the EKF value is a **Gaussian-approximate** normalising constant from moment filtering; the SQMC value is a particle likelihood estimate. They are not directly comparable as exact marginal likelihoods, so any raw gap must not be interpreted as a superiority verdict on its own.

## Headline prediction metrics (World Cup 2026 eligible)

### EKF
| Metric | Value |
|---|---|
| Mean Brier score | 0.5466 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | 0.1801 |
| Mean predictive log score | -3.211 |
| Exact-score accuracy | 0.07692 |
| Outcome accuracy | 0.5481 |
| Scored matches | 104 |

### SQMC
| Metric | Value |
|---|---|
| Mean Brier score | 0.7199 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | -0.07991 |
| Mean predictive log score | -4.186 |
| Exact-score accuracy | 0.07692 |
| Outcome accuracy | 0.4904 |
| Scored matches | 104 |

## Verdict (evidence-based)

- On the three-outcome Brier score, EKF had the lower mean (0.5466 vs 0.7199).
- Outcome accuracy was 0.5481 (EKF) vs 0.4904 (SQMC).
- The test logZ gap alone is not treated as a correctness verdict: the EKF reports a Gaussian-approximate logZ and the SQMC a particle estimate, so differences may reflect approximation quality and estimation noise rather than only the modelled correlations.
- Any remaining differences are attributed to correlated vs factorial dynamics only after the shared-input, shared-optimizer protocol is confirmed and prediction metrics are inspected.


## Method

Both methods consume one chronological, per-match dataset (train `[1980-01-01, 2024-01-01)`, test `[2024-01-01, 2026-06-11)`, prediction from June 11 2026). Same-day repeats are kept as separate `dt=0` rows. The EKF propagates only the two playing teams per match by each team's last-appearance gap; the SQMC shared-`dt` comparator propagates the full covariance per row.

Both used Adam with cosine decay, seed 0 and exactly 3 updates. SQMC gradients were averaged over 2 independently randomised replicas.

## Limitations

- EKF logZ is a Gaussian approximation; SQMC logZ is a particle estimate with finite-sample error.
- Different time-step conventions (only-playing-teams vs shared-dt) mean the two logZ scales are not exactly aligned.
- Headline prediction metrics are restricted to 2026 World Cup fixtures; other prediction matches are reported for completeness.