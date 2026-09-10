# SQMC–EKF Comparison Report

Generated: 2026-09-09 11:38:56
Dataset rows: 49521 (train 4702, test 305, prediction count 104).
Config: {"training_start_date": "1980-01-01", "test_start_date": "2024-01-01", "prediction_start_date": "2026-06-11", "n_particles": 128, "max_goals": 8, "seed": 0, "n_epochs": 50, "learning_rate": 0.02, "n_reps": 10, "include_friendly": true, "teams": "worldcup2026", "match_scale": 1, "gauss_hermite_degree": 32, "gpu": "A100", "colab_timeout": 14400, "setup_timeout": 900, "transfer_timeout": 600, "session": "sqmc_ekf", "repo_url": "https://github.com/ryantjx/rbsqmc.git", "repo_branch": "main", "source_commit": "846957bacb00a8673871aaf428115455832c9024", "run_id": "09092026_0235", "session_name": "sqmc_ekf_09092026_0235_9ab6df09b4", "resolved_utc": "2026-09-09T02:35:27.846808+00:00", "source_transport": "colab_git_bundle"}

## Training

| | EKF | SQMC |
|---|---|---|
| Final train logZ | -1.568e+04 | -1.499e+04 |
| Final test logZ | -1040 | -1005 |
| Best test logZ (epoch) | -1040 (50) | -970.9 (26) |
| Compile (s) / execute (s) | 0.9709 / 12.52 | 15.99 / 3657 |

> Note on logZ: the EKF value is a **Gaussian-approximate** normalising constant from moment filtering; the SQMC value is a particle likelihood estimate. They are not directly comparable as exact marginal likelihoods, so any raw gap must not be interpreted as a superiority verdict on its own.

## Headline prediction metrics (World Cup 2026 eligible)

### EKF
| Metric | Value |
|---|---|
| Mean Brier score | 0.5442 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | 0.1838 |
| Mean predictive log score | -3.131 |
| Exact-score accuracy | 0.07692 |
| Outcome accuracy | 0.5288 |
| Scored matches | 104 |

### SQMC
| Metric | Value |
|---|---|
| Mean Brier score | 0.6521 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | 0.02183 |
| Mean predictive log score | -3.523 |
| Exact-score accuracy | 0.06731 |
| Outcome accuracy | 0.5096 |
| Scored matches | 104 |

## Verdict (evidence-based)

- On the three-outcome Brier score, EKF had the lower mean (0.5442 vs 0.6521).
- Outcome accuracy was 0.5288 (EKF) vs 0.5096 (SQMC).
- The test logZ gap alone is not treated as a correctness verdict: the EKF reports a Gaussian-approximate logZ and the SQMC a particle estimate, so differences may reflect approximation quality and estimation noise rather than only the modelled correlations.
- Any remaining differences are attributed to correlated vs factorial dynamics only after the shared-input, shared-optimizer protocol is confirmed and prediction metrics are inspected.
