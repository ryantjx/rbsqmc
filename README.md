# RB-SQMC: Rao-Blackwellized Sequential Quasi-Monte Carlo

This repository contains implementations and experiments for **Rao-Blackwellized Sequential Quasi-Monte Carlo (RB-SQMC)**, a high-performance sequential inference method that combines the benefits of Quasi-Monte Carlo (QMC) sampling with Rao-Blackwellization techniques.

## Overview

Sequential Monte Carlo (SMC) and Quasi-Monte Carlo (QMC) methods suffer from the curse of dimensionality. This project demonstrates how RB-SQMC overcomes these limitations by:

- Reformulating high-dimensional problems to operate on lower-dimensional spaces during resampling
- Leveraging Rao-Blackwellization to marginalize out latent states analytically
- Achieving superior performance compared to standard SMC and SQMC methods

## Key Components

### Core Algorithms
- **Sequential Quasi-Monte Carlo (SQMC)**: QMC-based particle filtering with Hilbert curve sorting
- **Rao-Blackwellized SQMC (RB-SQMC)**: Combines QMC sampling with analytical marginalization
- **Sobol and Halton sequences**: Low-discrepancy point generation for QMC

### Applications
- **Football Match Prediction**: State-space model for predicting football match outcomes using bivariate Poisson likelihood with attack/defense latent states

## Repository Structure

- `rbpf_1` - old specification of the model. wrong implementation, but proof is correct.
- `rbpf_2` - Smoothing using grid search, not efficient at all.
- `rbpf_3` - Smoothing using gradient-based optimization.

## Documentation

- **[DISSERTATION.md](DISSERTATION.md)**: Complete dissertation covering SQMC theory, RB-SQMC methodology, and football prediction application
- **[src/scripts/hilbert_sort/HILBERT_SORT.md](src/scripts/hilbert_sort/HILBERT_SORT.md)**: Hilbert curve sorting documentation
- **[src/scripts/qmc/SCIPY.md](src/scripts/qmc/SCIPY.md)**: QMC implementation notes

## Dependencies

See `requirements.txt` for Python package dependencies.

## References

- Gerber, M., & Chopin, N. (2015). Sequential quasi Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 77(3), 509–579.
- Chopin, N., & Gerber, M. (2017). Sequential quasi-Monte Carlo: Introduction for Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion Processes. arXiv:1706.05305.
- Lemieux, C. (2009). *Monte Carlo and Quasi-Monte Carlo Sampling*. Springer.
- Duffield, S., Power, S., & Rimella, L. (2024). A state-space perspective on modelling and inference for online skill rating. *Journal of the Royal Statistical Society: Series C*, 73(5), 1262–1282.

## License

This project is part of academic research on sequential Monte Carlo methods.
