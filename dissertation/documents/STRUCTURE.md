# STRUCTURE

This document describes the structure of the dissertation.

Title: High Performance Sequential Quasi Monte Carlo with Rao-Blackwellization

Abstract: SQMC is an extension of the particle filter / smc algorithm by replacing the monte carlo sampling with a quasi monte carlo sampling. SQMC can also be re-interpreted as performing importance sampling in a state-space model. However, literature has shown that this method performs poorly in high dimensions. Given a fixed model, we can instead perform the importance sampling step on a smaller dimension which explains the entire model, thus retaining beneficial properties from SQMC while keeping the algorithm computationally feasible. We demonstrate this by performing Rao-Blackwellization on football model using particle filter, and show how we can use SQMC in this context whilst maintaining properties from QMC.

## Content

1. High-Performance SQMC
   1. QMC - Introduction to QMC literature, and the properties.
   2. SMC - Introduction to SMC
   3. SQMC - Introduction to SQMC
   4. Implementation of SQMC on GPU hardware
   5. Performance comparison of SQMC compared to SMC
2. Applications: Rao-Blackwellized SQMC on Football Matches
   1. Model - How the model can be rao-blackwellized
   2. Rao-Blackwellized SMC
      1. Rao-Blackwellized Filtering
      2. Smoothing - FFBSi
      3. Parameter Estimation - Limitations, progress, solutions (inverse-wishart)
   3. Rao-Blackwellized SQMC
      1. Filtering
   4. Performance
      1. Factorial Extended Kalman Filter - independence of teams
      2. Rao-Blackwellized SQMC
         1. smoothing issues
         2. log marginal likelihood maximization - using `cuthbertlib.resampling.autodiff` from Differentiable Particle Filtering without Modifying the Forward Pass to achieve an unbiased estimate.
         3. correlation term integrated to teams.