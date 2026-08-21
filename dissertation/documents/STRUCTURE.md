# STRUCTURE

This document describes the structure of the dissertation.

Title: High Performance Sequential Quasi Monte Carlo with Rao-Blackwellization

Abstract: SQMC is an extension of the particle filter / smc algorithm by replacing the monte carlo sampling with a quasi monte carlo sampling. SQMC can also be re-interpreted as performing importance sampling in a state-space model. However, literature has shown that this method performs poorly in high dimensions. Given a fixed model, we can instead perform the importance sampling step on a smaller dimension which explains the entire model, thus retaining beneficial properties from SQMC while keeping the algorithm computationally feasible. We demonstrate this by performing Rao-Blackwellization on football model using particle filter, and show how we can use SQMC in this context whilst maintaining properties from QMC.

## Content

1. High-Performance SQMC
   1. QMC - QMC literature, and the properties.
   2. SMC - SMC and a short section on bootstrap particle filter.
   3. SQMC - SQMC and how it is performing importance sampling at every time step. mention the limitations as seen in Gerber and Chopin's paper.
   4. Implementation and Performance comparison
2. Applications: Rao-Blackwellized SQMC on Football Matches
   1. Model - Initial model from sam, what are its assumptions, reframe the model and show how the football model can be rao-blackwellized. include kronecker product structure and show and it reduces the inference dimension.
   2. Rao-Blackwellized SMC
      1. Rao-Blackwellized PF
      2. Smoothing - FFBSi
      3. Parameter Estimation - what the moment I am having difficulty with the EM on the joint likelihood to converge, and I am considering using log marginal maximization instead, which is showing some sort of convergence and more sane results.
   3. Rao-Blackwellized SQMC
      1. Filtering
      2. Parameter estimation - similar to the previous section.
   4. Performance comparison
      1. Factorial Extended Kalman Filter - independence of teams
      2. Rao-Blackwellized SQMC - able to introduce correlation between teams in a scalable manner.
         1. smoothing issues
         2. log marginal likelihood maximization - using `cuthbertlib.resampling.autodiff` from Differentiable Particle Filtering without Modifying the Forward Pass to achieve an unbiased estimate.