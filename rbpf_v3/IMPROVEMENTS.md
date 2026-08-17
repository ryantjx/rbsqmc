1. log normalizing constant does not seem to be optimizing well. 1176 free parameters might be affecting the parameter estimation / optimization. 
   1. considered increasing the prior to make optimization better
   2. reduce number of free parameters by optimizing only the diagonals. this means that the initial params wont be informative of the correlation, but the final correlation can be estimated from the final filter. - still sucks
   3. What if i assume that the matrix is symmetric, and then optimize the diagonal? Optimizing the cholesky factor of the covariance matrix. `smoothing_cuthbert_full.py` already does this. high dimensional but has 1176 parameters.
   4. try impleenting a stop gradient on the normalization term in the transition (aside from the quadratic innovation term.) 