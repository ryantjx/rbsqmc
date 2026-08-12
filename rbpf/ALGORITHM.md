# RB-SQMC with Kronecker Structure

CAA: 31/07/2026

Version : 5

## 1 Background


## 2 Setup

We define a general discrete-time OU-process in $\mathbb{R}^{K \times M}$. Let $X_t \in \mathbb{R}^{K \times M}$ denote the latent state at time t. For simplicity, the initial distribution is a multivariate Gaussian

$$X_0 \sim \mathcal{N} (\mu_0, \Sigma_0)$$

where $\mu_0 \in \mathbb{R}^{K \times M}$ and $\Sigma_0 = \Gamma_0 \otimes B \in \mathbb{R}^{(K \times M) \times (K \times M)}$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{M \times M}$, is the initial covariance matrix between the latent states, and $B \in \mathbb{R}^{K \times K}$, a static covariance matrix within each latent state. In this particular scenario, we generalize the relationship with the latent states $X_t^{m}$ to be constant across all $M$ indices defined by the static parameter $B$. This simplification provides a significant computational benefits, as it reduces our computation of $\Sigma_0$ from a $O((K \times M) \times (K \times M))$ to a $O(M \times M)$ magnitude.

The state transition is a dicrete-time OU-process defined as 

$$X_t = \mu_0 + \Phi_t (X_{t-1} - \mu_0) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

where $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top$, $\Phi_t = \phi_t \otimes I_K$ and $\phi_t = \exp(-\kappa \Delta t)$, where $\Delta_t$ represents the time difference between each observation $y_t$. $Q_t \in \mathbb{R}^{(K \times M) \times (K \times M)}$ is also a non-diagonal covariance matrix inducing correlation across coordinates from the initial relationship $\Sigma_0$. We assume that our observations $y_t \in \mathbb{R}^{L \times 1}$ arrive asynchronously. At each time step $t$, only a subset of entities $\mathcal{O} \subseteq \{1,\dots,M\}$ and $\mid \mathcal{O} \mid = L$ with corresponding states $X_t^{\mathcal{O}} \in \mathbb{R}^{K \times L}$ is involved in the observation. The remaining states $\mathcal{R} = \{1,\dots,M\} \setminus \mathcal{O}$ and $\mid \mathcal{R} \mid = M - L$ with corresponding states $X_t^{\mathcal{R}} \in \mathbb{R}^{K \times (M - L)}$ are not involved in the observation. For simplicity, we define our likelihood function as a linear-Gaussian function of the latent states involved in an observation $X_t^{\mathcal{O}} \in \mathbb{R}^{K \times L}$,

$$y_t = H_t X_t^{\mathcal{O}} + \nu_t, \quad \nu_t \sim \mathcal{N}(0, P^{\mathcal{O}})$$

where $H_t = \text{diag}(h_1^\top, \ldots, h_L^\top) \in \mathbb{R}^{L \times (K \times L)}$ is a block-diagonal linear observation matrix with each $h_m^\top \in \mathbb{R}^{1 \times K}$ mapping entity $m$'s $K$-dimensional feature vector to a scalar, and $P^{\mathcal{O}} \in \mathbb{R}^{L \times L}$ is a non-diagonal static observation noise covariance. In general, this likelihood function $p(y_t^{\mathcal{O}} \mid x_t^{\mathcal{O}})$ is not restricted to linear functions because of the characteristics of the particle filter.

<!-- For simplicity, the likelihood function maps each observed entity's $K$-dimensional feature vector to a scalar via $g : \mathbb{R}^{K} \to \mathbb{R}$, applied independently per entity $m \in \mathcal{O}$,

$$y_t^m = g(X_t^{:,m}) + \nu_t^m, \quad \nu_t^m \sim \mathcal{N}(0, P^{\mathcal{O}})$$

Stacking over $\mathcal{O}$ gives $y_t \in \mathbb{R}^{L \times 1}$ with $P^{\mathcal{O}} \in \mathbb{R}^{L \times L}$ a non-diagonal static observation noise covariance. In general, the likelihood function $p(y_t^{\mathcal{O}} \mid x_t^{\mathcal{O}})$ is not restricted to linear functions because of the characteristics of the particle filter. -->

## 3 Rao-Blackwellized Particle Filter (RB-PF)

The RB-PF targets the marginal distribution of latent states $X_t^{\mathcal{O}} \subseteq X_t$ involved in an observation $y_t$. The joint distribution of latent states and observed can be factorized as

$$p(X_{0:T} \mid y_{1:T}) = p(X_{0:T}^{\mathcal{R}} \mid X^{\mathcal{O}}_{0:T}) p(X_{0:T}^{\mathcal{O}} \mid y_{1:T})$$

Conditional on a specific trajectory of latent states $X_{0:T}^{\mathcal{O}}$ involved in an observation, the remaining latent states $X_{0:T}^{\mathcal{R}}$ are conditionally independent of the observed data $y_{1:T}$. Since these dyanamics are linear and Gaussian, the conditional distribution of $X_{0:T}^{\mathcal{R}} \mid X^{\mathcal{O}}_{1:T}$ is deterministic and computed analytically using the Kalman filter.

### 3.1 Algorithm

For simplification, we choose the bootstrap particle filter as the proposal distribution $q(X_t^{\mathcal{O}} \mid X_{t-1}^{\mathcal{O}}) = p(X_t^{\mathcal{O}} \mid X_{t-1}^{\mathcal{O}})$. The algorithm simplifies to the following steps:

1. **Prediction (Over all states)**: Compute the predictive mean and covariance for the full state $X_t \mid y_{1:t-1}$ using the Kalman filter equations since it is linear and Gaussian.
   1. $$\mu_{t \mid t-1}^{(i)} = \mu_0 + \Phi_t (\mu_{t-1}^{(i)} - \mu_0)$$
   2. $$\Sigma_{t \mid t-1}^{(i)} = Q_t + \Phi_t \Sigma_{t-1}^{(i)} \Phi_t^\top$$
   <!-- 3. Extract the predictive mean and covariance for the latent states involved in an observation $X_t^{\mathcal{O},(i)} \mid y_{1:t-1} \sim \mathcal{N}(\mu_{t \mid t-1}^{\mathcal{O},(i)}, \Sigma_{t \mid t-1}^{\mathcal{O},(i)})$. -->
2. **Bootstrap particle sampling**: Extract and sample $(K \times L)$-dimensional latent states involved in an observation $X_t^{\mathcal{O},(i)} \mid y_{1:t-1}$ for each particle $i = 1, \ldots, N$.
   1. $$X_t^{\mathcal{O},(i)} \mid y_{1:t-1} \sim \mathcal{N}(\mu_{t \mid t-1}^{\mathcal{O},(i)}, \Sigma_{t \mid t-1}^{\mathcal{O},(i)})$$
3. **Compute Weights**: Compute unnormalized log-weights $\log(\tilde{w}_t^{(i)})$ for each particle $i = 1, \ldots, N$ using the likelihood function $p(y_t^{\mathcal{O}} \mid X_t^{\mathcal{O},(i)}, P)$ defined in the previous section.
   1. $$\log (\tilde{w}_t^{(i)}) = \log(w_{t-1}^{(i)}) + \log \mathcal{N} (y_t^{\mathcal{O}} \mid X_t^{\mathcal{O},(i)}, P)$$
   2. Normalize the weights $w_t^{(i)} = \tilde{w}_t^{(i)} / \sum_{j=1}^{N} \tilde{w}_t^{(j)}$.
4. **Exact Marginalization**: Condition full state on specific realization of $\mu_t^{\mathcal{O},(i)}$ and $\Sigma_t^{\mathcal{O},(i)}$ to compute the conditional distribution of the remaining latent states $X_t^{\mathcal{R},(i)} \mid X_t^{\mathcal{O},(i)} = \mu_t^{\mathcal{O},(i)}, y_{1:t-1}$ using the Kalman update equations.
   1. $$X_t^{\mathcal{R},(i)} \mid X_t^{\mathcal{O},(i)} = \mu_t^{\mathcal{O},(i)}, y_{1:t-1} \sim \mathcal{N}(\mu_t^{\mathcal{R} \mid \mathcal{O},(i)}, \Sigma_t^{\mathcal{R}\mathcal{R} \mid \mathcal{O},(i)})$$
   2. $\mu_t^{\mathcal{R} \mid \mathcal{O},(i)} = \mu_{t \mid t-1}^{\mathcal{R},(i)} + K_t^{(i)} (\mu_t^{\mathcal{O},(i)} - \mu_t^{\mathcal{O},(i)})$
   3. $\Sigma_t^{\mathcal{R}\mathcal{R} \mid \mathcal{O},(i)} = \Sigma_{t \mid t-1}^{\mathcal{R}\mathcal{R},(i)} - K_t^{(i)} \Sigma_{t \mid t-1}^{\mathcal{O}\mathcal{R},(i)}$
5. **Resampling**: Perform resampling on the particles $X_t^{\mathcal{O},(i)}$ based on the normalized weights $w_t^{(i)}$ and set $w_t^{(i)} = 1/N$ for $i = 1, \ldots, N$. Since the remaining particles would have been resampled prior to each step, all particles $X_t^{(i)}$ will have equal weights $w_t^{(i)} = 1/N$.

### 3.2 Proof of Kronecker Structure Preservation

In step 3, the exact gaussian marginalization involves the update of the remaining latent states $X_t^{\mathcal{R},(i)} \mid X_t^{\mathcal{O},(i)} = \mu_t^{\mathcal{O},(i)}, y_{1:t-1}$. From our initial distribution of $X_t$, we can decompose the joint distribution of the latent states into the from as follows,

$$\begin{pmatrix}X_t^{\mathcal{O}, (i)} \\ X_t^{\mathcal{R}, (i)}\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\mu_t^{\mathcal{O}, (i)} \\ \mu_t^{\mathcal{R}, (i)}\end{pmatrix}, \begin{pmatrix}\Sigma_t^{\mathcal{O}\mathcal{O}, (i)} & \Sigma_t^{\mathcal{O}\mathcal{R}, (i)} \\ \Sigma_t^{\mathcal{R}\mathcal{O}, (i)} & \Sigma_t^{\mathcal{R}\mathcal{R}, (i)}\end{pmatrix}\right)$$

where 

- $\Sigma_t^{\mathcal{O}\mathcal{O}, (i)} = \text{Var}(X_t^{\mathcal{O}, (i)} \mid y_{1:t-1}) \in \mathbb{R}^{(K \times L) \times (K \times L)}$
- $\Sigma_t^{\mathcal{R}\mathcal{R}, (i)} = \text{Var}(X_t^{\mathcal{R}, (i)} \mid y_{1:t-1}) \in \mathbb{R}^{(K \times (M-L)) \times (K \times (M-L))}$
- $\Sigma_t^{\mathcal{O}\mathcal{R}, (i)} = \text{Cov}(X_t^{\mathcal{O}, (i)}, X_t^{\mathcal{R}, (i)} \mid y_{1:t-1}) = (\Sigma_t^{\mathcal{R}\mathcal{O}, (i)})^\top \in \mathbb{R}^{(K \times L) \times (K \times (M-L))}$ since the covariance matrix is symmetric.

In the next step (step 4), we want to perform the Kalman update after obtaining noise free measurements of $X_t^{\mathcal{O},(i)} = \mu_t^{\mathcal{O},(i)}$. We perform the Kalman update on the remaining latent states $X_t^{\mathcal{R},(i)} \mid X_t^{\mathcal{O},(i)} = \mu_t^{\mathcal{O},(i)}, y_{1:t-1}$ to obtain the updated mean and covariance of the posterior distribution. By structuring the predictive covariance $\Sigma_t^{(i)} = \Gamma_t^{(i)} \otimes B$, we significantly reduce the computational complexity of the Kalman update from $O((K \times M)^3)$ to $O(M^3)$ during the inversion of $\Gamma_t^{\mathcal{O}\mathcal{O}, (i)} \in \mathbb{R}^{L \times L}$.

$$\begin{aligned}
K_t^{(i)} &= \Sigma_{t \mid t-1}^{\mathcal{O}\mathcal{R},(i)} (\Sigma_{t \mid t-1}^{\mathcal{O}\mathcal{O},(i)})^{-1} \\ &= (\Gamma_t^{\mathcal{O}\mathcal{R}, (i)} \otimes B) (\Gamma_t^{\mathcal{O}\mathcal{O}, (i)} \otimes B)^{-1} \\ &= (\Gamma_t^{\mathcal{O}\mathcal{R}, (i)} \otimes B) ((\Gamma_t^{\mathcal{O}\mathcal{O}, (i)})^{-1} \otimes B^{-1}) \quad \text{(inverse property)}\\ &= \Gamma_t^{\mathcal{O}\mathcal{R}, (i)} (\Gamma_t^{\mathcal{O}\mathcal{O}, (i)})^{-1} \otimes I_K \quad \text{(mixed-product property)}
\end{aligned}$$

The conditions for the Kronecker product structure to be preserved in the Kalman update are:

1. The initial covariance matrix $\Sigma_0$ must have a Kronecker product structure, i.e., $\Sigma_0 = \Gamma_0 \otimes B$.
2. $\Gamma_t^{\mathcal{O}\mathcal{O}, (i)} \in \mathbb{R}^{L \times L}$ must be positive definite (hence invertible).
3. $B \in \mathbb{R}^{K \times K}$ must be positive definite (hence invertible).

The filtering distribution for the full state conditional $X_t^{(i)} \mid y_{1:t}$ is then given by

$$p(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_t^{\mathcal{O}, (i)} \\ \mu_t^{\mathcal{R} \mid \mathcal{O}, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & \Sigma_t^{\mathcal{R} \mathcal{R} \mid \mathcal{O}, (i)} \end{bmatrix} \right)$$

## 4 Rao-Blackwellized Particle Filter Smoothing

At terminal time $T$, the forward filter approximates a weighted Gaussian mixture $p(X_T \mid y_{1:T}) \approx \sum_{i=1}^{N} w_T^{(i)} \mathcal{N}(\mu_T^{(i)}, \Sigma_T^{(i)})$. We can apply the Forward Filtering Backward Simulation (FFBSi) to obtain the the smoothing distribution $p(X_{0:T} \mid y_{1:T})$.

### 4.1 Algorithm

1. **Initialization**: At terminal time $T$, select which mixture component to sample from and then sample from the state of the component.
   1. Sample component index: $$I_T \sim \text{Categorical}(w_T^{(1)}, \ldots, w_T^{(N)})$$
   2. Sample state: $$X_T^* \sim \mathcal{N}(\begin{pmatrix} \mu_T^{\mathcal{O}, (I_T)} \\ \mu_T^{\mathcal{R} \mid \mathcal{O}, (I_T)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & \Sigma_T^{\mathcal{R} \mathcal{R} \mid \mathcal{O}, (I_T)} \end{bmatrix})$$
2. **Backward Simulation**: For $t = T-1, \ldots, 0$, the smoothed density is derived by Bayes' theorem,
   1. $$p(X_t \mid X_{t+1}^*, y_{1:T}) \propto p(X_t \mid y_{1:t}) p(X_{t+1}^* \mid X_t)$$
   2. Since the transition distribution is linear and Gaussian $p(X_{t+1}^* \mid X_t) = \mathcal{N}(X_t^* \mid X_t, Q_{t})$, the smoothed density yields a new Gaussian mixture
   3. $$p(X_t \mid X_{t+1}^*, y_{1:T}) \approx \sum_{i=1}^{N} w_{t \mid t+1}^{(i)} \mathcal{N}(X_t \mid \mu_{t \mid t+1}^{(i)}, \Sigma_{t \mid t+1}^{(i)})$$
   4. The backward weights are obtained by evaluating the predictive density of $X_{t+1}^*$ given the filtering distribution $p(X_{t+1} \mid y_{1:t})$,
   5. $$w_{t \mid t+1}^{(i)} \propto w_{t}^{(i)} \mathcal{N}(X_{t+1}^* \mid \mu_{t+1 \mid t}^{(i)}, \Sigma_{t+1 \mid t}^{(i)})$$
   6. The conditional mean and covariance follow the Rauch-Tung-Striebel (RTS) smoother equations,
      1. $$\mu_{t \mid t+1}^{(i)} = \mu_t^{(i)} + J_t^{(i)} (X_{t+1}^* - \mu_{t+1 \mid t}^{(i)})$$
      2. $$\Sigma_{t \mid t+1}^{(i)} = \Sigma_t^{(i)} - J_t^{(i)} \Sigma_{t+1 \mid t}^{(i)} (J_t^{(i)})^\top$$
      3. $$J_t^{(i)} = \Sigma_t^{(i)} \Phi_{t+1}^\top (\Sigma_{t+1 \mid t}^{(i)})^{-1}$$
   7. Repeat steps 1 and 2 until $t = 0$ to obtain the smoothed trajectory $X_{0:T}^*$.
      1.  $$X_t^* \sim \mathcal{N}(\mu_{t \mid t+1}^{(I_t)}, \Sigma_{t \mid t+1}^{(I_t)})$$

### 4.2 Proof of Kronecker Structure Preservation

We can decompose the backward sampling gain $J_t^{(i)} \in \mathbb{R}^{(K \times M) \times (K \times M)}$ as follows,

$$\begin{aligned}
J_t^{(i)} &= \Sigma_{t}^{(i)} \Phi_{t+1}^\top \Sigma_{t+1 \mid t}^{-1,(i)} \\ &= (\Gamma_t^{(i)} \otimes B) (\phi_{t+1}^\top \otimes I_K) (\Gamma_{t+1 \mid t}^{(i)} \otimes B)^{-1} \\ &= (\Gamma_t^{(i)} \phi_{t+1}^\top \otimes B) ((\Gamma_{t+1 \mid t}^{(i)})^{-1} \otimes B^{-1}) \quad \text{(inverse property)} \\ &= (\Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1}) \otimes I_K \quad \text{(mixed-product property)}
\end{aligned}$$

The conditional covariance $\Sigma_{t \mid t+1}^{(i)} = \Sigma_t^{(i)} - J_t^{(i)} \Sigma_{t+1 \mid t}^{(i)} (J_t^{(i)})^\top$ can be decomposed as follows,

$$\begin{aligned} \Sigma_{t \mid t+1}^{(i)} &= \Sigma_t^{(i)} - J_t^{(i)} \Sigma_{t+1 \mid t}^{(i)} (J_t^{(i)})^\top \\ &= (\Gamma_t^{(i)} \otimes B) - ((\Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1}) \otimes I_K) (\Gamma_{t+1 \mid t}^{(i)} \otimes B) ((\Gamma_{t+1 \mid t}^{(i)})^{-1} \phi_{t+1} \Gamma_t^{(i)} \otimes I_K) \\ &= (\Gamma_t^{(i)} \otimes B) - (\Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1} \Gamma_{t+1 \mid t}^{(i)} (\Gamma_{t+1 \mid t}^{(i)})^{-1} \phi_{t+1} \Gamma_t^{(i)}) \otimes B \quad \text{(mixed-product)} \\ &= (\Gamma_t^{(i)} \otimes B) - (\Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1} \phi_{t+1} \Gamma_t^{(i)}) \otimes B \\ &= (\Gamma_t^{(i)} - \Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1} \phi_{t+1} \Gamma_t^{(i)}) \otimes B \quad \text{(distributivity)} \\ &= \Gamma_{t \mid t+1}^{(i)} \otimes B \end{aligned}$$

where $\Gamma_{t \mid t+1}^{(i)} = \Gamma_t^{(i)} - \Gamma_t^{(i)} \phi_{t+1}^\top (\Gamma_{t+1 \mid t}^{(i)})^{-1} \phi_{t+1} \Gamma_t^{(i)}$ is the Schur complement of $\Gamma_{t+1 \mid t}^{(i)}$ and is also the predicted covariance of latent states.

The conditions for the Kronecker product structure to be preserved in the backward sampling update are:

1. The filtering covariance $\Sigma_t^{(i)}$ must have Kronecker structure, i.e., $\Sigma_t^{(i)} = \Gamma_t^{(i)} \otimes B$.
2. $\Gamma_{t+1 \mid t}^{(i)} \in \mathbb{R}^{M \times M}$ must be positive definite (hence invertible).
3. $B \in \mathbb{R}^{K \times K}$ must be positive definite (hence invertible).

## 5 Parameter Estimation - Expectation-Maximization (EM)

## Appendix

### 3.1 Kronecker Product Properties

Inverse Property: $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$

Mixed-Product Property: $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$

Distributivity: $(A \otimes B) + (C \otimes B) = (A + C) \otimes B$

https://en.wikipedia.org/wiki/Kronecker_product

## References

Mid-Price Estimation for European Corporate Bonds: A Particle Filtering Approach https://www.worldscientific.com/doi/abs/10.1142/S2382626619500059

West, M. and Harrison, J. (1997) Bayesian Forecasting and Dynamic Models (Springer Series in Statistics). 2nd Edition, Springer, Berlin. https://link.springer.com/book/10.1007/b98971


------------------------------------------------------------

## 4 Simulation Study

We evaluate the RB-PF with Kronecker structure on a simulated dataset. Latent states are generated from a correlated OU-process on $F = 10$ dimensions with $P = 2$ dimensions for the latent states involved in an observation, and $P = 2$ for the likelihood function.

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B$ with $\Gamma_0 \in \mathbb{R}^{10 \times 10}$ and $B \in \mathbb{R}^{2 \times 2}$. The state transition is governed by the OU-process defined as

$$X_t = \mu_0 + \Phi_t (X_{t-1} - \mu_0) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

where $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top$ and $\Phi_t = \phi_t \otimes I_P$ with $\phi_t = \exp(-\kappa \Delta t) \cdot I_F$. The likelihood function is defined as a non-linear function of inputs $X_t^{E}$. For simplicity, we utilize a bivariate Gaussian likelihood function for the observed data $y_t \in \mathbb{R}^2$:

$$G_t(y_t \mid x_t^{i}, x_t^{j}) = \frac{1}{2 \pi \sqrt{|R|}} \exp\left(-\frac{1}{2} (y_t - \mu)^\top R^{-1} (y_t - \mu)\right)$$

where $\mu = [x_t^{1, i} - x_t^{2, j}, x_t^{1, j} - x_t^{2, i}]^\top$ and $R$ is the observation noise covariance.

<!-- In this case, we use a bivariate poisson likelihood function for the observed data $y_t \in \mathbb{R}^2$:

$$G_t(y_t \mid x_t^{i}, x_t^{j}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{i}}}{y_t^{i}!} \frac{\lambda_2^{y_t^{j}}}{y_t^{j}!} \sum_{k=0}^{\min(y_t^{i}, y_t^{j})} \binom{y_t^{i}}{k} \binom{y_t^{j}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, i} - x_t^{\text{def}, j})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, j} - x_t^{\text{def}, i})$, $\lambda_3 = \exp(\beta)$. -->

We aim to estimate the following parameters: $\kappa$, $R$, $\mu_0$ and $\Sigma_t$, which decomposes into $\Gamma_t$ and $B$.

<!-- ## 5 Counterexample: Kronecker product covariance structure with time-varying $B_t$ and time-varying $\Gamma_t$

7. **Obtain new $\Sigma_t$ with Kronecker structure**
   1. $$\begin{aligned}P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top \\ &= \Phi_t (\Gamma_{t-1} \otimes B_{t-1}) \Phi_t^\top + (\Gamma_0 \otimes B_0) - \Phi_t (\Gamma_0 \otimes B_0) \Phi_t^\top\end{aligned}$$
   2. $\Gamma_{t} = \begin{bmatrix} \Gamma_t^{EE} & \Gamma_t^{ER} \\ \Gamma_t^{RE} & \Gamma_t^{RR} \end{bmatrix}$ and assume $B_{t}^{EE} = B_t^{RR} = B_t$
   3. **(NOT COMPLETED)** $$\begin{aligned}K_t &= P_{t}^{RE} (P_{t}^{EE})^{-1} \\ &= [\phi_t^{RE} (\Gamma_t^{RE} \otimes B_t) \phi_t^{RE} + (\Gamma_0^{RE} \otimes B_0)(I - \phi_t^{RE} {\phi_t^{RE}}^\top)] [(\Gamma_t^{EE} \otimes B_t)^{-1}] \end{aligned}$$  -->

## Additional Notes

- 290726: OU-process with asynchronous observations resulted in a non positive-definite matrix during the Prediction step i.e. $Q_t = (\Gamma_0 - \phi_t \Gamma_0 \phi_t^\top) \otimes B$. changed to a random walk. TrueSkill2 / Glicko avoid by modelling latent states for each index as non-correlated i.e. $Q_t = \text{diag}(\Gamma_0) \otimes B$ which is positive definite. This is a limitation of the OU-process with asynchronous observations.