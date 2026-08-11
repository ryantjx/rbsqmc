# RB-SQMC with Kronecker Structure

CAA: 27/07/26 19:00
Version : 3

## 1 Setup

We define a correlated OU-process in $\mathbb{R}^F$. Let $X_t \in \mathbb{R}^{P \times F}$ be the latent states at time $t$. The initial distribution is a multivariate Gaussian in $\mathbb{R}^{P \times F}$ with a Kronecker product structure,

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B \in \mathbb{R}^{P F \times P F}$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B \in \mathbb{R}^{P \times P}$, where $\Gamma_0$ is a time-varying covariance matrix and $B$ is a static covariance matrix within a index $i \in \{1, \ldots, P\}$

The state transition is governed by the OU-process defined as

$$X_t = \mu_0 + \Phi_t (X_{t-1} - \mu_0) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

where $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top$ and $\Phi_t = \phi_t \otimes I_P$ with $\phi_t = \exp(-\kappa \Delta t) \cdot I_F \in \mathbb{R}^{F \times F}$, where $Q_t \in \mathbb{R}^{PF \times PF}$ is also a non-diagonal covariance matrix.

Observations arrive at discrete time steps $t = 1, \ldots, T$. Let $y_t \in \mathbb{R}^H$ be the observed data at time $t$ where $H \leq F$ and $X_t^{E} \in \mathbb{R}^{H \times P}$ be the latent states involved in an observation. The likelihood function is defined as a non-linear function of inputs $X_t^{E}$,

$$G_t(y_t \mid X_t^{E})$$

$X_t^{R} \in \mathbb{R}^{(F-H) \times P}$ are the latent states not involved in an observation. This notation will be used in future sections.

## 2 Rao-Blackwellized Particle Filter (RB-PF)

The RB-PF targets the marginal distribution of latent states $X_t^{E} \subseteq X_t$ involved in an observation $y_t$. The joint distribution of latent states and observed can be factorized as 

$$p(X_{0:T} \mid y_{1:T}) = p(X_{0:T}^{-E} \mid X^{E}_{0:T}) p(X_{0:T}^{E} \mid y_{1:T})$$

where $X_{0:T}^{-E}$ are the latent states not involved in an observation. Since the dynamics of latent states are linear and Gaussian, the conditional distribution of $X_{0:T}^{-E} \mid X^{E}_{1:T}$ is deterministic and computed exactly using the Kalman filter.

### Boostrap Particle Filter

The particle filter performs the weight update as

$$\tilde{w}_t \propto w_{t-1} \cdot \frac{G_t(y_t \mid X_t) p(X_t \mid X_{t-1})}{q(X_t \mid X_{t-1})}$$

where $\tilde{w}_t$ is the unnormalized weight at time $t$. Under the boostrap particle filter, the proposal distribution is the transition distribution $q(X_t \mid X_{t-1}) = p(X_t \mid X_{t-1})$. The weight update simplifies to

$$\tilde{w}_t \propto w_{t-1} \cdot G_t(y_t \mid X_t)$$

$$\log (\tilde{w}_t) = \log(w_{t-1}) + \log G_t(y_t \mid X_t)$$

### Algorithm

**0 Initialization**: Our initial distribution follows a multivariate Gaussian with Kronecker structure. We generate $N$ particles with equal weights $w_{0}^{(i)} = 1/N$ for $i = \{1, \ldots, N\}$.

$$X_0^{(i)} \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B \in \mathbb{R}^{P \times P}$, where $\Gamma_0$ is a time-varying covariance matrix and $B$ is a static covariance matrix.

**2 Prediction**: Compute the predictive mean and covariance for the full state $X_t \mid y_{1:t-1}$ using the Kalman filter equations since it is linear and Gaussian.

$$\mu_{t \mid t-1}^{(i)} = \mu_0 + \Phi_t (\mu_{t-1}^{(i)} - \mu_0)$$

$$P_{t \mid t-1}^{(i)} = \Phi_t P_{t-1}^{(i)} \Phi_t^\top + Q_t = \Phi_t \Sigma_{t-1}^{(i)} \Phi_t^\top + Q_t$$

**3 Exact Gaussian Marginalization**: The predictive distribution for the latent states involved in an observation $X_t^{E,(i)} \mid y_{1:t-1}$ is a multivariate Gaussian conditional on the observed data $y_{1:t-1}$.

$$\begin{pmatrix}X_t^{E,(i)} \\ X_t^{R,(i)} \end{pmatrix} \mid y_{1:t-1} \sim \mathcal{N} \left( \begin{bmatrix} \mu_t^{E,(i)} \\ \mu_t^{R,(i)} \end{bmatrix}, \begin{bmatrix} P_t^{EE,(i)} & P_t^{ER,(i)} \\ P_t^{RE,(i)} & P_t^{RR,(i)} \end{bmatrix} \right)$$

- $P_t^{EE,(i)} = \text{Var}(X_t^{E,(i)} \mid y_{1:t-1}) \in \mathbb{R}^{HP \times HP}$
- $P_t^{RR,(i)} = \text{Var}(X_t^{R,(i)} \mid y_{1:t-1}) \in \mathbb{R}^{(F-H)P \times (F-H)P}$
- $P_t^{ER,(i)} = \text{Cov}(X_t^{E,(i)}, X_t^{R,(i)} \mid y_{1:t-1}) = (P_t^{RE,(i)})^\top \in \mathbb{R}^{HP \times (F-H)P}$ since the covariance matrix is symmetric.

**4 Bootstrap particle sampling**: Extract the predictive mean and covariance for the latent states involved in an observation $X_t^{E,(i)} \mid y_{1:t-1}$.

$$X_t^{E,(i)} \mid y_{1:t-1} \sim \mathcal{N}(\mu_t^{E,(i)}, P_t^{EE,(i)})$$

where 

$$\mu_t^{E,(i)} = \begin{bmatrix} \mu_{t \mid t-1}^{(i)}[f_1] \\ \mu_{t \mid t-1}^{(i)}[f_2] \\ \vdots \\ \mu_{t \mid t-1}^{(i)}[f_H] \end{bmatrix} \in \mathbb{R}^{HP}$$

$$P_t^{EE,(i)} = \begin{bmatrix} P_{t \mid t-1}^{(i)}[f_1,f_1] & P_{t \mid t-1}^{(i)}[f_1,f_2] & \cdots & P_{t \mid t-1}^{(i)}[f_1,f_H] \\ P_{t \mid t-1}^{(i)}[f_2,f_1] & P_{t \mid t-1}^{(i)}[f_2,f_2] & \cdots & P_{t \mid t-1}^{(i)}[f_2,f_H] \\ \vdots & \vdots & \ddots & \vdots \\ P_{t \mid t-1}^{(i)}[f_H,f_1] & P_{t \mid t-1}^{(i)}[f_H,f_2] & \cdots & P_{t \mid t-1}^{(i)}[f_H,f_H] \end{bmatrix} \in \mathbb{R}^{HP \times HP}$$ 

Sample the $HP$-dimensional latent states involved in an observation $X_t^{E,(i)} \mid y_{1:t-1}$ for each particle $i = 1, \ldots, N$.

**5 Weight Update**: Compute the unnormalized weights $\tilde{w}_t^{(i)}$ for each particle $i = 1, \ldots, N$ using the likelihood function $G_t(y_t \mid X_t^{E,(i)})$.

$$\log (\tilde{w}_t^{(i)}) = \log(w_{t-1}^{(i)}) + \log G_t(y_t \mid X_t^{E,(i)})$$

**6 Kalman Update**: By treating the sampled latent states $X_t^{E,(i)} = \hat{\mu}_t^{E,(i)}$ as noise-free measurements, we can then apply the standard Kalman update to the remaining latent states $X_t^{R,(i)} \mid X_t^{E,(i)} = \hat{\mu}_t^{E,(i)}, y_{1:t-1}$ for each particle $i = 1, \ldots, N$.

$$X_t^{R,(i)} \mid X_t^{E,(i)} = \hat{\mu}_t^{E,(i)}, y_{1:t-1} \sim \mathcal{N}(\mu_t^{R \mid E,(i)}, P_t^{RR \mid E})$$

where

$$\mu_t^{R \mid E,(i)} = \mu_{t \mid t-1}^{R,(i)} + K_t^{(i)} (\hat{\mu}_t^{E,(i)} - \mu_t^{E,(i)})$$

$$P_t^{RR \mid E,(i)} = P_t^{RR,(i)} - K_t^{(i)} P_{t \mid t-1}^{ER,(i)}$$

The exact Kalman Gain equation is given by decomposing the predictive covariance following the Kronecker product structure:

$$\begin{aligned}
K_t^{(i)} &= P_{t \mid t-1}^{RE,(i)} (P_{t \mid t-1}^{EE,(i)})^{-1} \\ &= (\Gamma_t^{RE, -} \otimes B) (\Gamma_t^{EE, -} \otimes B)^{-1} \\ &= (\Gamma_t^{RE, -} \otimes B) ((\Gamma_t^{EE, -})^{-1} \otimes B^{-1})\\ &= \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \otimes I_P
\end{aligned}$$

where $\Gamma_t^{EE, -} \in \mathbb{R}^{H \times H}$ and $\Gamma_t^{RE, -} \in \mathbb{R}^{(F-H) \times H}$ are the submatrices of $\Gamma_t^-$ satisfying $P_{t \mid t-1}^{EE,(i)} = \Gamma_t^{EE, -} \otimes B$ and $P_{t \mid t-1}^{RE,(i)} = \Gamma_t^{RE, -} \otimes B$.

The computational complexity of the Kalman update is $O(H^3)$ from the inversion of $\Gamma_t^{EE, -} \in \mathbb{R}^{H \times H}$ and $O(H^2 + (F-H)H)$ from storage of the covariance matrices.

The Kalman update is then applied for each particle $i = 1, \ldots, N$ to obtain the updated mean and covariance for the remaining latent states $X_t^{R,(i)} \mid X_t^{E,(i)} = \hat{\mu}_t^{E,(i)}, y_{1:t-1}$.

**7 Resampling**: Resample particles $X_t^{E,(i)}$ based on $\tilde{w}_t^{(i)}$ and set $w_t^{(i)} = 1/N$ for $i = 1, \ldots, N$.

## 3 RB-PF Smoothing

We can apply the Forward Filtering Backwards Sampling (FFBSi) to the Rao-Blackwellized particles. The forward filter approximates the distribution at time $T$ as a Gaussian mixture as stated previously.

$$p(X_T \mid y_{1:T}) \approx \sum_{i=1}^{N} w_T^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_T^{E, (i)} \\ m_T^{R \mid E, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_T^{RR \mid E} \end{bmatrix} \right)$$

### Algorithm 

**1 Initialization**:  To start the backward trajectory, we first **select which mixture component** (particle) to sample from, then **sample the actual state** from that component:

1. Sample component index: $I_T \sim \text{Categorical}(w_T^{(1)}, \ldots, w_T^{(N)})$
2. Sample state from component $I_T$: 
$$X_T^* = \begin{pmatrix} X_T^{E,*} \\ X_T^{R,*} \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu_T^{E, (I_T)} \\ m_T^{R \mid E, (I_T)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_T^{RR \mid E} \end{bmatrix}\right)$$

Since $X_T^{E,(I_T)}$ has zero covariance, $X_T^{E,*} = \mu_T^{E,(I_T)}$ exactly (degenerate/point mass). Only $X_T^{R,*}$ is randomly sampled from $\mathcal{N}(m_T^{R \mid E,(I_T)}, P_T^{RR \mid E})$. This gives us one complete smoothed state $X_T^*$ to begin the backward pass.

**2 Backward Simulation of $X_t^*$**: For $t = T-1 \ldots 0$, we sample $X_t^*$ from the conditional distribution $p(X_t \mid X_{t+1}^*, y_{1:t})$.

$$\begin{aligned}
p(X_t \mid X_{t+1}^*, y_{1:t}) &\propto p(X_{t+1}^* \mid X_t) p(X_t \mid y_{1:t}) \\ &= p(X_{t+1}^* \mid X_t) \sum_{i=1}^{N} w_t^{(i)} \mathcal{N}(X_t \mid \mu_t^{(i)}, \Sigma_t^{(i)})
\end{aligned}$$

Since $p(X_{t+1}^* \mid X_t) = \mathcal{N}(X_{t+1}^* \mid \mu_0 + \Phi_{t+1} (X_t - \mu_0), Q_{t+1})$ is linear Gaussian, the product yields a new Gaussian mixture

$$p(X_t \mid X_{t+1}^*, y_{1:t}) \approx \sum_{i=1}^{N} w_{t \mid t+1}^{(i)} \mathcal{N}(X_t \mid m_{t \mid t+1}^{(i)}, P_{t \mid t+1}^{(i)})$$

The backward weights are computed as

$$w_{t \mid t+1}^{(i)} \propto w_t^{(i)} \cdot \mathcal{N}\left(X_{t+1}^* \mid \mu_{t+1 \mid t}^{(i)}, P_{t+1 \mid t}^{(i)}\right)$$

where 

$$\mu_{t+1 \mid t}^{(i)} = \mu_0 + \Phi_{t+1} (\mu_t^{(i)} - \mu_0)$$

$$P_{t+1 \mid t}^{(i)} = \Phi_{t+1} P_t^{(i)} \Phi_{t+1}^\top + Q_{t+1}$$

The conditional mean and covariance follow the Rauch-Tung-Striebel (RTS) backward sampling equations for each particle $i = 1, \ldots, N$:

$$m_{t \mid t+1}^{(i)} = \mu_t^{(i)} + J_t^{(i)} \left(X_{t+1}^* - \mu_{t+1 \mid t}^{(i)}\right)$$

$$\Sigma_{t \mid t+1}^{(i)} = \Sigma_t^{(i)} - J_t^{(i)} P_{t+1 \mid t}^{(i)} {J_t^{(i)}}^\top$$

$$J_t^{(i)} = \Sigma_t^{(i)} \Phi_{t+1}^\top P_{t+1 \mid t}^{(i), -1}$$

**Sampling:**
1. Sample component index: $I_t \sim \text{Categorical}\left(w_{t \mid t+1}^{(1)}, \ldots, w_{t \mid t+1}^{(N)}\right)$
2. Sample state: $X_t^* \sim \mathcal{N}\left(m_{t \mid t+1}^{(I_t)}, \Sigma_{t \mid t+1}^{(I_t)}\right)$

## Appendix

### 2.1 Preserving Kronecker Structure

<!-- For the following proof, we simplify the equations to have $B \in \mathbb{R}^{H \times H}$. -->

Assuming $\Phi_t = \phi_t \otimes I_H$ where $\phi_t \in \mathbb{R}^{F \times F}$ and $I_H \in \mathbb{R}^{H \times H}$, we can decompose the predictive covariance into a Kronecker product structure as follows:

$$\begin{aligned}
P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + Q_t \\
&= \Phi_t (\Gamma_{t-1} \otimes B) \Phi_t^\top + (\Gamma_0 \otimes B) - \Phi_t (\Gamma_0 \otimes B) \Phi_t^\top \\
&= (\phi_t \otimes I_H)(\Gamma_{t-1} \otimes B)(\phi_t^\top \otimes I_H) + (\Gamma_0 \otimes B) - (\phi_t \otimes I_H)(\Gamma_0 \otimes B)(\phi_t^\top \otimes I_H) \\
&= (\phi_t \Gamma_{t-1} \phi_t^\top) \otimes (I_H B I_H) + \Gamma_0 \otimes B - (\phi_t \Gamma_0 \phi_t^\top) \otimes (I_H B I_H) && \text{(mixed-product)} \\
&= (\phi_t \Gamma_{t-1} \phi_t^\top) \otimes B + \Gamma_0 \otimes B - (\phi_t \Gamma_0 \phi_t^\top) \otimes B\\
&= (\phi_t \Gamma_{t-1} \phi_t^\top + \Gamma_0 - \phi_t \Gamma_0 \phi_t^\top) \otimes B && \text{(distributivity)}\\
&= \Gamma_t^{-} \otimes B
\end{aligned}$$

where $\Gamma_t^{-} = \phi_t \Gamma_{t-1} \phi_t^\top + \Gamma_0 - \phi_t \Gamma_0 \phi_t^\top$. We also used the **mixed-product property** of Kronecker products: $(A \otimes B)(C \otimes D) = AC \otimes BD$ and the **distributivity property**: $A \otimes C + B \otimes C = (A + B) \otimes C$. Since the marginal distribution of latent states is conditionally Gaussian, we can decompose the predictive covariance into a block matrix form as follows:

$$P_{t \mid t-1} = \begin{bmatrix} P_{t \mid t-1}^{EE} & P_{t \mid t-1}^{ER} \\ P_{t \mid t-1}^{RE} & P_{t \mid t-1}^{RR} \end{bmatrix} = \begin{bmatrix} \Gamma_t^{EE, -} & \Gamma_t^{ER, -} \\ \Gamma_t^{RE, -} & \Gamma_t^{RR, -} \end{bmatrix} \otimes B$$

Following our Kalman update, we can decompose the Kalman gain as

$$\begin{aligned}
K_t &= P_{t \mid t-1}^{RE} (P_{t \mid t-1}^{EE})^{-1} \\ 
&= (\Gamma_t^{RE, -} \otimes B) (\Gamma_t^{EE, -} \otimes B)^{-1} \\ 
&= (\Gamma_t^{RE, -} \otimes B) ((\Gamma_t^{EE, -})^{-1} \otimes B^{-1}) && \text{(inverse property)} \\ 
&= \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \otimes I_P && \text{(mixed-product)}
\end{aligned}$$

where $\Gamma_t^{R \mid E, -} = \Gamma_t^{RR, -} - \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \Gamma_t^{ER, -}$. We also used the **inverse property** of Kronecker products: $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$. 

The filtering distribution for the full state conditional on the observed data $y_{1:t}$ can be represented as

$$p(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_t^{E, (i)} \\ m_t^{R \mid E, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_t^{RR \mid E} \end{bmatrix} \right)$$

where 

- $P_t^{EE \mid E} = P_t^{EE} - P_t^{EE}(P_t^{EE})^{-1} P_t^{EE} = 0$
- $P_t^{ER \mid E} = P_t^{ER} - P_t^{EE}(P_t^{EE})^{-1} P_t^{ER} = 0$, and 
- $P_t^{RE \mid E} = P_t^{RE} - P_t^{RE}(P_t^{EE})^{-1} P_t^{EE} = 0$. The covariance matrix $\begin{bmatrix} 0 & 0 \\ 0 & P_t^{RR \mid E} \end{bmatrix} \in \mathbb{R}^{HP \times HP}$ where $P_t^{RR \mid E} \in \mathbb{R}^{(F-H)P \times (F-H)P}$.

The conditions for the Kronecker product structure to be preserved are:

1. The initial covariance matrix $\Sigma_0$ must have a Kronecker product structure, i.e., $\Sigma_0 = \Gamma_0 \otimes B$.
2. $\Gamma_t^{EE, -}$ must be positive definite (hence invertible).
3. $B$ must be positive definite (hence invertible) for the Kalman gain to preserve Kronecker structure for the latent states involved in an observation.

### 3.1 Preserving Kronecker Structure in Smoothing

Assuming the filtering covariance maintains the Kronecker structure $\Sigma_t^{(i)} = \Gamma_t \otimes B$ and the predictive covariance is $P_{t+1 \mid t}^{(i)} = \Gamma_{t+1}^{-} \otimes B$, we can decompose the backward sampling gain following the Kronecker product structure:

$$\begin{aligned}
J_t^{(i)} &= \Sigma_t^{(i)} \Phi_{t+1}^\top P_{t+1 \mid t}^{(i), -1} \\
&= (\Gamma_t \otimes B)(\phi_{t+1}^\top \otimes I_P)(\Gamma_{t+1}^{-} \otimes B)^{-1} \\
&= (\Gamma_t \otimes B)(\phi_{t+1}^\top \otimes I_P)((\Gamma_{t+1}^{-})^{-1} \otimes B^{-1}) && \text{(inverse property)} \\
&= \Gamma_t \phi_{t+1}^\top (\Gamma_{t+1}^{-})^{-1} \otimes I_P && \text{(mixed-product)}
\end{aligned}$$

where we used the **mixed-product property** $(A \otimes B)(C \otimes D) = AC \otimes BD$ and the **inverse property** $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$.

The conditional covariance for backward sampling can be decomposed as:

$$\begin{aligned}
\Sigma_{t \mid t+1}^{(i)} &= \Sigma_t^{(i)} - J_t^{(i)} P_{t+1 \mid t}^{(i)} {J_t^{(i)}}^\top \\
&= (\Gamma_t \otimes B) - (\Gamma_t \phi_{t+1}^\top (\Gamma_{t+1}^{-})^{-1} \otimes I_P)(\Gamma_{t+1}^{-} \otimes B)(\Gamma_{t+1}^{-})^{-1} \phi_{t+1} \Gamma_t \otimes I_P) \\
&= (\Gamma_t \otimes B) - (\Gamma_t \phi_{t+1}^\top (\Gamma_{t+1}^{-})^{-1} \Gamma_{t+1}^{-} (\Gamma_{t+1}^{-})^{-1} \phi_{t+1} \Gamma_t) \otimes B \\
&= (\Gamma_t - \Gamma_t \phi_{t+1}^\top (\Gamma_{t+1}^{-})^{-1} \phi_{t+1} \Gamma_t) \otimes B \\
&= \Gamma_{t \mid t+1} \otimes B
\end{aligned}$$

where $\Gamma_{t \mid t+1} = \Gamma_t - \Gamma_t \phi_{t+1}^\top (\Gamma_{t+1}^{-})^{-1} \phi_{t+1} \Gamma_t$, which is also the Schur complement of $\Gamma_{t+1}^{-}$.

The conditions for the Kronecker product structure to be preserved in smoothing are:

1. The filtering covariance $\Sigma_t^{(i)}$ must have Kronecker structure, i.e., $\Sigma_t^{(i)} = \Gamma_t \otimes B$.
2. The predictive covariance $\Gamma_{t+1}^{-}$ must be positive definite (hence invertible).
3. $B$ must be positive definite (hence invertible) for the backward gain to preserve Kronecker structure.

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

## 5 Counterexample: Kronecker product covariance structure with time-varying $B_t$ and time-varying $\Gamma_t$

7. **Obtain new $\Sigma_t$ with Kronecker structure**
   1. $$\begin{aligned}P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top \\ &= \Phi_t (\Gamma_{t-1} \otimes B_{t-1}) \Phi_t^\top + (\Gamma_0 \otimes B_0) - \Phi_t (\Gamma_0 \otimes B_0) \Phi_t^\top\end{aligned}$$
   2. $\Gamma_{t} = \begin{bmatrix} \Gamma_t^{EE} & \Gamma_t^{ER} \\ \Gamma_t^{RE} & \Gamma_t^{RR} \end{bmatrix}$ and assume $B_{t}^{EE} = B_t^{RR} = B_t$
   3. **(NOT COMPLETED)** $$\begin{aligned}K_t &= P_{t}^{RE} (P_{t}^{EE})^{-1} \\ &= [\phi_t^{RE} (\Gamma_t^{RE} \otimes B_t) \phi_t^{RE} + (\Gamma_0^{RE} \otimes B_0)(I - \phi_t^{RE} {\phi_t^{RE}}^\top)] [(\Gamma_t^{EE} \otimes B_t)^{-1}] \end{aligned}$$ 