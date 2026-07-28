# RB-SQMC with Kronecker Structure

CAA: 24/07/26 16:54
Version : 2

## 1 Setup

We define a correlated OU-process in $\mathbb{R}^F$. Let $X_k \in \mathbb{R}^{P \times F}$ be the latent states at time $k$. The initial distribution is a multivariate Gaussian in $\mathbb{R}^{P \times F}$ with a Kronecker product structure,

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B \in \mathbb{R}^{P F \times P F}$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B \in \mathbb{R}^{P \times P}$, where $\Gamma_0$ is a time-varying covariance matrix and $B$ is a static covariance matrix within a index $i \in \{1, \ldots, P\}$

The state transition is governed by the OU-process defined as 

$$X_k = \mu_0 + \Phi_k (X_{k-1} - \mu_0) + \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, Q_k)$$

where $Q_k = \Sigma_0 - \Phi_k \Sigma_0 \Phi_k^\top$ and $\Phi_k = \phi_k \otimes I_P$ with $\phi_k = \exp(-\kappa \Delta t) \cdot I_F \in \mathbb{R}^{F \times F}$, where $Q_k \in \mathbb{R}^{PF \times PF}$ is also a non-diagonal covariance matrix.

Observations arrive at discrete time steps $k = 1, \ldots, T$. Let $y_k \in \mathbb{R}^H$ be the observed data at time $k$ where $H \leq F$ and $X_k^{E} \in \mathbb{R}^{H \times F}$ be the latent states involved in an observation. The likelihood function is defined as a non-linear function of inputs $X_k$,

$$G_k(y_k \mid X_k)$$

## 2 RB-PF

The RB-PF targets the marginal distribution of latent states $X_k^{E} \subseteq X_k$ involved in an observation $y_k$. The joint distribution of latent states and observed can be factorized as 

$$p(X_{0:T}, y_{1:T}) = p(X_{0:T}^{-E} \mid X^{E}_{0:T}) p(X_{0:T}^{E} \mid y_{1:T})$$

where $X_{0:T}^{-E}$ are the latent states not involved in an observation. Since the dynamics of latent states are linear and Gaussian, the conditional distribution of $X_{0:T}^{-E} \mid X^{E}_{1:T}$ is deterministic and computed exactly using the Kalman filter.

### Boostrap Particle Filter

The particle filter performs the weight update as

$$\tilde{w}_t \propto w_{t-1} \cdot \frac{G_t(y_t \mid X_t) p(X_t \mid X_{t-1})}{q(X_t \mid X_{t-1})}$$

where $\tilde{w}_t$ is the unnormalized weight at time $t$. Under the boostrap particle filter, the proposal distribution is the transition distribution $q(X_t \mid X_{t-1}) = p(X_t \mid X_{t-1})$. The weight update simplifies to

$$\tilde{w}_t \propto w_{t-1} \cdot G_t(y_t \mid X_t)$$

$$\log (\tilde{w}_t) = \log(w_{t-1}) + \log G_t(y_t \mid X_t)$$

### Algorithm

**0 Initalization**: Our initiali distribution follows a multivariate Gaussian with Kronecker structure. We generate $N$ particles with equal weights $w_{0}^{(f, i)} = 1/N$ for $f = \{1, \ldots, F\}$ and $i = \{1, \ldots, N\}$.

$$X_0^{(f, i)} \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B \in \mathbb{R}^{P \times P}$, where $\Gamma_0$ is a time-varying covariance matrix and $B$ is a static covariance matrix.

**2 Prediction**: Compute the predictive mean and covariance for the full state $X_t \mid y_{1:t-1}$ using the Kalman filter equations since it is linear and Gaussian.

$$\mu_{t \mid t-1}^{(f, i)} = \mu_0 + \Phi_t (\mu_{t-1}^{(f, i)} - \mu_0)$$

$$P_{t \mid t-1}^{(f, i)} = \Phi_t P_{t-1}^{(f, i)} \Phi_t^\top + Q_t = \Phi_t \Sigma_{t-1}^{(f, i)} \Phi_t^\top + Q_t$$

**3 Exact Gaussian Marginalization**: The predictive distribution for the latent states involved in an observation $X_t^{(E, i)} \mid y_{1:t-1}$ is a multivariate Gaussian conditional on the observed data $y_{1:t-1}$.

$$\begin{pmatrix}X_t^{(E, i)} \\ X_t^{(R, i)} \end{pmatrix} \mid y_{1:t-1} \sim \mathcal{N} \left( \begin{bmatrix} \mu_t^{(E, i)} \\ \mu_t^{(R, i)} \end{bmatrix}, \begin{bmatrix} P_t^{(EE, i)} & P_t^{(ER, i)} \\ P_t^{(RE, i)} & P_t^{(RR, i)} \end{bmatrix} \right)$$

- $P_t^{(EE, i)} = \text{Var}(X_t^{(E, i)} \mid y_{1:t-1}) \in \mathbb{R}^{H \times H}$
- $P_t^{(RR, i)} = \text{Var}(X_t^{(R, i)} \mid y_{1:t-1}) \in \mathbb{R}^{(F-H) \times (F-H)}$
- $P_t^{(ER, i)} = \text{Cov}(X_t^{(E, i)}, X_t^{(R, i)} \mid y_{1:t-1}) = P_t^{(RE, i)} \in \mathbb{R}^{H \times (F-H)}$ since the covariance matrix is symmetric.

**4 Bootstrap particle sampling**: Extract the predictive mean and covariance for the latent states involved in an observation $X_t^{(E, i)} \mid y_{1:t-1}$.

$$X_t^{(E, i)} \mid y_{1:t-1} \sim \mathcal{N}(m_t^{(E, i)}, P_t^{(EE, i)})$$

where 

$$m_t^{(E, i)} = \begin{bmatrix} \mu_{t \mid t-1}^{(f_1, i)} \\ \mu_{t \mid t-1}^{(f_2, i)} \\ \vdots \\ \mu_{t \mid t-1}^{(f_H, i)} \end{bmatrix} \in \mathbb{R}^H$$

$$P_t^{(EE, i)} = \begin{bmatrix} P_{t \mid t-1}^{(f_1 f_1, i)} & P_{t \mid t-1}^{(f_1 f_2, i)} & \cdots & P_{t \mid t-1}^{(f_1 f_H, i)} \\ P_{t \mid t-1}^{(f_2 f_1, i)} & P_{t \mid t-1}^{(f_2 f_2, i)} & \cdots & P_{t \mid t-1}^{(f_2 f_H, i)} \\ \vdots & \vdots & \ddots & \vdots \\ P_{t \mid t-1}^{(f_H f_1, i)} & P_{t \mid t-1}^{(f_H f_2, i)} & \cdots & P_{t \mid t-1}^{(f_H f_H, i)} \end{bmatrix} \in \mathbb{R}^{H \times H}$$ 

Sample the $H$-dimensional latent states involved in an observation $X_t^{E, i} \mid y_{1:t-1} $ for each particle $i = 1, \ldots, N$.

**5 Weight Update**: Compute the unnormalized weights $\tilde{w}_t^{(i)}$ for each particle $i = 1, \ldots, N$ using the likelihood function $G_t(y_t \mid X_t^{E, i})$.

$$\log (\tilde{w}_t^{(i)}) = \log(w_{t-1}^{(i)}) + \log G_t(y_t \mid X_t^{E, i})$$

**6 Kalman Update**: By treating the sampled latent states $X_t^{E, i} = \mu_t^{E, i}$ as noise-free measurements, we can then apply the standard Kalman update to the remaining latent states $X_t^{R, i} \mid X_t^{E, i} = \mu_t^{E, i}, y_{1:t-1}$ for each particle $i = 1, \ldots, N$.

$$X_t^{R, i} \mid X_t^{E, i} = \mu_t^{E, i}, y_{1:t-1} \sim \mathcal{N}(m_t^{R \mid E, (i)}, P_t^{R \mid E})$$

where

$$m_t^{R \mid E, (i)} = m_{t \mid t-1}^{R} + K (\mu_t^{E, (i)} - m_t^{E})$$

$$P_t^{R \mid E} = P_t^{RR} - K P_{t \mid t-1}^{ER}$$

where $K = P_{t \mid t-1}^{RE} (P_{t \mid t-1}^{EE})^{-1}$ is the Kalman gain. The Kalman update is performed for each particle $i = 1, \ldots, N$.

**7 Resampling**: Resample particles $X_t^{E, i}$ based on $\tilde{w}_t^{(i)}$ and set $w_t^{(i)} = 1/N$ for $i = 1, \ldots, N$. 

### Preserving Kronecker Structure

For the following proof, we simplify the equations to have $B \in \mathbb{R}^{P \times P}$.

In our exact Gaussian Marginalization step, our predictive covariance can be decomposed using the **mixed-product property** of Kronecker products:

$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

Assuming $\Phi_t = \phi_t \otimes I_P$ where $\phi_t \in \mathbb{R}^{F \times F}$ and $I_P \in \mathbb{R}^{P \times P}$, we have:

$$\begin{aligned}
P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + Q_t \\
&= \Phi_t (\Gamma_{t-1} \otimes B) \Phi_t^\top + (\Gamma_0 \otimes B) - \Phi_t (\Gamma_0 \otimes B) \Phi_t^\top \\
&= (\phi_t \otimes I_P)(\Gamma_{t-1} \otimes B)(\phi_t^\top \otimes I_P) + (\Gamma_0 \otimes B) - (\phi_t \otimes I_P)(\Gamma_0 \otimes B)(\phi_t^\top \otimes I_P) \\
&= (\phi_t \Gamma_{t-1} \phi_t^\top) \otimes (I_P B I_P) + \Gamma_0 \otimes B - (\phi_t \Gamma_0 \phi_t^\top) \otimes (I_P B I_P) \\
&= (\phi_t \Gamma_{t-1} \phi_t^\top) \otimes B + \Gamma_0 \otimes B - (\phi_t \Gamma_0 \phi_t^\top) \otimes B \\
&= \underbrace{(\phi_t \Gamma_{t-1} \phi_t^\top + \Gamma_0 - \phi_t \Gamma_0 \phi_t^\top)}_{\Gamma_t^{-}} \otimes B \\
&= \Gamma_t^{-} \otimes B
\end{aligned}$$

where $\Gamma_t^{-} = \phi_t \Gamma_{t-1} \phi_t^\top + \Gamma_0 - \phi_t \Gamma_0 \phi_t^\top$. Marginalizing out the latent states involved in an observation $X_t^{E, i}$, we have

$$P_{t \mid t-1} = \begin{bmatrix} P_{t \mid t-1}^{EE} & P_{t \mid t-1}^{ER} \\ P_{t \mid t-1}^{RE} & P_{t \mid t-1}^{RR} \end{bmatrix} = \begin{bmatrix} \Gamma_t^{EE, -} & \Gamma_t^{ER, -} \\ \Gamma_t^{RE, -} & \Gamma_t^{RR, -} \end{bmatrix} \otimes B$$

using the **block matrix multiplication property** of Kronecker products. Following our Kalman update, we can decompose the Kalman gain as 

$$\begin{aligned}
K_t &= P_{t \mid t-1}^{RE} (P_{t \mid t-1}^{EE})^{-1} \\ &= (\Gamma_t^{RE, -} \otimes B) (\Gamma_t^{EE, -} \otimes B)^{-1} \\ &= \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \otimes I_P
\end{aligned}$$

where $\Gamma_t^{R \mid E, -} = \Gamma_t^{RR, -} - \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \Gamma_t^{ER, -}$. This simplification allows us to only compute $B$ and use the Kronecker product structure for the full covariance matrix $\Sigma_t$.

The filtering distribution for the full state conditional on the observed data $y_{1:t}$ can be represented as

$$p(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_t^{E, (i)} \\ m_t^{R \mid E, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_t^{R \mid E} \end{bmatrix} \right)$$

where $P_t^{EE \mid E} = P_t^{EE} - P_t^{EE}(P_t^{EE})^{-1} P_t^{EE} = 0$, $P_t^{ER \mid E} = P_t^{ER} - P_t^{EE}(P_t^{EE})^{-1} P_t^{ER} = 0$, and $P_t^{RE \mid E} = P_t^{RE} - P_t^{RE}(P_t^{EE})^{-1} P_t^{EE} = 0$. The covariance matrix $\begin{bmatrix} 0 & 0 \\ 0 & P_t^{R \mid E} \end{bmatrix} \in \mathbb{R}^{FP \times FP}$ where $P_t^{R \mid E} \in \mathbb{R}^{(F-H)P \times (F-H)P}$.

The conditions for the Kronecker product structure to be preserved are:

1. The initial covariance matrix $\Sigma_0$ must have a Kronecker product structure, i.e., $\Sigma_0 = \Gamma_0 \otimes B$.
2. $\Gamma_t^{EE, -}$ must be positive definite (hence invertible).
3. $B$ must be positive definite (hence invertible) for the Kalman gain to preserve Kronecker structure for the latent states involved in an observation.

## 3 RB-PF Smoothing

We can apply the Forward Filtering Backwards Sampling (FFBSi) to the Rao-Blackwellized particles. The forward filter approximates the distribution at time $T$ as a Gaussian mixture as stated previously.

$$p(X_T \mid y_{1:T}) \approx \sum_{i=1}^{N} w_T^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_T^{E, (i)} \\ m_T^{R \mid E, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_T^{R \mid E} \end{bmatrix} \right)$$


### Algorithm 

**1 Initialization**: The forward filter at time $T$ approximates the posterior as a mixture of $N$ Gaussians:

$$p(X_T \mid y_{1:T}) \approx \sum_{i=1}^{N} w_T^{(i)} \mathcal{N}\left(\mu_T^{(i)}, P_T\right)$$

To start the backward trajectory, we first **select which mixture component** (particle) to sample from, then **sample the actual state** from that component:

1. Sample component index: $I_T \sim \text{Categorical}(w_T^{(1)}, \ldots, w_T^{(N)})$
2. Sample state from component $I_T$: 

$$X_T^* = \begin{pmatrix} X_T^{E,*} \\ X_T^{R,*} \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu_T^{E, (I_T)} \\ m_T^{R \mid E, (I_T)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_T^{R \mid E} \end{bmatrix}\right)$$

Since $X_T^{E,(I_T)}$ has zero covariance, $X_T^{E,*} = \mu_T^{E,(I_T)}$ exactly (degenerate/point mass). Only $X_T^{R,*}$ is randomly sampled from $\mathcal{N}(m_T^{R \mid E,(I_T)}, P_T^{R \mid E})$. This gives us one complete smoothed state $X_T^*$ to begin the backward pass.

**2 Backward Sampling**: For $t = T-1, \ldots, 1$, we sample the smoothed state given the future trajectory $X_{t+1}^*$. Since $X_t^E$ is a sampled particle (degenerate) and $X_t^R$ is Gaussian, we handle them separately.

First, compute backward weights by evaluating the transition density:

$$w_{t \mid t+1}^{(i)} \propto w_t^{(i)} \cdot p(X_{t+1}^{E,*} \mid X_t^{E,(i)}) \cdot \mathcal{N}\left(X_{t+1}^{R,*} \mid \mu_{t+1 \mid t}^{R,(i)}, P_{t+1 \mid t}^{RR}\right)$$

where the predictive mean and covariance for particle $i$ are:
$$\mu_{t+1 \mid t}^{(i)} = \mu_0 + \Phi_{t+1}(\mu_t^{(i)} - \mu_0), \quad P_{t+1 \mid t} = \Phi_{t+1} P_t \Phi_{t+1}^\top + Q_{t+1}$$

Sample an index $I_t \sim \text{Categorical}(w_{t \mid t+1}^{(1)}, \ldots, w_{t \mid t+1}^{(N)})$ and set $X_t^{E,*} = \mu_t^{E,(I_t)}$.

**3 Conditional Sampling of $X_t^{R,*}$**: Given $X_t^{E,*}$ and $X_{t+1}^*$, sample $X_t^{R,*}$ from the conditional Gaussian. Using the partitioned transition:

$$\begin{pmatrix} X_{t+1}^{E,*} \\ X_{t+1}^{R,*} \end{pmatrix} \mid X_t \sim \mathcal{N}\left(\mu_0 + \Phi_{t+1}(X_t - \mu_0), Q_{t+1}\right)$$

The conditional distribution $X_t^R \mid X_t^E, X_{t+1}^*$ is Gaussian with mean and covariance derived from the Rauch-Tung-Striebel (RTS) smoother applied to the $R$ block:

$$m_{t \mid T}^{R,(I_t)} = m_t^{R,(I_t)} + J_t^R (X_{t+1}^{R,*} - \mu_{t+1 \mid t}^{R,(I_t)})$$

$$\Sigma_{t \mid T}^{R} = \Sigma_t^{R \mid E} - J_t^R \Sigma_{t+1 \mid t}^{RR} (J_t^R)^\top$$

where $J_t^R = \Sigma_t^{R \mid E} \Phi_{t+1}^{RR} (\Sigma_{t+1 \mid t}^{RR})^{-1}$ is the smoother gain for the $R$ component. Sample $X_t^{R,*}$ from this conditional distribution:

$$X_t^{R,*} \sim \mathcal{N}\left(m_{t \mid T}^{R,(I_t)}, \Sigma_{t \mid T}^{R}\right)$$

This completes the backward step, giving us the full smoothed state $X_t^* = (X_t^{E,*}, X_t^{R,*})$ with covariance $\Sigma_{t \mid T} = \begin{bmatrix} 0 & 0 \\ 0 & \Sigma_{t \mid T}^{R} \end{bmatrix} = \Gamma_{t \mid T} \otimes B$. Repeat Steps 2-3 for $t = T-1, \ldots, 1$ to obtain one complete smoothed trajectory.

**Preserving Kronecker Structure in Smoothing**: The smoothed covariance for particle $I_t$ can be written as:

$$P_{t \mid T} = \begin{bmatrix} 0 & 0 \\ 0 & P_{t \mid T}^{R} \end{bmatrix}$$

where $P_{t \mid T}^{R}$ is obtained from the RTS smoother recursion. Starting from the filtering covariance $P_t^{R \mid E} = \Gamma_t^{R \mid E, -} \otimes B$ where $\Gamma_t^{R \mid E, -} = \Gamma_t^{RR, -} - \Gamma_t^{RE, -}(\Gamma_t^{EE, -})^{-1}\Gamma_t^{ER, -}$, the smoothed covariance follows:

$$P_{t \mid T}^{R} = P_t^{R \mid E} + J_t^R (P_{t+1 \mid T}^{R} - P_{t+1 \mid t}^{RR}) (J_t^R)^\top$$

The smoother gain decomposes using Kronecker structure:

$$J_t^R = P_t^{R \mid E} \Phi_{t+1}^{RR} (P_{t+1 \mid t}^{RR})^{-1} = \underbrace{\Gamma_t^{R \mid E, -} \phi_{t+1}^{RR} (\Gamma_{t+1}^{RR, -})^{-1}}_{\tilde{J}_t^R} \otimes I_P$$

Therefore, the smoothed covariance preserves Kronecker structure:

$$P_{t \mid T}^{R} = \left[\Gamma_t^{R \mid E, -} + \tilde{J}_t^R (\Gamma_{t+1 \mid T}^{RR} - \Gamma_{t+1}^{RR, -}) (\tilde{J}_t^R)^\top\right] \otimes B = \Gamma_{t \mid T}^{RR} \otimes B$$

This shows the Kronecker structure is preserved throughout the backward pass with the same $B$ matrix. 


## 4 Counterexample: Kronecker product covariance structure with time-varying $B_t$ and time-varying $\Gamma_t$

7. **Obtain new $\Sigma_t$ with Kronecker structure**
   1. $$\begin{aligned}P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top \\ &= \Phi_t (\Gamma_{t-1} \otimes B_{t-1}) \Phi_t^\top + (\Gamma_0 \otimes B_0) - \Phi_t (\Gamma_0 \otimes B_0) \Phi_t^\top\end{aligned}$$
   2. $\Gamma_{t} = \begin{bmatrix} \Gamma_t^{EE} & \Gamma_t^{ER} \\ \Gamma_t^{RE} & \Gamma_t^{RR} \end{bmatrix}$ and assume $B_{t}^{EE} = B_t^{RR} = B_t$
   3. **(NOT COMPLETED)** $$\begin{aligned}K_t &= P_{t}^{RE} (P_{t}^{EE})^{-1} \\ &= [\phi_t^{RE} (\Gamma_t^{RE} \otimes B_t) \phi_t^{RE} + (\Gamma_0^{RE} \otimes B_0)(I - \phi_t^{RE} {\phi_t^{RE}}^\top)] [(\Gamma_t^{EE} \otimes B_t)^{-1}] \end{aligned}$$ 