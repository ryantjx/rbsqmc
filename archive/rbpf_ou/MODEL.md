## RB-SQMC with Kronecker Structure

## 1 Setup

**Initial**

For $m = 1, \ldots, M$ teams. $X_0^m = (X_0^{m,\text{att}}, X_0^{m,\text{def}})$ is the initial latent state for team $m$ and is distributed as

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\mu_0 \in \mathbb{R}^{M \times 2}$ and $\Sigma_0 = \gamma_0 \otimes B \in \mathbb{R}^{2M \times 2M}$ where $\Gamma_0 \in \mathbb{R}^{M \times M}$ and $B \in \mathbb{R}^{2 \times 2}$.

**Transition Distribution**

OU-process where covariance depends on time difference between previous observation.

$$X_t = \mu_0 + \phi_t (X_{t-1} - \mu_0) + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

where $\phi_t = \exp(-\kappa \Delta t)$, $\Delta_t$ is the time difference between current observation and the previous observation, and $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^T$.

**Likelihood**

Bivariate Poisson likelihood for observed goals. $y_t = (y_t^{\text{h}}, y_t^{\text{a}})$ is the observed goals for the home and away teams at time $t$.

$$G_t(y_t \mid x_t^{\text{h}}, x_t^{\text{a}}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{\text{h}}}}{y_t^{\text{h}}!} \frac{\lambda_2^{y_t^{\text{a}}}}{y_t^{\text{a}}!} \sum_{k=0}^{\min(y_t^{\text{h}}, y_t^{\text{a}})} \binom{y_t^{\text{h}}}{k} \binom{y_t^{\text{a}}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, \text{h}} - x_t^{\text{def}, \text{a}})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, \text{a}} - x_t^{\text{def}, \text{h}})$, $\lambda_3 = \exp(\beta)$.

## 2 RB-SQMC

**1 Prediction**

$$X_t \sim \mathcal{N}(\mu_t, \Sigma_t)$$

$$\mu_{t + 1 \mid t} = \mu_0 + \phi_{t + 1} (\mu_{t \mid t} - \mu_0)$$

$$\Sigma_{t + 1 \mid t} = \Sigma_0 - \Phi_{t + 1} \Sigma_0 \Phi_{t + 1}^T$$

**2 Bootstrap Particle Sampling**

$$\begin{pmatrix}X_{t + 1}^{\mathcal{O}} \\ X_{t + 1}^{\mathcal{R}}\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\mu_{t + 1 \mid t}^{\mathcal{O}} \\ \mu_{t + 1 \mid t}^{\mathcal{R}}\end{pmatrix}, \begin{pmatrix}\Sigma_{t + 1 \mid t}^{\mathcal{OO}} & \Sigma_{t + 1 \mid t}^{\mathcal{OR}} \\ \Sigma_{t + 1 \mid t}^{\mathcal{RO}} & \Sigma_{t + 1 \mid t}^{\mathcal{RR}}\end{pmatrix}\right)$$

$$X_{t + 1}^{\mathcal{O}} \sim \mathcal{N}(\mu_{t + 1 \mid t}^\mathcal{O}, \Sigma_{t + 1 \mid t}^\mathcal{O})$$

**3 Compute weights**

$$\log \tilde{w}_{t + 1}^{(i)} = \log w_{t + 1}^{(i)} + \log G_{t + 1}(y_{t + 1} \mid X_{t + 1}^{\mathcal{O}})^{(i)}$$

**4 Exact Marginalization**

$$X_{t + 1}^{\mathcal{R}} \mid X_{t + 1}^{\mathcal{O}} \sim \mathcal{N}(\mu_{t + 1 \mid t}^{\mathcal{R} \mid \mathcal{O}}, \Sigma_{t + 1 \mid t}^{\mathcal{RR} \mid \mathcal{O}})$$

where

1. $\mu_{t + 1 \mid t}^{\mathcal{R} \mid \mathcal{O}} = \mu_{t + 1 \mid t}^{\mathcal{R}} + K_t (X_{t + 1}^{\mathcal{O}} - \mu_{t + 1 \mid t}^{\mathcal{O}})$
2. $\Sigma_{t + 1 \mid t}^{\mathcal{RR} \mid \mathcal{O}} = \Sigma_{t + 1 \mid t}^{\mathcal{RR}} - K_t \Sigma_{t + 1 \mid t}^{\mathcal{OR}}$

## 3 RB-SQMC Smoothing

**Forward filtering backward simulation (FFBSi)** to sample backwards in time from the smoothing distribution $p(X_{0:T} \mid y_{1:T})$.

**1 Initialization**: Sample particle index $I_T$ using the normalized weights $w_T^{(i)}$. Draw full terminal state from corresponding Gaussian mixture $\sum_{i=1}^N w_T^{(i)} \mathcal{N}(\mu_T^{(i)}, \Sigma_T)$.

$$X_T^{*, (I_T)} \sim \mathcal{N}(\mu_T^{(I_T)}, \Sigma_T)$$

**2 Backward Sampling**: From T-1 to 0, sample particle index $I_t$ using the backward kernel $p(X_t \mid X_{t + 1}^*, y_{1:t})$ and draw full state from corresponding Gaussian mixture $\sum_{i=1}^N w_t^{(i)} \mathcal{N}(\mu_t^{(i)}, \Sigma_t)$.

$$\begin{aligned}
p(X_t \mid X_{t + 1}^*, y_{1:t}) &\propto p(X_{t + 1}^* \mid X_t) p(X_t \mid y_{1:t}) \\ &\propto p(X_{t + 1}^* \mid X_t) \sum_{i=1}^N w_t^{(i)} \mathcal{N}(\mu_t^{(i)}, \Sigma_t)
\end{aligned}$$

Since transition $p(X_{t + 1}^* \mid X_t)$ is Gaussian, the backward kernel is a Gaussian mixture with $N$ components.

$$p(X_t \mid X_{t + 1}^*, y_{1:t}) \approx \sum_{i=1}^N w_{t \mid t+1}^{(i)} \mathcal{N}(\mu_{t \mid t+1}^{(i)}, \Sigma_{t \mid t+1})$$

where $w_{t \mid t+1}^{(i)} \propto w_t^{(i)} \mathcal{N}(X_{t + 1}^* \mid \mu_{t + 1 \mid t}^{(i)}, \Sigma_{t + 1 \mid t})$. Based on RTS equations, 

$$\mu_{t \mid t+1}^{(i)} = \mu_t^{(i)} + J_t (X_{t + 1}^* - \mu_{t + 1 \mid t}^{(i)})$$

$$\Sigma_{t \mid t+1} = \Sigma_t - J_t \Sigma_{t + 1 \mid t} J_t^{(i)T}$$

where $J_t = \Sigma_t \Phi_{t + 1}^T (\Sigma_{t + 1 \mid t})^{-1}$.

## 4 RB-SQMC Parameter Estimation

Use the Rauch-Tung-Striebel (RTS) smoother to compute smoothed state estimates and the expected complete log-likelihood to estimate parameters $\theta = (\mu_0, \gamma_0, B, \kappa, \alpha, \beta)$.

Time Series and its Applications
1. Parameter Estimation with non-linear likelihood - smoothing 
   1. Newton Raphson
   2. Shumway and Stoffer (1982)

## Notes

- 290726: OU-process with asynchronous observations resulted in a non positive-definite matrix during the Prediction step i.e. $Q_t = (\Gamma_0 - \phi_t \Gamma_0 \phi_t^\top) \otimes B$. changed to a random walk. TrueSkill2 / Glicko avoid by modelling latent states for each index as non-correlated i.e. $Q_t = \text{diag}(\Gamma_0) \otimes B$ which is positive definite. This is a limitation of the OU-process with asynchronous observations.