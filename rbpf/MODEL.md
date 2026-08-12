# RBPF SQMC

We model the latent attack/defence strengths of $M$ football teams, $X_t^m = (X_t^{m,\text{att}}, X_t^{m,\text{def}})$ for team $m$ at time $t$, evolving as a random walk. Observed goals are modelled by a bivariate Poisson distribution.

## 1 Setup

**Initial**

$X_0^m = (X_0^{m,\text{att}}, X_0^{m,\text{def}})$ is the initial latent state for team $m$ and is distributed as

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

**Transition Distribution**

$$X_t = X_{t-1} + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, \Delta_t Q)$$

where $\Delta_t$ is the time difference, $Q = \Gamma_Q \otimes B_Q$ is a non-diagonal Kronecker product covariance matrix, where $\Gamma_Q \in \mathbb{R}^{M \times M}$ and $B_Q \in \mathbb{R}^{2 \times 2}$. This implies that team evolutions are correlated differently while all teams share the same covariance evolution between attack and defence.

**Likelihood**

$y_t = (y_t^{\text{h}}, y_t^{\text{a}})$ is the observed goals for the home $X_t^{\text{h}}$ and away $X_t^{\text{a}}$ teams at time $t$.

$$G_t(y_t \mid x_t^{\text{h}}, x_t^{\text{a}}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{\text{h}}}}{y_t^{\text{h}}!} \frac{\lambda_2^{y_t^{\text{a}}}}{y_t^{\text{a}}!} \sum_{k=0}^{\min(y_t^{\text{h}}, y_t^{\text{a}})} \binom{y_t^{\text{h}}}{k} \binom{y_t^{\text{a}}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, \text{h}} - x_t^{\text{def}, \text{a}})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, \text{a}} - x_t^{\text{def}, \text{h}})$, $\lambda_3 = \exp(\beta)$.

Since only $X_t^{\mathcal{O}_t} = (X_t^{\text{h}}, X_t^{\text{a}})$ enter the likelihood, the remaining latent states are represented analytically as a Gaussian conditional.

## 2 RB-PF

**1 Prediction**

$$\mu_{t \mid t -1 } = \mu_{t - 1 \mid t - 1}$$

$$\Sigma_{t \mid t - 1} = \Sigma_{t - 1 \mid t - 1} + Q$$

**2 Bootstrap Particle Sampling**: Assume that the proposal distribution is the same as the transition distribution $q (X_t \mid X_{t-1}) = p(X_t \mid X_{t-1}) = \mathcal{N}(X_{t-1}, Q)$.

$$\begin{pmatrix}X_{t}^{\mathcal{O}_t} \\ X_{t}^{\mathcal{R}_t}\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\mu_{t \mid t - 1}^{\mathcal{O}_t} \\ \mu_{t \mid t - 1}^{\mathcal{R}_t}\end{pmatrix}, \begin{pmatrix}\Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{O}_t} & \Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{R}_t} \\ \Sigma_{t \mid t - 1}^{\mathcal{R}_t \mathcal{O}_t} & \Sigma_{t \mid t - 1}^{\mathcal{R}_t \mathcal{R}_t}\end{pmatrix}\right)$$

**3 Compute weights**

$$\log \tilde{w}_{t}^{(i)} = \log w_{t - 1}^{(i)} + \log G_{t}(y_{t} \mid X_{t}^{\mathcal{O}_t, (i)})$$

**4 Exact Marginalization**

$$X_t^{\mathcal{R}_t} \mid X_{t}^{\mathcal{O}_t} \sim \mathcal{N}(\mu_{t \mid t - 1}^{\mathcal{R}_t \mid \mathcal{O}_t}, \Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t \mid \mathcal{O}_t})$$

- $\mu_{t \mid t - 1}^{\mathcal{R}_t \mid \mathcal{O}_t} = \mu_{t \mid t - 1}^{\mathcal{R}_t} + K_t (X_{t}^{\mathcal{O}_t} - \mu_{t \mid t - 1}^{\mathcal{O}_t})$
- $\Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t \mid \mathcal{O}_t} = \Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t} - K_t \Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{R}_t}$

**5 Resampling**: Resample particles according to their weights.

## 3 RB-PF Smoothing

We use Forward Filtering Backward Simulation (FFBSi) to sample backwards in time from the smoothing distribution $p(X_{0:T} \mid y_{1:T})$.


**1 Initialization**: Sample particle index $I_T$ using the normalized weights $w_T^{(i)}$. Draw full terminal state from corresponding Gaussian mixture $p(X_{0:T} \mid y_{1:T}, \Theta^{(k)}) \approx \sum_{i=1}^N w_T^{(i)} \mathcal{N}(\mu_T^{(i)}, \Sigma_T)$.

$$X_T^{*, (I_T)} \sim \mathcal{N}(\mu_T^{(I_T)}, \Sigma_T)$$

**2 Backward Sampling**: From T-1 to 0, sample particle index $I_t$ using the backward kernel $p(X_t \mid X_{t + 1}^*, y_{1:t})$ and draw full state from corresponding Gaussian mixture $\sum_{i=1}^N w_t^{(i)} \mathcal{N}(\mu_t^{(i)}, \Sigma_t)$.

$$\begin{aligned}
p(X_t \mid X_{t + 1}^*, y_{1:t}) &\propto p(X_{t + 1}^* \mid X_t) p(X_t \mid y_{1:t}) \\ &\propto p(X_{t + 1}^* \mid X_t) \sum_{i=1}^N w_t^{(i)} \mathcal{N}(\mu_t^{(i)}, \Sigma_t)
\end{aligned}$$

Since transition $p(X_{t + 1}^* \mid X_t)$ is Gaussian, the backward kernel is a Gaussian mixture with $N$ components.

$$p(X_t \mid X_{t + 1}^*, y_{1:t}) \approx \sum_{i=1}^N w_{t \mid t+1}^{(i)} \mathcal{N}(\mu_{t \mid t+1}^{(i)}, \Sigma_{t \mid t+1})$$

where $w_{t \mid t+1}^{(i)} \propto w_t^{(i)} \mathcal{N}(X_{t + 1}^* \mid \mu_{t + 1 \mid t}^{(i)}, \Sigma_{t + 1 \mid t})$. Based on RTS equations, 

$$\mu_{t \mid t+1}^{(i)} = \mu_t^{(i)} + J_t (X_{t + 1}^* - \mu_{t + 1 \mid t}^{(i)})$$

$$\Sigma_{t \mid t+1} = \Sigma_t - J_t \Sigma_{t + 1 \mid t} J_t^T$$

where $J_t = \Sigma_t(\Sigma_t + Q)^{-1}$ is the RTS smoother gain.

## 4 Parameter Estimation - Expectation-Maximization (EM)

Let $\Theta = (\mu_0, \Sigma_0, \Gamma_Q, B_Q, \alpha, \beta)$ be the model parameters. We can estimate $\Theta$ using maximum likelihood estimation (MLE) or Bayesian inference.

**1 E-step**: Approximate the smoothing distribution $p(X_{0:T} \mid y_{1:T}, \Theta^{(k)})$ via FFBSi (Section 3), then the expected complete log-likelihood via Monte Carlo:

$$\begin{aligned}A(\Theta \mid \Theta^{(k)}) &= E(\log p(X_{0:T}, y_{1:T} \mid \Theta) \mid y_{1:T}, \Theta^{(k)}) \\ &= \int \log p(X_{0:T}, y_{1:T} \mid \Theta) p(X_{0:T} \mid y_{1:T}, \Theta^{(k)}) dX_{0:T} \\ &\approx \frac{1}{N} \sum_{i=1}^{N} \log p(X_{0:T}^{(i)}, y_{1:T} \mid \Theta) \\ &= \frac{1}{N} \sum_{i=1}^N \left[\log p_{\mu_0, \Sigma_0}(X_{0}^{(i)}) + \sum_{t=1}^T \log p_{\mu_t, \Sigma_t}(X_{t}^{(i)} \mid X_{t-1}^{(i)}) + \sum_{t=1}^T \log p_{\alpha, \beta}(y_{t} \mid X_{t}^{\mathcal{O}_t, (i)}) \right] \end{aligned}$$

**2 M-step**: Maximize $A(\Theta \mid \Theta^{(k)})$ with respect to $\Theta$ to obtain the updated parameter estimates $\Theta^{(k+1)}$.

$$\Theta^{(k+1)} = \arg \max_{\Theta} A(\Theta \mid \Theta^{(k)}) = \arg \max_{\mu_0, \Sigma_0} A_{init} + \arg \max_{\Gamma_Q, B_Q} A_Q + \arg \max_{\alpha, \beta} A_{obs}$$

## 5 SQMC