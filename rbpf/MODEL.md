# RBPF SQMC

We model the latent attack/defence strengths of $M$ football teams, $X_t^m = (X_t^{m,\text{att}}, X_t^{m,\text{def}})$ for team $m$ at time $t$, evolving as a random walk. Observed goals are modelled by a bivariate Poisson distribution.

## 1 Setup

**Initial**

$X_0^m = (X_0^{m,\text{att}}, X_0^{m,\text{def}})$ is the initial latent state for team $m$ and is distributed as

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B_0$ is the Kronecker product of the $2 \times 2$ covariance matrix $\Gamma_0$ and the $M \times M$ correlation matrix $B_0$.

**Transition**

Transition follows an OU-process with a time delta $\Delta t$ shared across all teams.

$$X_{t-1} = \mu_0 + \phi_t (X_{t-1} - \mu_0) + \epsilon_t$$

where $\phi_t = \exp(- \kappa \Delta t)$, $\epsilon_t \sim \mathcal{N}(0, Q)$ and $Q = (1 - \phi_t^2) \Sigma_0 $ is the covariance matrix of the process noise.

**Likelihood**

$y_t = (y_t^{\text{h}}, y_t^{\text{a}})$ is the observed goals for the home $X_t^{\text{h}}$ and away $X_t^{\text{a}}$ teams at time $t$.

$$G_t(y_t \mid x_t^{\text{h}}, x_t^{\text{a}}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{\text{h}}}}{y_t^{\text{h}}!} \frac{\lambda_2^{y_t^{\text{a}}}}{y_t^{\text{a}}!} \sum_{k=0}^{\min(y_t^{\text{h}}, y_t^{\text{a}})} \binom{y_t^{\text{h}}}{k} \binom{y_t^{\text{a}}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, \text{h}} - x_t^{\text{def}, \text{a}})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, \text{a}} - x_t^{\text{def}, \text{h}})$, $\lambda_3 = \exp(\beta)$.

Since only $X_t^{\mathcal{O}_t} = (X_t^{\text{h}}, X_t^{\text{a}})$ enter the likelihood, the remaining latent states are represented analytically as a Gaussian conditional.

## 2 RB-PF

**1 Prediction**

$$\mu_{t \mid t -1 } = \mu_0 + \phi_t (\mu_{t-1}  - \mu_0)$$


$$\Sigma_{t \mid t - 1} = \phi_t^2 \Sigma_{t-1 \mid t-1} + (1 - \phi_t^2) \Sigma_0$$

**2 Bootstrap Particle Sampling**: Assume that the proposal distribution is the same as the transition distribution $q (X_t \mid X_{t-1}) = p(X_t \mid X_{t-1}) = \mathcal{N}(\mu_0 + \phi_t (X_{t-1} - \mu_0), Q)$.

$$\begin{pmatrix}X_{t}^{\mathcal{O}_t} \\ X_{t}^{\mathcal{R}_t}\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\mu_{t \mid t - 1}^{\mathcal{O}_t} \\ \mu_{t \mid t - 1}^{\mathcal{R}_t}\end{pmatrix}, \begin{pmatrix}\Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{O}_t} & \Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{R}_t} \\ \Sigma_{t \mid t - 1}^{\mathcal{R}_t \mathcal{O}_t} & \Sigma_{t \mid t - 1}^{\mathcal{R}_t \mathcal{R}_t}\end{pmatrix}\right)$$

**3 Compute weights**

$$\log \tilde{w}_{t}^{(i)} = \log w_{t - 1}^{(i)} + \log G_{t}(y_{t} \mid X_{t}^{\mathcal{O}_t, (i)})$$

**4 Exact Marginalization**

$$X_t^{\mathcal{R}_t} \mid X_{t}^{\mathcal{O}_t} \sim \mathcal{N}(\mu_{t \mid t - 1}^{\mathcal{R}_t \mid \mathcal{O}_t}, \Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t \mid \mathcal{O}_t})$$

- $\mu_{t \mid t - 1}^{\mathcal{R}_t \mid \mathcal{O}_t} = \mu_{t \mid t - 1}^{\mathcal{R}_t} + K_t (X_{t}^{\mathcal{O}_t} - \mu_{t \mid t - 1}^{\mathcal{O}_t})$
- $\Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t \mid \mathcal{O}_t} = \Sigma_{t \mid t - 1}^{\mathcal{R}_t\mathcal{R}_t} - K_t \Sigma_{t \mid t - 1}^{\mathcal{O}_t \mathcal{R}_t}$

where $K = \Sigma^{\mathcal{R}_t \mathcal{O}_t} (\Sigma^{\mathcal{O}_t \mathcal{O}_t})^{-1}$ is the Kalman gain.

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

### 3.1 FFBSi 

In `cuthbert`, it is using exact backward kernel, so there is no sampling of the weighted Gaussian mixture.

## 4 Parameter Estimation - Expectation-Maximization (EM)

### 4.1 Maximize the Joint Likelihood

Let $\Theta = (\Gamma_0, B, \alpha, \beta)$ be the model parameters, where $\Sigma_0 = \Gamma_0 \otimes B$ is the Kronecker product of the $M \times M$ team covariance $\Gamma_0$ and the $2 \times 2$ attack/defence covariance $B$. The initial mean $\mu_0$ is held fixed (not estimated). We can estimate $\Theta$ using maximum likelihood estimation (MLE) or Bayesian inference.

**1 E-step**: Approximate the smoothing distribution $p(X_{0:T} \mid y_{1:T}, \Theta^{(k)})$ via FFBSi (Section 3), then the expected complete log-likelihood via Monte Carlo using $M$ smoothed trajectories $X_{0:T}^{(i)}$.

$$\begin{aligned}Q(\Theta \mid \Theta^{(k)}) &= E(\log p(X_{0:T}, y_{1:T} \mid \Theta) \mid y_{1:T}, \Theta^{(k)}) \\ &= \int \log p(X_{0:T}, y_{1:T} \mid \Theta) p(X_{0:T} \mid y_{1:T}, \Theta^{(k)}) dX_{0:T} \\ &\approx \frac{1}{N} \sum_{i=1}^{N} \log p(X_{0:T}^{(i)}, y_{1:T} \mid \Theta) \\ &= \frac{1}{M} \sum_{i=1}^M \left[\log p_{\mu_0, \Sigma_0}(X_{0}^{(i)}) + \sum_{t=1}^T \log p_{\mu_t, \Sigma_t}(X_{t}^{(i)} \mid X_{t-1}^{(i)}) + \sum_{t=1}^T \log p_{\alpha, \beta}(y_{t} \mid X_{t}^{\mathcal{O}_t, (i)}) \right] \end{aligned}$$

> Problem: log-likelihood function terms contain different dimensions (transition is $2M \times 2M$ while observation is $2 \times 2$). This affects the convergence of the EM.

**log potential function**

$$\log p(X_t \mid X_{t-1}) = -\frac{d}{2} \log 2\pi - \frac{1}{2} \log |\Sigma_t| - \frac{1}{2} (X_t - \mu_t)^T \Sigma_t^{-1} (X_t - \mu_t)$$

**2 M-step**: Maximize $Q(\Theta \mid \Theta^{(k)})$ with respect to $\Theta$ to obtain the updated parameter estimates $\Theta^{(k+1)}$.


### 4.1.2 Modified Loss Function: Adding Inverse-Wishart Prior

$$Q(\Theta \mid \Theta^{(k)}) = \frac{1}{M} \sum_{i=1}^M \left[\log p_{\mu_0, \Sigma_0}(X_{0}^{(i)}) + \sum_{t=1}^T \log p_{\mu_t, \Sigma_t}(X_{t}^{(i)} \mid X_{t-1}^{(i)}) + \sum_{t=1}^T \log p_{\alpha, \beta}(y_{t} \mid X_{t}^{\mathcal{O}_t, (i)}) + \underbrace{\log p(\Gamma_0) + \log p(B)}_{\text{Inverse-Wishart Priors}} \right]$$

### 4.2 Maximizing the Marginal Likelihood

**Biased Estimate for an Exact Model**

In our previous set-up, we aim to maximize the joint distribution,

$$p(X_{0:T}, y_{1:T} \mid \Theta) = p(X_0 \mid \Theta) \prod_{t=1}^T p(X_t \mid X_{t-1}, \Theta) p(y_t \mid X_t^{\mathcal{O}_t}, \Theta)$$

Instead now we aim to maximize the marginal distribution, which is estimated by the particle filter's accumulated log normalizing constant $\log \hat{Z}_N(\Theta) = \log p(y_{1:T}\mid\Theta)$.

$$\hat{\Theta} = \arg \min_{\Theta} - \log \hat{Z}_N(\Theta) = \arg \min_{\Theta} - \log p(y_{1:T} \mid \Theta).$$

where $\hat{Z}_N(\Theta) = \sum_{i=1}^N \tilde{w}_T^{(i)}$, $p(y_{1:T} \mid \Theta) = \prod_{t=1}^{T} p(y_t \mid y_{1:t-1}, \Theta)$, $p(y_t \mid y_{1:t-1}, \Theta) \approx \frac{1}{N}\sum_{i=1}^{N} \tilde{w}_t^{(i)}$ and $\tilde{w}_t^{(i)} = w_{t-1}^{(i)}\, G_t\bigl(y_t \mid X_t^{\mathcal{O}_t,(i)}\bigr)$

However, the estimate $\hat{\Theta}$ is biased
  1. $\log \hat{Z}_N(\Theta)$ is consistent as $N \to \infty, \hat{Z}_N \to Z(\Theta)$ but $E[\hat{Z}_N(\Theta)] \neq Z(\Theta)$ for finite $N$, where $Z(\Theta) = p(y_{1:T} \mid \Theta)$ is the true normalizing constant. resampling induces correlation between previous time step.
  2. gradient for $\hat{\Theta}$ biased due to non-differentiable resampling.

can use `cuthbertlib.resampling.autodiff` to make resampling differentiable and unbiased.


## 5 Implementation Notes

**140826: M-step overfitting**

- transition loss dominates the M-step term by collapsing the transition covariance to a delta function ($\Sigma_t \to 0$). this inflates the determinant term which causes the log-likelihood to increase.
- observation likelihood being maximized - improving from -4898 to -4867, so M-step fits better observation scores.
- transition term goes positive -> degenerate term.
- solution: introduce a regularization term to the transition covariance matrix (e.g. Wishart prior) to prevent it from collapsing to 0.
  - $\log p(\Theta \mid X, y) = \underbrace{\log p(X, y \mid \Theta)}_{\text{joint likelihood}} + \underbrace{\log p(\Theta)}_{\text{prior parameters}}$
  - Since we have a Kronecker product for covariance, we will introduce a Wishart prior on $\Gamma_0$ and $B$.
- Key links from today
  - wishart : https://rich-d-wilkinson.github.io/MATH3030/7-2-the-wishart-distribution.html
  - On Particle Methods for Parameter Estimation in State-Space Models: https://arxiv.org/pdf/1412.8695
  - Differentiable Particle Filtering without Modifying the Forward Pass: https://arxiv.org/pdf/2106.10314
  - On backward smoothing algorithms: https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2F23-AOS2324
  - autodiff resampling: https://github.com/state-space-models/cuthbert/blob/main/cuthbertlib/resampling/autodiff.py
- probability densities can be greater than 1.
  - consider uniform $U(0, 0.5)$. log density is height x width = 2.0.
- Improvements / Fixes
  - rewrite smoother - backward sampling not the same because of the fixed particle
  - issue with `dt==0`. The cleanest solution is to group all matches on the same date into one latent-state time point and process their observation likelihoods sequentially. Otherwise, zero-time backward selection must respect the singular support, for example by tracing compatible ancestry.
- meeting notes: GD is trying to find the parameters to maximize the joint distribution i.e. state and observation likelihood. E -step is maximizing the marginal likelihood $p(y_{1:t} \mid \theta)$.
- If we combine EM with smoothing, we are trying to find the parameters that fit the distribution. but this will lead to overfitting?

**150826**
- transition loss still scales the covariance matrix to 0
- did a review with 5.6 SOL Ultra High and realized a few implementation errors with the filter. Also would probably need to reimplement the backward smoother.
- A Backward-Simulation Based
Rao-Blackwellized Particle Smoother for
Conditionally Linear Gaussian Models (2012): https://users.aalto.fi/~ssarkka/pub/rb-smoother-sysid.pdf

## 6 Appendix

### What the hell is a Wishart prior and distribution?

https://rich-d-wilkinson.github.io/MATH3030/7-2-the-wishart-distribution.html

The problem we are experiencing is a degenerate covariance matrix.

$$\log p(X_t \mid X_{t-1}) = -\frac{d}{2} \log 2\pi - \frac{1}{2} \log |Q| - \frac{1}{2} \big(X_t - \mu_0 - \phi_t (X_{t-1} - \mu_0)\big)^T Q^{-1} \big(X_t - \mu_0 - \phi_t (X_{t-1} - \mu_0)\big)$$

where $\phi_t = \exp(-\kappa \Delta t)$ and $Q = (1 - \phi_t^2)\,\Sigma_0$. As $Q \to 0$, the $\log |Q| \to + \infty$. the MLE is trying to maximize the log-likelihood, so this drives $Q \to 0$ to increase the log-likelihood. this creates a degenerate covariance matrix, where our means are overfitted to the data and the covariance is collapsed to a delta function.

We fix by switching from maximizing the likelihood to maximizing the posterior (MAP) by introducing a prior on the covariance matrix. we penalize the log-likelihood so it cannot collapse $Q \to 0$ for free.

$$\log p(\Theta \mid X, y) = \underbrace{\log p(X, y \mid \Theta)}_{\text{joint likelihood}} + \underbrace{\log p(\Theta)}_{\text{prior parameters}}$$

Wishart is a distribution over precision matrix, inverse covariance. Inverse-Wishart is a distribution over a covariance matrix. It is the same to use a Wishard / Inverse-Wishard prior on the covariance matrix. Since our parameter to be estimated is a covariance matrix, our conjugate prior should be using an inverse-wishart. The inverse-Wishart distribution $\mathcal{IW}(\Sigma_0 \mid \nu, S)$ for a $d \times d$ matrix is parameterized by degrees of freedom $\nu$ and scale matrix $S$.

$$\log p(\Sigma_0) = - \frac{\nu + d + 1}{2} \log |\Sigma_0| - \frac{1}{2} \text{tr}(S \Sigma_0^{-1})$$

Since we have a Kronecker product for covariance, we will introduce a Wishart prior on $\Gamma_0$ and $B$. Issue 1: You cannot couple the two prior together into a single inverse-wishart prior because the prior on $\Sigma_0 = \Gamma_0 \otimes B$ does not factor into independent priors on $\Gamma_0$ and $B$ (trace term couples two matrices together)

Issue 2: You cannot use a single inverse-wishart prior on $\Gamma_0$

$$p(\Gamma_0, B) = p(\Gamma_0) | \det J_f$$

because Kronecker map $(\Gamma_0, B) \mapsto \Gamma_0 \otimes B$ is many-to-one, so a change of variables Jacobian is singular and ill-defined. placing independent inverse-wishart priors on $\Gamma_0$ and $B$ avoids both problems and resolves the scale ambiguity.

$$\log p(\Gamma_0) = - \frac{\nu_\Gamma + d_\Gamma + 1}{2} \log |\Gamma_0| - \frac{1}{2} \text{tr}(S_\Gamma \Gamma_0^{-1})$$

$$\log p(B) = - \frac{\nu_B + d_B + 1}{2} \log |B| - \frac{1}{2} \text{tr}(S_B B^{-1})$$

where $d_B = 2$ and $d_\Gamma = M$ are the dimensions of the matrices.


$$Q(\theta_k,\theta) = \int \log p_\theta(x_{0:T},y_{0:T}) p_{\theta_k}(x_{0:T}\mid y_{0:T}) dx_{0:T}$$