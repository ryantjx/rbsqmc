# High-Performance Sequential Quasi-Monte Carlo

Describe the SQMC algorithm. Demonstrate its empirical performance in JAX compared to current methods. Address the limitations of SMC and QMC in high-dimensional problems, and show how the problem can be reformulated with the points mentioned in the paper by Gerber and Chopin (2017) to reduce the effective dimension of the problem, allowing for better performance of SQMC. Demonstrate this application to football match prediction.

## Sequential Quasi-Monte Carlo (SQMC)

### Background

#### Quasi-Monte Carlo (QMC) and Sequential Monte Carlo (SMC)

#### Parallelization of Seqeuntial Monte Carlo

#### Performance Comparison

- Seqeuntial Monte Carlo
- Sequential Quasi-Monte Carlo

## Application: Football Match Prediction using a Sequential Quasi-Monte Carlo approach

SQMC suffer from the inherent curse of dimensionality - SQMC relies on importance sampling at every step, which performs poorly in high dimensions because of greater discrepancy between the proposal and the target distributions.

A practical improvement of SQMC appears when we are able to reformulate the problem to operate on a lower dimension during the resampling step, achieving the benefits of QMC whilst solving for the high-dimensional problem. (Gerber, M., Chopin, N. 2017) (3.1)

In the following example, we demonstrate how our problem can be reformulated to operate on a lower dimension step, reducing the effective dimension of the problem from $\mathbb{R}^{2F \times 2F}$ to $\mathbb{R}^{2 \times 2}$, overcoming the curse of dimensionality faced by SMC and QMC.

### Model

Define the latent states as $x_t = (x_t^{\text{attack}}, x_t^{\text{defense}})^\top \in \mathbb{R}^{2}$. The observed data is the number of goals scored by the home team and away team, denoted as $y_t = (y_t^{\text{home}}, y_t^{\text{away}})^\top$.

**Initial distribution** follows a multivariate Gaussian distribution. $\Sigma_0 \in \mathbb{R}^{2F \times 2F}$ is a non-diagonal covariance matrix that contains the long-run correlation between teams. $\Sigma_0^f \in \mathbb{R}^{2 \times 2}$ represents the covariance matrix of between attack and defense for a single team, constant across all teams. The initial distribution can be represented as a Kronecker product of $\Sigma_0^f$ and $\Sigma_{\text{teams}} \in \mathbb{R}^{F \times F}$, which is the covariance matrix between teams.

$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

$$x_0^f = \begin{bmatrix} x_0^{f, attack} \\ x_0^{f, defense} \end{bmatrix} \sim \mathcal{N}(\mu_0^f, \Sigma_0^f)$$

$$\Sigma_0 = \Sigma_{\text{teams}} \otimes \Sigma_0^f = \begin{bmatrix} \sigma_{11}\Sigma_0^f & \sigma_{12}\Sigma_0^f & \cdots & \sigma_{1F}\Sigma_0^f \\ \sigma_{21}\Sigma_0^f & \sigma_{22}\Sigma_0^f & \cdots & \sigma_{2F}\Sigma_0^f \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{F1}\Sigma_0^f & \sigma_{F2}\Sigma_0^f & \cdots & \sigma_{FF}\Sigma_0^f \end{bmatrix}$$



**Transition distribution** follows an OU-process with a stationary distribution $\mathcal{N}(\mu_0, \Sigma_0)$ and a mean-reversion parameter $\kappa$ and $\Delta t$ represents the time difference between the previous and current observation.

$$x_t = x_{t-1} + \phi_t(\mu_0 - x_{t-1}) + \epsilon_t$$

$$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \phi_t(x_{t-1} - \mu_0), Q_t)$$

where $\epsilon_t \sim \mathcal{N}(0, Q_t)$, $\phi_t = \exp(- \kappa \Delta t)$, $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$ and $\Phi_t = \text{diag}(\phi_t)$.

**Likelihood function** is a bivariate Poisson distribution,

$$G_t(y_t \mid x_t^{h(t)}, x_t^{a(t)}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{h(t)}}}{y_t^{h(t)}!} \frac{\lambda_2^{y_t^{a(t)}}}{y_t^{a(t)}!} \sum_{j=0}^{\min(y_t^{h(t)}, y_t^{a(t)})} \binom{y_t^{h(t)}}{j} \binom{y_t^{a(t)}}{j} j! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^j$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, h(t)} - x_t^{\text{def}, a(t)})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, a(t)} - x_t^{\text{def}, h(t)})$, $\lambda_3 = \exp(\beta)$.

### Sequential Quasi-Monte Carlo (SQMC)

$$p(x^{h(t)}, x^{a(t)} \mid y_{1:t}) \propto G_t(y_t \mid x^{h(t)}, x^{a(t)}) p(x^{h(t)}, x^{a(t)} \mid y_{1:t-1})$$

#### Rao-Blackwellized Sequential Quasi-Monte Carlo (RB-SQMC)

We define $E_k = \{x^{h(t)}, x^{a(t)}\}$ as the set of latent states involved in the observation at time $k$. The joint distribution of the latent states can be factorized as

$$p(x_{0:t}, y_{1:t}) = p(x^{E_t}_{0:t} \mid y_{1:t}) p(x^{-E_t}_{0:t} \mid x^{E_t}_{0:t})$$

From the given model above, the transition distribution is linear Gaussian and the mean and covariance of the latent states not involved in the observation are deterministic and observation independent. Latent states not involved in the observation can be marginalized out analytically,

$$p(x^{-E_t}_{0:t + 1} \mid x^{E_t}_{0:t}) = \mathcal{N}(\mu_0 + \phi_{t + 1} (x^{E_t}_{0:t} - \mu_0), Q_{t+1})$$

Using the bootstrap filter, we can sample latent states involved in the observation and compute the weight update step using the likelihood function $G_t$.

$$w_t = p(x^{E_t}_{0:t+1} \mid y_{1:t+1}) \propto G_{t+1}(y_{t+1} \mid x^{E_t}_{0:t+1}) p(x^{E_t}_{0:t+1} \mid y_{1:t})$$

Thereafter, resampling is performed on the propagated particles, subsequently rsetting the weights to $w_t^{(i)} = 1 / N$ for $N$ particles.

##### Smoothing

At terminal time, we obtain a mixture of Gaussians for the latent states. 

$$p(x_{T} \mid y_{1:T}) = \sum_{i=1}^{N} w_T^{(i)} p(x_{T} \mid y_{1:T}) \sim \mathcal{N}(\mu_T, \Sigma_T)$$

Smoothing can be performed by using the Forward Filtering Backward Simulation (FFBSi) algorithm. We can obtain the smoothed latent state by recursively sampling backwards from the terminal time $T$ to the initial time $0$.

$$\begin{aligned}
p(x_{t} \mid x_{t+1} y_{1:T}) &\propto p(x_{t} \mid x_{t+1}, y_{1:T}) p(x_{t+1} \mid y_{1:T}) \\ &\propto p(x_{t} \mid x_{t+1}) \sum_{i=1}^{N} w_t^{(i)} p(x_{t}^{(i)} \mid y_{1:T}) \\ & \approx \sum_{i=1}^{N} w_{t \mid t+1}^{(i)} \mathcal{N}(x_{t} \mid m_{t \mid t+1}^{(i)}, \Sigma_{t \mid t+1}^{(i)})
\end{aligned}$$

$$w_{t \mid t+1}^{(i)} \approx w_t^{(i)} \mathcal{N}(x_{t+1} \mid \mu_0 + \phi_{t+1}(x_t - \mu_0), \Phi_{t+1} \Sigma_t \Phi_{t+1} + Q_{t+1})$$

$$m_{t \mid t+1}^{(i)} = \mu_t + J_t (x_{t+1} - \Phi_{t+1} \mu_k)$$

$$\Sigma_{t \mid t+1}^{(i)} = \Sigma_t - J_t \Phi_{t+1} \Sigma_t$$

where $J_t = \Sigma_t (\Sigma_t + Q_{t+1})^{-1}$.

$$x_k^{(i)} \sim \mathcal{N}(m_{t \mid t+1}^{(i)}, \Sigma_{t \mid t+1}^{(i)})$$


### Performance of SMC, SQMC, RB-SQMC

<!-- ### Rao-Blackwellized Sequential Monte Carlo (RB-SMC)

The RB-SMC computes the marginal posterior distribution of latent states not involved in the observation in closed form. Defining $E_k = \{x^{h(k)}, x^{a(k)}\}$ as the set of latent states involved in the observation at time $k$, we can write the posterior distribution of latent states as

$$p(x^{E_k}_{0:k} \mid y_{1:k}) = p(x^{-E_k}_{0:k} \mid x^{E_k}_{0:k}) p(E^{k}_{0:k} \mid y_{1:k})$$

Since the transition distribution is linear, the conditional distribution $p(x^{-E_k}_{0:k} \mid x^{E_k}_{0:k})$ is deterministic and can be computed in closed form.

For non-playing teams, the mean $\mu_k$ and covariance matrix $\Sigma_k$ are deterministic and particle-independent.

$$\mu_k^{-E_k} = \mu_{k-1}^{-E_k} + \phi_k(\mu_0^{-E_k} - \mu_{k-1}^{-E_k})$$

$$\Sigma_k^{-E_k} = \Sigma_{0}^{-E_k} - \Phi_k^{-E_k} \Sigma_{0}^{-E_k} \Phi_k^{-E_k}$$

For playing teams, we perform a bivariate Gaussian sampling of particles and compute the weight update step using the likelihood function.

1. Sample: $x_k^{E_k, (i)} \sim \mathcal{N}(\mu_{k \mid k-1}, \Sigma_{k \mid k-1})$
2. Weight update (Bootstrap filter): $w_k^{(i)} = G_k(y_k \mid x_k^{E_k, (i)})$
3. Resample: Resample particles based on $w_k^{(i)}$ and reset $w_k^{(i)} = 1/N$ for $N$ particles.
4. Condition: Condition the full Gaussian distribution on the resampled particle values $x_k^{E_k}$, setting $\mu_k^{E_k} = x_k^{E_k}$ and $\Sigma_{k}^{E_k} = 0$.

We then perform resampling on the particles based on $w_k$, subsequently resetting $w_k^i = 1/N$ for $N$ particles. After resampling, we condition the full Gaussian distribution on the resampled particle values $x_k^{E_k}$, setting $\mu_k^{E_k} = x_k^{E_k}$ and $\Sigma_{k}^{E_k} = 0$. This reflects no residual uncertainty in the states of teams that have just played.

#### RB-SMC Smoothing -->

## References

<!-- [state-space-models/cuthberto-carlos]([https://github.com/state-space-models/cuthberto-carlos]) -->

Gerber, M., Chopin, N.: Sequential quasi Monte Carlo. J. R. Stat. Soc. Ser. B. Stat. Methodol. 77(3), 509–579 (2015). https://doi.org/10.1111/rssb.12104

Gerber, M., Chopin, N.: Convergence of Sequential Quasi-Monte Carlo Smoothing Algorithms. arXiv:1506.06117 (2015). https://arxiv.org/abs/1506.06117

Chopin, N., Gerber, M.: Sequential quasi-Monte Carlo: Introduction for Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion Processes. arXiv:1706.05305 (2017). https://arxiv.org/abs/1706.05305

Lemieux, C.: Monte Carlo and Quasi-Monte Carlo Sampling. Springer (2009). https://doi.org/10.1007/978-0-387-78165-5

Samuel Duffield, Samuel Power, Lorenzo Rimella, A state-space perspective on modelling and inference for online skill rating, Journal of the Royal Statistical Society Series C: Applied Statistics, Volume 73, Issue 5, November 2024, Pages 1262–1282, https://doi.org/10.1093/jrsssc/qlae035

https://github.com/AdrienCorenflos/parallel-Hilbert