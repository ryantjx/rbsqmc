# RB-SQMC

1. Time Varying $\Sigma_t = \Gamma_t \otimes B_t$ where $\Gamma_t \in \mathbb{R}^{F \times F}$ and $B_t \in \mathbb{R}^{2 \times 2}$ are both static parameters. Assumes that covariance structure is proportional to the time-varying $B_t$.
2. Static $B_0$, $\Sigma_t = \Gamma_t \otimes B_0$ where $\Gamma_t \in \mathbb{R}^{F \times F}$ is a time-varying parameter and $B_0 \in \mathbb{R}^{2 \times 2}$ is a static parameter.
3. $\Sigma_0 = \Gamma_0 \otimes B_0$ where $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B_0 \in \mathbb{R}^{2 \times 2}$ are both static parameters. Assumes that the covariance structure and covariance block for each team is constant.
4. Full $\Sigma_t$ where $\Sigma_t \in \mathbb{R}^{2F \times 2F}$ is a static parameter. Most flexible but most expensive to compute.

## Model 3 - 220726

$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

Defining $\Sigma_t = \Gamma_t \otimes B_t$ where $\Gamma_t \in \mathbb{R}^{F \times F}$ and $B_t \in \mathbb{R}^{2 \times 2}$ are both time-varying parameters.

Transition distribution is a time-varying follows OU-process with a stationary distribution $\mathcal{N}(\mu_0, \Sigma_0)$ and a mean-reversion parameter $\kappa$. The transition distribution can be represented as

<!-- $$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \Phi_t(x_{t-1} - \mu_0), Q_t)$$

where $\Phi_t = \exp(-\kappa \Delta t)$, $\phi_t = \text{diag}(\Phi_t)$ and $Q_t = \Sigma_0 - \Phi_t \Sigma_{0} \Phi_t^\top$. -->

$$x_t \mid x_{t-1} \sim \mathcal{N}(m_{t \mid t-1}, P_{t \mid t-1})$$

where 
- $m_{t \mid t-1} = \mu_0 + \Phi_t(x_{t-1} - \mu_0) = \mu_0 + \Phi_t (\mu_{t-1} - \mu_0)$
- $P_{t \mid t-1} = \Sigma_t - \Phi_t P_{t-1 \mid t-1} \Phi_t^\top = \Sigma_t - \Phi_t \Sigma_{t-1} \Phi_t^\top = \Gamma_t \otimes B_t - \Phi_t (\Gamma_{t-1} \otimes B_{t-1}) \Phi_t^\top$.
- $\Phi_t = \exp(-\kappa \Delta t)$ and $\phi_t = \text{diag}(\Phi_t)$

Likelihood function to be $G_t (y_t^{(i)}, y_t^{(j)} \mid x_t^{(i)}, x_t^{(j)})$, where there is an implicit correlation between $x_t^{(i)}$ and $x_t^{(j)}$ via $G_t$.

**Algorithm**

1. Prediction: Predictive mean and covariance for full state
   1. $$m_{t \mid t-1} = \mu_0 + \Phi_t (\mu_{t-1} - \mu_0)$$
   2. $$P_{t \mid t-1} = \Sigma_t - \Phi_t \Sigma_{t-1} \Phi_t^\top$$
2. Bootstrap Proposal Sampling: Sample from 2D coordinate of predictive distribution for each pair of teams $(i, j)$. Sample both teams together due to the correlation between $x_t^{(i)}$ and $x_t^{(j)}$ via the likelihood function $G_t$. Since particles have equal weights, we can jointly sample from the predictive distribution for each pair of teams $(i, j)$.
   1. $$\mu_{t}^{(i, j)} \mid \mu_{t-1}^{(i, j)} \sim \mathcal{N}\left(\begin{bmatrix} m_{t \mid t-1}^{(i)} \\ m_{t \mid t-1}^{(j)} \end{bmatrix}, \begin{bmatrix} P_{t \mid t-1}^{(i, i)} & P_{t \mid t-1}^{(i, j)} \\ P_{t \mid t-1}^{(j, i)} & P_{t \mid t-1}^{(j, j)} \end{bmatrix}\right)$$
   2. $P_{t \mid t-1}^{(i, i)} = \Gamma_t^{(i, i)} \otimes B_t - \Phi_t^{(i)} (\Gamma_{t-1}^{(i, i)} \otimes B_{t-1}) \Phi_t^{(i)^\top}$
   3. $P_{t \mid t-1}^{(i, j)} = \Gamma_t^{(i, j)} \otimes B_t - \Phi_t^{(i)} (\Gamma_{t-1}^{(i, j)} \otimes B_{t-1}) \Phi_t^{(j)^\top}$

## Model 2
$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

Defining $\Sigma_t = \Gamma_t \otimes B_t$ where $\Gamma_t \in \mathbb{R}^{F \times F}$ and $B_t \in \mathbb{R}^{2 \times 2}$ are both static parameters, we have

$$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \Phi_t(x_{t-1} - \mu_0), Q_t)$$

where $\Phi_t = \exp(-\kappa \Delta t)$, $\phi_t = \text{diag}(\Phi_t)$ and $Q_t = \Sigma_t - \Phi_t \Sigma_{t-1} \Phi_t^\top$.

$$x_t \mid y_{1:t-1} \sim \mathcal{N}(m_{k \mid k-1}, P_{k \mid k-1})$$

where $m_{k \mid k-1} = \mu_0 + \Phi_k (m_{k-1 \mid k-1} - \mu_0)$ and $P_{k \mid k-1} = \Phi_k P_{k-1 \mid k-1} \Phi_k^\top + Q_t$.

## Model 1

Define $x_t = (x_1, x_2)^T \in \mathbb{R}^2$ as the latent states. Assuming that the prior distribution is a multivariate Gaussian distribution with $F$ dimensions, the initial distribution can be represented as

$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B_0$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B_0 \in \mathbb{R}^{2 \times 2}$, which represents the covariance matrix between teams and the covariance matrix of between attack and defense for a single team, respectively.

For the transition distribution, we assume that the latent states follow an OU-process with a stationary distribution $\mathcal{N}(\mu_0, \Sigma_0)$ and a mean-reversion parameter $\kappa$. The transition distribution can be represented as

<!-- $$x_t = x_{t-1} + \phi_t(\mu_0 - x_{t-1}) + \epsilon_t$$ -->

$$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \phi_t(x_{t-1} - \mu_0), Q_t)$$

where $\epsilon_t \sim \mathcal{N}(0, Q_t)$, $\phi_t = \exp(- \kappa \Delta t)$, $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$ and $\Phi_t = \text{diag}(\phi_t)$.

The observed data is denoted as $y_t = (y_1, y_2)^T \in \mathbb{R}^2$. The likelihood function is a bivariate Poisson distribution,

$$G_t(y_t \mid x_t^{h(t)}, x_t^{a(t)}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{h(t)}}}{y_t^{h(t)}!} \frac{\lambda_2^{y_t^{a(t)}}}{y_t^{a(t)}!} \sum_{j=0}^{\min(y_t^{h(t)}, y_t^{a(t)})} \binom{y_t^{h(t)}}{j} \binom{y_t^{a(t)}}{j} j! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^j$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, h(t)} - x_t^{\text{def}, a(t)})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, a(t)} - x_t^{\text{def}, h(t)})$, $\lambda_3 = \exp(\beta)$.

**Problem**: Observation model is a non-linear bivariate Poisson distribution, which may not preserve the Kronecker structure of the covariance matrix.

**Set-up**: Assuming a linear observation model 

$$y_k = \mathcal{H} x_k + \nu_k$$

where $\mathcal{H} = I_n \otimes H$ and $\nu_k \sim \mathcal{N}(0, R)$. The innovation covariance is 

$$\begin{aligned}
S_k 
&= \mathcal{H} P^{-} \mathcal{H}^T + R \\
&= (I_n \otimes H) Q_k (I_n \otimes H)^T + R \\
&= (I_n \otimes H) (\Sigma_0 - \Phi_k \Sigma_0 \Phi_k) (I_n \otimes H)^T + R \\
&= (I_n \otimes H) (\Gamma \otimes B - \Phi_k (\Gamma \otimes B) \Phi_k) (I_n \otimes H)^T + \Gamma \otimes V \\
&= \Gamma \otimes (H B H^T - \Phi_k H B H^T \Phi_k + V)
\end{aligned}$$

So only $B$ is updated and $\Gamma$ is fixed, preserving the Kronecker structure of the covariance matrix.


Issue:

Assuming the kronecker structure,

$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B_0$ where $\Gamma_0 \in \mathbb{R}^{F \times F}$ and $B_0 \in \mathbb{R}^{2 \times 2}$. The transition distribution is 

$$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \phi_t(x_{t-1} - \mu_0), \Sigma_t)$$

where $\phi_t = \exp(- \kappa \Delta t)$, $\Sigma_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$ and $\Phi_t = \text{diag}(\phi_t)$

The likelihood function is a bivariate Poisson distribution,

$$G_t(y_t \mid x_t^{h(t)}, x_t^{a(t)}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{h(t)}}}{y_t^{h(t)}!} \frac{\lambda_2^{y_t^{a(t)}}}{y_t^{a(t)}!} \sum_{j=0}^{\min(y_t^{h(t)}, y_t^{a(t)})} \binom{y_t^{h(t)}}{j} \binom{y_t^{a(t)}}{j} j! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^j$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, h(t)} - x_t^{\text{def}, a(t)})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, a(t)} - x_t^{\text{def}, h(t)})$, $\lambda_3 = \exp(\beta)$.

There are issues with why Kronecker structure does not hold for our model.

1. OU-transition is index-dependent
   1. $\phi_k = \exp(-\kappa \Delta t_k)$ is index-dependent, thus it differs for each index which results in a non-constant $\Phi_k$ matrix, which breaks the Kronecker structure of $B_t$.
2. pair specific non-linear observation model
   1. pair updates breaks the Kronecker structure since we cannot perform a linear update of covariance matrix.
   2. we will have different $B_t^h$ and $B_t^a$ for home and away teams, which breaks the Kronecker structure of $B_t$.

If we initially have a constant parameter $\Sigma_0 = \Gamma_0 \otimes B_0$ over all time steps,

1. conceptually does not work - attack and defense strength vary for each team $B_{i,t}$. i.e. should not be the same for all teams $B_{i,t} \neq B_0$
2. OU transition is compatible with the stationary model
   1. $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^T$ where $\Phi_t = \text{diag}(\phi_t)$ and $\phi_t = \exp(-\kappa \Delta t)$ 
   2. predictive covariance of $x_t \mid y_{1:t-1}$ is $P_t^{-} = \Phi_t P_{t-1} \Phi_t^T + Q_t$
   3. if we assume $P_{t-1} = \Sigma_0$, then $P_t^{-} = \Phi_t \Sigma_0 \Phi_t^T + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^T = \Sigma_0$
   4. shows that covariance update is deterministic thus maintaining the same covariance structure over time.
3. pair specific non-linear observation model
   1. pair updates breaks the Kronecker structure since we will have $x_t^h \pm x_t^a$ from the likelihood function
   2. we will have different $B_t^h$ and $B_t^a$ for home and away teams, which breaks the Kronecker structure of $B_t$.
   3. there may also be covariance from the observation model $p(x^h_t, x_t^a \mid y_t)$, which then breaks the structure because we cannot update as one shared $B_0$.