# RBPF with OU-process and Kornecker product covariance structure

$$p(x_{0:T}, y_{1:T}) = p(x_0) p(x_{1:T}^{-E_k} \mid x^{E_k}) p(y_{1:T} \mid x_{1:T}^{E_k})$$

## Models

1. time varying $\Sigma_t$ and static $B$ - keeps Kronecker structure with varying correlation between teams (original idea)
2. time varying $\Sigma_t$ and $B_t$ - loses simplification of Kronecker product structure but more flexible

### Model 1 - time varying $\Sigma_t$ and static $B$ - keeps Kronecker structure with varying correlation between teams

Let $x_t = (x_t^{\text{attack}}, x_t^{\text{defense}})^\top \in \mathbb{R}^2$ be the latent states. The joint distribution of latent states and observed data can be represented as

Initial distribution is a multivariate Gaussian with $F$ dimensions, where $F$ is number of teams. We assume that the covariance matrix follows a Kronecker product structure,

$$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\Sigma_0 = \Gamma_0 \otimes B_0 \in \mathbb{R}^{2F \times 2F}$ is a Kronecker product of $\Gamma_0 \in \mathbb{R}^{F \times F}$, the covariance between teams and $B_0 \in \mathbb{R}^{2 \times 2}$, the covariance between attack and defense for a single team. We assume that the covariance structure between attack and defense of all teams is the same.

**Transition distribution** defined using the OU-process

$$x_t = \mu_0 + \Phi_t (x_{t-1} - \mu_0) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

where $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$ and $\Phi_t = \exp(-\kappa \Delta t)$. 

Let $x_t^{E} = (x_t^{(h)}, x_t^{(a)})$ be the latent states of teams involved in an observation. Let $y_t = (y_t^{(h)}, y_t^{(a)})^\top$ be the observed data at time $t$. likelihood function $G_t$ defined as a non-linear function of inputs $x_t^{E}$.

$$G_t(y_t \mid x_t^{E_t})$$

#### Algorithm

1. Initial Distribution
   1. $$x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$
   2. $$\Sigma_0 = \Gamma_0 \otimes B_0 \qquad \Gamma_0 \in \mathbb{R}^{F \times F}, B_0 \in \mathbb{R}^{2 \times 2}$$
   3. where $B_0$ is a common covariance matrix between attack and defense for all teams and $\Gamma_0$ is a covariance matrix between teams.
2. Transition Distribution
   1. $$x_t \mid x_{t-1} \sim \mathcal{N}(\mu_0 + \Phi_t(x_{t-1} - \mu_0), Q_t)$$
   2. where $\Phi_t = \exp(-\kappa \Delta t)$, $\phi_t = \text{diag}(\Phi_t)$ and $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$.
3. Prediction
   1. $$x_t \mid y_{1:t-1} \sim \mathcal{N}(m_{t \mid t-1}, P_{t \mid t-1})$$
   2. $m_{t \mid t-1} = \mu_0 + \Phi_t (m_{t-1 \mid t-1} - \mu_0) = \mu_0 + \Phi_t (\mu_{t-1} - \mu_0)$ and 
   3. $P_{t \mid t-1} = \Phi_t P_{t-1 \mid t-1} \Phi_t^\top + Q_t = \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t$.
   4. We then have the predictive distribution for the full state as a multivariate Gaussian condtional on the observed data $y_{1:t-1}$.
   5. $$\begin{pmatrix}x_t^{E} \\ x_t^{R} \end{pmatrix} \mid y_{1:t-1} \sim \mathcal{N} \left( \begin{bmatrix} m_t^{E} \\ m_t^{R} \end{bmatrix}, \begin{bmatrix} P_t^{EE} & P_t^{ER} \\ P_t^{RE} & P_t^{RR} \end{bmatrix} \right)$$
      1. $P_t^{EE} = \text{Var}(x_t^{E} \mid y_{1:t-1}) \in \mathbb{R}^{2 \times 2}$
      2. $P_t^{RR} = \text{Var}(x_t^{R} \mid y_{1:t-1}) \in \mathbb{R}^{(F-2) \times (F-2)}$
      3. $P_t^{ER} = \text{Cov}(x_t^{E}, x_t^{R} \mid y_{1:t-1}) = P_t^{ER} \in \mathbb{R}^{2 \times (F-2)}$
4. Bootstrap particle sampling - perform bootstrap sampling for each particle involved in teams $h$ and $a$.
   1. $$\mu_t^{E} \sim \mathcal{N} (m_t^{E}, P_t^{EE}) = \mathcal{N}\left(\begin{bmatrix} m_{t \mid t-1}^{h} \\ m_{t \mid t-1}^{a} \end{bmatrix}, \begin{bmatrix} P_{t \mid t-1}^{hh} & P_{t \mid t-1}^{ha} \\ P_{t \mid t-1}^{ah} & P_{t \mid t-1}^{aa} \end{bmatrix}\right)$$
   2. $P_{t \mid t-1}^{hh} = \phi_t^{hh} \Sigma_{t-1}^{hh} (\phi_t^{hh})^\top + \Sigma_0^{hh} - \phi_t^{hh} \Sigma_0^{hh} (\phi_t^{hh})^\top$
   3. $P_{t \mid t-1}^{ha} = \phi_t^{ha} \Sigma_{t-1}^{ha} (\phi_t^{ha})^\top + \Sigma_0^{ha} - \phi_t^{ha} \Sigma_0^{ha} (\phi_t^{ha})^\top$
   4. $\phi_t = \text{diag}(\Phi_t)$
5. **Unnormalized weights**
   1. $$\log(\tilde{w}_{t}^{E}) = \log (w_{t-1}^{E}) + \log G_t (y_t \mid x_t^{E})$$
6. **Exact Gaussian Marginalization** update full predictive 
   1. $$x_t^{R} \mid x_t^{E} = \mu_{t}^{E}, y_{1:t-1} \sim \mathcal{N} (m_{t}^{R \mid E, (i)}, P_{t}^{R \mid E})$$
   2. $m_t^{R \mid E, (i)} = m_t^{R} + K (\mu_t^{E, (i)} - m_t^{E})$
   3. $P_t^{R \mid E} = P_t^{RR} - K P_t^{ER}$
   4. $K = P_t^{RE} (P_t^{EE})^{-1}$ is the Kalman gain.
7. **Obtain new $\Sigma_t$ with Kronecker structure**
   1. If $\Sigma_t = \Sigma_t \otimes B$ and $\Sigma_0 = \Sigma_0 \otimes B$,
   2. $$\begin{aligned}P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top \\ &= \Phi_t (\Gamma_{t-1} \otimes B) \Phi_t^\top + (\Gamma_0 \otimes B) - \Phi_t (\Gamma_0 \otimes B) \Phi_t^\top \\ &= \Gamma_t^{-} \otimes B\end{aligned}$$
   3. where $\Gamma_t^{-} = \phi_t \Gamma_{t-1} \phi_t^\top + \Gamma_0 + \phi_t \Gamma_0 \phi_t^\top$
   4. $\Gamma_{t}^{-} = \begin{bmatrix} \Gamma_t^{EE, -} & \Gamma_t^{ER, -} \\ \Gamma_t^{RE, -} & \Gamma_t^{RR, -} \end{bmatrix}$
   5. $P_t^{EE} = \Gamma_t^{EE, -} \otimes B$, $P_t^{RR} = \Gamma_t^{RR, -} \otimes B$ and $P_t^{RE} = \Gamma_t^{RE, -} \otimes B$.
   6. Then our updates using the Kalman equations will be simplified to
   7. $$\begin{aligned}K_t &= P_{t}^{RE} (P_{t}^{EE})^{-1} \\ &= (\Gamma_t^{RE, -} \otimes B) (\Gamma_t^{EE, -} \otimes B)^{-1} \\ &= \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \otimes I_2 \end{aligned}$$
   8. $P_t^{R \mid E} = \Gamma_t^{R \mid E} \otimes B$
      1. $\Gamma_t^{R \mid E} = \Gamma_t^{RR, -} - \Gamma_t^{RE, -} (\Gamma_t^{EE, -})^{-1} \Gamma_t^{ER, -}$
   9. This simplification allows us to only compute $B$ and use the Kronecker product structure for the full covariance matrix $\Sigma_t$.
8. **Resampling** - resample particles $\mu_t^{E}$ based on $\tilde{w}_t^{E}$ and set $w_t^{E} = 1/N$.
9.  The filtering distribution for the full state conditional on the observed data $y_{1:t}$ can be represented as
      1. $$p(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \mathcal{N} \left(\begin{pmatrix} \mu_t^{E, (i)} \\ m_t^{R \mid E, (i)} \end{pmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & P_t^{R \mid E} \end{bmatrix} \right)$$
      2. $P_t^{EE \mid E} = P_t^{EE} - P_t^{EE}(P_t^{EE})^{-1} P_t^{EE} = 0$
      3. $P_t^{ER \mid E} = P_t^{ER} - P_t^{EE}(P_t^{EE})^{-1} P_t^{ER} = 0$
      4. $P_t^{RE \mid E} = P_t^{RE} - P_t^{RE}(P_t^{EE})^{-1} P_t^{EE} = 0$

### Model 2 - Kronecker product covariance structure with time-varying $B_t$ and time-varying $\Gamma_t$

7. **Obtain new $\Sigma_t$ with Kronecker structure**
   1. $$\begin{aligned}P_{t \mid t-1} &= \Phi_t \Sigma_{t-1} \Phi_t^\top + \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top \\ &= \Phi_t (\Gamma_{t-1} \otimes B_{t-1}) \Phi_t^\top + (\Gamma_0 \otimes B_0) - \Phi_t (\Gamma_0 \otimes B_0) \Phi_t^\top\end{aligned}$$
   2. $\Gamma_{t} = \begin{bmatrix} \Gamma_t^{EE} & \Gamma_t^{ER} \\ \Gamma_t^{RE} & \Gamma_t^{RR} \end{bmatrix}$ and assume $B_{t}^{EE} = B_t^{RR} = B_t$
   3. $$\begin{aligned}K_t &= P_{t}^{RE} (P_{t}^{EE})^{-1} \\ &= [\phi_t^{RE} (\Gamma_t^{RE} \otimes B_t) \phi_t^{RE} + (\Gamma_0^{RE} \otimes B_0)(I - \phi_t^{RE} {\phi_t^{RE}}^\top)] [(\Gamma_t^{EE} \otimes B_t)^{-1}] \end{aligned}$$