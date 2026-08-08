# Rao-Blackwellized Particle Filter for Online Skill Rating

## 1 Introduction
Duffield et. al (2024) proposes a state-space approach on modelling and inference of online skill rating. The authors introduces *factorial approximation* as the necessary bias to scale computations for a high-dimensional problem.

## 2 Setup

Let $X_t = (X_t^a, X_t^d) \in \mathbb{R}^{K \times 2}$ denote the latent state at time $t$, where $X_t^a$ represents the attacking component and $X_t^d$ represents the defensive component. The initial distribution is a multivariate Gaussian

$$X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

where $\mu_0 \in \mathbb{R}^{K \times 2}$ and $\Sigma_0 = \Gamma_0 \otimes B \in \mathbb{R}^{(K \times 2) \times (K \times 2)}$ is a Kronecker product representing the initial covariance between the dimensions. We define a general discrete-time OU process in $\mathbb{R}^{K \times 2}$ for the latent state evolution of teams.

$$X_{t} = \mu_0 + \Phi_{t} (X_{t-1} - \mu_0) + \epsilon_t \qquad \epsilon_t \sim \mathcal{N}(0, Q_{t})$$

where $Q_t = \Sigma_0 - \Phi_t \Sigma_0 \Phi_t^\top$ is the process noise covariance which becomes larger if a team has not been active for a long time. $\Phi_t = \text{diag}(\phi_{t,1}, \ldots, \phi_{t,K})$ is a diagonal matrix with $\phi_{t} = \exp(- \kappa \Delta t)$ where $\Delta t$ is the time since the last match and $\kappa$ is a hyperparameter controlling the rate of mean reversion. We assume that our observations $Y_t \in \mathbb{R}^{2}$ involves latent states of two teams, $X_t^{\mathcal{O}} = (X_t^{h}, X_t^{a})$, where $h$ and $a$ are the home and away teams respectively. The remaining states $X_t^{\mathcal{R}} = X_t \setminus X_t^{\mathcal{O}}$ are the latent states of the other teams. We follow the bivariate Poisson likelihood model defined in (Karlis and Ntzoufras, 2003) to model the match outcome $y_t = (y_t^h, y_t^a) \in \mathbb{N}^2$ given the latent states of the two teams.

$$G_t(y_t \mid x_t^{h}, x_t^{a}) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y_t^{h}}}{y_t^{h}!} \frac{\lambda_2^{y_t^{a}}}{y_t^{a}!} \sum_{j=0}^{\min(y_t^{h}, y_t^{a})} \binom{y_t^{h}}{j} \binom{y_t^{a}}{j} j! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^j$$

where $\lambda_1 = \exp(\alpha + x_t^{\text{att}, h} - x_t^{\text{def}, a})$, $\lambda_2 = \exp(\alpha + x_t^{\text{att}, a} - x_t^{\text{def}, h})$, $\lambda_3 = \exp(\beta)$.

## 3 Rao-Blackwellized Particle Filter (RB-PF)

The RB-PF targets the marignal distrivution of latent states $X_t^{\mathcal{O}} \subseteq X_t$ involved in an observation $y_t$. The joint distribution of the latent states can be factorized as

$$p(X_t{0 : T} \mid y_{1 : T}) = p(X_t^{\mathcal{O}} \mid y_{1 : T}) p(X_t^{\mathcal{R}} \mid X_t^{\mathcal{O}})$$

Conditional on a specific trajectory of latent states $X_{0:T}^{\mathcal{O}}$, the remaining latent states $X_{0:T}^{\mathcal{R}}$ are conditionally independent of the observed data $y_{1:T}$.



## 4 Filtering

## 5 Smoothing