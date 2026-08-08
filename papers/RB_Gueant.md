# Rao-Blackwellised particle filtering for models with asynchronous observations

**Adrien Corenflos**  
*July 8, 2026*

---

## 1. Original Model Context

The original model (Guéant and Pu, 2019) describes complex latent Gaussian spatio-temporal dynamics and asynchronous convoluted observations of different spatial locations. The primary challenge in such continuous-time state-space models with discrete observation triggers is maintaining a tractable posterior distribution over the latent function when observations arrive asynchronously and independently across distinct coordinates.

---

## 2. Simplified Model

To analyse the Rao-Blackwellisation structure without the overhead of the full model, we define a simplified discrete-time correlated random walk in $\mathbb{R}^D$. Let $X_k \in \mathbb{R}^D$ denote the latent state at time step $k$. The state transition is governed by linear Gaussian dynamics:

$$X_k = X_{k-1} + \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, Q), \tag{1}$$

where $Q$ is a non-diagonal $D \times D$ covariance matrix inducing correlation across the coordinates.

Observations arrive asynchronously such that at each discrete time step $k$, only a single coordinate $d_k \in \{1, \ldots, D\}$ is observed. For the sake of simplicity, and to allow for comparison with ground truth, we assume that the observation $y_k \in \mathbb{R}$ is conditionally Gaussian:

$$y_k = X_k^{d_k} + \nu_k, \quad \nu_k \sim \mathcal{N}(0, R), \tag{2}$$

where $X_k^{d_k}$ denotes the $d_k$-th element of the state vector $X_k$, and $R$ is a scalar observation variance.

---

## 3. Inference Algorithms

### 3.1 Standard Particle Filter (PF)

A standard bootstrap particle filter approximates the posterior $p(X_{0:k} \mid y_{1:k})$ via a set of $N$ weighted particles $\{X_k^{(i)}, w_k^{(i)}\}_{i=1}^N$.

1. **Prediction:** Sample $X_k^{(i)} \sim \mathcal{N}(X_{k-1}^{(i)}, Q)$ for $i = 1, \ldots, N$.

2. **Update:** Compute unnormalized log-weights based on the 1D observation:

$$\log \tilde{w}_k^{(i)} = \log w_{k-1}^{(i)} + \log \mathcal{N}\left(y_k \mid [X_k^{(i)}]_{d_k}, R\right). \tag{3}$$

Normalize weights $w_k^{(i)}$.

3. **Resampling:** Perform resampling on the particles $\{X_k^{(i)}\}$ according to weights $\{w_k^{(i)}\}$ at every step, subsequently resetting $w_k^{(i)} = 1/N$.

---

### 3.2 Rao-Blackwellized Particle Filter (RB-PF)

The RB-PF targets the marginal posterior of the sequence of sampled coordinates and recovers the complementary in closed form. Let $U_k = X_k^{d_k}$ be the realization of the coordinate observed at time $k$. The posterior factorizes as:

$$p(X_{0:k} \mid y_{1:k}) = p(X_{0:k}^{-d} \mid U_{1:k}) \, p(U_{1:k} \mid y_{1:k}). \tag{4}$$

The particle filter approximates the augmented distribution $p(U_{1:k} \mid y_{1:k})$ using weighted samples $\{U_{1:k}^{(i)}, w_k^{(i)}\}_{i=1}^N$. Conditional on a specific trajectory $U_{1:k}^{(i)}$, the remaining state dimensions are analytically tracked via a Gaussian distribution $\mathcal{N}(\mu_k^{(i)}, P_k)$. Because the dynamics are linear and the observed coordinates are identical across particles, the covariance matrix $P_k$ is deterministic and particle-independent. Under a bootstrap proposal, the algorithm takes the following form.

1. **Prediction (Prior over full state):** Compute the predictive mean and covariance for the full state:

$$\mu_{k|k-1}^{(i)} = \mu_{k-1}^{(i)}, \quad P_{k|k-1} = P_{k-1} + Q. \tag{5}$$

Extract prior statistics to form the predictive distribution for the specific observed coordinate $d_k$, $p(U_k \mid U_{1:k-1}) = \mathcal{N}(m_k^{(i)}, s_k^{(i)})$, where

$$m_k^{(i)} = [\mu_{k|k-1}^{(i)}]_{d_k} \quad \text{and} \quad s_k^{(i)} = [P_{k|k-1}]_{d_k,d_k}.$$

2. **Bootstrap Proposal Sampling:** Sample the 1D coordinate realization $u_k^{(i)}$ from its predictive prior:

$$u_k^{(i)} \sim \mathcal{N}(m_k^{(i)}, s_k^{(i)}). \tag{6}$$

3. **Weight Update:** Compute unnormalized log-weights based on the 1D observation of the sampled coordinate, identical to the standard particle filter:

$$\log \tilde{w}_k^{(i)} = \log w_{k-1}^{(i)} + \log \mathcal{N}\left(y_k \mid u_k^{(i)}, R\right). \tag{7}$$

Normalize weights $w_k^{(i)}$ in log-space.

4. **Exact Marginalization:** Condition the full state on the specific realization $u_k^{(i)}$. By treating $u_k^{(i)}$ as a noise-free measurement of $X_k^{d_k}$, we can apply the standard Kalman update. The gain is $K = \frac{1}{s_k} P_{k|k-1,(:,d_k)}$.

$$\mu_k^{(i)} = \mu_{k|k-1}^{(i)} + K \left(u_k^{(i)} - m_k^{(i)}\right) \tag{8}$$

$$P_k = P_{k|k-1} - K \, P_{k|k-1,(d_k,:)}. \tag{9}$$

> **Note:** This operation forces the $d_k$-th element of $\mu_k^{(i)}$ to equal exactly $u_k^{(i)}$. Correspondingly, the $d_k$-th row and column of $P_k$ collapse to exactly zero, reflecting no residual uncertainty in the value of the sampled coordinate.

5. **Resampling:** Perform resampling on the conditional statistics $\{\mu_k^{(i)}\}$ based on $\{w_k^{(i)}\}$, subsequently resetting $w_k^{(i)} = 1/N$.

---

### 3.3 Compute-Efficient Lazy Evaluation

In high-dimensional spaces, updating the full state mean vector $\mu_k^{(i)}$ eagerly at every observation step requires $\mathcal{O}(ND)$ operations. An exact lazy evaluation strategy defers these updates until a coordinate is explicitly queried, preserving exactness by conditioning on all intermediate coordinates sampled since the last update.

Let $\tau(d)$ denote the time index when coordinate $d \in \{1, \ldots, D\}$ was last observed. Because the latent dynamics are linear and Gaussian, the predictive covariance matrix $P_k$ and the Kalman gain vectors $K_k$ are deterministic and independent of the particle realizations.

For each particle $i$, the sequence of sampled innovations $\Delta_t^{(i)} = u_t^{(i)} - m_t^{(i)}$ is retained. When coordinate $d_k$ is observed at time $k$, its exact conditional prior mean $m_k^{(i)}$ is computed by accumulating the historical innovations:

$$m_k^{(i)} = \mu_{\tau(d_k)}^{(i),d_k} + \sum_{t=\tau(d_k)+1}^{k-1} K_t^{d_k} \Delta_t^{(i)} \tag{10}$$

The prior variance is the diagonal element $s_k^{(i)} = [P_{k|k-1}]_{d_k,d_k}$ from the globally tracked covariance. Following the sampling of $u_k^{(i)}$, the innovation $\Delta_k^{(i)}$ is stored, the tracked mean is updated for the queried dimension to $\mu_k^{(i),d_k} = u_k^{(i)}$, and the time index advances via $\tau(d_k) = k$. This formulation is analytically identical to the standard RB update.

---

### 3.4 Backward Sampling (Smoothing)

To obtain the smoothing distribution $p(X_{0:T} \mid y_{1:T})$, we apply Forward Filtering Backward Simulation (FFBSi) to the Rao-Blackwellized particles. The forward filter approximates the distribution at time $k$ as a Gaussian mixture $\sum_{i=1}^N w_k^{(i)} \mathcal{N}(\mu_k^{(i)}, P_k)$. Because the observation model and timings are identical across particles, the state covariance $P_k$ is deterministic and independent of the particle index $i$.

1. **Initialization:** At the terminal time $T$, sample a particle index $I_T$ with probability proportional to the final filtering weights $w_T^{(i)}$. Draw the terminal full state $X_T^*$ from the corresponding conditional Gaussian:

$$X_T^* \sim \mathcal{N}\left(\mu_T^{(I_T)}, P_T\right). \tag{11}$$

2. **Backward Simulation:** For $k = T-1$ down to $0$, the smoothed density given the future sampled trajectory state $X_{k+1}^*$ is derived via Bayes' theorem:

$$p(X_k \mid X_{k+1}^*, y_{1:k}) \propto p(X_{k+1}^* \mid X_k) \sum_{i=1}^N w_k^{(i)} \mathcal{N}\left(X_k \mid \mu_k^{(i)}, P_k\right). \tag{12}$$

Because the latent transition $p(X_{k+1}^* \mid X_k) = \mathcal{N}(X_{k+1}^* \mid X_k, Q)$ is linear Gaussian, the product yields a new Gaussian mixture:

$$p(X_k \mid X_{k+1}^*, y_{1:k}) \approx \sum_{i=1}^N w_{k|k+1}^{(i)} \mathcal{N}\left(X_k \mid m_{k|k+1}^{(i)}, \Sigma_{k|k+1}\right). \tag{13}$$

The backward mixing weights are obtained by evaluating the predictive density of $X_{k+1}^*$:

$$w_{k|k+1}^{(i)} \propto w_k^{(i)} \mathcal{N}\left(X_{k+1}^* \mid \mu_k^{(i)}, P_k + Q\right), \tag{14}$$

where the weights $\{w_{k|k+1}^{(i)}\}_{i=1}^N$ are subsequently normalized. The conditional mean and covariance follow standard Rauch-Tung-Striebel (RTS) smoothing equations. Given the deterministic smoother gain $J_k = P_k(P_k + Q)^{-1}$, these are computed as:

$$m_{k|k+1}^{(i)} = \mu_k^{(i)} + J_k \left(X_{k+1}^* - \mu_k^{(i)}\right), \tag{15}$$

$$\Sigma_{k|k+1} = P_k - J_k P_k. \tag{16}$$

Sample an index $I_k$ according to the backward weights $w_{k|k+1}^{(i)}$, and simulate the state:

$$X_k^* \sim \mathcal{N}\left(m_{k|k+1}^{(I_k)}, \Sigma_{k|k+1}\right). \tag{17}$$

Repeating this backward pass multiple times produces independent realizations from the joint smoothing distribution. The structural independence of $P_k$, $P_k + Q$, and $J_k$ from the particle indices ensures the backward pass can be vectorized efficiently alongside the lazy evaluation scheme.

---

## 4. Simulation Study

We evaluate the standard PF and the RB-PF for $N = 10\,000$ on a $D = 10$-dimensional system over 150 time steps, executing systematic resampling unconditionally at every iteration. Ground truth inference is obtained via an exact Kalman Filter. Both filters use the standard bootstrap proposal mechanism, isolating Rao-Blackwellisation as the only point of algorithmic divergence. Results are illustrated in Table 1 and Figure 1 and code to reproduce is here. Additionally, the lazy implementation was $\sim 3.5$ times more efficient on this example.

**Table 1:** Root Mean Square Error (RMSE) against ground truth trajectory and exact inference.

| Filter | RMSE (vs True State) | RMSE (vs Kalman Filter) |
|--------|---------------------|------------------------|
| Kalman Filter (Exact) | 6.8645 | — |
| Standard PF | 7.8682 | 2.8071 |
| RBPF | 6.8550 | 0.1091 |

---

## References

Guéant, O. and Pu, J. (2019). *Mid-price estimation for european corporate bonds: a particle filtering approach.*

---

## Figure

**Figure 1:** First component mean estimates for the different methods. Vertical dashed lines correspond to observation times for the first component.

```
Filtering the 1st Coordinate of a 10-D Correlated Random Walk
True State
Kalman Filter
Standard PF
RBPF (Naive)
RBPF (Lazy)

70 |
60 |
50 |
40 |
30 |
20 |
10 |
 0 |________________________________________________
   0    20    40    60    80   100   120   140
              Time Step
```
