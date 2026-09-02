# Dissertation Feedback Log

This document is a continuous, reverse-chronological record of feedback on the dissertation. New feedback should be added immediately below this description so that the most recent review remains at the top. Each entry records the review timestamp, the material reviewed, the overall assessment, detailed findings, and any parts that were confirmed as correct.

## 2026-09-01 — Supervisor notes review (`ryan_notes.txt`)

Source: [`ryan_notes.txt`](./ryan_notes.txt). The supervisor had read Sections 2.1, 2.2, and 2.5, plus earlier RB-SMC discussion. This entry records the action items and the current todo list. Content-level changes are tracked here and applied in the source only where agreed; grammatical fixes have been applied directly.

### Write

% The football problem is a pairwise comparison problem with a high-dimensional latent state. The general state-space model notation and the filtering and smoothing objectives were introduced in Section~\ref{sec:background}; here we specialise the relevant structure to football. Let $x_t^k$ denote the latent state of team $k$ at matchtime $t$, and let $\mathcal{O}_t = \{i_t,j_t\}$ denote the two teams involved in match $t$. The observation likelihood therefore has the local form $G_t(y_t \mid x_t^{\mathcal{O}_t})$.

% The factorial hidden Markov model (FHMM) was introduced by Ghahramani and Jordan \citep{ghahramanijordan1995factorialhmm,ghahramanijordan1997factorialhmm}. A factorial state-space model (fSSM) is the corresponding general-state-space construction. Its initial distribution and transition law factorise across latent components, while an observation may depend jointly on several components. In the football setting, this structure would give

% \begin{equation}
% p(x_{0:T}^{1:K}, y_{1:T}) = \left[ \prod_{k=1}^K p(x_0^k) \prod_{t=1}^T p(x_t^k \mid x_{t-1}^k) \right] \prod_{t=1}^T G_t(y_t \mid x_t^{\mathcal{O}_t}).
% \end{equation}

% The factorisation expresses \emph{a-priori} independence of the team processes: the initial states are independent and each team evolves independently of the others. It does not imply posterior independence. Once a match result is observed, the likelihood couples the latent states of the two participating teams. Results from later matches can transmit this dependence through the network of opponents, even though each individual likelihood contribution involves only two teams.

% Duffield, Power and Rimella \citep{duffieldpowerrimella2024factorialssm} apply the factorial construction to online skill rating and sports. Their main computational approximation is to project the filtering distribution back into a factorial form after every pairwise update. They maintain a separate marginal filtering distribution for each team, propagate the marginals of $i_t$ and $j_t$ to the next matchtime, and assimilate the result in the joint distribution of the playing pair. The two marginal distributions are then extracted from this joint update and stored independently for subsequent matches. In this final marginalisation, or ``unpairing'', step, the posterior dependence between the two teams is discarded. The approximation is therefore not the pairwise assimilation itself; it is the repeated projection of the coupled posterior back to a product of team-level marginals.

% This approximation is computationally attractive because it avoids representing the full joint distribution over all teams. It is nevertheless restrictive for international football. Shared players, clubs, leagues, tournaments and common external conditions provide plausible sources of dependence between national teams that are not represented by an a-priori independent team prior. More generally, even when the prior team processes are independent, match outcomes create posterior dependence that a factorial filter removes after each update. The resulting team marginals may still be useful for prediction, but they do not retain the joint uncertainty induced by the observed competition history.

% The model developed in this chapter relaxes the factorial assumption by allowing dependence between teams through the initial covariance $\Sigma_0 = \Gamma_0 \otimes B$. When $\Gamma_0$ is non-diagonal, the model is not an fSSM under the definition above; it is a correlated Gaussian state-space model with a sparse pairwise likelihood. Rao--Blackwellisation preserves this cross-team dependence in the Gaussian component and samples only the four coordinates of the playing pair. RB-SQMC then applies low-discrepancy sampling to these four coordinates while retaining the complete component means and covariance needed by later matches. It consequently avoids Duffield et al.'s repeated factorial projection while remaining feasible for the $2K$-dimensional state. Here, ``exact'' means that the correlated model is not replaced by a factorial-independence approximation; for finite particle numbers, RB-SMC and RB-SQMC remain Monte Carlo approximations with sampling error.

% This distinction motivates the comparison between the factorial approximation and the correlated RB-SQMC procedure in the remainder of the chapter.

### Todo list

- [X] **Chase the factorial-SSM citation.** The text attributes the factorial state-space model (fSSM) to Duffield, Power and Rimella (2024). The supervisor notes they apply it to sports and are unlikely to have introduced it; Lorenzo Rimella has a separate JMLR paper on factorial state-space models that also cites the origin. Verify the true source and cite the original methodological reference. Do not fabricate a bib entry — confirm against the literature first.
  - **Document changes:** State that Ghahramani and Jordan introduced the factorial hidden Markov model (FHMM) in the 1995 NIPS paper, expanded in their 1997 *Machine Learning* article. Describe the fSSM as the general-state-space extension of this construction, and Duffield *et al.* as an application and formalisation for online skill rating and sports.
  - Add verified bibliography entries for Ghahramani and Jordan (1995, 1997). The 1997 journal article is the main methodological citation; cite the 1995 conference version when referring to the earliest publication.
  - Distinguish the **factorial model structure** from the **factorial inference approximation**: the prior and transition laws factorise across latent components, but a joint observation can induce posterior dependence between them.
- [X] **Revamp the Duffield/independence narrative (Chapter 3 `Background`).** The point of Duffield *et al.* is less the particular SSM form than the recursive projection back to independence. Explain (i) their inference procedure, (ii) what it implies for teams, and (iii) why the independence approximation is unsatisfactory for football. Position this work as making do with their approximation while retaining exactness via RB + SQMC. **The purpose of this work is to make do with the approximation and see how results differ; however because of the high dimensionality of the system we put a RB + SQMC procedure in place, allowing to handle the system as if it was of low dimension while retaining exactness, contrarily to them.**
  - **Document changes:** Explain Duffield *et al.*'s recursion explicitly: maintain independent player marginals; propagate the two players in the next match; assimilate the result in their joint distribution; then marginalise or “unpair” the result to restore independent player marginals. The discarded cross-player covariance is the approximation.
  - Explain the consequence for football: every match creates posterior dependence between the participating teams, and repeated matches transmit information through the competition graph. A factorial filter retains each team's marginal update but discards this joint dependence after each match.
  - Explain why this is a modelling limitation for national-team football. Shared players, clubs, leagues, tournaments, and common external conditions provide plausible sources of dependence between teams that cannot be represented by an a-priori independent team prior.
  - State explicitly that the proposed model with non-diagonal $\Gamma_0$ is not factorial under this definition. It is a correlated Gaussian state-space model with a sparse pairwise likelihood, or a relaxation of the factorial model.
  - Position RB-SQMC as the comparison method: it preserves the correlated Gaussian dependence through the Rao--Blackwell covariance update and samples only the four coordinates of the playing pair. It therefore avoids Duffield *et al.*'s repeated projection back to independent teams while remaining computationally feasible in the high-dimensional system.
  - Use “exact” carefully: RB-SQMC targets the non-factorial model without a factorial-independence approximation; finite particle SMC/SQMC still has Monte Carlo error and is not finite-$N$ exact.
- **Draft replacement text for Chapter 3 `Background`:**
  ```latex
  In a general state-space model, the joint distribution of latent states and observations can be factorized as
  \begin{equation}
  p(x_{0:T},y_{1:T}) = p(x_0)\prod_{t=1}^T p(x_t\mid x_{t-1})p(y_t\mid x_t).
  \end{equation}
  The general state-space model and the filtering and smoothing objectives were introduced in Section~\ref{sec:background}; here we specialise the discussion to football. Let $x_t^k$ denote the latent state of team $k$ at matchtime $t$, and let $\mathcal{O}_t=\{i_t,j_t\}$ denote the teams involved in match $t$. The likelihood therefore has the local form $G_t(y_t\mid x_t^{\mathcal{O}_t})$.

  The factorial hidden Markov model was introduced by Ghahramani and Jordan, and the factorial state-space model is its general-state-space extension. Its initial distribution and transition law factorize across latent components, while an observation may depend jointly on a local subset of components. In the football setting, this structure gives
  \begin{equation}
  p(x_{0:T}^{1:K},y_{1:T}) = \left[\prod_{k=1}^K p(x_0^k)\prod_{t=1}^T p(x_t^k\mid x_{t-1}^k)\right]\prod_{t=1}^T G_t(y_t\mid x_t^{\mathcal{O}_t}).
  \end{equation}

  This factorization expresses \emph{a-priori} independence of the team processes: initial team states are independent and each team evolves independently of the others. It does not imply posterior independence. Once a match result is observed, its likelihood couples the latent states of the two participating teams. Results from later matches can transmit this dependence through the network of opponents, even though each individual likelihood contribution involves only two teams.

  Duffield, Power and Rimella apply this factorial construction to online skill rating and sports. Their key computational approximation is the repeated projection of the filtering distribution back into factorial form after each pairwise update. They maintain a separate marginal filtering distribution for each team, propagate the marginals of $i_t$ and $j_t$ to the next matchtime, and assimilate the result in the joint distribution of the playing pair. They then extract the two marginal distributions from this joint update and store them independently for subsequent matches. This final marginalisation, or ``unpairing'', discards the posterior dependence between the two teams. Thus, the approximation is not the pairwise assimilation itself; it is the repeated projection of the coupled posterior back to a product of team-level marginals.

  This approximation is computationally attractive because it avoids representing the full joint distribution over all teams. It is nevertheless restrictive for international football. Shared players, clubs, leagues, tournaments and common external conditions provide plausible sources of dependence between national teams that are not represented by an a-priori independent team prior. More generally, even when the prior team processes are independent, match outcomes create posterior dependence that a factorial filter removes after each update. The resulting team marginals may remain useful for prediction, but they do not retain the joint uncertainty induced by the observed competition history.

  The model developed in this chapter relaxes the factorial assumption by allowing dependence between teams through the initial covariance $\Sigma_0=\Gamma_0\otimes B$. When $\Gamma_0$ is non-diagonal, the model is not an fSSM under the definition above; it is a correlated Gaussian state-space model with a sparse pairwise likelihood. Rao--Blackwellisation preserves this cross-team dependence in the Gaussian component and samples only the four coordinates of the playing pair. RB-SQMC therefore avoids Duffield et al.'s repeated factorial projection while remaining feasible for the $2K$-dimensional state. Here, ``exact'' means that the correlated model is not replaced by a factorial-independence approximation; for finite particle numbers, RB-SMC and RB-SQMC remain Monte Carlo approximations with sampling error.
  ```
- [X] **Specialise the football `Background` (Chapter 3).** The general state-space model is already introduced in Chapter 2 (`sec:background`); remove its re-introduction in the football chapter and specialise to football directly. *Merge of QMC and SQMC sections (`sec:qmc`/`sec:sqmc`) is out of scope per author decision.*
- [X] **A-priori independence wording (Chapter 3).** "latent states of index $k$ are independent" should read "a priori independent, but matches induce dependence". *Done 2026-09-01.*
- [X] **Purge negative characterisations and hallucinated terminology (Chapter 3).** (i) "it controls dependence but is not itself the score correlation" — replaced with a positive statement of what $\beta$ is. (ii) "rather than an exact application of their transition-sufficiency result" — the term "transition-sufficiency" does not appear in Chopin and Gerber (2017); the passage was rewritten without it, the projected ordering is labelled a dimension-reduction heuristic, and the locality assumption is stated positively. *Done 2026-09-01.*
- [ ] **Smoothing benefits of SQMC (Discussion).** SQMC's statistical benefit is concentrated in the filtering pass because the effective dimension is small under RB; the smoothing pass is full-dimensional and loses the QMC advantage. Consider fewer filtering particles in SQMC (e.g. SMC 10,000/100 vs SQMC 100/100). Add to the Discussion.

### Structural notes from the supervisor

- **Merge SQMC and QMC** into one section to avoid repetition (deferred per author decision).
- The football chapter re-introduces the general SSM although it is already covered; specialise instead.

### Confirmed as correct / positive

- The supervisor confirmed the earlier RB-SMC discussion; no corrections raised for that material in this pass.

## 2026-08-27 21:53:50 BST — Equation and derivation review of Chapter 2

Reviewed [`2_football_model_with_sqmc.tex`](../drafts/chapters/2_football_model_with_sqmc.tex) for mathematical accuracy only. No changes were made to the chapter during the review.

### Overall assessment

The Gaussian-conditioning and Kronecker-product algebra is mostly correct, but the chapter is not yet a mathematically valid complete derivation. Several filtering, SQMC-weighting, and parameter-gradient equations require correction.

### Critical issues

#### 1. State-space factorizations have inconsistent time indices

In equations (2.1) and (2.2), $p(x_1)$ is followed by a product beginning at $t=1$, which introduces $p(x_1\mid x_0)$. Either use

\[
p(x_0)\prod_{t=1}^T p(x_t\mid x_{t-1})p(y_t\mid x_t),
\]

or retain $p(x_1)$ and start the transitions at $t=2$.

Equation (2.2) also contains a free, unbound $k$ in $p(y_t\mid x_t^k)$. For football, the likelihood depends on two teams, so this term should involve the relevant subset $x_t^{\mathcal O_t}$, not one unspecified $k$.

#### 2. The Rao–Blackwell covariance recursion is incomplete

The Gaussian conditioning equations are correct. However, only the $\mathcal R\mathcal R$ conditional block is given. To justify the subsequent induction, the full filtered covariance must be defined:

\[
\Sigma_t
=
\Sigma_{t\mid t-1}
-
\Sigma_{t\mid t-1}^{:\mathcal O}
(\Sigma_{t\mid t-1}^{\mathcal O\mathcal O})^{-1}
\Sigma_{t\mid t-1}^{\mathcal O:}.
\]

Its $\mathcal O$ rows and columns are zero because those coordinates were sampled and are fixed within that mixture component. This update preserves the Kronecker structure, but that essential induction step is currently missing.

#### 3. The bootstrap proposal is conditioned on the wrong object

The expression

\[
p(x_t^{\mathcal O}\mid x_{t-1}^{\mathcal O})
\]

is generally not the required proposal. The playing pair changes with $t$, and correlated non-playing teams carry information. The proposal is the predictive marginal conditional on particle $i$'s full component or history:

\[
q_t^{(i)}(x_t^{\mathcal O_t})
=
\mathcal N\!\left(
\mu_{t\mid t-1}^{\mathcal O_t,(i)},
\Sigma_{t\mid t-1}^{\mathcal O_t\mathcal O_t}
\right).
\]

#### 4. The SMC weight recursion uses the wrong time index

The right-hand side of the log-weight equation should use $w_{t-1}^{(i)}$, not $w_t^{(i)}$, unless resampling immediately preceded propagation. In the latter case, the inherited numerical weight is $1/N$.

#### 5. The SQMC ancestor permutation is omitted

After Hilbert sorting, $a_i$ indexes the sorted particle array. The actual ancestor is therefore $\sigma_t(a_i)$, not merely $a_i$. The standard SQMC recursion explicitly uses $X_{t-1}^{\sigma_t(A_t^n)}$; see Gerber and Chopin's [SQMC algorithm](https://arxiv.org/abs/1706.05305).

#### 6. The SQMC weights double-count the previous weights

Ancestor weights have already been used during resampling. Under a bootstrap proposal, the new numerical weights are therefore

\[
\widetilde w_t^{(i)}\propto
G_t(y_t\mid x_t^{\mathcal O_t,(i)}),
\]

not $w_t^{(i)}G_t$. The cited SQMC algorithm likewise weights only by the new potential after resampling.

#### 7. Four dimensions are sufficient for propagation, but not demonstrably for Hilbert sorting

The random propagation dimension is indeed four. However, the Rao–Blackwellized particle state is the full component mean $\mu_{t-1}^{(i)}\in\mathbb R^{2K}$. Since the playing pair changes and future means inherit information from non-playing teams, sorting only $x_{t-1}^{\mathcal O_t}$ is not justified by the derivation. A proof that this projection is a sufficient Markov state is needed; otherwise this is a heuristic rather than an application of standard SQMC theory.

#### 8. The normalizing-constant and gradient equations are incorrect

With normalized weights and resampling at every step, the normalizing-constant estimator should not be defined as

\[
\widehat Z_T=\sum_i\widetilde w_T^{(i)}.
\]

Instead, under the chapter's weight convention it has the form

\[
\widehat Z_T
=
\prod_{t=1}^T
\left(\sum_{i=1}^N\widetilde w_t^{(i)}\right),
\]

with a $1/N$ factor in each term depending on how the unnormalized weights are defined.

Moreover,

\[
\nabla_\Theta\log p(y_{1:T}\mid\Theta)
=
\nabla_\Theta\log\widehat Z_T
\]

is not a samplewise equality. Stop-gradient resampling makes automatic differentiation of a specially constructed surrogate $\log\widehat Z_T$ recover the Poyiadjis path-space score estimator. That estimator is consistent, but generally not finite-$N$ unbiased. The stop-gradient correction must also appear explicitly in the resampled weights. This is how the result is stated by [Ścibior and Wood](https://arxiv.org/abs/2106.10314).

#### 9. The path-space score omits the initial state

Fisher's identity is correct, subject to the usual regularity assumptions, but the path-space approximation uses $p(x_{1:T},y_{1:T})$. It must include $x_0$, especially because $\Gamma_0$ appears in $p(x_0)$. In the Rao–Blackwellized filter, analytically marginalized coordinates must also be integrated in the score or supplied using a valid smoothing or Rao–Blackwell calculation.

### Additional issues

- The RTS equations are correct, but the transition from $t$ to $t+1$ uses $\phi_{t+1}$ under the chapter's earlier indexing, whereas the smoother uses $\phi_t$.

- The text preceding the smoother says one-step backward simulation cannot be used, but the proposed Gaussian-mixture backward kernel does exactly perform one-step backward draws. What is unavailable is discrete point-particle FFBSi without retaining the Gaussian component.

- The full state dimension is $2K$, not $2K\times2K$; the latter is the covariance dimension.

- $\mu_0=(0,0)$ is a per-team mean. The full mean is $1_K\otimes(0,0)$.

- $\kappa$ is missing from the parameter vector $\Theta$.

- $\Gamma_0\otimes B$ is scale non-identifiable because $c\Gamma_0\otimes(B/c)$ gives the same covariance. A constraint such as $\det(B)=1$ is required.

- The claimed $\mathcal O(N+K)$ complexity is unsupported as written. Updating $N$ full $K$-team means costs $\mathcal O(NK)$, while materializing the covariance update costs $\mathcal O(K^2)$.

- In the likelihood, $\alpha$ appears symmetrically in both scoring intensities, so it is a baseline scoring intercept, not a home-advantage parameter. Likewise, $\beta$ is the log shared-Poisson intensity; the actual score correlation is not simply $\beta$.

### Equations confirmed as correct

- The bivariate Poisson probability mass function.

- The scalar OU transition and stationary process covariance $Q_t=(1-\phi_t^2)\Sigma_0$.

- The conditional Gaussian mean and Schur-complement covariance.

- The Kronecker factorization
  \[
  K_t=\widetilde K_t\otimes I_2
  \]
  and the corresponding conditional covariance, assuming $B$ and the relevant $\Gamma$ block are invertible.

- The Gaussian-mixture backward weights and RTS conditional mean and covariance, apart from the $\phi$ index noted above.

### Exact changes to make in Chapter 2

The following changes implement the corrections identified above. They are written as replacements or insertions for [`2_football_model_with_sqmc.tex`](../drafts/chapters/2_football_model_with_sqmc.tex); they have not been applied to that file.

#### A. Replace equations (2.1) and (2.2)

Replace equation (2.1) with

\[
p(x_{0:T},y_{1:T})
=
p(x_0)\prod_{t=1}^T
p(x_t\mid x_{t-1})p(y_t\mid x_t).
\]

For independent latent factors with a likelihood depending on the local subset $\mathcal O_t$, replace equation (2.2) with

\[
p(x_{0:T}^{1:K},y_{1:T})
=
\left[
\prod_{k=1}^K
p(x_0^k)
\prod_{t=1}^T p(x_t^k\mid x_{t-1}^k)
\right]
\prod_{t=1}^T
p(y_t\mid x_t^{\mathcal O_t}).
\]

Define $\mathcal O_t$ as the indices involved in observation $t$. For a football match, $\mathcal O_t=\{i_t,j_t\}$. Also state explicitly that the proposed football model is not a factorial state-space model under the preceding independence definition when $\Gamma_0$ is non-diagonal. It is a correlated Gaussian state-space model with a local or sparse likelihood, or equivalently a relaxation of the factorial model.

#### B. Correct the dimensions and identify the covariance parameterization

Replace the initial-state definition by

\[
x_0\in\mathbb R^{2K},\qquad
x_0\sim\mathcal N_{2K}(\mu_0,\Sigma_0),
\]

where

\[
\mu_0=\mathbf 1_K\otimes
\begin{pmatrix}0\\0\end{pmatrix},
\qquad
\Sigma_0=\Gamma_0\otimes B.
\]

State the required conditions

\[
\Gamma_0\succ0,
\qquad
B\succ0,
\qquad
\det(B)=1.
\]

The last condition removes the scale non-identifiability because

\[
(c\Gamma_0)\otimes(B/c)=\Gamma_0\otimes B.
\]

If positive-semidefinite rather than positive-definite factors are intended, all inverses in the derivation must be replaced by generalized inverses and the Gaussian distributions must be treated as potentially degenerate.

#### C. Make the OU time indexing explicit

Define

\[
\Delta t_t=t_t-t_{t-1},
\qquad
\phi_t=\exp(-\kappa\Delta t_t),
\]

and retain

\[
x_t\mid x_{t-1}
\sim
\mathcal N\!\left(
\mu_0+\phi_t(x_{t-1}-\mu_0),
(1-\phi_t^2)\Sigma_0
\right).
\]

This makes $\phi_t$ the coefficient for the transition from $t-1$ to $t$. Consequently, every smoother equation for the transition from $t$ to $t+1$ must use $\phi_{t+1}$.

#### D. Correct the interpretation of the bivariate Poisson parameters

Keep the probability mass function and intensities unchanged, but replace the descriptions of $\alpha$ and $\beta$ with:

> The parameter $\alpha$ is a common log scoring-rate intercept because it enters both $\lambda_1$ and $\lambda_2$ symmetrically. The parameter $\beta$ is the log intensity of the shared Poisson component, $\lambda_3=\exp(\beta)$; it controls dependence but is not itself the score correlation.

If $\alpha$ is intended to represent home advantage, it must enter only the home-team intensity, for example

\[
\lambda_1
=
\exp(\eta+\alpha+x_t^{\mathrm{att},i}-x_t^{\mathrm{def},j}),
\qquad
\lambda_2
=
\exp(\eta+x_t^{\mathrm{att},j}-x_t^{\mathrm{def},i}),
\]

where $\eta$ is the common scoring intercept.

#### E. Use time-varying observed and complementary sets

Throughout the RB-SMC and RB-SQMC derivations, replace $\mathcal O$ and $\mathcal R$ by

\[
\mathcal O_t=\{i_t,j_t\},
\qquad
\mathcal R_t=\{1,\ldots,K\}\setminus\mathcal O_t.
\]

Interpret $x_{0:T}^{\mathcal O}$ in the path factorization as the time-varying collection $\{x_t^{\mathcal O_t}\}_{t=0}^T$. Since the likelihood depends only on that collection, add

\[
p(x_{0:T}^{\mathcal R}
\mid x_{0:T}^{\mathcal O},y_{1:T})
=
p(x_{0:T}^{\mathcal R}
\mid x_{0:T}^{\mathcal O}).
\]

#### F. Replace the RB-SMC prediction, proposal, conditioning, and weight equations

For particle or Gaussian-mixture component $i$, use

\[
\mu_{t\mid t-1}^{(i)}
=
\mu_0+\phi_t(\mu_{t-1}^{(i)}-\mu_0),
\]

\[
\Sigma_{t\mid t-1}
=
\phi_t^2\Sigma_{t-1}
+(1-\phi_t^2)\Sigma_0.
\]

Replace the proposal equation by

\[
q_t^{(i)}(x_t^{\mathcal O_t})
=
p(x_t^{\mathcal O_t}\mid I_{t-1}=i,y_{1:t-1})
=
\mathcal N\!\left(
\mu_{t\mid t-1}^{\mathcal O_t,(i)},
\Sigma_{t\mid t-1}^{\mathcal O_t\mathcal O_t}
\right).
\]

For the chapter's propagate-weight-resample ordering, replace the log-weight recursion by

\[
\log\widetilde w_t^{(i)}
=
\log w_{t-1}^{(i)}
+
\log G_t(y_t\mid x_t^{\mathcal O_t,(i)}),
\]

followed by

\[
w_t^{(i)}
=
\frac{\widetilde w_t^{(i)}}
{\sum_{n=1}^N\widetilde w_t^{(n)}}.
\]

After resampling, call the carried weights something distinct, such as $\bar w_t^{(i)}=1/N$, so that the pre-resampling filtering weights $w_t^{(i)}$ remain available for smoothing.

#### G. Insert the full covariance and mean update needed by the Kronecker proof

Define the full team-space gain

\[
\overline K_t
=
\Gamma_{t\mid t-1}^{:\mathcal O_t}
\left(
\Gamma_{t\mid t-1}^{\mathcal O_t\mathcal O_t}
\right)^{-1},
\]

where $\Gamma^{:\mathcal O_t}$ contains all rows and the two observed-team columns. Then update the complete component mean using

\[
\mu_t^{(i)}
=
\mu_{t\mid t-1}^{(i)}
+
(\overline K_t\otimes I_2)
\left(
x_t^{\mathcal O_t,(i)}
-
\mu_{t\mid t-1}^{\mathcal O_t,(i)}
\right).
\]

The rows of $\overline K_t$ belonging to $\mathcal O_t$ form the identity, so the observed entries of $\mu_t^{(i)}$ equal the sampled entries $x_t^{\mathcal O_t,(i)}$.

Define the full filtered covariance by

\[
\Gamma_t
=
\Gamma_{t\mid t-1}
-
\Gamma_{t\mid t-1}^{:\mathcal O_t}
\left(
\Gamma_{t\mid t-1}^{\mathcal O_t\mathcal O_t}
\right)^{-1}
\Gamma_{t\mid t-1}^{\mathcal O_t:},
\]

and hence

\[
\Sigma_t=\Gamma_t\otimes B.
\]

This equation supplies the missing induction step. It also produces zero rows and columns for the sampled teams within each conditional Gaussian component. The existing $\mathcal R_t\mathcal R_t$ Schur complement is the corresponding nonzero submatrix of this full update.

#### H. Replace the computational-complexity claim

Replace the claims of $\mathcal O(N+K)$ computation with:

> The Kronecker representation replaces dense operations on a $2K\times2K$ covariance matrix by operations on the $K\times K$ matrix $\Gamma_t$ and the $2\times2$ matrix $B$. The covariance recursion is shared across particles. With dense storage, updating the covariance costs $\mathcal O(K^2)$ per observation after a $2\times2$ solve, while updating all particle means costs $\mathcal O(NK)$. Thus the method reduces constants and avoids particle-specific covariance recursions, but it is not $\mathcal O(N+K)$ as currently written.

#### I. Correct the smoother explanation and time index

Replace the paragraph claiming that one-step backward simulation is unavailable with:

> Discrete point-particle FFBSi cannot be applied directly because each filtering particle represents a Gaussian component rather than a complete state. Nevertheless, a one-step backward kernel is available in closed form: first select a Gaussian component using its backward weight, and then draw the full state from that component's conditional Gaussian distribution.

Define

\[
P_t=\Sigma_t,
\qquad
A_{t+1}=\phi_{t+1}I_{2K},
\]

\[
a_{t+1}^{(i)}
=
\mu_0+\phi_{t+1}(\mu_t^{(i)}-\mu_0),
\]

and

\[
R_{t+1}
=
\phi_{t+1}^2P_t
+
(1-\phi_{t+1}^2)\Sigma_0.
\]

Then use the backward component weights

\[
w_{t\mid t+1}^{(i)}
\propto
w_t^{(i)}
\mathcal N(X_{t+1}^*\mid a_{t+1}^{(i)},R_{t+1}),
\]

the gain

\[
J_t=P_tA_{t+1}^{\mathsf T}R_{t+1}^{-1}
=P_t\phi_{t+1}R_{t+1}^{-1},
\]

and

\[
\mu_{t\mid t+1}^{(i)}
=
\mu_t^{(i)}
+
J_t(X_{t+1}^*-a_{t+1}^{(i)}),
\]

\[
\Sigma_{t\mid t+1}
=
P_t-J_tR_{t+1}J_t^{\mathsf T}.
\]

At the terminal time, write

\[
I_T\sim\operatorname{Categorical}(w_T^{1:N}),
\qquad
X_T^*\sim\mathcal N(\mu_T^{(I_T)},\Sigma_T).
\]

State that one complete smoothed trajectory is obtained by one backward pass and that the pass must be repeated independently to obtain $M$ trajectories. Do not say that the displayed algorithm automatically produces $N$ trajectories.

#### J. Correct the state dimension in the RB-SQMC section

Replace

> rather than the full state dimension $2K\times2K$

with

> rather than the full state dimension $2K$; the full covariance matrix has dimension $2K\times2K$.

Retain the statement that the RQMC propagation point has dimension $1+4=5$.

#### K. Replace the Hilbert-sort and ancestor equations

For the standard SQMC justification, sort the full RB particle state, which is the full component mean. Define

\[
\psi:\mathbb R^{2K}\to[0,1]^{2K}
\]

and choose $\sigma_t$ such that

\[
h(\psi(\mu_{t-1}^{(\sigma_t(1))}))
\le\cdots\le
h(\psi(\mu_{t-1}^{(\sigma_t(N))})).
\]

Use the previous normalized weights in sorted order:

\[
W_{t-1}^{(\sigma_t(1:N))}.
\]

Sort the complete QMC pairs $(u_t^{(i)},v_t^{(i)})$ by $u_t^{(i)}$, preserving the pairing between each $u_t^{(i)}$ and $v_t^{(i)}$. Define

\[
a_i
=
\min\left\{
j:
\sum_{k=1}^j
W_{t-1}^{(\sigma_t(k))}
\ge u_t^{(i)}
\right\},
\qquad
b_i=\sigma_t(a_i).
\]

The propagation equation must then use $b_i$:

\[
x_t^{\mathcal O_t,(i)}
=
\mu_{t\mid t-1}^{\mathcal O_t,(b_i)}
+
L_t\Phi^{-1}(v_t^{(i)}).
\]

Sorting only the four coordinates of the current pair may instead be retained as a heuristic, but then the text must explicitly label it a projected Hilbert sort and remove the claim that it follows directly from the standard full-state SQMC convergence argument.

#### L. Replace the RB-SQMC weight equation

Because ancestor weights were already used in inverse-transform resampling, replace the numerical weight equation with

\[
\log\widetilde w_t^{(i)}
=
\log G_t(y_t\mid x_t^{\mathcal O_t,(i)}),
\]

followed by normalization:

\[
w_t^{(i)}
=
\frac{\widetilde w_t^{(i)}}
{\sum_{n=1}^N\widetilde w_t^{(n)}}.
\]

Do not add the old ancestor weight a second time.

#### M. Replace the parameter vector and Fisher-score approximation

Replace the static parameter definition with

\[
\Theta=(\Gamma_0,B,\kappa,\alpha,\beta),
\]

subject to

\[
\Gamma_0\succ0,
\qquad
B\succ0,
\qquad
\det(B)=1,
\qquad
\kappa>0.
\]

Keep Fisher's identity, but write the complete path consistently:

\[
\nabla_\Theta\log p(y_{1:T}\mid\Theta)
=
\int
p(x_{0:T}\mid y_{1:T},\Theta)
\nabla_\Theta
\log p(x_{0:T},y_{1:T}\mid\Theta)
\,dx_{0:T}.
\]

If complete ancestral trajectories are available, replace the path-space approximation with

\[
\widehat S_{N,T}(\Theta)
=
\sum_{i=1}^N
w_T^{(i)}
\nabla_\Theta
\log p(\widetilde x_{0:T}^{(i)},y_{1:T}\mid\Theta),
\]

where $\widetilde x_{0:T}^{(i)}$ is the complete ancestral lineage of terminal particle $i$. State that this estimator is consistent under standard assumptions but generally biased for finite $N$.

For the Rao–Blackwellized filter, whose particles are Gaussian components rather than complete paths, replace the complete-data term by the conditional expectation

\[
\widehat S_{N,T}^{\mathrm{RB}}(\Theta)
=
\sum_{i=1}^N w_T^{(i)}
\mathbb E\!\left[
\nabla_\Theta
\log p(X_{0:T},y_{1:T}\mid\Theta)
\,\middle|\,
\text{RB lineage }i,
y_{1:T}
\right],
\]

or explicitly state that full trajectories are first generated using the RB backward sampler and then inserted into the preceding path-space approximation.

#### N. Replace the normalizing-constant and stop-gradient claims

When resampling occurs at every step and the carried numerical weights are $1/N$, define

\[
\widehat Z_0=1,
\qquad
\widehat Z_t
=
\widehat Z_{t-1}
\left[
\frac1N
\sum_{i=1}^N
G_t(y_t\mid x_t^{\mathcal O_t,(i)})
\right].
\]

Equivalently,

\[
\log\widehat Z_T
=
\sum_{t=1}^T
\log\left[
\frac1N
\sum_{i=1}^N
G_t(y_t\mid x_t^{\mathcal O_t,(i)})
\right].
\]

If adaptive resampling or nonuniform carried weights are used, replace $1/N$ by the corresponding normalized pre-propagation weights and state that convention explicitly.

Delete the equality

\[
\nabla_\Theta\log p(y_{1:T}\mid\Theta)
=
\nabla_\Theta\log\widehat Z_T.
\]

For stop-gradient resampling, define $\operatorname{sg}(z)$ as numerically equal to $z$ with zero derivative. If $b_i$ is the selected ancestor, the corrected post-resampling surrogate log-weight is

\[
\ell_{t,\mathrm{res}}^{(i)}
=
-\log N
+
\log w_{t-1}^{(b_i)}
-
\operatorname{sg}\!\left(
\log w_{t-1}^{(b_i)}
\right).
\]

Its forward value is $-\log N$, but its derivative retains the resampling score. The next surrogate unnormalized log-weight is

\[
\widetilde\ell_t^{(i)}
=
\ell_{t,\mathrm{res}}^{(i)}
+
\log G_t(y_t\mid x_t^{\mathcal O_t,(i)}).
\]

The corrected claim should be

\[
\widehat S_{N,T}^{\mathrm{SGR}}(\Theta)
:=
\operatorname{AD}_\Theta
\left[
\log\widehat Z_T^{\mathrm{SGR}}
\right]
\xrightarrow[N\to\infty]{\mathbb P}
\nabla_\Theta\log p(y_{1:T}\mid\Theta),
\]

under the regularity assumptions of the particle score estimator and with the proposal-gradient treatment required by the stop-gradient-resampling construction. Describe this as a consistent estimator of the score, not an exact equality or a generally finite-$N$ unbiased estimator.
