# Complete dissertation review: `ST980_5754603.tex`

**Review date:** 11 September 2026  
**Primary standard:** `dissertation/documents/TONE.md`  
**Material reviewed:** the root document, abstract, introduction, Chapters 2--4, all five appendices, bibliography, metadata and preamble, plus the compiled 66-page PDF.  
**Build status:** the dissertation compiles to an A4 PDF, but it does so with material warnings and unfinished content. The abstract contains 171 body words and is within the 200-word limit. The compiled document contains approximately 10,426 counted words including headings and captions.

The dissertation has a promising and coherent technical core: an accelerator implementation of SQMC, followed by a Rao--Blackwellised construction for a correlated football state-space model. The present draft is not yet submission-ready. The main risk is not length but evidential alignment. The CPU/GPU experiment is a valid test of backend performance, but the abstract currently describes it ambiguously as establishing SQMC's “advantage”; the intended end-to-end comparison with the original `cuthberto-carlos` EKF is also valid, but its interpretation must be limited to the two complete procedures rather than attributed to one component. The status of the test-likelihood trace depends on whether it influenced epoch or hyperparameter selection and should be documented. Monte Carlo uncertainty is largely absent, and the conclusion is only about 100 words. The limitations discussion acknowledges computational cost but omits several limitations that directly affect the scope of the principal conclusions.

# 1 Immediate Fixes

The changes in this section can be made to the current write-up without collecting new data or running substantial new experiments. Where a claim cannot be supported by the current evidence, the immediate fix is to narrow or remove it rather than leave it unqualified.

## 1.1 Resolve submission and build defects

1. **Replace the acknowledgements placeholder.** `chapters/acknowledgements.tex` currently contains only “Acknowledgements here.” This is visible as an almost blank page in the PDF. Either write the acknowledgements or remove the chapter entirely. A placeholder cannot remain in a submitted dissertation.

2. **Replace the automatic date.** `utils/metadata.tex` uses `\today`. Freeze the actual submission date so that recompilation does not silently change the title page. Also revise the title to sentence-style capitalisation: “High-performance sequential quasi-Monte Carlo with applications to football match modelling”. “Applications to” reads more naturally than “applications in”, “high-performance” needs a hyphen as a compound modifier, and the title should use the same lower-case treatment of “sequential quasi-Monte Carlo” as the body.

3. **Enable the lists of figures and tables.** The manuscript contains numerous figures and tables, but `\listoffigures` and `\listoftables` are commented out. Both should normally be enabled in a document of this length. Check that their placement is consistent with the department template.

4. **Remove duplicate equation anchors.** Many `equation` and `align` environments contain `\nonumber`. This suppresses the printed number without safely suppressing the equation counter and produces repeated PDF destinations such as `equation.2.2`, `equation.2.3`, `equation.3.12` and `equation.3.19`. Use `equation*`, `align*` or `gather*` for equations that will never be cited. Retain numbered environments only for equations that have a label and are referred to. This is both a PDF-navigation defect and a violation of the tone guide’s rule that numbered equations should have a role in the argument.

5. **Fix the remaining layout warnings.** The build reports an overfull line in Chapter 3 around the definition of the two-dimensional state, an overfull configuration table of roughly 32 pt, and an underfull Appendix B caption. Rewrap the opening model sentence; use a width-aware table (`tabularx`, narrower descriptions or a smaller but still legible table font) for the configuration; and shorten the Appendix B caption. Inspect the final PDF after each change.

6. **Correct the bibliography entry types.** `plainnat.bst` does not recognise the two `@software` entries (`cuthbertocarlos2026` and `ghq`). Convert them to compatible `@misc` entries or use a bibliography style that supports software. Include version or commit, release date, repository URL and access date. The current build emits two BibTeX warnings.

7. **Remove the duplicated `xcolor` package declaration.** `utils/preamble.tex` loads `xcolor` twice, once without and once with `dvipsnames`. Retain only the latter declaration.

8. **Do not call Appendix B “Implementation Code” while it contains no code.** Both `\lstinputlisting` commands are commented out, so the appendix contains a screenshot of a GitHub issue and two statements about files that are not reproduced. Either rename it “Implementation and reproducibility” and give commit-specific paths, environment and commands, or remove it because the code is supplied separately. A tracking-issue screenshot is not useful scholarly evidence and should be replaced by a textual repository reference. The sentence “This appendix reproduces the core implementation files” is currently false.

9. **Remove dead drafting comments before submission.** The sources contain “TO DO”, alternative paragraphs, placeholder table rows and long blocks of obsolete prose. Although comments do not appear in the PDF, they make the authoritative argument unclear and increase the risk that an outdated assertion is later restored. Keep development history in version control, not in the submission source.

10. **Use one spelling convention.** Active text alternates between British and American forms, including “randomized”, “parallelized” and “parameterization”. Change these to “randomised”, “parallelised” and “parameterisation”. Preserve original spellings only in titles, quotations and code identifiers.

## 1.2 Rewrite the abstract so that every claim is supported

The abstract is within the word limit but currently overstates three results.

- “propagates particles deterministically” is incomplete because the implementation uses randomised QMC scrambles. Say that SQMC replaces independent Monte Carlo inputs with randomised low-discrepancy point sets and imposes an ordering before resampling.
- “achieving an error rate smaller than” is too unconditional. The rate depends on regularity, randomisation and dimension. Use “can attain lower integration error than SMC under regularity conditions” or state the precise theorem and conditions.
- “GPU, TPU and multi-core CPU” is not supported by the reported experiments, which describe an NVIDIA A100 and an Intel CPU. Remove TPU unless an actual TPU result appears in the dissertation. Do not describe CPU as a “parallel accelerator” without explaining the threading implementation.
- “substantiate its advantage on a toy filtering model” is ambiguous: a reader is likely to interpret “its advantage” as SQMC's statistical advantage over SMC, whereas the experiment measures the same SQMC implementation on CPU and GPU. No SMC experiment is required for the intended hardware question. Replace the phrase with the result actually sought: at 2,048 particles, GPU execution of SQMC was 6.23--13.37 times faster than CPU execution across the five tested dimensions.
- “Rao--Blackwellisation overcomes high-dimension limitations” is too strong. Only four coordinates are sampled, but full component means and a full between-team covariance are retained; the four-dimensional projected Hilbert ordering is not formally justified in the manuscript; and no RB-SMC or full-state ablation demonstrates that Rao--Blackwellisation caused an SQMC gain. Use “reduces the propagation dimension to four” and explicitly call the projected ordering a heuristic unless a theorem is supplied.
- “does not outperform” needs its evaluation scope. State that RB-SQMC had worse Brier score, predictive log score and outcome accuracy on the 104-match prediction period, under one fitted run, while its noisy normalising-constant estimate was not directly comparable with the EKF approximation.
- The last sentence should state the contribution precisely: implementation in `jax`, accelerator benchmark and exploratory correlated football application. Avoid implying that code not yet merged into `cuthbert` is already part of a released package.

The revised abstract should use a problem--method--result--conclusion structure and retain at least one quantitative computational result and one quantitative predictive result. If statistical SQMC-versus-SMC performance is deliberately outside scope, say so through precise hardware language and do not imply that Chapter 2 tests it.

## 1.3 Rebuild the introduction around explicit questions and contributions

The introduction is only about 470 words and presently operates as a long chapter map. It needs an argumentative opening and explicit research questions.

1. **State the problem before presenting the method.** Open with the tension the dissertation investigates: SQMC may improve integration accuracy, but ordering costs and high effective dimension can erase that benefit. Sparse pairwise observations offer a possible route to lower-dimensional propagation.

2. **Separate three distinct problems.** The draft conflates (a) importance-weight degeneracy in high-dimensional particle filters, (b) deterioration of QMC efficiency with effective dimension, and (c) the computational cost of the Hilbert sort. Explain each mechanism separately. The sentence saying SQMC’s dimensional problem is “most likely” the curse of dimensionality that affects SMC is too speculative and does not distinguish the methods.

3. **Add explicit research questions.** The current evidence can answer narrowly framed questions such as:

   - How much faster is a shared `jax` SQMC implementation on an A100 GPU than on the reported CPU at matched particle counts and horizons?
   - Can a correlated football state-space model be filtered by sampling only the playing-team coordinates while analytically retaining uncertainty about non-playing teams?
   - Under the reported single-run design, how do RB-SQMC forecasts compare with the factorial EKF baseline?

   Do not ask whether SQMC is more statistically efficient than SMC unless that comparison is added.

4. **List inherited and original components separately.** Identify SMC, SQMC, Hilbert ordering, Rao--Blackwellised filtering, the bivariate Poisson model and the EKF baseline as inherited. Identify the `jax` implementation, accelerator benchmark, correlated Kronecker football formulation, projected RB-SQMC implementation and empirical comparison as the dissertation’s work. “This work is part of a contribution” is too vague to establish originality.

5. **Qualify the time-uniform claim.** The introduction currently describes a “safety guarantee” based on a linear-Gaussian problem and then generalises its motivation to long-horizon problems. State the assumptions, norm, order of limits and model class of the cited result. Do not suggest that finite-`N` SQMC is uniformly accurate merely because an asymptotic uniform-in-time result holds.

6. **Make the chapter map accurate.** Describe Chapter 2 as an accelerator-backend study rather than a statistical SQMC-versus-SMC study. Chapter 3 does not directly demonstrate that weight degeneracy was mitigated, and Chapter 4 contains almost no discussion. Describe what the chapters actually show after those chapters have been revised.

## 1.4 Correct and qualify the Chapter 2 theory

### State-space and SMC exposition

- Define `m_0`, `m_t` and `g_t` as densities, with their domains, not merely as “distributions”. Use `\mid` rather than `|` in the joint density. State the proposal densities beside the target densities and say whether all densities are with respect to Lebesgue measure.
- The filtering target is introduced, but `\pi_t(d\mathbf{x}_t)` then appears without defining whether `\mathbf{x}_t` means the current state or the complete path. Use one object consistently. If the chapter estimates `p(x_t\mid y_{1:t})`, do not switch silently to a path-space target.
- The SIS weight recursion should use the ancestor history or path when the proposal is path-dependent. If the presentation is intentionally restricted to a Markov proposal, say so.
- Replace “resampling has a huge impact” and the unsupported assertion that variance “diverges exponentially fast” with a cited, qualified statement. Also fix subject--verb agreement: “the variance ... remains”.
- Algorithm 1 describes independent-uniform inverse-CDF draws and is therefore multinomial resampling. Name it. SQMC later requires the ordered first coordinates of an RQMC point set, which is a different construction; explicitly distinguish the two uses.
- Algorithm 2 has no declared inputs or output and does not state whether resampling is always performed or triggered by ESS. Add the model, observations, particle count, proposals and random inputs as inputs, and return the weighted particle approximation and normalising-constant estimate.

### QMC and RQMC exposition

- Correct the RQMC unbiasedness equation: the integrand on the right is written as `f(\mathbf{u}_n)` although `\mathbf{u}` is the integration variable. Define expectation over the randomisation.
- Replace “a vector with low-discrepancy” with “a member of a low-discrepancy point set”. A single point does not have discrepancy in the sense being used.
- Replace “cover the entire support ... with lesser values” with a precise statement about more even coverage. QMC does not generally “obtain a better estimate for any function”; Koksma--Hlawka requires bounded Hardy--Krause variation and can be uninformative in high dimension.
- Verify every stated RQMC rate against the exact cited theorem. The claim that *any* discrepancy-preserving scramble has variance `O(N^{-2}(\log N)^{d-1})` for all square-integrable integrands is too broad. State the point-set class, scrambling scheme, integrand smoothness and whether the bound concerns variance, RMSE or MSE. The paragraph currently moves between all three.
- Explain why those ordinary RQMC rates do or do not transfer to sequential resampling. Do not present a static integration rate as an SQMC rate without the SQMC theorem’s conditions.
- “limited to dimensions `d <= 1111`” should be attributed to the particular direction-number table used by this implementation, not to the Sobol’ construction generally.
- State whether the first Sobol’ point is retained. The source contains two contradictory commented alternatives; this is a reproducibility choice and should be active prose.

### SQMC exposition

- Correct the Hilbert-locality explanation. Continuity of the forward Hilbert curve supports a locality statement in one direction; an arbitrary pseudo-inverse need not map every pair of nearby multidimensional points to nearby scalar keys. State only the property required by the cited SQMC convergence argument.
- The claim that any coordinate-wise monotone bijection is “discrepancy-preserving” needs a source and a precise definition. It maps the state into the cube, but it does not preserve star discrepancy literally in general. Explain the actual regularity condition used by the theorem and the transformation used in code.
- Algorithm 3 says “QMC or RQMC” but the surrounding error and uncertainty claims require randomisation. State the actual algorithm evaluated. Define `\Gamma_0`, `\Gamma_t`, `G_t`, `\psi`, `h`, `W_t`, the random point dimensions, inputs and outputs before the algorithm.
- Replace “Algorithm 3” with `Algorithm~\ref{alg:sqmc}` so numbering remains stable.
- Reconcile the contradictory rate statements. The abstract and introduction say the error rate is smaller than `O_P(N^{-1/2})`; the properties section says it is the same order with a smaller constant; then it says a faster rate may hold. Present the strongest result actually justified by the cited conditions, followed by the weaker general result. A “smaller constant” is not a universal theorem.
- The time-uniform paragraph needs the exact model and assumptions. Define `\hat\eta_t^N`, `\eta_t` and the norm in the displayed result. Explain that the limit is asymptotic in particle count and is not the same as a guarantee for a fixed computational budget.

## 1.5 Make the Chapter 2 experiment match its claims

1. **Rename the empirical question.** The experiment establishes backend acceleration for this SQMC implementation, not “the advantage of SQMC”. Use “accelerator performance of the SQMC implementation” throughout.

2. **Restore the currently commented interpretation and limitations paragraph.** It is among the strongest pieces of writing in the source. It correctly states that the `scipy` comparison confounds library and backend, no 10 ms configuration succeeds, high-dimensional error remains large, the 62-bit Hilbert key becomes extremely coarse, only one data set is used per dimension, and the results are hardware-specific. These qualifications must be visible, not comments.

3. **Fully specify the environment.** The main text or Appendix D should report CPU model rather than only “Intel Xeon at 2.20 GHz”, physical and logical cores used, GPU model, operating environment, `jax`/`jaxlib`, CUDA and XLA versions, precision (`float32` or `float64`), compilation policy, device synchronisation call, host--device placement, thread settings and whether CPU/GPU runs shared the same process. “More information ... in the Appendix” is currently inaccurate because Appendix D omits most of this.

4. **Define every symbol in the RMSE.** `R`, `T`, `P_{tj}` and the reference `\mu_{tj}` are not defined beside equation (2.6). Say that the Kalman filtering mean and variance are the analytic reference, state `R=16` if that is the actual number of randomisations, and explain why division by `P_{tj}` is appropriate. “Held-out” in the caption also needs a precise selection protocol.

5. **Clarify budget selection.** Explain which repetitions selected the particle count under each budget and which repetitions estimated error. If the same 16 randomisations were used for selection and evaluation, do not call the result held out. State that budgets apply to median device execution rather than end-to-end latency.

6. **Report uncertainty consistently.** Runtime plots use seven timing runs and IQR bands; error uses bootstrap intervals conditional on one data set. Explain what source of variation each interval represents and what it excludes. The bootstrap over 16 randomisations does not represent variability across observation data sets or hardware sessions.

7. **Add a compact numeric results table from existing outputs.** Record the cross-over particle count and speed-up range by dimension, plus the corresponding RMSE. This makes the central result recoverable without reading a plot and supplies numbers for the abstract and conclusion.

8. **Remove causal language from implementation comparisons.** The `scipy` CPU/`jax` GPU comparison changes generator implementation, object-construction policy and hardware simultaneously. It cannot attribute the gain to the GPU. The shared `jax` CPU/GPU comparison isolates the backend more closely and should carry the principal claim.

9. **Do not call the accuracy comparison “time to reach a target error” unless an error target is fixed in advance.** The current prose instead reports the runtime at a fixed particle count and selects configurations under runtime budgets. Rename the subsection or state the exact target and interpolation rule.

## 1.6 Correct Chapter 3’s model and algorithm narrative

### Background and computational claims

- Correct “a priori independent (independent given any observations)”. A priori independence means independence before observing data; pairwise likelihoods induce posterior dependence. The parenthetical currently says the opposite of the intended point.
- Remove both claims that the representation changes complexity from `O(NK)` to `O(N+K)` “terms”. The object being counted is undefined and the later algorithm carries an `N x 2K` array of particle-specific means. A dense mean update is at least `O(NK)` and a dense team-covariance update is `O(K^2)` after a `2 x 2` solve. State time and memory separately.
- The motivation for prior correlations needs tightening. National teams do not share players, so “shared players” is misleading. Clubs and leagues might generate correlated national-team form through player development, but the mechanism is not encoded directly. Present these as possible omitted common factors, not as evidence that a freely estimated dense `\Gamma_0` is correct.
- Distinguish dependence created by match outcomes from dependence imposed in the stationary prior. Transitive comparison information can arise under a posterior even if team processes are initially independent. This does not by itself justify a dense prior covariance.

### Model definition

- Fix “2-dimensional” to “two-dimensional”, the extra space in “`2K` -dimensional”, “paramterization”, “the parameter `\alpha` and `\beta` are”, and repeated “as mentioned”. These are representative of a larger grammatical pass.
- Define the order of the `2K` state coordinates once. The Kronecker expressions depend on whether states are ordered team-major or skill-major.
- Define the meaning and sign of attack and defence explicitly. Because defence enters with a minus sign, a larger defence coordinate represents stronger defence. Use that fact when later defining total strength.
- Explain the bivariate Poisson parameters in the fitted model. `\alpha` is a common scoring intercept as currently written, not home advantage. The text introduces hypothetical `\alpha^h` and `\alpha^a` but the parameter vector contains only `\alpha`; remove the hypothetical extension from the model definition or clearly label it as future work. State that `\exp(\beta)` is the shared Poisson intensity and constrain it consistently.
- State how neutral-site World Cup matches are encoded as “home” and “away”. A common intercept may be reasonable for neutral matches, but the order must not be interpreted as home advantage.
- Discuss identifiability beyond the Kronecker scale. Fixing `\mu_0=0` anchors location, but attack/defence contrasts and the intercept can still be weakly identified. State the actual unconstrained-to-constrained parameterisation used for `\Gamma_0`, `B`, `\kappa` and `\beta`, and any jitter or eigenvalue floor used for positive definiteness.
- A dense `48 x 48` `\Gamma_0` has 1,176 free entries before the two parameters of `B`. This is a substantial model relative to 4,702 training matches. State whether it is regularised, structured or initialised from external information. If it is not, explicitly identify weak identification and overfitting as limitations.
- Explain the modelling consequence of using `\Sigma_0` both as the initial covariance and as the stationary OU covariance. This forces the prior dependence pattern to be the long-run dependence pattern restored between observations.

### RB-SMC filtering

- Rewrite the first paragraph of Section 3.3. `x_{0:T}^{\mathcal O}` is ambiguous because `\mathcal O_t` changes over time. Define the sampled sequence as `(x_t^{\mathcal O_t})_{t=1}^T` and the complementary coordinates at each time. Do not imply a fixed partition of the complete trajectory.
- Make the operation ordering consistent. The RB-SMC section says “propagate--weight--resample”, while Chapter 2’s bootstrap filter and the RB-SQMC section use “resample--propagate--weight”. Pick the implemented convention and write the weights and normalising-constant recursion for that convention.
- Replace manual bold numbered paragraphs with an algorithm environment containing inputs, outputs and execution order. Include initialisation at `t=0`, log-sum-exp normalisation, the resampling schedule and the returned likelihood/filter summaries.
- The Kalman conditioning formula omits particle superscripts from several predictive means. Because means are particle-specific, use them consistently.
- Replace explicit matrix inverses with linear solves in both prose and pseudocode, and state any Cholesky/jitter stabilisation.
- Replace “once a team has played a match, its skill level is exactly known” with “conditional on a particle’s sampled playing-team coordinates, those coordinates have zero residual Gaussian covariance within that mixture component”. Posterior uncertainty remains across particles. The present wording wrongly suggests inferential certainty.
- Correct the cost claim at the end of the section. The relevant covariance update uses a `2 x 2` team-space solve and dense products of order `K^2`; it is not naturally `O(K^3)`. Updating all particle means is `O(NK)`. The comparison with `O((2K)^3)` is inconsistent with Appendix A, which correctly describes a `2 x 2` solve.

### Smoothing

- Repair the backward-kernel derivation. The product should first contain `N(x_t\mid\mu_t^{(i)},\Sigma_t) p(x_{t+1}\mid x_t)`. It can then be factorised into the predictive density `N(x_{t+1}\mid a_{t+1}^{(i)},R_{t+1})` times the conditional Gaussian in `x_t`. The current displayed substitution replaces the transition density with a factor that does not depend on `x_t`, so the derivation as written is invalid even though the subsequent RTS formula has the intended form.
- Clarify whether saved filtering weights are pre- or post-resampling. If resampling is performed every step and stored weights are all `1/N`, the backward weights must be described accordingly.
- “M independent smoothed trajectories” is only conditional on the same fitted filter approximation and independent backward random draws. It does not mean independent evidence about parameter uncertainty. Qualify the statement.
- The section says the method is “extremely parallelisable” even though each path is sequential in reverse time and requires `O(N)` mixture evaluation per time. Say that paths can be parallelised across `M`, while each path retains a sequential `T` recursion.
- The proposed RB-SQMC smoother uses dimension `1+2K`, so it loses the four-dimensional filtering advantage and may be poorly suited to QMC. This is currently stated as if it were a routine extension. Mark it as an untested proposal and move it to future work unless it was implemented and evaluated.

### RB-SQMC filtering

- The active text says the four-dimensional projected sort satisfies the cited conditions, while several commented drafts correctly call it a heuristic because non-playing component means affect future matches. Resolve this in favour of the cautious statement unless a formal proof is supplied. The local likelihood alone does not show that four current coordinates are a sufficient ordering state for the future transition.
- Define the map `\psi_t` exactly as implemented: centring/scaling rule, treatment of zero variance, clipping, quantisation and Hilbert key resolution. Chapter 2’s implementation limitation says a packed 62-bit key becomes very coarse at large dimension; the four-dimensional football key should be documented separately.
- State whether points are freshly independently scrambled at every match and how keys are split for initialisation, resampling and propagation.
- Add the initial `t=0` RB-SQMC step. The current five steps begin with an update-time point set and never define the initial particle mixture.
- “Recover the non-playing teams’ coordinates in closed form” is inaccurate if the algorithm stores conditional Gaussian moments rather than coordinates. Say that it updates their conditional mean and covariance analytically.
- Appendix A’s final sentence says the only approximation is finite QMC resampling error. That is too strong: finite-particle filtering error, projected ordering, parameter-estimation error, numerical quantisation and model misspecification also matter. Narrow the statement to the correctness of the Gaussian inverse-CDF propagation conditional on an ancestor.

## 1.7 Correct the parameter-estimation account

1. **Do not equate the implemented gradient with exact marginal-likelihood maximisation.** The text says “we therefore maximise” the exact likelihood by backpropagating through `\log \widehat Z_T`. In a finite-particle run this is a stochastic surrogate/score estimator. State its target, finite-`N` bias, variance and consistency conditions.

2. **Describe the stop-gradient construction precisely.** Merely blocking gradients through discrete ancestor indices is not a full explanation of the Ścibior--Wood estimator. Show the actual surrogate-weight correction or cite the exact proposition and map each mathematical term to the implementation. State whether the implementation was numerically checked against finite differences on a small model.

3. **Fix the trajectory notation.** Equation (3.20) uses `x_{0:T}^{(i)}` but the following sentence defines `\widetilde{x}_{0:T}^{(i)}`. Use one symbol. More importantly, explain how the analytically marginalised coordinates contribute to the score for `\Gamma_0` and `B`; a sampled playing-team lineage alone is not obviously the complete-data score for the full correlated Gaussian state.

4. **Define the initial likelihood contribution.** The normalising-constant recursion starts with `\widehat Z_0=1`, although parameters occur in the initial state distribution. Explain whether initial proposal weights cancel exactly and why.

5. **Report optimisation diagnostics already available.** Give initialisation, parameter transforms, gradient clipping, convergence criterion, objective variability across 15 scrambles and the reason for exactly 50 epochs. A noisy curve that reaches its best test value at epoch 49 is not evidence of convergence.

6. **Remove causal language about smoothing bias.** Path degeneracy can produce a high-variance or biased finite-particle score/covariance estimate, but the draft asserts a particular bias in `\Gamma_0` without evidence about direction. Say “can make the estimate unstable or dominated by few ancestors” unless a simulation establishes bias.

## 1.8 Repair the football experimental design description and results

### Data lineage and split

- State the exact frozen data commit or checksum, licence, retrieval date and unit of observation in the main chapter, not only the AI declaration and appendix.
- Explain how a 49,521-row source becomes 4,702 training, 305 test and 104 prediction matches. It appears that historical data may have been restricted to the 48 tournament teams. If so, this is selection using future tournament membership and excludes matches involving other opponents; document and justify it because it can materially bias strength estimation.
- List exclusions with counts: pre-1980 matches, teams outside the retained set, unresolved identifiers, duplicate records, abandoned matches, extra-time/penalty treatment and scores above eight. Explain whether the 4--6 third-place score includes extra time or penalties and whether it is compatible with the score model.
- Describe team-name reconciliation and geopolitical succession. International football data require rules for names such as United States/USA, Ivory Coast/Côte d’Ivoire, DR Congo, historical states and renamed associations.
- Clarify same-day match ordering. Retaining repeats with zero elapsed time does not resolve their within-day order; an arbitrary CSV order may affect sequential updates.
- There is no validation set, and the test log-likelihood is plotted at every epoch with a “best test epoch”. Monitoring is not leakage by itself if 50 epochs, all hyperparameters and the final forecast checkpoint were fixed without reference to this curve. It becomes model selection if the curve influenced stopping, checkpoint choice, hyperparameters or which run was reported. State explicitly which case applies. If the test trace influenced any decision, rename it validation and reserve the 104-match tournament as the final test. If it was descriptive only, retain the pre-specified final-epoch value as the comparison and label the maximum as a noisy diagnostic rather than evidence of generalisation.

### Fairness of the comparison

- Complete the opening sentence of Section 3.7; it currently ends “for the World Cup 2026” without punctuation or a stated purpose.
- Remove “biased Gaussian EKF baseline” as a label. Every finite-particle and approximate method has error. Use “factorial EKF approximation” and then explain its approximation explicitly.
- The intended comparison with the original `cuthberto-carlos` model is a legitimate end-to-end benchmark: it asks whether the proposed RB-SQMC pipeline improves on the existing procedure. State that purpose explicitly. Appendix D nevertheless claims that “any difference” is attributable specifically to the correlated Rao--Blackwellised construction. That narrower attribution does not follow because the two pipelines also differ in prior dependence, posterior factorisation, likelihood integration, stochastic versus deterministic objective, and possibly time-step convention. Replace only the component-level attribution; do not present the end-to-end comparison itself as invalid.
- State whether both methods use exactly the same likelihood, transition clock, static parameter constraints, initialisation, friendly-match weighting and score truncation. A commented paragraph says the methods use different time-step conventions; if that is true, it is a major active limitation and must not remain hidden in a comment.
- Do not present raw normalising-constant values as a clean model-fit comparison when one is an EKF approximation and the other is a noisy particle estimate. “Best test likelihood” especially selects an extreme favourable SQMC draw. Report the final pre-specified checkpoint and uncertainty across scrambles, or treat the likelihood curves as optimisation diagnostics only.

### Prediction metrics and interpretation

- Check the definition of “predictive log-score”. Values of `-3.131` and `-3.664` appear unusually low for a three-category home/draw/away log score and may instead be exact-score log scores. Verify the code and label. If it is a scoreline log score, define `p(y_i^{home},y_i^{away})`, not `p_{i,o_i}` over three outcomes.
- State whether the `0,...,8` score grid is renormalised after truncation and how much probability lies outside it. Otherwise both the Brier probabilities and log score may be improperly normalised.
- Add uncertainty to all 104-match comparisons. At minimum, use a paired bootstrap over matches for differences in Brier score, log score and accuracy, while acknowledging dependence within a tournament. For RB-SQMC, also separate across-scramble Monte Carlo variation from match-sampling variation.
- Report calibration, as required by the tone guide: reliability plots or calibration intercept/slope for home/draw/away probabilities, and possibly ranked probability score. Accuracy alone discards probability quality.
- The result paragraph should state the actual differences: RB-SQMC’s Brier score is 0.1436 higher, its reported log score is 0.533 lower, and its accuracy is 5.76 percentage points lower. Then state that uncertainty has not yet been quantified and no population-level superiority claim follows from one tournament.
- The final-match case study is too prominent relative to the aggregate evidence and encourages anecdotal interpretation. Retain it as an illustration of forecast shape, not evidence of comparative performance. Make “red outline” and the caption’s “red cross” consistent.
- Use “GitHub”, not “Github”, and give a commit-specific link or appendix table rather than saying all results are somewhere in a repository.

### Qualitative evaluation and rankings

- Remove the claim that EKF trajectories are “linear” because the filter is deterministic and RB-SQMC means are “more stochastic, representing random evolution”. A posterior mean can be jagged or smooth under either method; visual roughness is not evidence of a more realistic stochastic process. Describe the observed paths and uncertainty bands instead.
- Figure 3.5 shows point trajectories without posterior intervals. Add uncertainty bands or clearly state that only means are plotted. Comparing a deterministic approximation with one noisy particle realisation otherwise confounds posterior dynamics with Monte Carlo noise.
- The correlation plot needs values, uncertainty or stability across runs. The statement that regional pairs “evolve together” is a post-hoc interpretation of an estimated prior covariance, not an established mechanism. Explain how `\Gamma_0` was regularised and whether signs persist across seeds.
- Define “total strength” before the ranking comparison and justify attack plus defence under the likelihood’s sign convention.
- FIFA ranking after the tournament is not an independent gold standard; it incorporates overlapping match information and a different rating objective. Describe it as an external descriptive reference, not validation of latent-state correctness.
- Replace top-ten anecdotal overlap with pre-specified summaries such as Spearman or Kendall rank correlation, top-`k` overlap and rank displacement, with uncertainty where possible. Do not infer method validity because selected teams “look plausible”.
- Correct “Elo-stye”, capitalise “Go” only if referring to the game, and avoid the unsupported claim that the FIFA method is simply Elo without describing FIFA’s actual update rule.

## 1.9 Expand and correct the limitations section now

The answer to whether limitations are appropriately addressed is **partly, but not sufficiently**. The computational-cost paragraph is concrete: it names 512 particles, an A100 and a 1.6-hour optimisation run, and it states that multiple seeds and scrambles were not run. That is good practice. However, computational cost is not the only limitation and may not be the one with the largest effect on validity.

Add separate paragraphs covering the following, with the affected conclusion and a remedy in each paragraph:

- **Scope of the end-to-end comparison:** EKF versus RB-SQMC is appropriate when the estimand is “Does the proposed complete pipeline outperform the original `cuthberto-carlos` pipeline?” Because model dependence and inference change together, the result should not additionally be attributed to SQMC, Rao--Blackwellisation or removal of the independence approximation. Controlled component ablations are optional future work if the dissertation wants to explain *why* the pipelines differ; they are not required for the stated end-to-end comparison.
- **Test-set reuse:** monitoring and selecting by test likelihood makes that set validation data. Remedy: pre-specify the final checkpoint or introduce a separate validation period.
- **Single stochastic fit:** one seed/finite-particle run cannot establish numerical stability. Remedy: repeat fits across optimisation seeds and RQMC scrambles and report between-run dispersion.
- **Projected sorting heuristic:** the four-dimensional playing-team key may omit information in non-playing component means relevant to future observations. Remedy: compare projected ordering with full-state ordering on smaller `K`, random ordering and RB-SMC.
- **Data selection and team identity:** restriction to the 48 tournament teams and unreported matching/exclusion rules may distort learned strengths. Remedy: publish a flow table and sensitivity analysis including all historical opponents.
- **Model misspecification:** stationary OU dynamics, a shared attack--defence covariance, constant scoring/shared-Poisson parameters, no covariates, neutral venue handling and a dense time-invariant cross-team covariance are strong assumptions. Remedy: posterior predictive checks and targeted alternatives.
- **Parameter identification and regularisation:** a dense `\Gamma_0` is large relative to the effective data and may explain implausible correlations/rankings. Remedy: shrinkage, low-rank/confederation structure or priors, with sensitivity to regularisation.
- **Forecast sample size and dependence:** 104 matches from one tournament are not 104 independent draws from a stable population. Remedy: rolling-origin evaluation over several historical tournaments or seasons.
- **Metric and truncation uncertainty:** verify score-grid normalisation and log-score definition; report calibration and uncertainty. Remedy: code-to-equation audit and paired intervals.
- **Hardware scope:** speed-ups are conditional on one A100/CPU/software session, exclude compilation and transfer, and may not transfer to laptop or TPU settings. Remedy: narrow the claim and report end-to-end as well as steady-state timing.

Remove “the simplest solution is ... better hardware”. More hardware may reduce Monte Carlo error or permit replications, but it does not resolve model misspecification, ambiguity about what component caused an end-to-end difference, or test-selection concerns if the test curve influenced decisions. Present more compute as one resource trade-off, not a methodological solution.

The paragraph listing particle stitching, EM, marginal particle filters and PaRIS currently reads as a catalogue. For each method, state which observed bottleneck it addresses, what approximation it introduces, and whether it is compatible with the required correlated filtering distribution. In particular, the marginal particle filter does not merely “keep only the current marginals”, and EM does not generically converge in fewer iterations; qualify those statements.

## 1.10 Replace the conclusion in full

Chapter 4 is visibly underdeveloped and is the most urgent prose revision. It is about 105 words, does not answer explicit research questions, reports no numbers, does not synthesise theory with evidence and has no adequate limitations or future-work discussion.

Write a two- to four-page conclusion with the following order:

1. Restate the two or three research questions in one paragraph.
2. Answer the accelerator question quantitatively and condition it on the A100/CPU setup, matched work, excluded compilation/transfer and tested particle grid.
3. State what the Rao--Blackwellised construction contributes mathematically: four sampled playing-team coordinates with analytical Gaussian updates for the remainder, while noting that projected Hilbert ordering remains heuristic unless proved.
4. Answer the end-to-end football comparison with all principal metrics and its scope. It is valid to conclude that the original `cuthberto-carlos` EKF pipeline performed better than the proposed RB-SQMC pipeline on the reported forecasts. Do not infer from that result alone that the EKF’s “inherent assumptions are valid”. At most, it is consistent with the simpler pipeline being adequate under this data and design; the difference could arise from any combination of model structure, approximation, optimisation noise or misspecification of the correlated model.
5. State the contribution at the right level: implementation and exploratory evaluation, not proof of universal accelerator or predictive superiority.
6. Give a prioritised limitations synthesis rather than referring only to compute.
7. End with two or three directly motivated future studies: controlled RB-SMC/RB-SQMC ablations, repeated rolling-origin football evaluation with regularised covariance, and filtering-focused SQMC where the low effective dimension can be retained. Explain what each would establish.

Delete promotional closing language about “continuous development” unless it is linked to a concrete released version and reproducibility record. The conclusion must close the dissertation’s statistical argument, not the software project’s roadmap.

## 1.11 Repair the appendices

- **Appendix A, Proposition A.1:** The parameter-count calculation is useful, but the proof’s final sentence contrasts `O(K^2)` with `O(K^2)` and then refers vaguely to a `2K x 2K` solve. Rewrite it to distinguish storage, the `2 x 2` Schur solve, dense rank-two covariance update and `N` particle-mean updates. Make it consistent with the main chapter.
- **Appendix A, Proposition A.2:** A deterministic QMC point is not itself a random `Unif([0,1]^4)` vector. State the proposition for ordinary random input, then separately explain the randomised-net marginal property. Do not claim that deterministic invertibility automatically preserves low discrepancy under an unbounded Gaussian inverse CDF. Narrow the proposition to exact Gaussian transformation of a uniform vector.
- **Appendix B:** Remove the issue screenshot and false claim that source is reproduced, or include the actual short extracts and make the appendix analytically useful. Full code belongs in the electronic supplement.
- **Appendix C:** It says it reports all 104 fixtures, but only the knockout rounds (31 matches) appear and “World Cup 2026 fixtures” is an empty section. Either include all 104 or change every claim to “knockout-stage fixtures”. Several captions say “win--loss predictions” while draws are allowed; use “outcome and modal-score predictions”.
- **Appendix C:** The table entries are scorelines, not merely predicted outcomes. Explain whether modal scorelines or modal three-way outcomes are shown; those can disagree.
- **Appendix D:** Add missing environment, parameter transforms, initial values, data checksum and commands. Replace the overclaim about attribution. Use `\texttt{max_goals}` and `\texttt{match_scale}` rather than mathematical text for code identifiers.
- **Appendix E:** The pre-tournament table declares four columns but supplies three. Change `{clll}` to `{cll}`. Explain why only top twenty are shown if the text calls these “full” rankings. Include a date/source for the pre-tournament external ranking if one is later added.

## 1.12 Complete a sentence-level tone and notation pass

The following are not exhaustive, but they illustrate recurring changes required by `TONE.md`:

- Use “sequential quasi-Monte Carlo”, not “Sequential Quasi-Monte Carlo”, in ordinary prose and headings unless it begins a sentence.
- Replace “we can expect the algorithm to reach faster convergence for a smaller class of functions” with the exact convergence result and its function class.
- Replace “This safety guarantee” with “Under the assumptions of [result], this asymptotic bound motivates...”.
- Replace “performance advantage ... decreases in increasing dimensions” with “the observed or theoretical advantage may deteriorate as effective dimension increases”.
- Replace “This provides the motivation for this dissertation” with a sentence specifying the unresolved question.
- Replace “The most likely factor” with a cited mechanism or explicitly marked hypothesis.
- Replace “a key point to note”, “as mentioned before”, “as stated before”, “similarly in Section”, “so then” and “the simplest solution” with direct claims.
- Fix agreement and punctuation: “these two characteristics mean”; “the raw results do not”; “the weighted particle mean ... appears”; “the EKF states were”; remove “Empirically,,”; add punctuation after bold paragraph labels.
- Use `\texttt{jax}`, `\texttt{scipy}`, `\texttt{numpy}` and repository names consistently. Use “GPU” and “CPU” as hardware nouns, not package names.
- Use one notation for particle index (`n` or `i`), one notation for normalised weights (`w` or `W`) and one notation for observations and realised states. At present these change across algorithms.
- Define transpose as `\top` consistently instead of mixing `T` and `\top`.
- Use `\operatorname{Unif}` or `\mathcal U` consistently and use `[0,1)^d` only when the half-open cube matters.
- Put punctuation in surrounding prose rather than inside displayed equations, following the project’s tone guide.
- Give every consequential displayed equation a label and later use, or make it unnumbered. Avoid consecutive displays without an interpretive sentence.
- Captions should identify data, sample size, method and uncertainty summary but should not perform the interpretation. Ensure all visual abbreviations are defined.

# 2 Overall Feedback

## 2.1 What is already working

The dissertation has a defensible research spine. The link between sparse pairwise observations and lower-dimensional propagation is statistically interesting, and the distinction between a correlated Gaussian state model and a factorial posterior approximation has the potential to be an original MSc-level contribution. The `jax` benchmark is more carefully designed than a simple wall-clock anecdote: it includes warm-up, synchronisation, matched implementations for the principal backend comparison, repeated timings and explicit exclusion of compilation and transfer. The football chapter uses chronological splits and reports a negative predictive result rather than hiding it. The computational limitation is quantified. The draft also has a reasonable set of primary methodological citations and a compact abstract within the formal limit.

These strengths should be made more visible by narrowing claims. At present, strong work is obscured by claims that the design cannot identify. A careful statement that an A100 accelerates a shared implementation by a measured factor is stronger than an unsupported claim that accelerator SQMC is generally superior. Likewise, a candid exploratory finding that the correlated particle model underperformed a factorial EKF on one tournament is more credible than inferring that the EKF assumptions are “valid”.

## 2.2 The central argument needs one consistent hierarchy

The manuscript currently shifts among four possible theses:

1. SQMC is statistically more accurate than SMC.
2. GPUs make SQMC computationally practical.
3. Rao--Blackwellisation makes SQMC effective in a high-dimensional sparse model.
4. The original `cuthberto-carlos` EKF pipeline outperforms the proposed RB-SQMC pipeline on the reported football forecasts.

The current experiments directly support a restricted, hardware-conditional version of (2) and an end-to-end version of (4): on the reported 104 forecasts, the original EKF pipeline outperformed the proposed RB-SQMC pipeline on the stated metrics. They do not test (1), and that is acceptable if statistical SQMC-versus-SMC comparison is outside scope. They also do not isolate the contribution of Rao--Blackwellisation or establish adequacy of the EKF assumptions. The final dissertation should therefore make (2) the established computational result, describe (3) as the methodological construction, and frame (4) as a comparison of complete pipelines. Claim (1) should remain literature motivation rather than an empirical contribution.

A suitable overarching statement would be: *This dissertation implements accelerator-compatible SQMC in `jax`, measures its backend performance, and investigates whether a Rao--Blackwellised four-coordinate propagation scheme makes SQMC usable for a correlated football state-space model. On the reported A100 configuration the implementation is substantially faster than on CPU at moderate particle counts; in the single football comparison, however, the correlated RB-SQMC fit is computationally expensive and predicts worse than the factorial EKF baseline.* This is precise, original enough to be interesting and fully compatible with a candid limitations section.

## 2.3 The literature review is technically broad but insufficiently evaluative

Chapter 2 explains many ingredients, but it often reads as a catalogue of SMC, QMC, RQMC, Halton, Sobol’ and Hilbert curves. Reorganise the literature around the dissertation’s decisions:

- What error improvement can randomised SQMC guarantee, under what regularity and dimensional conditions?
- Why does Hilbert ordering create the relevant computational bottleneck?
- Which accelerator operations are local/parallel and which remain global?
- What existing dimension-reduction or Rao--Blackwellised SQMC results apply to sparse observations, and where does the proposed projected sort depart from them?
- What is sacrificed by factorial projection in online skill models, and which baselines can isolate that sacrifice?

For each central source, state its result, assumptions, relationship to the implemented choice and remaining gap. In particular, separate static RQMC error theory from sequential SQMC theory; distinguish theoretical complexity from measured accelerator runtime; and distinguish the fSSM model factorisation from the repeated approximate posterior projection. This would turn the chapter from competent notes into a research argument.

## 2.4 The experimental programme needs controlled ablations

If further computation is possible, the highest-value new experiments are not simply longer versions of the same optimisation. They are comparisons that identify causes.

### Optional extension A: only if a statistical SQMC-versus-SMC claim is desired

If the author later wants to make an empirical claim about statistical SQMC superiority, add SMC-CPU and SMC-GPU to the linear-Gaussian benchmark. Compare SMC and SQMC at matched particle count and matched wall-clock budget using analytic Kalman truth. Repeat across observation data sets and randomisations. Report RMSE, variance across runs, effective sample size, runtime and time to a pre-specified error. This is not required for the present CPU-versus-GPU SQMC question.

### Optional extension B: explain which football component causes the difference

The existing EKF-versus-RB-SQMC comparison is sufficient for an end-to-end benchmark against `cuthberto-carlos`. If the dissertation additionally wants to identify why one pipeline performs better, use a two-by-two design where possible:

| Dependence representation | Gaussian/moment inference | Particle inference |
| --- | --- | --- |
| Factorial/independent | Existing factorial EKF | Factorial SMC or SQMC |
| Correlated | Correlated EKF/assumed-density approximation | RB-SMC and RB-SQMC |

RB-SMC versus RB-SQMC at fixed fitted parameters would test whether QMC adds value after Rao--Blackwellisation without mixing that question with parameter optimisation. Comparing projected and full-state Hilbert ordering on a smaller subset of teams would test the central heuristic. These ablations would strengthen mechanistic interpretation but are optional if the research question remains explicitly end to end.

### Priority C: test numerical and predictive stability

At fixed fitted parameters, vary particles, scrambles and seeds. Then repeat complete fits for a smaller number of seeds. Use rolling-origin forecasts over multiple historical tournaments rather than one World Cup. Report distributions of metric differences, not only one table. This separates Monte Carlo variability, optimisation variability and variation across football periods.

### Priority D: interrogate the model

Perform posterior predictive checks for goal totals, draw frequency, tail scores and calibration by favourite strength. Compare diagonal, confederation-block, shrinkage and low-rank structures for `\Gamma_0`. Assess sensitivity to friendlies, historical start date, score truncation, mean-reversion rate and inclusion of non-tournament opponents. These checks are more informative about the implausible RB-SQMC rankings than simply increasing GPU time.

## 2.5 Uncertainty must be separated into distinct layers

The dissertation should explicitly distinguish:

- football outcome uncertainty represented by the predictive score distribution;
- latent-state and parameter uncertainty under the model;
- finite-particle Monte Carlo/RQMC error;
- variability across RQMC scrambles;
- stochastic optimisation variability;
- numerical error from precision, quantisation and approximate quadrature;
- sampling uncertainty from evaluating only one 104-match tournament;
- structural uncertainty from model misspecification and data-selection rules.

At present, “stochastic” is sometimes used as if particle-path roughness demonstrates real process uncertainty, and bootstrap intervals conditional on one synthetic data set are easy to read as general uncertainty. Each result should say which layer its interval or repetition captures and which it does not. This change would materially improve the statistical maturity of the write-up even before new experiments are run.

## 2.6 Limitations should constrain conclusions, not sit after them

The current limitations are mostly listed after a strong conclusion has already been stated. Move qualifications next to the claims they affect. For example:

- Immediately after the GPU speed-up, state that compilation, transfer, other hardware and other implementations are outside scope.
- Immediately after the likelihood table, state that the two approximations are not numerically commensurate and that test-epoch selection favours noisy extrema.
- Immediately after the prediction table, state the single-run and single-tournament uncertainty.
- Immediately after the correlation plot, state the dense-covariance identification and post-hoc interpretation risks.
- Immediately after the four-dimensional RB-SQMC construction, state that the projected sort is a heuristic if no proof is available.

The dedicated limitations subsection should then synthesise their combined effect. Its job is to say which conclusions survive. The backend acceleration conclusion probably survives in a narrow, hardware-conditional form. The claim that RB-SQMC is worse for football remains descriptive for this run, not general. The claim that the EKF assumptions are valid does not survive the current design.

## 2.7 Future discussion should be driven by questions the results exposed

The most useful future-work section would not be a list of algorithms. It would pursue the following questions:

1. **Does projected ordering preserve the benefit of SQMC?** Compare four-dimensional projected ordering, full component-mean ordering, random ordering and SMC on tractable smaller systems. This directly tests the methodological assumption.
2. **Where is the SQMC benefit concentrated?** The filtering pass uses four propagation coordinates, whereas the proposed smoother uses `1+2K` coordinates. Evaluate filtering and smoothing separately. It may be preferable to use SQMC for filtering and an ordinary Rao--Blackwellised smoother thereafter.
3. **Is the poor football result numerical or structural?** Freeze shared parameters and increase particles/scrambles; then regularise `\Gamma_0`; then add model covariates. This staged design separates finite-particle error, overfitting and model misspecification.
4. **Can accelerator speed translate into statistical efficiency?** Compare time-to-pre-specified RMSE for SMC and SQMC, including compilation and transfer in a cold-start analysis and excluding them in a repeated-filter analysis.
5. **When is retaining cross-team covariance predictively valuable?** Simulate data with known cross-team dependence of increasing strength, then analyse real rolling-origin football periods. This identifies regimes where the factorial approximation fails.
6. **Can parameter learning be made stable without full smoothing?** Compare the path-space score, forward-only smoothing/PaRIS, fixed-lag or stitching methods on a small model with known parameters. Judge bias, variance, memory and time rather than assuming one is preferable.

These questions follow directly from the negative and ambiguous results and would make the discussion intellectually stronger than recommending “more compute”.

## 2.8 Suggested revised structure

The existing chapter order can remain, but each chapter needs a clearer function.

1. **Introduction:** problem, gap, three research questions, scoped contributions and chapter map.
2. **SQMC theory and accelerator implementation:** only the background needed to justify the algorithm; implementation mapping; reproducible benchmark design; CPU/GPU and, if added, SMC/SQMC results; chapter-level limitations and consequence for the sparse model.
3. **Correlated sparse football model:** factorial baseline and its approximation; correlated model and identifiability; RB-SMC; projected RB-SQMC with theoretical boundary; parameter estimation; auditable data design; controlled comparison; quantitative and qualitative results; limitations.
4. **Discussion and conclusion:** direct answers, synthesis across computation and statistics, contribution, limitations that constrain inference, and prioritised future work.
5. **Appendices:** proofs that are genuinely deferred, compact reproducibility configuration, complete supplementary tables only when referenced, and no source-code duplication or issue screenshots.

Add short opening and closing paragraphs to Chapters 2 and 3. Each opening should state the chapter’s question; each closing should state what was established, what remains unresolved and why the next chapter follows.

## 2.9 Likely assessment impact

In its current form, the manuscript demonstrates substantial technical work but would lose marks for logical alignment, experimental design, notation, grammar and interpretation. The most serious assessment risks are the nearly absent conclusion, reuse of test data, unsupported complexity/rate statements, failure to isolate SQMC from hardware or model differences, incomplete data lineage, empty code appendix and overstatement of what the negative football result establishes.

The route to a strong dissertation is therefore:

1. correct mathematical and build defects;
2. narrow every claim to the evidence currently available;
3. rewrite the introduction and conclusion around explicit questions;
4. activate the full limitations analysis;
5. make data and computation reproducible;
6. add RB-SMC/SMC ablations and repeated uncertainty estimates if compute permits;
7. complete a final British-English, notation, caption and PDF-layout pass.

The project’s originality is most credible as a combination of implementation, correlated sparse-state formulation and candid empirical evaluation. The write-up should lean into that contribution. It does not need to claim that SQMC universally wins, that four-dimensional sorting is theoretically exact or that the EKF assumptions have been validated. A precise account of where the method accelerated computation, where it failed predictively and which unresolved mechanisms explain that gap would be a much stronger statistical dissertation.
