# Tone and Style Guide for a UK Statistics Dissertation

This document defines the expected voice, argumentation and presentation for
the dissertation. It should be used when drafting, revising and reviewing every
chapter. It interprets the criteria in ASSESSMENT.md and the formal guidance in
REQUIREMENTS.md as practical writing rules.

The target is a distinction-level MSc dissertation in statistics: technically
precise, logically structured, appropriately critical and close to the style of
a paper in the *Journal of the Royal Statistical Society*. The prose should
demonstrate mastery without exaggeration and originality without novelty claims
that the evidence cannot support.

## 1. Core voice

Write in a formal, measured and analytical voice.

- Prefer precise claims to emphatic claims.
- Explain the statistical reason for each methodological choice.
- Distinguish established results, implementation decisions, empirical
  findings and the author's interpretation.
- State assumptions before relying on them.
- State limitations where they affect the validity or scope of a conclusion.
- Use technical language when it increases precision, but define specialised
  terms at first use.
- Write for a statistically literate reader who may not already know SQMC,
  accelerator programming or football modelling.

The writing should sound confident because it is supported, not because it is
absolute. Phrases such as "the results indicate", "under these assumptions" and
"within the range examined" are preferable to unsupported declarations such as
"this clearly proves" or "the method is always superior".

### Preferred

> At matched particle counts, RB-SQMC produced a lower empirical RMSE than
> RB-SMC in this experiment. The difference was largest at moderate particle
> counts, although the additional sorting cost reduced the wall-clock advantage
> on the CPU.

### Avoid

> RB-SQMC is obviously much better and completely solves the accuracy problem.

## 2. British academic English

Use British English consistently.

- Use forms such as "modelling", "normalisation", "optimisation",
  "randomised", "behaviour", "centre", "analyse" and "factorised".
- Use "programme" for a programme of study, but "program" for computer code.
- Prefer "while" for directness; use "whilst" only when it reads naturally.
- Use the serial comma only where it removes ambiguity.
- Use single quotation marks for quoted terms, with double quotation marks only
  for a quotation within a quotation.
- Treat collective nouns consistently. In technical prose, singular agreement
  is usually clearest: "the data set contains" and "the committee recommends".
- Use an en dash for a genuine range or relationship in typeset prose, such as
  "pages 10–14" or "attack–defence covariance". Use a hyphen in compound
  modifiers, such as "low-discrepancy points".

Do not alternate between British and American spellings. Code identifiers and
titles in the bibliography must retain their original spelling.

## 3. Authorial stance

Prefer active constructions when responsibility matters:

> We compare SMC and SQMC at matched particle counts.

> Section 4 introduces the observation model.

> The experiment uses a chronological train-test split.

The first-person plural is acceptable for choices made in the dissertation,
even for a single-author work, but it should be used sparingly. Avoid
anthropomorphising equations or results: a table may "show" a numerical pattern,
but it does not "believe", "want" or "prove".

Passive voice is useful when the procedure matters more than the actor:

> Hyperparameters were selected using the validation period.

Do not use passive voice to conceal an important decision:

> Three failed runs were excluded because they produced non-finite likelihoods.

is more informative than:

> Some runs were removed.

## 4. Build an argument, not a catalogue

Each chapter and section should have a clear argumentative function. A strong
section normally follows this progression:

1. identify the statistical or computational problem;
2. explain why it matters to the dissertation aim;
3. introduce the relevant method or evidence;
4. state the assumptions under which it applies;
5. derive or describe the result;
6. interpret the result; and
7. connect it to the next stage of the argument.

Open sections with their purpose rather than a generic history. Close them with
the consequence for the dissertation rather than a summary of headings.

### Preferred transition

> The full football state grows with the number of teams, whereas each
> likelihood contribution depends on only four latent coordinates. This sparse
> observation structure motivates the Rao-Blackwellisation developed next.

### Avoid

> The next section discusses Rao-Blackwellisation.

The introduction should formulate the problem, motivate it statistically,
identify the gap or unresolved question, state the aims and contributions, and
give a concise map of the dissertation. The conclusion should answer those
aims, synthesise the theoretical and empirical findings, acknowledge material
limitations and identify justified directions for further work. It should not
introduce a new method or merely repeat the abstract.

## 5. Demonstrate statistical reasoning

The prose should expose the reasoning behind the analysis. For every important
method, answer the following questions in the text:

- What inferential or predictive target is being estimated?
- Why is this method appropriate for that target and data structure?
- Which assumptions are required?
- How are uncertainty and Monte Carlo error represented?
- Which competing method forms the relevant baseline?
- Which diagnostic could reveal failure?
- What does the result establish, and what does it not establish?

Do not present a method as appropriate merely because it is sophisticated.
Relate its structure to the problem. For example, SQMC is relevant not simply
because it uses low-discrepancy points, but because Rao-Blackwellisation reduces
the effective propagation dimension of the football filter.

Different kinds of uncertainty must not be conflated. Distinguish, where
relevant:

- observation or aleatoric uncertainty;
- posterior or parameter uncertainty;
- Monte Carlo error;
- numerical error;
- variability across randomised QMC scrambles; and
- uncertainty caused by data limitations or model misspecification.

Use "bias", "variance", "consistency", "efficiency", "calibration" and
"significance" only in their statistical senses, unless another meaning is
made explicit.

## 6. Claims and evidential strength

Match each verb to the evidence.

| Evidence | Appropriate wording | Wording to avoid |
| --- | --- | --- |
| A formal derivation under stated conditions | "establishes", "implies", "is sufficient for" | "suggests" when a result has actually been proved |
| A finite simulation | "indicates", "is consistent with", "was observed" | "proves", "guarantees", "always" |
| An empirical association | "is associated with", "coincides with" | "causes", unless the design supports causality |
| A benchmark on named hardware | "was faster on the reported configuration" | "is faster" without qualification |
| A literature result | "Gerber and Chopin (2015) show that..." | Presenting another author's result as original |
| An interpretation | "One explanation is...", "This may reflect..." | Stating speculation as fact |

Quantify comparisons wherever possible. Replace "substantially faster" with the
speed-up, interval and benchmark conditions. Replace "more accurate" with the
metric, particle count or wall-clock budget, and uncertainty across repetitions.

Negative and null results should be reported directly. They are evidence about
the limits of the method, not material to be hidden.

## 7. Literature: synthesise and evaluate

The assessment criteria reward integration and evaluation of literature, not a
sequence of paper summaries. Organise the review around questions, assumptions
or methodological contrasts.

### Descriptive

> Author A applies SMC. Author B proposes SQMC. Author C uses a football model.

### Evaluative and integrated

> Standard SMC accommodates the non-Gaussian score likelihood but retains the
> usual Monte Carlo error rate. SQMC can improve integration accuracy, although
> its advantage deteriorates as the effective dimension grows. The sparse
> football likelihood creates an opportunity to address this limitation by
> sampling only the playing-team block.

For each central source, make clear:

- which result or idea is being used;
- the assumptions and scope of that result;
- how it relates to competing work;
- what remains unresolved; and
- how the dissertation adopts, tests or extends it.

Cite the source next to the supported claim. Cite original methodological
sources where feasible rather than relying only on textbooks or secondary
accounts. Every cited work must appear in the reference list, and the reference
list should contain only works cited in the dissertation.

Avoid excessive quotation. Statistical writing should normally paraphrase,
synthesise and attribute.

## 8. Equations and notation

Notation should be well chosen, stable and economical.

- Define every symbol before or immediately after first use.
- Use one symbol for one concept and do not change notation between chapters.
- Distinguish random variables, their realisations, parameters and estimators.
- State dimensions when they are not obvious.
- State the domain of important mappings and distributions.
- Number equations that are referred to later.
- Refer to numbered equations as "equation (3.2)", not "the equation above".
- Punctuate displayed equations as part of the surrounding sentence.
- Introduce an equation with its purpose; do not leave it to speak for itself.
- Follow a derivation with a statistical interpretation.
- Index observations from \(1\): write \(y_{1:t}=(y_1,\ldots,y_t)\), so that
  filtering is \(p(x_t\mid y_{1:t})\) and smoothing is \(p(x_{0:T}\mid y_{1:T})\).
  Do not write \(y_{0:t}\) unless an observation at time \(0\) is actually used.
- Use a single symbol for the initial density throughout: write \(m_0(x_0)\)
  in the joint density and in the weight function \(G_0\), not \(f_0\) in one
  place and \(\mu_0\) in another.
- Use \(m_0\) for the initial distribution, \(m_t\) for the transition density,
  and \(g_t\) for the observation density. Define the proposal densities
  \(q_0\) and \(q_t\) alongside them, since the weight functions
  \(G_0=m_0/q_0\) and \(G_t=m_t g_t/q_t\) depend on both.
- Write index sets in the standard form \(t\in\{0,\ldots,T\}\), not
  \(t\in 0:T\) or \(0:T=\{0,\ldots T\}\).
- Write conditional densities as \(p_t(x_t\mid x_{t-1})\); do not repeat the
  conditioning symbol, as in \(p_t(x_t\mid d\mid x_t)\).

### Preferred

> Let \(X_t\in\mathbb R^d\) denote the latent state. Conditional on
> \(X_{t-1}=x_{t-1}\), its transition density is
> \[
> f_t(x_t\mid x_{t-1}).
> \]
> This Markov assumption makes the filtering distribution recursively
> computable.

Avoid strings of equations without connective prose. Routine algebra may be
moved to an appendix, but the main text must retain the steps needed to
understand the argument.

Theorems and propositions should state their conditions precisely. Distinguish
a theorem cited from the literature from a proposition proved in the
dissertation. Do not label an empirical regularity as a theorem.

## 9. Algorithms and implementation

Algorithms should connect mathematical notation to reproducible computation.
For each principal algorithm:

- state its inputs and outputs;
- give the steps in execution order;
- identify stochastic inputs and their dimensions;
- state the computational cost of material operations;
- explain any numerical-stability measures;
- identify approximations or implementation-specific departures from theory;
  and
- refer to the repository file implementing it.

Use short code extracts only where syntax clarifies an implementation choice
that prose or pseudocode cannot. The dissertation should explain the
statistical design; it should not reproduce entire source files. In accordance
with REQUIREMENTS.md, full code belongs in the separate electronic submission.

Implementation claims must name the relevant environment. For GPU results,
report the hardware, software versions, precision, compilation treatment,
warm-up policy, repetition count and synchronisation procedure. Separate
algorithmic complexity from measured runtime.

### Preferred

> The ordered CDF is computed by a device prefix scan. Hilbert sorting therefore
> resolves the multidimensional ordering problem but does not remove the
> cumulative-weight operation.

### Avoid

> The GPU parallelises everything and makes the algorithm \(O(1)\).

## 10. Data and experimental design

The applied chapters should make the data lineage and design auditable.
Describe:

- the source, coverage and unit of observation;
- inclusion and exclusion criteria;
- missingness, corrections, duplicates and team-identity matching;
- temporal ordering and the prevention of information leakage;
- training, validation and test periods;
- baselines and ablations;
- the choice of evaluation metrics; and
- sensitivity analyses and known data limitations.

Use a chronological split for a sequential forecasting problem unless a
different design is carefully justified. Do not tune on the final test set.
Explain why each baseline isolates the effect under study. For example, an
SMC-GPU comparison is required to distinguish the effect of SQMC from the
effect of accelerator hardware.

The prose should show ownership of design decisions:

> Matches with unresolved team identifiers were excluded before the temporal
> split; Appendix B lists the mapping rules and affected records.

## 11. Results and interpretation

Separate reporting from interpretation, while keeping them close enough for
the argument to remain clear.

1. State the comparison and metric.
2. Report the numerical result with appropriate uncertainty.
3. Describe the visible pattern.
4. Interpret it in relation to the research question.
5. Discuss plausible alternatives, limitations or sensitivity.

Do not narrate every cell of a table. Direct the reader to the result that
answers the research question.

### Preferred

> At \(N=2^{12}\), SQMC reduced median RMSE from 0.084 to 0.061 across 32
> independent scrambles. Its GPU runtime was nevertheless 18% higher than that
> of SMC-GPU because sorting dominated at this particle count. Thus SQMC improved
> statistical efficiency but not time-to-accuracy in this configuration.

### Avoid

> Table 4 has many results. SQMC has 0.061 and SMC has 0.084, so SQMC is best.

Use "statistically significant" only when a defined inferential procedure
supports it. Practical importance should be discussed separately from
statistical significance.

For predictive results, report proper scoring rules and calibration as well as
classification accuracy where appropriate. A football forecast should not be
judged solely by whether the most probable outcome occurred.

## 12. Tables and figures

Follow the presentation style of a recent *Journal of the Royal Statistical
Society* article, as required by REQUIREMENTS.md.

- Number every table and figure.
- Refer to each one by number in the main text.
- Give each a self-contained caption that explains its content but does not
  perform the substantive interpretation.
- Define abbreviations, symbols, units and uncertainty summaries in the caption
  or notes.
- Label axes and include units.
- Use consistent scales and visual encodings across comparable panels.
- Avoid decorative graphics and unnecessary three-dimensional effects.
- Use legible type and line widths at the final A4 size.
- State the number of repetitions and whether bars represent standard errors,
  standard deviations, quantiles or confidence intervals.

The text should interpret why a pattern matters. The caption should enable the
reader to identify what is displayed.

## 13. Paragraph and sentence style

Each paragraph should advance one main claim. Place that claim near the
beginning, support it, then connect it to the wider argument.

- Prefer sentences of moderate length.
- Vary sentence structure without sacrificing clarity.
- Place qualifications next to the claim they qualify.
- Replace vague pronouns such as "this" with the specific result or mechanism.
- Avoid stacked nouns when a short clause is clearer.
- Remove throat-clearing phrases such as "It is interesting to note that".
- Avoid rhetorical questions, conversational fillers and exclamation marks.
- Avoid "obviously", "clearly", "simply" and "trivially" unless the statement
  truly is immediate for the intended reader.
- Avoid repeated signposting such as "as mentioned previously".
- Use abbreviations only when they recur often enough to aid reading.

Prefer:

> The resampling step introduces a global synchronisation point.

to:

> It should be noted that there is a global synchronisation point which is
> introduced by the resampling step.

## 14. Scope, originality and limitations

Originality may lie in the formulation, implementation, experimental design,
application, synthesis or interpretation. State the contribution precisely:

> This dissertation implements Hilbert-ordered SQMC in JAX and evaluates
> whether Rao-Blackwellisation preserves its advantage in a correlated football
> state-space model.

Do not imply that every component is novel. Identify inherited methods and
original contributions separately.

Limitations should be specific and consequential. Explain:

- which conclusion is affected;
- why the limitation matters;
- whether its likely direction is known; and
- what experiment or data would address it.

### Weak

> More work is needed.

### Strong

> The projected four-dimensional Hilbert ordering is a dimension-reduction
> heuristic rather than an exact sufficient representation of the full
> transition. Comparing it with full-state ordering at smaller team counts
> would show whether the projection removes information material to
> resampling.

## 15. Required document-level conventions

The following requirements constrain the writing and presentation:

- The dissertation should be a single, page-numbered PDF on A4 paper.
- The total length is normally 40–80 pages.
- The first page must give the title, author, student number and supervisor.
- The abstract must not exceed 200 words.
- Any preface and acknowledgements belong on the second page.
- References must be collected at the back.
- Substantial code must be submitted separately and be sufficient to reproduce
  the results.
- Supplementary files should include a concise README explaining how to
  reproduce the analysis.

These are compliance requirements, not substitutes for a coherent argument.

## 16. Chapter-specific tone

### Abstract

Use a compact problem-method-results-conclusion structure. State what was done
and the principal quantitative findings. Do not include a literature review,
undefined abbreviations, equations or claims absent from the dissertation.
Keep the final abstract within 200 words.

### Introduction

Move from the substantive problem to the statistical challenge, relevant gap,
aims, contributions and document structure. Do not overstate societal impact or
promise results that later chapters do not deliver.

### Background and literature

Be explanatory but selective. Derive the material needed for later chapters
and critically compare the assumptions of alternative methods. The chapter
should read as a foundation for the research question, not as general lecture
notes detached from it.

### Methodology

Be exact. Define the model, assumptions, targets, algorithms and complexity.
Explain why the method is appropriate and where its theoretical guarantees
end.

### Implementation

Be reproducible and hardware-specific. Connect code operations to mathematical
steps, report numerical choices and avoid promotional language about software
or GPUs.

### Results

Be quantitative, restrained and comparative. Report uncertainty, baselines,
failures and sensitivity analyses. Distinguish a statistical improvement from
a computational improvement.

### Discussion and conclusion

Be synthetic and candid. Answer the research questions, relate results to the
literature, state the contribution at the correct level, and identify
limitations that materially affect interpretation.

## 17. Revision checklist

Before considering a section complete, check:

- Does its first paragraph state the problem or purpose?
- Does every claim have a derivation, citation or empirical result?
- Are inherited results clearly distinguished from original work?
- Are notation and terminology consistent with earlier chapters?
- Are assumptions stated before use?
- Is each empirical comparison fair and reproducible?
- Are uncertainty and limitations reported?
- Does the section interpret evidence rather than merely describe it?
- Does each table, figure and equation have a role in the argument?
- Is every table and figure numbered, captioned and cited in the text?
- Are UK spelling, grammar and typography consistent?
- Can any sentence be shortened without losing meaning?
- Does the final paragraph explain the consequence for the next section or the
  dissertation aim?

At document level, also check:

- Is the abstract no more than 200 words?
- Are the introduction and conclusion aligned with the stated aims?
- Does the literature review evaluate and integrate the relevant literature?
- Does the applied analysis document data quality and design choices?
- Are code and reproduction instructions ready for separate submission?
- Does the final PDF satisfy the A4, pagination and front-matter requirements?

## 18. Governing principle

The dissertation should make it easy for a critical statistical reader to
reconstruct the argument: what was assumed, what was derived, what was
implemented, what was observed and how strongly the evidence supports the
conclusion. Precision, transparency and coherent reasoning take priority over
ornament or apparent certainty.
