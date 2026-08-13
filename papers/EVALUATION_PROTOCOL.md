# Shared evaluation and statistical protocol

Frozen on 2026-08-12 before the final headline runs.  The two papers use the
same hygiene rules but answer different questions.

## Data access

- A training procedure may read only the coordinates, domain geometry,
  observation indices, and noisy values at those indices.
- A geometry-derived basis, distance field, graph, or kernel may use the entire
  known domain.  It may not use the unobserved target values.
- Hyperparameters and uncertainty calibration are selected from observed values
  only (marginal likelihood, leave-one-observation-out, or an explicitly logged
  split of the observations).
- Synthetic targets and masks are generated from independent fixed seeds.  A
  matched-basis simulator is labelled a sanity check, never an OOD result.

## Fair comparisons

- Report the exact observation count as well as the nominal ratio.
- Use identical masks and noise draws across models within a seed.
- A wrong-geometry control receives the same feature budget and downstream model
  capacity as the correct-geometry version whenever possible.
- Report training time and trainable parameter count.  Where capacities are not
  matched, add a matched-size baseline or state the mismatch in the table.
- Never select a checkpoint or architecture using values on the final unseen
  geometry or held-out field.  Pilot results used for redesign are kept in the
  iteration log and removed from confirmatory status.

## Uncertainty paper (A)

Primary outcomes are held-out NLL, 50/80/95% coverage and ECE, interval width at
matched coverage, uncertainty--absolute-error Spearman correlation, selective
risk, and downstream error under a fixed acquisition budget.  Point NRMSE is a
secondary outcome.  Coverage is also stratified by room/region and predictive
scale.  Both raw and observation-only calibrated posteriors are shown.

## Neural geometry paper (B)

Primary outcomes are unseen-geometry NRMSE, globally normalized boundary and
shadow RMSE, and absolute high-band coefficient RMSE normalized by the global
field scale.  Relative high-band error is reported only if that band contains at
least 1% of the represented target energy.  Results are separated into same
resolution, cross-resolution, and changed-topology tests.

### Absolute-effect amendment (frozen 2026-08-13)

The Well early-40 exposed a missing gate in the original protocol: all methods
can have NRMSE approximately one while small paired differences are statistically
stable. Such comparisons are not positive evidence. For every new external
Paper-B task, before any pairwise significance is interpreted, the proposed
method must satisfy both:

- macro NRMSE at most 0.8;
- at least 20% MSE skill relative to the strongest trivial constant,
  persistence, or interpolation baseline.

The thresholds were set after rejecting The Well early-40 and apply only to
future external tasks; they are not retroactively used to select among existing
results. Failing the gate classifies the whole task as an ineffective stress
test regardless of parameter efficiency, win count, or p-value.

## Statistics

Headline tables use at least three independent mask/noise/training seeds.  The
unit of pairing is `(seed, task, split, ratio)`.  For every claimed improvement,
report the mean paired relative improvement, a two-sided paired permutation test
(exact sign enumeration when the number of independent seed aggregates is
small), and a 95% bootstrap confidence interval over seeds.  Task-level samples
within a seed are not treated as independent replicas.  Multiple secondary
metrics are descriptive; no isolated secondary p-value is used as a headline.

## Negative and exploratory results

Every redesign records the prior configuration, outcome, diagnosis, and change
in `papers/paper_a/ITERATIONS.md` or `papers/paper_b/ITERATIONS.md`.  Runs used to
choose the next design are exploratory.  At least one fresh seed or fresh task is
reserved for the final confirmation whenever compute permits.
