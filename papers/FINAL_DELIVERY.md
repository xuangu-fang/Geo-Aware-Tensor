# Final tensor-refocused two-paper delivery

> **2026-08-13 absolute-effect correction:** Paper B's The Well early-40 result
> is rejected, not a public confirmation. Paired CP has NRMSE 0.99175, only
> about 1.64% approximate explained variance and 1.96% MSE skill over
> time-scaled persistence. Small paired p-values among near-null predictors are
> not evidence of useful reconstruction. Paper B currently has controlled
> mechanism evidence only.

## Post-feedback three-round extension

Paper A has been extended from a CP-only model to operator-spectral Bayesian
Tucker with an explicit small core and a conditional exact Gaussian core
posterior. On the dense-core Tucker task, ten fresh seeds give 0.125±0.015
NRMSE at 2% observations, versus 0.367±0.055 geometry CP, 0.612±0.028 flat
operator GP, 1.712±0.201 wrong-geometry Tucker, and 1.976±0.131 discrete
Tucker. At 1%, an observation-matched `(3,4,4)` core gives 0.676±0.072 versus
0.752±0.030 flat GP.

Paper B received two explicit phase-envelope extensions. Learned rank-2
envelopes improve moving-envelope reconstruction over paired CP
(0.644±0.118 versus 0.897±0.033) but do not beat IP-NF (0.624±0.027);
fixed-RBF envelope Tucker is also negative. Following the predeclared gate,
neither extension is promoted to the main Paper-B method. Both papers now
share mismatch-by-observation phase diagrams and the same baseline/seed
discipline. See [the Chinese three-round ledger](zh/三轮持续迭代记录.md) and
[machine-readable results](longterm_results/summary.json).

The two papers now share one non-negotiable idea: geometry enters the factors of
a traditional CP/Tucker model, and no flat joint regressor is allowed to stand
in for the tensor decomposition. The stories differ in inference and physical
mechanism, not in whether the learned object is a tensor.

## Paper A — operator-factor Bayesian CP

**Question.** Can topology and boundary conditions regularize Bayesian CP mode
factors when only 0.5--2% of a physical tensor is observed?

**Method.** Each CP factor is expanded in the eigenfunctions of its mode
operator and receives eigenvalue-shaped Gaussian shrinkage. Factor means are
MAP estimates; component amplitudes have a conditional Gaussian posterior; a
diagonal factor Laplace correction and strict observation-only split calibration
provide predictive dispersion. An observed-data operator posterior followed by
CP-ALS is used only as an optimization initialization.

**Frozen evidence.** On ten fresh seeds at 2% observations, geometry-aware
Bayesian CP obtains approximately 0.198 NRMSE versus 0.406 for a dense operator
GP, 2.459 for identical CP with permuted geometry, and 1.406 for discrete
Bayesian CP. All ten paired seeds favor the proposed model. At 0.5%, point error
is neutral relative to the flat GP, while calibrated NLL, coverage, and
uncertainty ranking are better. Periodic-sector missingness, Tucker-format
mismatch, and public Active Matter point recovery are mixed or negative.
Automatic rank determination failed and is not claimed.

Start at [Paper A draft](paper_a/DRAFT_TUCKER.md), [tensor audit](paper_a/TENSOR_REFOCUS.md),
[T1--T4 log](paper_a/TENSOR_ITERATIONS.md), [tables](paper_a/tensor_results/TABLES.md),
and [artifact manifest](paper_a/tensor_results/MANIFEST.json).

## Paper B — geometry-conditioned phase CP

**Question.** Can explicit neural tensor factors transfer to unseen domain
geometries and a new mesh resolution from 1% observations?

**Method.** Geometry, time, and geometry-conditioned spatial factors remain
separate. Shortest-path phase enters the spatial factor; paired sine/cosine
space-time carriers use the angle-addition identity inside an explicit CP
contraction. No joint-coordinate residual bypasses the tensor model.

**Frozen evidence.** On ten fresh 24→32 cross-resolution seeds for an
independently generated eikonal harmonic field with a 6% off-model moving
residual, paired geometry CP reaches 0.0952±0.0144 NRMSE. Ordinary geometry CP
reaches 0.1113±0.0213 (14.5% reduction, exact paired p=0.0391), monolithic IP-NF
0.1825±0.0173 (47.9%, p=0.00195), wrong-geometry paired CP 1.5598±0.1827, and
raw F-INR-style Tucker 1.8167±0.2535. On a harder moving-envelope field, the
monolithic IP-NF beats tensor models; that limitation is retained prominently.

Start at [Paper B draft](paper_b/DRAFT.md), [tensor audit](paper_b/TENSOR_REFOCUS.md),
[T1--T5 log](paper_b/ITERATIONS.md), [structured statistics](paper_b/results/tensor_t5_confirm_summary.json),
[result figure](paper_b/results/tensor_t5_confirm_crossres.png), and
[artifact manifest](paper_b/MANIFEST.md).

## Shared scientific boundary

- Paper A contributes operator-resolved Bayesian factor priors and calibrated
  factor/core uncertainty; Paper B contributes phase-aligned neural factors and
  cross-geometry/resolution transfer.
- Correct-versus-wrong geometry and tensor-versus-flat controls are mandatory in
  both papers.
- The main positive regimes are intentionally moderate and physically
  multilinear. The harder regimes where dense GP or monolithic INR wins remain
  part of the evidence and delimit the claims.
- Existing Bayesian CP with side information, F-INR, and neural Tucker are
  treated as prior art. The differentiator is operator/geodesic structure
  inside mode factors, not tensor factorization alone.
- Public-data pilots are external stress tests, not substitutes for the causal
  geometry controls that the available public tensors lack.

The shared [tensor-core contract](TENSOR_CORE_REFOCUS.md), [iteration ledger](TENSOR_REFOCUS_PROGRESS.md),
[evaluation protocol](EVALUATION_PROTOCOL.md), and [literature/data audit](../reports/LITERATURE_AND_DATA.md)
record the decisions and provenance.

## Validation

```bash
cd /home/ubuntu/project/Geo-Aware-Tensor
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python -m pytest -q
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python -m compileall -q src experiments
```

The current code audit reports 21 passing tests and successful compilation.
