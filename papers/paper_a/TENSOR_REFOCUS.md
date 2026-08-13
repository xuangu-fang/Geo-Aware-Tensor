# Paper A tensor refocus audit

Date: 2026-08-12

## Why the current headline method is an operator GP, not yet a Bayesian tensor factorization paper

`ExactFeatureBayes` constructs a feature matrix from all retained product
eigenfunctions and fits

\[
y=\Phi_{\Omega}w+\epsilon,\qquad
w\sim\mathcal N(0,\operatorname{diag}\tau).
\]

This is a useful finite Hilbert-space GP.  If `w` is reshaped into a spectral
coefficient tensor, it can be called a full Tucker core with fixed factor
matrices, but that description hides the important missing structure:

1. **No explicit low tensor rank.** The spectral core is dense; every retained
   product coefficient is independent a priori.  Its parameter count grows as
   \(\prod_m K_m\), not \(R\sum_m K_m\) for CP or
   \(\sum_m K_mR_m+\prod_mR_m\) for Tucker.
2. **No learned mode factors.** Operator eigenvectors are fixed design features,
   not priors over CP/Tucker factor functions.  The posterior cannot say which
   latent multilinear component explains a field pattern.
3. **No tensor-rank ARD.** Spectral powers shrink frequency, but there is no
   shared precision \(\alpha_r\) tying component `r` across every tensor mode,
   so irrelevant CP components cannot be pruned.
4. **No low-rank core posterior.** Exact Gaussian uncertainty lives on an
   unstructured vector of product weights.  It does not expose component or
   multilinear-rank uncertainty.
5. **The main control is another kernel.** Correct/wrong bases test geometry,
   but not whether tensor factorization itself contributes beyond an operator GP.

The existing `BayesianSpectralCP` is closer to the intended paper because it has
mode-wise random factors.  Its first POC failed for identifiable reasons:
independent mean-field distributions over multiplicative factors amplify
Monte-Carlo noise; reciprocal CP scaling is not removed in the variational
posterior; learned smoothness can collapse; and it lacks shared rank ARD.

## Refocused formulation

The minimal proposed model is a geometry-aware empirical-Bayesian CP:

\[
\mathcal X(i_1,\ldots,i_M)
=\sum_{r=1}^{R}a_r\prod_{m=1}^{M}f_{mr}(i_m)+\epsilon,
\]

\[
f_{mr}=\Phi_m u_{mr},\qquad
u_{mr}\sim\mathcal N\!\left(0,
[\alpha_r(I+\Lambda_m)^{p_m}]^{-1}\right),
\]

\[
a_r\mid\alpha_r\sim\mathcal N(0,\alpha_r^{-1}),\qquad
\alpha_r\sim\operatorname{Gamma}(a_0,b_0).
\]

Each factor column is normalized on its complete mode grid, leaving `a_r` as
the identifiable component amplitude.  Mode factors are fitted under their
operator priors.  Conditional on them, amplitudes receive an exact Gaussian
posterior and evidence/ARD updates.  A later round adds a diagonal
Gauss–Newton/Laplace correction for mode-factor uncertainty.  This keeps the
model recognizable as Bayesian CP rather than inventing a new deep architecture.

## Required ablations and what each proves

| Method | Tensor low rank | Correct geometry | Bayesian/ARD | Purpose |
|---|---:|---:|---:|---|
| Geo Bayesian CP | yes | yes | yes | proposed |
| Wrong-geometry Bayesian CP | yes | no | yes | geometry contribution |
| Discrete Bayesian CP | yes | no side information | yes | operator side-information contribution |
| Geo CP without ARD | yes | yes | conditional posterior only | ARD contribution |
| Dense operator GP | no | yes | exact | tensor-rank contribution |
| Neural CP / SIREN | yes/no | no | no | flexible point baselines |

The main table must allow both deductions:

- Geo Bayesian CP versus dense operator GP isolates low-rank tensor structure.
- Geo Bayesian CP versus wrong/discrete Bayesian CP isolates correct geometry.

## Data tensor requirements

Experiments must preserve explicit modes, not flatten all spatial points into a
single graph coordinate.  Primary controlled tensors use three semantic modes:
bounded time/range and periodic angle, or time×two periodic spatial modes.  At
least one task must include non-random missing fibers/sectors, where tensor
sharing and boundary/topology are identifiable inductive biases.  Public Active
Matter remains a 3-way `time×x×y` tensor.  The two-room graph×time experiment is
retained as an operator-GP appendix result, not the core tensor claim.

## Four-round success criteria

1. Round 1 must show an explicit spectral CP runs and identify whether simple
   conditional Bayesian amplitudes are stable below 1%.
2. Round 2 must demonstrate automatic component pruning or honestly record its
   failure, and compare uncertainty before/after factor correction.
3. Round 3 must show, on identical masks, that correct geometry and low-rank
   tensor structure are separately useful in at least one structured sparse
   regime; no claim may be inferred from a single favorable seed.
4. Round 4 must repeat the selected configuration on fresh seeds and a public
   tensor, aggregate paired seed statistics, and preserve negative regimes.

The intended paper claim is modest: **traditional Bayesian CP becomes more
sample-efficient and better regularized when its mode factors live in
operator-defined function spaces, especially under structured subpercent
missingness.** Operator geometry is a prior on tensor factors, not a replacement
for tensor factorization.

## Literature boundary

- Classical fully Bayesian CP uses hierarchical shared precisions for automatic
  CP-rank determination: [Zhao, Zhang and Cichocki, 2015](https://arxiv.org/abs/1401.6497).
- CP completion with side-information subspaces shows why low-dimensional fiber
  spans reduce sample complexity: [Budzinskiy and Zamarashkin, 2022](https://arxiv.org/abs/2206.12486).
- Sparse Bayesian Tucker models learn multilinear ranks by hierarchical group
  shrinkage: [Zhao, Zhang and Cichocki, 2015](https://arxiv.org/abs/1505.02343).
- The previous implementation remains a Hilbert-space/operator-GP baseline:
  [Solin and Särkkä, 2020](https://link.springer.com/article/10.1007/s11222-019-09886-w).

Our differentiator must therefore be precise: the side-information subspaces are
not arbitrary covariates but operator eigenfunctions that encode topology,
boundary conditions and frequency-dependent Bayesian shrinkage, with calibrated
prediction evaluated in the extreme-sparse physical-field setting.
