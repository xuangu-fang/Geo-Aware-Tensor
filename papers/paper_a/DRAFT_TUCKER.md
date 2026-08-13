# Operator-Geometry Bayesian Tucker for Extreme-Sparse Tensor Completion

> **Canonical Paper-A draft, 2026-08-13.** The earlier CP manuscript remains in
> `DRAFT.md` as an evidence ledger. This draft reflects the frozen Tucker method,
> ten-seed confirmations, structured-missing and high-noise stress tests, and
> the negative active-acquisition result.

## Abstract

Bayesian Tucker completion shares information across tensor modes, but ordinary
factor-table priors treat a circle, a bounded interval and a spatial mesh as
interchangeable index sets. We make Tucker decomposition geometry-aware by
representing every mode factor in eigenfunctions of a mode-specific physical
operator and regularizing spectral coefficients according to operator energy.
The multilinear core remains explicit. Mode factors are maximum-a-posteriori
estimates; conditional on them, the small Tucker core has an exact empirical
Bayesian Gaussian posterior. On a controlled order-three tensor with
multilinear rank `(4,5,5)`, 2% noisy observations and ten fresh seeds, the
proposed model obtains `0.125±0.015` held-out NRMSE, compared with
`0.367±0.055` for geometry-aware CP, `0.612±0.028` for a flat operator GP,
`1.712±0.201` for an identical Tucker model with permuted geometry and
`1.976±0.131` for discrete Tucker. With an entirely unobserved central block,
it obtains `0.174±0.074` versus `0.631±0.030` for the flat GP. With observation
noise increased from 10% to 30% of field standard deviation, it obtains
`0.487±0.091` versus `0.661±0.025`. Conditional core integrated-variance
acquisition fails against random sampling, showing that core uncertainty does
not capture nonlinear factor uncertainty. The evidence supports a focused
claim: correct mode operators and an explicit non-diagonal core jointly reduce
sample complexity when the field is approximately multilinear; it does not
support fully Bayesian factors, automatic rank selection, or universal public
PDE performance.

## 1. Motivation

For an order-(M) tensor, classical Tucker writes

\[
Y_{i_1\ldots i_M}\approx
\left\langle\mathcal G,
U_1(i_1)\otimes\cdots\otimes U_M(i_M)\right\rangle.
\]

Its low multilinear rank is useful under sparse observations, but the prior on
each row of (U_m) usually ignores what the index means. At 1--2% observations,
the data cannot reliably relearn periodic seams, boundary conditions or
operator smoothness.

We change the factor space, not the tensor decoder. For mode (m), let

\[
\mathcal A_m\phi_{mk}=\lambda_{mk}\phi_{mk},\qquad
\Phi_m(i,k)=\phi_{mk}(x_i).
\]

We parameterize

\[
U_m=\Phi_mW_m,
\qquad
p(W_{mkr})\propto
\exp\{-\tfrac12(1+\lambda_{mk})^pW_{mkr}^2\}.
\]

Thus geometry determines which mode factors are plausible. Product features
are formed only after each mode has been reduced to a small factor table.

## 2. Contributions

1. **Operator-defined Tucker factors.** Each mode factor lives in a physical
   operator eigenspace with frequency-dependent prior precision.
2. **Explicit geometry-aware Bayesian Tucker.** A small, inspectable core
   relaxes CP's superdiagonal restriction while retaining multilinear sharing.
3. **Conditional core inference and causal controls.** Given learned factors,
   core inference is exact Gaussian empirical Bayes. Correct, permuted,
   discrete, CP-restricted and flat-operator controls isolate geometry and
   tensor structure.
4. **Extreme-sparsity stress evidence.** Frozen ten-seed experiments cover
   random 2% observations, a connected missing spatial block and 30% noise.

We deliberately do not claim fully Bayesian factors, automatic rank discovery,
or successful active sensing.

## 3. Method

### 3.1 Explicit operator Tucker

For order three,

\[
\widehat Y_{ijk}=\sum_{a=1}^{R_1}\sum_{b=1}^{R_2}
\sum_{c=1}^{R_3}G_{abc}
(\Phi_1W_1)_{ia}(\Phi_2W_2)_{jb}(\Phi_3W_3)_{kc}.
\]

Factors and the core are first optimized by MAP under squared observation loss
and Sobolev operator penalties. Columns are normalized to unit grid RMS to
remove factor/core scale ambiguity. An observation-only flat operator posterior
followed by projected HOSVD initializes the factors; the dense initializer is
discarded and its cost is included.

### 3.2 Conditional Bayesian core

With factors fixed, define the row design

\[
z_{ijk}=U_1(i)\otimes U_2(j)\otimes U_3(k).
\]

Writing (g=\mathrm{vec}(\mathcal G)), the conditional model is Bayesian linear
regression:

\[
\Sigma_g=(\beta Z_\Omega^\top Z_\Omega+\alpha I)^{-1},
\qquad
\mu_g=\beta\Sigma_gZ_\Omega^\top y_\Omega.
\]

Scalar core precision (alpha) and noise precision (eta) are updated by
evidence maximization. Predictive mean and variance follow analytically. LOO
residuals provide observation-only variance scaling. The accurate description
is therefore **operator-regularized MAP factors with a conditional Bayesian
Tucker core**, not a fully Bayesian Tucker posterior.

### 3.3 Baselines and ablations

- **Geometry CP:** the same factor spaces but a superdiagonal/restricted core.
- **Wrong Tucker:** the same ranks and inference with permuted operator
  eigenfunctions.
- **Discrete Tucker:** identity factor bases and no operator energy.
- **Flat operator GP:** correct product-operator features with no multilinear
  factorization.

All models share masks, noisy observations, normalizations and held-out metrics.

## 4. Experimental design

The controlled tensor has shape `20×28×36`, corresponding to a Neumann time
interval, a Dirichlet bounded range and a periodic angular mode. Independent
smooth factors and a dense `(4,5,5)` core generate a field that is Tucker-low-
rank but not low CP-rank. Values are standardized. Unless stated otherwise,
observed values receive Gaussian noise with standard deviation 10% of the
observed field standard deviation.

The main random-mask configuration and its ranks were frozen before seeds
10--19. Stress-test settings were inspected on seeds 20--24 and then frozen
before seeds 25--29. Metrics are computed only on unobserved entries; the block
mask removes a connected central spatial rectangle from the eligible
observation set.

## 5. Results

### 5.1 Observation phase transition

| Observation | Geo-BTucker | Geo-CP | Flat GP | Wrong Tucker |
|---:|---:|---:|---:|---:|
| 1% | **0.676±0.072** | 0.809±0.072 | 0.752±0.030 | 1.462±0.097 |
| 2% | **0.125±0.015** | 0.367±0.055 | 0.612±0.028 | 1.712±0.201 |

At 2%, Geo-BTucker improves over geometry CP by 65.9%, over the flat GP by
79.6%, and over wrong geometry by 92.7%. Each ten-seed exact two-sided paired
permutation test gives `p=0.001953`. The sharp improvement between 1% and 2%
is an identifiability transition: roughly 200 observations must estimate a
100-coefficient core and nonlinear factors, whereas 2% provides roughly 400.
We report the transition rather than hiding the weaker 1% regime.

### 5.2 Structured missing block

| Model | Held-out NRMSE |
|---|---:|
| Geo-BTucker | **0.174±0.074** |
| Geometry CP | 0.515±0.103 |
| Flat operator GP | 0.631±0.030 |
| Wrong Tucker | 1.640±0.119 |

Geo-BTucker improves over the flat GP by 72.5%, with a seed-bootstrap 95% CI
of `[64.9%,77.7%]` and exact paired `p=0.001953`. One seed reaches 0.369,
showing that extrapolation variance is real, but all ten paired comparisons
favor the proposed model.

### 5.3 Thirty-percent observation noise

| Model | Held-out NRMSE |
|---|---:|
| Geo-BTucker | **0.487±0.091** |
| Flat operator GP | 0.661±0.025 |
| Geometry CP | 0.969±0.134 |
| Wrong Tucker | 1.631±0.111 |

The improvement over the flat GP is 26.4%, bootstrap CI `[18.1%,33.1%]`,
with exact paired `p=0.003906`. The smaller margin than at 10% noise is expected:
factor MAP estimates themselves become noisy, while only the core is Bayesian.

![Frozen stress-test results](../longterm_results/round7_headline.png)

## 6. What the ablations establish

The causal chain is compact:

- Tucker over CP isolates the value of a non-superdiagonal core.
- Correct over permuted Tucker isolates operator geometry within the same model.
- Tucker over flat operator GP shows that the gain is not merely an operator
  feature dictionary; multilinear compression matters.
- The 1% result bounds the claim: correct structure cannot overcome an
  underidentified core/factor problem at arbitrary sparsity.

## 7. Conditional uncertainty and active acquisition: negative result

We reserve a fixed 20% evaluation set, observe 1%, and acquire another 1% from
the remaining pool. Conditional integrated variance greedily applies the exact
rank-one covariance reduction

\[
\Delta(z)=
\frac{z^\top\Sigma_g G_{\rm eval}\Sigma_gz}
{\sigma^2+z^\top\Sigma_gz}.
\]

Every acquisition strategy is finally evaluated by the same correct-operator
Tucker model.

| Acquisition | Fixed-evaluation NRMSE |
|---|---:|
| Correct-operator core-IV | 0.206±0.014 |
| Wrong-operator core-IV | 0.330±0.256 |
| Random | **0.137±0.010** |

Core-IV is optimal only conditional on fixed factors. Its concentrated samples
hurt the later nonlinear factor refit. We therefore remove active acquisition
from the claim rather than adding a diversity heuristic after seeing the result.

## 8. External and irregular-domain boundary

Low global graph modes on The Well acoustic scattering do not represent its
localized high-frequency dynamics; correct graph Tucker does not obtain useful
held-out reconstruction. A separate irregular wave/elliptic gate finds that
correct operators beat wrong/bounding-box operators, but simple coordinate/SDF
functional CP is substantially stronger. These are negative appendices, not
evidence for the controlled Tucker headline.

The remaining major weakness is therefore external validity. A suitable public
dataset must expose genuine tensor modes with known mode operators and a field
that is plausibly low multilinear rank. If such a dataset is unavailable, the
paper should be positioned as a controlled mechanism study, not retrofitted to
an incompatible PDE benchmark.

## 9. Limitations

- The strongest evidence uses a model-aligned controlled generator.
- Factors are MAP estimates; uncertainty is conditional on them.
- Ranks are fixed from validation and are not automatically inferred.
- At 1%, improvement over the flat GP is modest and absolute error is high.
- Conditional core uncertainty does not support active sensing after factor
  refitting.
- Public complex-wave reconstruction remains negative.

## 10. Reproduction

```bash
export PYTHONPATH=src
PY=/home/ubuntu/project/yanjiu/.venv/bin/python

$PY experiments/run_tensor_bayes.py \
  --output runs/reproduce_a_random --task tucker \
  --models geo_btucker,wrong_btucker,discrete_btucker,geo_bcp_noard,flat_geo_gp \
  --ratios .02 --masks random --seeds 10,11,12,13,14 \
  --tucker-ranks 4,5,5 --steps 500 --reg .002 --noise .1 --init flat_gp

$PY experiments/analyze_longterm_iterations.py
$PY experiments/analyze_round7.py
```

Machine-readable statistics are in
[`../longterm_results/summary.json`](../longterm_results/summary.json) and
[`../longterm_results/round7_summary.json`](../longterm_results/round7_summary.json).
The complete decision ledger is
[`../zh/第七轮发表导向迭代报告.md`](../zh/第七轮发表导向迭代报告.md).

## References

- Zhao, Zhang, and Cichocki. Bayesian CP Factorization of Incomplete Tensors
  with Automatic Rank Determination. *TPAMI*, 2015.
- Zhao et al. Bayesian Sparse Tucker Models for Dimension Reduction and Tensor
  Completion. 2015.
- Budzinskiy and Zamarashkin. Tensor train completion of multidimensional
  arrays using tensor networks with side information. 2022.
- Solin and Särkkä. Hilbert space methods for reduced-rank Gaussian process
  regression. *Statistics and Computing*, 2020.
