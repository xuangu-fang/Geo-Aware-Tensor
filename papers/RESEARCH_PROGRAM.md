# Two-paper research program: geometry-aware field reconstruction

Status: working design, 2026-08-12. This document fixes the separation between
the papers before the long experiment campaigns, so later framing is not chosen
only after seeing favorable numbers.

## Shared problem, deliberately different scientific questions

Both papers study a continuous field `u : X_1 x ... x X_M -> R^C` observed at a
small set `Omega`. They share data loaders, masks, and elementary baselines, but
they should not share the same headline claim, model selection metric, or primary
experiment.

| | Paper A: Bayesian geometry | Paper B: neural geometry |
|---|---|---|
| Question | What can be inferred reliably from 0.1–0.5% measurements, and where/at what geometric scale is information missing? | How can a neural field preserve fine scales and generalize across meshes, resolutions, and domain geometry? |
| Primary object | Posterior measure over spectral coefficients/functions | Deterministic multiscale representation and decoder |
| Geometry role | Defines prior covariance, identifiable subspaces, and information gain | Defines stable coordinates, frequency bands, boundary transport, and equivariant adaptation |
| Main metric | NLL, conditional coverage, selective risk, active-sensing regret | bandwise spectral error, high-frequency RMSE, boundary/seam error, zero-shot resolution/geometry error |
| Main regime | Extreme sparse/noisy or connected missing regions | Moderate sparse data with high-frequency targets and geometry shift |
| Main baseline | Matched/mismatched GP, RFF-GP, Bayesian last layer, ensembles | SIREN/FINER-style INR, Fourier MLP, neural CP, graph/mesh INR, wrong geometry |
| Main failure to avoid | Accurate mean with meaningless uncertainty | Smooth interpolation that erases high frequencies or breaks on a new mesh |
| Not claimed | Universal best point reconstruction | Calibrated posterior or Bayesian epistemic guarantees |

## Paper A working formulation

Working title: **Geometry-Resolved Bayesian Spectral Tensor Fields under Extreme
Sparse Sensing**.

For self-adjoint mode operators `A_m phi_mk = lambda_mk phi_mk`, construct a
product basis indexed by `k=(k_1,...,k_M)` and joint energy

```text
Phi_k(x) = product_m phi_m,k_m(x_m),
Lambda_k = sum_m tau_m lambda_m,k_m.
```

The core prior should become hierarchical/band adaptive rather than a single
Sobolev slope:

```text
w_k | a_b, p_b ~ Normal(0, [a_b (1 + Lambda_k)^p_b]^-1),  k in band b,
y_Omega | w, sigma(x) ~ Normal(Phi_Omega w, diag(sigma(x)^2)).
```

The posterior decomposition is the key scientific output:

```text
Var[u(x)|D] = Phi(x)^T Sigma_w Phi(x) + sigma(x)^2,
U_b(x) = Phi_b(x)^T Sigma_w,bb Phi_b(x).
```

`U_b(x)` says *which geometric scales* remain uncertain. For linear-Gaussian
observations, candidate sensor information gain has a closed form based on the
posterior variance or log determinant. This creates a coherent chain:

```text
operator geometry -> spectral posterior -> scale-resolved uncertainty
                  -> selective prediction / sensor acquisition.
```

Predeclared success gates:

1. At <=0.5% observations, beat the best matched-capacity point baseline by at
   least 15% NRMSE on two tasks, or by 10% NLL with matched NRMSE.
2. Achieve 90–98% empirical coverage for nominal 95% intervals after using only
   a validation/calibration split; report conditional coverage in observed,
   missing-region, boundary, and high-frequency strata.
3. Uncertainty must rank error: positive Spearman correlation and lower risk than
   baselines when abstaining on the top 10–30% most uncertain points.
4. Variance/EIG acquisition must reduce reconstruction error faster than random,
   space-filling, and wrong-geometry acquisition under an identical budget.
5. A wrong operator must measurably hurt at least one of NLL, conditional
   coverage, or acquisition regret. Otherwise “geometry-resolved” is unsupported.

Recommended tasks are domains with holes/branches, periodic seams, heteroscedastic
noise, and contiguous missing regions. Random i.i.d. masks alone are insufficient.

## Paper B working formulation

Working title: **Band-Adaptive Geometry Neural Tensor Fields for High-Frequency
Reconstruction on Changing Domains**.

Represent each mode in a geometry basis, split by operator energy, and give each
band a residual adapter:

```text
F_m(x) = sum_b g_m,b(x, Lambda_b) [Phi_m,b(x) A_m,b
                                  + R_m,b(Phi_m,<=b(x))],
u(x_1,...,x_M) = <G, tensor_product_m F_m(x_m)>.
```

The adapter must be residual and band limited. A curriculum first fits low
operator energies, then unlocks higher bands according to validation spectral
residual. Across changing meshes/domains, sign and within-eigenspace rotations
make raw eigenvectors non-identifiable; the transferable input should therefore
use eigenspace/projector or heat-kernel features where required.

The paper's causal story is:

```text
geometry supplies intrinsic multiscale coordinates
-> band-adaptive residuals counter coordinate-MLP spectral bias
-> tensor factors keep the decoder sample efficient
-> intrinsic features transfer across sampling density and domain discretization.
```

Predeclared success gates:

1. Reduce high-frequency-band error by >=20% over SIREN/Fourier-INR/neural CP on
   at least two tasks without degrading total NRMSE by more than 2%.
2. Demonstrate zero-shot resolution and nonuniform-sampling transfer: train on one
   mesh/resolution, test by querying another, with no dense target fine-tuning.
3. Demonstrate a genuine geometry shift (hole radius/location, boundary shape,
   graph connectivity, or sphere discretization), not merely new field values.
4. Correct geometry must beat coordinate-matched wrong geometry and a random
   spectral basis; parameter and training budgets must be reported.
5. Show the mechanism through band energy/error plots, gate activation, boundary
   error, and a frequency curriculum ablation.

This paper should not headline Bayesian uncertainty. Ensembles may be used only
as a diagnostic, not as the contribution.

## Common experimental safeguards

1. Fix masks/noise before model construction and serialize them.
2. Fit normalization from observations or training trajectories only.
3. Tune on validation seeds/geometries, then freeze before test seeds/geometries.
4. Compare both equal-parameter and strongest-reasonable baselines.
5. Report all attempted rounds, including negative reformulations.
6. Separate reconstruction, interpolation, missing-region extrapolation, and
   cross-geometry generalization; they are different claims.
7. Use at least five final test seeds for synthetic tasks and cluster bootstrap
   by trajectory/geometry for public physical data.
8. Keep `runs/paper_a_*` and `runs/paper_b_*` isolated and attach config hashes.

## Literature pressure points

Paper A must distinguish itself from uncertainty-aware INRs and probabilistic
neural fields by making *operator-frequency-resolved uncertainty and downstream
information acquisition* central. Relevant comparators include
[Uncertainty-aware Continuous INRs](https://proceedings.mlr.press/v238/xu24b.html),
[Geometric Neural Process Fields](https://openreview.net/forum?id=yvGkEB3C26),
[FlowGINO](https://openreview.net/forum?id=gOZlqUCFQ6), and classical
[spatial-GP sensor placement](https://proceedings.mlr.press/v124/longi20a.html).

Paper B must distinguish itself from generic arbitrary-domain neural operators.
Relevant pressure comes from
[GNOT](https://proceedings.mlr.press/v202/hao23c.html),
[NUNO](https://proceedings.mlr.press/v202/liu23o.html),
[Fourier operators on arbitrary domains](https://proceedings.mlr.press/v235/lingsch24a.html),
and work directly addressing
[coordinate-MLP frequency bias](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0525fa17a8dbea687359116d01732e12-Abstract-Conference.html).
The claim therefore cannot merely be “we use eigenfeatures.” It needs measurable
high-band recovery and transfer across discretizations/geometries.
