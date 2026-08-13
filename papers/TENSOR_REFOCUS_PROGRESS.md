# Tensor-refocus iteration ledger

Date started: 2026-08-12

This ledger is the cross-paper decision record for the second development
campaign.  It prevents a successful geometry coordinate experiment from being
silently relabeled as a tensor result.  Every retained headline model must have
an explicit multilinear contraction and must separately survive a wrong-geometry
control.

## Shared causal checklist

| Question | Paper A control | Paper B control |
|---|---|---|
| Does low-rank tensor structure help? | operator Bayesian CP vs dense operator GP | conditional CP/Tucker vs monolithic IP-NF |
| Does correct geometry help? | correct vs permuted/discrete operator bases | geodesic vs Euclidean spatial factor |
| Does the probabilistic/neural component help? | ARD and factor-uncertainty ablations | raw F-INR-style factors and core ablations |
| Is the result more than one lucky seed? | paired fresh-seed statistics | paired fresh-seed statistics |

## Paper A iteration record

### A-T1: random-initialized operator Bayesian CP — negative

- Object: explicit CP over time × bounded range × periodic angle, with
  operator-spectral mode factors and a conditional Gaussian amplitude posterior.
- At 1% random observations (seed 0, rank 8, short optimization), correct
  geometry improved NRMSE from 1.359 (permuted geometry) and 1.427 (discrete
  factors) to 1.063.
- The dense operator GP reached 0.539 and ARD retained all eight components.
- Decision: geometry inside CP was useful, but the tensor and pruning claims
  failed.  Do not promote this round.

### A-T2: operator-posterior to CP-ALS initialization — mixed positive

- Single change: initialize the same CP objective by fitting the observed-data
  operator posterior, reshaping its mean as a tensor, applying CP-ALS, and
  projecting the factors back into their mode bases.  No target outside the
  observation mask is used, and no model component is added.
- At 0.5% random observations over three seeds, NRMSE was 0.660 for correct
  geometry, 1.356 for wrong geometry, and 0.756 for the dense operator GP.
  Correct CP beat both controls in all three seeds.
- Under periodic-sector missingness, correct CP was 0.814 versus 0.809 for the
  dense GP and 1.336 for wrong geometry. Thus geometry remained useful, but the
  low-rank advantage did not survive that mask.
- ARD and no-ARD point predictions were effectively identical, the effective
  rank remained at the cap of eight, and conditional-amplitude 95% coverage was
  only 0.06--0.14.  The factor means are useful, but an amplitude-only posterior
  is not an adequate Bayesian uncertainty model.
- Decision: preserve random-mask point recovery as the positive regime and the
  sector result as a limitation. T3 targets factor uncertainty and tests rank
  shrinkage without changing the predictor architecture.

### A-T3: factor uncertainty and ARD falsification

- A diagonal factor Laplace correction improves uncertainty ranking but remains
  under-dispersed without calibration.
- Type-II component ARD retains every component at rank caps 8 and 12 and
  collapses coverage. It is removed from the proposed configuration and retained
  as a negative result.

### A-T4: strict split calibration and frozen confirmation

- Preliminary normalization, initialization, factor fitting, and posterior use
  only 75% of observed entries; the remaining 25% set one dispersion scale. The
  final point model is refit on all observations. The flat GP receives the same
  protocol.
- On ten fresh seeds at 2% random observations, geometry Bayesian CP is about
  0.198 NRMSE versus 0.406 flat operator GP, 2.459 wrong-geometry CP, and 1.406
  discrete Bayesian CP; all ten paired differences favor the proposed model.
- At 0.5%, point recovery is neutral versus the flat GP, but NLL, coverage, and
  uncertainty-based selective prediction improve. Periodic-gap, Tucker-format,
  and public Active Matter point results remain mixed or negative.

## Paper B iteration record

### B-T1: explicit conditional neural CP/Tucker — pilot positive

- Object: geometry, time, and geometry-conditioned spatial factors combined by
  CP or a small Tucker core.  The spatial factor receives geodesic phase and
  boundary distance.  This is a conditional tensor factorization because
  `X(x; g)` depends on the geometry instance; it is not an independent three-way
  Tucker model.
- Protocol: narrow-wall family, six training and three unseen geometries,
  2% observations, same 24-grid resolution, seed 100, 300 optimization steps.
- Unseen NRMSE: Tucker 0.328, CP 0.364, diagonal-core Tucker 0.469,
  monolithic IP-NF 0.843, raw Neural-CP 1.191, wrong-geometry tensor 1.368,
  no-phase tensor 1.730, and SIREN 1.508.
- The full Tucker model used 16.8k parameters versus 7.6k for IP-NF; this pilot
  establishes a quality signal, not yet a parameter-efficiency claim.
- Decision: the correct geometry and multilinear contraction tests both pass on
  one seed.  The full core beats the diagonal core, so cross-mode interactions
  matter.  Confirm only after fair F-INR-style and discrete tensor baselines,
  fresh seeds, and 24-to-higher-resolution transfer.

### B-T2: band-gated Tucker and stronger factor baseline — no method gain

- The main causal ordering repeated on seed 101: unseen NRMSE was 0.428 for
  Tucker, 0.469 for CP, 0.838 for monolithic IP-NF, and 1.502 for wrong geometry.
  A parameter-comparable raw F-INR-style Tucker baseline reached 3.720.
- Five learned band gates all converged near 0.82, so the proposed gate neither
  selected nor sparsified phase bands. It adds no supported method claim.
- Decision: remove band gating from the selected method. The simpler plain
  conditional Tucker is the T3 candidate; T3 changes the evidence, not the
  architecture, by running frozen multi-seed cross-resolution tests.

### B-T3: frozen cross-resolution confirmation — mixed/contract failure

- Over seeds 100--104 at 24-to-32 resolution and 2% observations, plain Tucker
  reached `0.713 ± 0.024` NRMSE. It strongly beat wrong geometry
  (`1.414 ± 0.051`) and raw F-INR Tucker (`1.963 ± 1.019`).
- Monolithic IP-NF reached `0.615 ± 0.029`, 13.8% better than Tucker. The
  geometry and tensor-factor controls pass separately, but the required
  tensor-versus-geometry-aware-flat comparison fails.
- Diagnosis: IP-NF receives the joint traveling phase `d-c t` directly, while
  separate time and spatial factors must approximate it through their core.

### B-T4: exact trigonometric paired factors — negative

- A paired CP used the exact identity
  `sin(k(d-c t)) = sin(kd)cos(kct) - cos(kd)sin(kct)` for five bands and three
  speeds, without adding a joint decoder.
- Seed 100 unseen NRMSE was 0.818 for paired CP, 0.695 for plain Tucker, and
  0.612 for IP-NF; wrong paired geometry was 1.898.
- The paired factors preserve a strong correct-geometry signal but cannot model
  the moving Gaussian envelope efficiently. The predeclared pilot failed and
  was not expanded to more seeds.

### B-T5: moderate low-rank physical regime — confirmed

The first four rounds show that the hardest moving-envelope task favors a joint
INR. Following the project objective, T5 changes the *data regime*, not model
complexity: an independently generated intrinsic standing/damped-wave family
whose time × intrinsic-space structure is physically low rank, with a 6%
off-model moving residual. On ten fresh seeds, 1% observations and 24-to-32
resolution transfer, paired geometry CP reaches 0.0952 NRMSE versus 0.1113
ordinary geometry CP, 0.1825 IP-NF, 1.5598 wrong paired geometry, and 1.8167 raw
F-INR Tucker. This confirms the narrower claim that geometry-aware tensor
regularization is sample efficient when the field has compatible multilinear
structure. The moving-envelope result remains an explicit limitation.

## Current interpretation

The shared thesis is now empirically falsifiable: operator or intrinsic geometry
must improve the factors of a classical multilinear model, not merely provide
better coordinates to a flat regressor. Frozen ten-seed evidence now supports
this thesis in both directions under the moderate regimes stated above. It does
not support universal superiority: sub-1% factor identifiability, structured
gaps, Tucker-format mismatch, and moving localized envelopes remain explicit
scope boundaries.
