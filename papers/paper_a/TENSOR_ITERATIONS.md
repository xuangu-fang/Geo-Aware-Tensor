# Paper A tensor-core iteration log

This log begins after [`TENSOR_REFOCUS.md`](TENSOR_REFOCUS.md) and follows the
shared [`../TENSOR_CORE_REFOCUS.md`](../TENSOR_CORE_REFOCUS.md) contract.  The
operator-GP campaign remains preserved in `ITERATIONS.md`; it is now a precursor
baseline rather than the paper's central method.

## T1 — minimal conditional Bayesian spectral CP

### Formulation

The tensor remains explicitly indexed and CP-factorized:

\[
Y_{ijk}=\sum_{r=1}^{R}a_r(\Phi_1u_{1r})_i
(\Phi_2u_{2r})_j(\Phi_3u_{3r})_k+\epsilon_{ijk}.
\]

Adjacent parenthesized factors above are multiplied; explicitly,
\(a_r\prod_m(\Phi_mu_{mr})_{i_m}\). Factor means are MAP estimates under
operator-frequency penalties.  Conditional on normalized factors, amplitudes
have an exact Gaussian posterior and MacKay-style component ARD updates.

### Data tensor and hypothesis

`operator_cp_tensor` is a 20×28×36 time×bounded-range×periodic-angle tensor.  It
has rank-four operator factors plus a mild localized residual.  Shared random
and missing periodic-sector masks use 0.5% and 1% observations.  The falsifying
hypothesis was that correct geometry and explicit low rank would both improve
over wrong/discrete Bayesian CP and a dense operator GP.

### Experiment

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py --output runs/paper_a_tensor_round1 \
  --task cp --ratios .005,.01 --masks random,periodic_gap --seeds 0,1,2 \
  --rank 10 --steps 1800 \
  --models geo_bcp,geo_bcp_noard,wrong_bcp,discrete_bcp,flat_geo_gp \
  --ard-cycles 1
```

The run was deliberately interrupted after all three random-0.5% seeds and part
of random-1% because the cheap falsification had already failed; stdout is the
authoritative partial record and no complete manifest is claimed.

### Result and reflection

At random 0.5%, geometry-aware CP NRMSE was 1.139, 1.381 and 1.016.  Wrong CP
was 1.446, 1.584 and 1.256; discrete CP was 1.421, 1.382 and 1.170.  Geometry
therefore helped *inside CP*, but flat operator GP was 0.646, 0.758 and 0.863.
ARD and no-ARD predictions were numerically identical and effective rank stayed
at the cap.  At random 1% seed 0, geometry CP improved to 0.708 but flat GP was
still 0.539.

Diagnosis: sparse multiplicative optimization, rather than the CP representation
itself, dominated. Random factor initialization almost never reached the basin
of the low-rank operator tensor. Conditional amplitude ARD cannot prune useful
components before factor columns become meaningful.

### Next change

Only the optimizer initialization changes: first obtain an observation-only
dense operator posterior mean, compress it by ordinary CP-ALS, project those
factors into each operator eigenspace, then optimize the *same* Bayesian CP
likelihood on the original observations.  This is an optimization repair, not a
new generative-model component.

## T2 — operator-posterior CP-ALS initialization

### Cheap falsifying pilot

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py --output runs/paper_a_tensor_round2_init_smoke \
  --task cp --ratios .005 --masks random --seeds 0 --rank 8 --steps 500 \
  --models geo_bcp,wrong_bcp,flat_geo_gp --init flat_gp
```

On the identical 0.5% mask, correct geometry Bayesian CP reached 0.543 NRMSE,
wrong-geometry Bayesian CP 1.396, and flat geometry GP 0.646.  Thus the cheap
pilot passes both causal checks: correct geometry matters within CP and explicit
CP compression improves over the flat geometry model.  This is exploratory and
must be repeated across seeds before it supports a claim.

### Next change

Run the same initialization across multiple random and structured masks without
changing the method.  If the result replicates, T3 will address the still-failed
ARD and factor uncertainty; if it does not, no additional model machinery will
be added to rescue it.

### Three-seed result

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py --output runs/paper_a_tensor_round2_init \
  --task cp --ratios .005 --masks random,periodic_gap --seeds 0,1,2 \
  --rank 8 --steps 800 \
  --models geo_bcp,geo_bcp_noard,wrong_bcp,flat_geo_gp --init flat_gp
```

The initializer was computed separately for correct and wrong models from their
corresponding feature GP, always using observed values only. Its cost is included
in the recorded elapsed time for CP.

| mask | correct BCP | wrong BCP | flat geometry GP |
|---|---:|---:|---:|
| random 0.5% | 0.660 | 1.356 | 0.756 |
| periodic gap 0.5% | 0.814 | 1.336 | 0.809 |

Correct CP beats the flat GP on all three random-mask seeds, but not reliably on
the periodic gap; seed 2 is a clear loss. Correct geometry strongly beats
permuted side information in both masks. This verifies a tensor contribution in
one regime, not universally.

ARD remains a failure: ARD/no-ARD point predictions are nearly identical and
the estimated effective rank stays at eight. More importantly, a posterior only
over component amplitudes is grossly overconfident. Correct-CP 95% coverage is
roughly 0.06--0.14; even no-ARD is only 0.22--0.47. This is direct evidence that
mode-factor uncertainty cannot be omitted from a Bayesian tensor claim.

### T3 change

Keep the same factors and point estimator, add a diagonal Gauss--Newton/Laplace
posterior over every spectral factor coefficient, and propagate it through the
CP product with a delta method. Separately test a larger rank cap and repeated
ARD/factor cycles; do not call ARD successful unless components are actually
removed without harming prediction.

## T3 — factor posterior correction; ARD falsified

### Experiment

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py --output runs/paper_a_tensor_round3_laplace \
  --task cp --ratios .005 --masks random --seeds 0,1,2 \
  --rank 8 --steps 800 \
  --models geo_bcp,geo_bcp_noard,wrong_bcp,flat_geo_gp \
  --init flat_gp --factor-laplace --ard-cycles 1
```

### Result

The diagonal Gauss--Newton correction leaves point means unchanged, as intended.
For geometry CP without ARD, coverage becomes 0.633, 0.616 and 0.330 with NLL
2.39, 3.67 and 13.57. This improves over the amplitude-only conditional
posterior but remains seriously under-dispersed. Uncertainty nevertheless ranks
error: 50%-selective RMSE reductions are 0.37, 0.53 and 0.50.

Type-II component ARD fails decisively: effective rank remains eight, coverage
is only 0.105, 0.160 and 0.122, and NLL is two orders of magnitude too large.
Increasing the cap to 12 and alternating three factor/ARD cycles also retains
all 12 components. The failure is not hidden in the final method. In this sparse
non-conjugate CP setting, conditional evidence collapses component/noise
uncertainty before factors are well identified.

### Reflection and next change

The final POC does **not** claim automatic rank recovery. Fixed-rank Bayesian CP
with operator-factor priors is the primary model; ARD is a documented negative
ablation. The one T4 change is observation-only split calibration: reserve 25%
of observed entries, fit the same factor-Laplace model to the remaining 75%, and
scale final predictive standard deviations by the held-observation 95% residual
quantile. Then refit the identical point model on all observations. This changes
dispersion only, not reconstruction.

## T4 — split-calibrated factor posterior and confirmation

### Calibration pilot

At random 0.5%, seed 0, split calibration changes correct no-ARD Bayesian CP to
coverage 0.985, NLL 0.726 and width 2.57, while preserving NRMSE 0.533. The flat
GP has coverage 0.935, NLL 0.803 and width 2.34. Wrong-geometry CP requires width
6.16 for coverage 0.964 and has NLL 1.72. Thus uncertainty now reflects the
wrong geometry's larger error rather than becoming narrowly overconfident.

The calibration scale is learned only from observations excluded from the
preliminary fit. Full-model target values remain unseen. The preliminary fit and
its corresponding correct/wrong operator-GP initializer are included in runtime.

### Format and public-data pilots

A Tucker-generated tensor is an intentional CP-format mismatch. At 0.5% seed 0,
rank-10 geometry CP obtains 0.928 NRMSE, wrong CP 1.336 and flat GP 0.842. Correct
geometry still helps within CP, but CP compression does not beat the flat model.
This delimits the tensor-structure claim to CP-plausible data and motivates a
future Bayesian spectral Tucker implementation rather than silently increasing
CP rank.

On public Active Matter at 1% seed 0, geometry CP is also mixed: NRMSE 1.384,
versus 2.300 wrong CP, 1.251 discrete CP and 1.340 flat GP. Its calibrated
coverage is 0.931, compared with 0.748 wrong and 0.312 discrete CP. Geometry
helps UQ and the same CP architecture, but not the best point baseline.

### Fresh confirmation and frozen outcome

The final strict protocol applies the same 75/25 observed-only split calibration
to every Bayesian method. Preliminary normalization is training-only; final
refits use all observations and their own all-observed normalization.

```bash
# Hardest primary UQ setting
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py \
  --output runs/paper_a_tensor_round4_confirm_random --task cp \
  --ratios .005 --masks random --seeds 10,11,12,13,14 --rank 8 --steps 800 \
  --models geo_bcp_noard,wrong_bcp,discrete_bcp,flat_geo_gp \
  --init flat_gp --factor-laplace --split-calibration

# Structured-gap limitation
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py \
  --output runs/paper_a_tensor_round4_confirm_gap --task cp \
  --ratios .005 --masks periodic_gap --seeds 10,11,12 --rank 8 --steps 800 \
  --models geo_bcp_noard,wrong_bcp,discrete_bcp,flat_geo_gp \
  --init flat_gp --factor-laplace --split-calibration
```

At 0.5%, fresh geometry CP NRMSE is 0.751±0.113 versus 0.725±0.038 flat GP:
point performance is neutral, and the earlier T2 advantage is classified as
exploratory. However, geometry CP has NLL 0.967 versus 1.047, 95% coverage 0.933
versus 0.882, and 50%-selective gain 0.433 versus 0.277. Wrong and discrete CP
have NRMSE 1.527 and 1.424, establishing the correct-geometry contribution
inside the identical tensor format.

The periodic gap stays mixed: geometry CP 0.805 versus flat GP 0.797 NRMSE,
with coverage 0.821 versus 0.881. The method does not extrapolate a completely
unseen angular sector as reliably as the flat posterior.

Seeds 20--24 were then used to compare candidate observation ratios. Because
that exploratory check selected 2% as the headline regime, those seeds are not
confirmatory evidence. After fixing 2%, rank 8, 800 steps, split calibration and
all four baselines, the final run used ten entirely new seeds 30--39:

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_tensor_bayes.py \
  --output runs/paper_a_tensor_final_confirm_2pct --task cp \
  --ratios .02 --masks random \
  --seeds 30,31,32,33,34,35,36,37,38,39 --rank 8 --steps 800 \
  --models geo_bcp_noard,wrong_bcp,discrete_bcp,flat_geo_gp \
  --init flat_gp --factor-laplace --split-calibration
```

Here factors become identifiable: geometry CP reaches 0.198±0.037 NRMSE versus
0.406±0.031 flat GP, 2.459±0.381 wrong CP and 1.406±0.049 discrete CP. It also
has mean NLL −0.158 versus 0.387 flat, coverage 0.983 versus 0.948, and narrower
intervals (1.080 versus 1.569). Geometry CP improves NRMSE over flat GP by 51.2%
(seed-bootstrap CI 45.0--57.1%) and over discrete CP by 85.9% (84.3--87.3%).
All ten paired NRMSE differences have the same sign; exact two-sided paired
permutation \(p=0.001953\). Correct CP also beats wrong-geometry CP by 91.9%
(90.8--92.9%, \(p=0.001953\)), separately establishing the geometry contribution.

The final predictor has 256 CP coefficients, versus 512 retained flat-product
features and 680 discrete-CP coefficients. End-to-end mean runtime is 8.19 s for
geometry CP and 0.49 s for flat GP; the preliminary calibration fit and dense
initializer are included. Compact prediction therefore does not mean cheaper
training in this proof of concept.

No further model changes are made after this confirmation.
