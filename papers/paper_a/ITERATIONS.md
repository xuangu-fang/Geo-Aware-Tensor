# Paper A iteration log

Frozen protocol: [`../EVALUATION_PROTOCOL.md`](../EVALUATION_PROTOCOL.md).  All
hyperparameter selection and calibration below uses noisy observations only.
The unobserved target is used only for final metrics.  Pilot outcomes are kept
here even when they motivated a redesign.

## Predeclared Paper-A story and initial success gate

Paper A asks whether an operator-spectral Bayesian field model can provide
*geometry-resolved uncertainty* under 0.1--0.5% observations, and whether that
uncertainty supports selective prediction and sensing.  This is deliberately
orthogonal to Paper B's neural/high-frequency generalization story.

The initial gate required correct geometry to improve held-out NLL/calibration
and either selective risk or active acquisition, relative to wrong geometry,
matched random features, Euclidean GP and neural baselines.  Point NRMSE was a
secondary outcome.  A positive result had to survive a genuinely misspecified
task, not only a target synthesized from the learner's eigenbasis.  Results
below force a narrower final claim: no universal 0.1% or misspecified-task
active-learning claim is supported.

## Round 1 — direct transfer of the POC formulation (negative)

Configuration: two rooms separated by an impermeable wall with a narrow lower
passage and an internal obstacle; 24 time steps, about 1,000 spatial nodes.  The
target combines diffusion/oscillation and a localized off-prior transient.
Models used only the first 256 joint-low-energy features.  Empirical Bayes used
training marginal likelihood and no calibration correction.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a.py --output runs/paper_a_round1 \
  --ratios 0.001,0.0025,0.005 --masks random,room_imbalance \
  --seeds 0,1,2,3,4 \
  --models geo_spectral,wrong_geometry,rff,rbf_gp,siren,neural_cp \
  --selector evidence --no-calibration --max-features 256 --steps 1000
```

Outcome: the geometry model did not transfer.  Under random 0.5% observations
its NRMSE was about 1.33 versus about 0.95 for the Euclidean RBF GP; under room
imbalance it was worse still.  Coverage varied widely and was not calibrated.

Diagnosis: joint-energy truncation discarded high temporal/low spatial and low
temporal/high spatial combinations.  A single isotropic Sobolev scale conflated
oscillation and diffusion.  Marginal likelihood on very few training values
also favored brittle noise/scale combinations.

Revision: retain the full graph×time product, use an oscillatory temporal basis,
learn separate temporal and graph powers by leave-one-observation-out (LOO)
predictive likelihood, and add observation-only calibration.

## Round 2 — anisotropic full spectrum and conditional calibration

This round is explicitly a *controlled matched-basis sanity check*.  Most signal
is generated from graph modes with integer-frequency temporal forcing; a
localized transient and nonlinear mixture create only mild mismatch.  Correct,
wrong-geometry and RFF models receive matched feature budgets (1,440 features).

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a.py --output runs/paper_a_round2 \
  --ratios 0.001,0.0025,0.005 --masks random,room_imbalance \
  --seeds 0,1,2,3,4 \
  --models geo_spectral,wrong_geometry,rff,rbf_gp,siren,neural_cp \
  --selector loo --max-features 1440 --steps 1000
```

Outcome: random-mask NRMSE fell to about 0.54 at 0.25% and 0.17 at 0.5%, versus
about 0.98 and 0.95 for RBF.  Under 90/10 room imbalance, geometry was strong at
0.5% (about 0.66 versus 0.98 RBF) but failed at 0.1% (about 1.42).  Correct
geometry typically improved uncertainty/error ranking, but observation-only
conditional coverage was not uniformly 95% (roughly 0.82--0.96 in important
regimes).  Therefore this is evidence of implementation correctness and a
special model-correct regime, not robust real-world superiority.

Calibration revision: derive global and predictive-scale-bin multipliers solely
from exact LOO standardized residuals, with small-bin shrinkage to the global
factor.  Both raw, global-calibrated and conditional-calibrated results are
retained; calibration never sees held-out targets.

## Round 3a — fresh heterogeneous numerical dynamics (mixed/negative)

The fresh task uses a known heterogeneous conductivity operator but generates
targets through nonlinear reaction–diffusion dynamics, chirped moving forcing,
and a late localized shock.  It is not a Gaussian draw from the fitted prior.
Seeds 41--45 were not used during Rounds 1--2.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a.py \
  --output runs/paper_a_round3_static_heterogeneous --task heterogeneous \
  --ratios .001,.0025,.005 --masks random,room_imbalance \
  --seeds 41,42,43,44,45 \
  --models geo_spectral,wrong_geometry,rff,rbf_gp,siren,neural_cp \
  --selector loo --steps 1000
```

Outcome: under random observations, neural CP has the best point mean at 0.25%
and 0.5%; geometry does not pass the original two-task NRMSE gate.  Correct
geometry consistently beats wrong geometry, however.  In the geometry-specific
90/10 room-imbalance setting at 0.25%, geometry is about 0.58 NRMSE versus about
0.67 wrong-geometry, 0.67 RBF and about 0.70 neural CP.  This supports a narrower
claim about biased coverage of a barrier domain.  Exact paired statistics are
generated by `experiments/analyze_paper_a.py`.

## Round 3b — point acquisition (negative)

Every strategy shares the same nested initial observations and budgets.  Each
acquired set is evaluated by both a common geometry evaluator and a common RBF
evaluator to separate acquisition from reconstruction.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a.py \
  --output runs/paper_a_round3_active_heterogeneous --task heterogeneous \
  --seeds 41,42,43,44,45 --selector loo --active \
  --active-start .001 --active-budgets .002,.0035,.005 \
  --active-batch 8 --active-pool 512
```

Outcome: geometry integrated-variance (IV) essentially ties random at 0.2% and
loses at 0.35/0.5%; Euclidean space filling is strongest at 0.5%.  Myopic maximum
variance is substantially worse.  Likely causes are posterior misspecification,
pooled/batched IV, and an action space that permits arbitrary space–time points
rather than physical sensors.  This result rules out a broad active-learning
headline.

## Round 3c — persistent sensors and exact grouped IV

The action is changed to a physically meaningful one: install a spatial sensor
that reveals all 24 times.  Its exact score is

\[
 \Delta(H)=\operatorname{tr}\!\left[G S H^\top
 (HSH^\top+\sigma^2I)^{-1}HS\right],
\]

so within-sensor temporal redundancy is included.  Sensors are selected one at
a time.  The controlled spectral task provides the model-correct design check;
the heterogeneous sensor smoke is retained as a robustness failure.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a.py --output runs/paper_a_round3_sensor_spectral \
  --task spectral --seeds 41,42,43,44,45 --selector loo --active \
  --active-mode sensors --active-start .001 --active-budgets .002,.005 \
  --active-batch 1
```

At seed 41 and 0.5%, exact geometry-IV reached 0.143 NRMSE versus 0.228 random;
the five-seed aggregate and paired test are in `results/TABLES.md`.  In the
heterogeneous smoke, exact geometry-IV is still not reliably better than wrong
geometry/random.  Thus the defensible conclusion is: grouped IV works when the
operator posterior is adequate, and its failure is a useful misspecification
diagnostic.

## External stress test — public Active Matter

The Well Active Matter concentration field is spatially subsampled to 32×32;
its known periodic x/y topology defines the correct basis.  This is an external
stress test, not proof of physical noise calibration.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_paper_a_real.py --output runs/paper_a_round3_real_active \
  --ratios .001,.0025,.005 --seeds 41,42,43,44,45 --max-features 512
```

Point recovery is weak (NRMSE near or above one), although geometry tends to
improve within-dataset NLL over wrong geometry and conditional coverage is near
0.9--0.95.  Negative NLL values are possible because continuous Gaussian NLL is
density- and unit-dependent; only within-dataset deltas are interpreted.

## Final limitations forced by the iterations

- No universal claim at 0.1%: performance and calibration can collapse with too
  few observations or severe covariate imbalance.
- Conditional LOO variance scaling is empirical and does not give conditional
  coverage guarantees.
- Integrated variance optimizes the assumed posterior; it can select harmful
  points under operator/dynamics mismatch.
- Graph construction and full domain geometry are assumed known.
- The strongest results are a matched controlled task and a barrier-domain
  imbalance regime; public-field point reconstruction remains inconclusive.

## Fresh confirmatory seeds for headline configurations

After the exploratory campaign and frozen protocol, independent seeds 46--50
were run only for headline configurations. They were merged with seeds 41--45;
no architecture or hyperparameter grid was changed after inspecting them.

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python experiments/run_paper_a.py \
  --output runs/paper_a_confirm_matched --task spectral \
  --ratios .0025,.005 --masks random --seeds 46,47,48,49,50 \
  --models geo_spectral,wrong_geometry,rbf_gp,neural_cp --selector loo --steps 1000

PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python experiments/run_paper_a.py \
  --output runs/paper_a_confirm_heterogeneous --task heterogeneous \
  --ratios .0025 --masks room_imbalance --seeds 46,47,48,49,50 \
  --models geo_spectral,wrong_geometry,rbf_gp,neural_cp --selector loo --steps 1000

PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python experiments/run_paper_a.py \
  --output runs/paper_a_confirm_sensor_spectral --task spectral \
  --seeds 46,47,48,49,50 --selector loo --active --active-mode sensors \
  --active-start .001 --active-budgets .005 --active-batch 1
```

With ten paired seeds, matched-task geometry vs RBF gives p=0.0020 at both
0.25% and 0.5%; heterogeneous room-imbalance at 0.25% gives p=0.0039 vs RBF and
p=0.0391 vs neural CP; controlled grouped-IV gives p=0.0020 vs random. It still
ties wrong-geometry IV (p=0.916), which prevents a topology-specific acquisition
claim.
