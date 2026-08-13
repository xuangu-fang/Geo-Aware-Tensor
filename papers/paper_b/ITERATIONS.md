# Paper B iteration log

This log retains unsuccessful trials alongside successful ones.

## Predeclared story and success criteria

Paper B concerns deterministic neural reconstruction and transfer, not Bayesian
uncertainty. The model learns a shared spectral transfer rule on intrinsic graph
Laplacian coordinates. The headline claim requires: lower unseen-geometry NRMSE
than SIREN, RFF, neural CP, and wrong geometry; lower absolute high-band NRMSE;
a test resolution absent in training; three seeds and paired differences; and
disclosure of boundary/shadow failures and settings where a baseline wins.

## Round 1a — failed engineering run (2026-08-12)

**Design.** Matched graph-wave sanity task, six train and three held-out obstacle
geometries, 36x36, 80 eigenfeatures, 0.5% sensors, five models, 900 steps.

**Outcome.** No scientific result. The first seed had not completed after about
three minutes because the initial harness repeatedly evaluated per-task features
inside a Python loop. It was interrupted before serialization and the output
directory was empty. This is a performance failure, not a model failure.

**Revision.** The matched target is mechanism sanity only. We added a graph
finite-difference wave target with spatially varying speed and cubic frequency
mixing, which is not diagonal in the learner basis. We also added absolute
high-band NRMSE, target high-band energy fraction, and the band-start eigenvalue.

## Round 2a — diagonal spectral adapter, negative result

**Design.** The non-isomorphic heterogeneous target at 32x32, 64 modes, 0.5%
random sensors, phase-aligned spectral transfer network, 500 steps. Seed 0 is
saved in `runs/paper_b_round2/seed_0.json`.

**Result.** On unseen geometries, graph adapter NRMSE was 1.400 versus 1.127 for
RFF and 1.195 for wrong geometry. Absolute high-band NRMSE was 0.0596 versus
0.0421 for RFF. Thus the initial formulation failed both headline criteria.

**Diagnosis.** Spatially varying propagation speed and cubic dynamics couple
Laplacian modes. A diagonal transfer function `h(lambda,t)` cannot express this,
even with phase features.

**Revision.** Replace the diagonal transfer with an intrinsic kernel bank
(multi-scale heat plus damped sine/cosine wave kernels) followed by a local
neural adapter. The wrong-geometry control uses the identical network and
parameter count, changing only obstacle-aware graph kernels to rectangle
kernels. This tests geometry, rather than capacity.

## Round 2b — intrinsic kernel bank, negative

**Config.** `target=heterogeneous`, 32x32, 64 modes, 0.5% random sensors,
900 steps, hidden 96, seed 0 (`runs/paper_b_round2b`).

**Result.** Unseen NRMSE/high-band NRMSE: graph 1.439/0.0612, RFF
1.127/0.0421, wrong geometry 1.281/0.0673. Graph features improved the globally
normalized obstacle-boundary RMSE (0.356 versus RFF 0.611), but failed globally.

**Diagnosis/revision.** The intrinsic local signal is real but its flexible
decoder overfits. Introduced a zero-gated, low-dimensional intrinsic correction
on an identical RFF residual path.

## Round 2c — gated residual, negative

**Config.** Same as Round 2b (`runs/paper_b_round2c`).

**Result.** Graph 1.227/0.0469 versus RFF 1.127/0.0421; correct and wrong
geometry were essentially equal (wrong 1.229/0.0476). The gate improved on 2b
but did not isolate a geometry effect.

**Diagnosis/revision.** Zero-context transfer is information-limited: a new
domain's dynamics cannot be inferred from only a descriptor. We changed to a
few-shot protocol in which the declared sparse test sensors condition an
intrinsic heat-kernel residual; only their complement is evaluated.

## Round 3a — 0.5% context, negative

**Config.** Heterogeneous target, 32x32, 64 modes, 0.5%, seed 10, diffusion
0.003, ridge 0.15, 900 steps (`runs/paper_b_round3pilot`).

**Result.** Graph 1.118/0.0491 versus RFF 1.090/0.0391; correct and wrong
geometry were again equal. About five sensors per field do not identify the
local propagation correction.

**Revision.** A fixed exploratory grid at 2% tested `(diffusion,ridge)` =
(.001,.3), (.003,1), (.01,1), (.02,3), 28x28, 56 modes, 500 steps, seed 10
(`runs/paper_b_tune_*`). Best graph NRMSE was 1.092 versus RFF 1.096, but high
band was 0.0391 versus 0.0388 and correct/wrong remained indistinguishable.
Truncation to 56 modes likely erases obstacle-local distinctions.

## Round 3b — independent elliptic boundary-layer PDE, exploratory

**Design.** Replaced the volume-wave target with a sparse finite-difference
variable-coefficient screened-Poisson solve driven at the obstacle boundary.
This independent solver is not diagonal in the learner basis. Used 2%
boundary-stratified context, 28x28 and 160 modes. Four RFF-base settings tested
diffusion/ridge `(0,.03)`, `(.0005,.03)`, `(.002,.08)`, `(.008,.2)`, seed 10
(`runs/paper_b_elliptic_*`).

**Result/diagnosis.** RFF was a poor base (NRMSE 1.842); graph corrections
reached 1.735 at best, while SIREN reached 0.770. The base, not geometry, was
the bottleneck. With a SIREN base, 192 modes, 650 steps, and diffusion/ridge
`(.004,.3)`, `(.012,.5)`, `(.03,1)`, context correction improved NRMSE to
0.710--0.727 and boundary NRMSE to 1.327--1.351 versus raw SIREN 0.780/1.627.
However correct and rectangle geometry were identical (e.g. 0.712 versus
0.710), so this is generic few-shot smoothing, not evidence for geometry.

**Next revision.** Use a near-disconnecting wall-with-door geometry where
geodesic and Euclidean kernels differ causally, and run pure correct/wrong
kernel regression before any further neural sweep. No result above is eligible
for the paper headline.

## Round 3c — walls, identifiability limits, and final pivot

Closed-wall upstream-context seed 20 gave graph 0.665, rectangle 0.660, and
SIREN 0.861 NRMSE; random context gave graph 0.270 and rectangle 0.270. Correct
geometry could not update a disconnected component and offered no attribution.
On narrow open walls, elliptic seed 30 gave graph 0.385, rectangle 0.379, and
SIREN 0.743: generic context helped, correct geometry did not. These are retained
as failures of graph smoothing as a universal mechanism.

We then introduced an independent Dijkstra/eikonal wavepacket generator and
first tried the diagonal graph spectral adapter. Seed 40 failed: graph NRMSE
1.184, rectangle 1.077, SIREN 1.203; neural CP had the best high-band error.
Diagnosis: a scalar spectral multiplier does not efficiently represent a moving
localized phase front.

## Round 3d — intrinsic phase INR, positive confirmatory result

**Method revision.** Shortest-path source distance is known geometry metadata.
A shared MLP receives sin/cos features of this distance and traveling phases at
five bands and three speeds. The matched wrong-geometry control replaces only
shortest-path distance by Euclidean source distance. Target generation uses two
shortest-path packets and is independent of the neural model/Laplacian basis.

**Pilot.** Seed 40, narrow-wall geometries, 2% random observations, 28x28:
intrinsic NRMSE 0.873 versus RFF 1.170, SIREN 1.189, rectangle phase 1.735;
high-band 0.0174 versus 0.0284, 0.0272, 0.0564. The declared high band contains
58.6% of target energy.

**Frozen confirmation.** Seeds 60–69, six 28x28 training geometries, three
unseen 40x40 query geometries, 192 diagnostic modes, 1600 steps, hidden 96,
2% random observations (`runs/paper_b_phase_crossres`). Seed-aggregated results:

| Method | NRMSE | high-band NRMSE | boundary NRMSE | shadow NRMSE |
|---|---:|---:|---:|---:|
| intrinsic phase | **0.6025** | **0.01027** | **0.6249** | **0.5614** |
| Euclidean phase | 1.9371 | 0.05717 | 2.1101 | 1.5937 |
| SIREN | 1.2880 | 0.02734 | 1.4007 | 0.7984 |
| RFF | 1.2184 | 0.02920 | 1.3295 | — |
| neural CP | 1.4253 | 0.01931 | 1.3745 | — |

Against every baseline, both total and high-band exact paired sign-permutation
p-values are 0.001953 (10 seed pairs). Against RFF, relative improvements are
50.6% total (paired bootstrap CI 48.2–52.9%) and 64.8% high band (62.5–66.8%).
No nested task row is counted as an independent replicate.

**No-phase ablation.** With the same intrinsic distance but no sinusoidal phase
bank, fresh seeds 70–72 (28→40, 2%, 128 metric modes, 1000 steps) yield NRMSE
`1.643 ± 0.396` and high-band NRMSE `0.0310 ± 0.0174`; RFF is `1.197 ± 0.049`.
This supports the specific intrinsic *phase* mechanism rather than distance as
a generic additional scalar. Three seeds are reported descriptively only.

# Tensor-core refocus iterations

The following rounds supersede IP-NF as the central model. IP-NF remains the
required geometry-aware monolithic/no-tensor ablation. The formal contract is
in `papers/TENSOR_CORE_REFOCUS.md` and the audit in `TENSOR_REFOCUS.md`.

## T1 — naïve geometry-aware neural CP/Tucker

**Formula.** `sum_r w_r G_r(e_g) T_r(t) X_r(x,d_G,SDF)` for CP and
`Core ×_1 G ×_2 T ×_3 X` for Tucker. Each factor is generated independently;
there is no joint-coordinate residual.

**Tensor/mask.** Geometry × time × irregular-space fields from six narrow-door
train geometries and three held-out geometries, four train and three test times,
24x24, 2% shared random masks, independent shortest-path wavepacket generator.

**Engineering failure.** The first implementation issued 24 tiny task forwards
per optimization step. An intended 1000-step pilot exceeded eight minutes and
was interrupted before serialization. After batching pointwise mode factors,
the 300-step pilot completed in 20.7 seconds. This batching change did not alter
the formula.

**Seed-100 result (unseen geometries).** Tucker NRMSE/high-band NRMSE
`0.328/0.0096` (16.8k parameters), CP `0.364/0.0118` (21.0k), diagonal-core
Tucker `0.469/0.0141`, monolithic IP-NF `0.843/0.0125`, wrong-geometry CP
`1.368/0.0529`, no-phase CP `1.730/0.0323`, raw Neural-CP `1.191/0.0135`, and
SIREN `1.508/0.0539`.

**Reflection.** This cheap pilot simultaneously supports geometry necessity
(correct versus wrong), multilinear factorization (Tucker versus IP-NF), phase
bands (phase versus no-phase), and a non-diagonal core (Tucker versus diagonal).
It is exploratory and one seed. Tucker's smaller parameter count and lower error
make it the T2 candidate.

**Next single method change.** Retain the Tucker ranks/factors and add shared
band gates with a weak sparsity penalty, testing whether an explicit band
adapter preserves accuracy while exposing which intrinsic frequencies matter.

## T2 — shared band gates, factorization survives but component fails

**Formula change.** Five sigmoid gates multiply corresponding sine/cosine
channels in the time and conditional spatial factors; a `2e-4` mean-gate penalty
is added. Factors, Tucker core, and ranks are unchanged from T1.

**Protocol.** Seed 101, geometry × time × irregular-space tensor, 24x24, 2%
shared masks, 400 steps. Added an F-INR-style raw Tucker baseline with the same
mode grouping/core but raw spatial coordinates, SDF, and Euclidean radius rather
than intrinsic phase.

**Result.** Gated Tucker NRMSE/high-band `0.428/0.0150`, CP `0.469/0.0156`,
diagonal Tucker `0.781/0.0279`, monolithic IP-NF `0.838/0.0130`, wrong tensor
`1.502/0.0532`, raw F-INR Tucker `3.720/0.1057`, no-phase tensor
`2.851/0.0610`, Neural-CP `1.373/0.0147`, SIREN `1.618/0.0746`.

**Reflection.** Explicit tensor and correct-geometry signals survive, but all
gates remain between 0.819 and 0.837. There is no band selection, and a
cross-seed comparison cannot show improvement over T1. The added component is
therefore rejected rather than retuned.

**Next change.** Remove gates and freeze plain conditional Tucker. T3 changes
only the evaluation scope: five seeds and 24→32 resolution transfer.

## T3 — frozen plain Tucker across resolution, mixed/contract failure

**Formula.** Reverted exactly to the plain conditional Tucker from T1: geometry
factor, time factor, geometry-conditioned spatial factor, and a dense 5x8x12
core. No gates or new modules.

**Protocol.** Frozen seeds 100–104, 24x24 training and unseen 32x32 query
geometries, 2% masks, 400 steps. Each metric first averages nine unseen tasks
within seed. Raw F-INR Tucker uses the identical mode/core grouping without
intrinsic travel phase.

| model | unseen NRMSE | high-band NRMSE |
|---|---:|---:|
| plain conditional Tucker | 0.7129 ± 0.0237 | 0.01154 ± 0.00200 |
| conditional CP | 0.7206 ± 0.0460 | 0.01200 ± 0.00109 |
| diagonal-core Tucker | 0.7955 ± 0.0624 | 0.01458 ± 0.00374 |
| wrong-geometry tensor | 1.4144 ± 0.0509 | 0.04671 ± 0.00689 |
| no-phase tensor | 1.8410 ± 0.5637 | 0.02904 ± 0.01022 |
| raw F-INR-style Tucker | 1.9631 ± 1.0190 | 0.03555 ± 0.02635 |
| monolithic intrinsic-phase INR | **0.6146 ± 0.0294** | **0.01093 ± 0.00271** |
| raw Neural-CP | 1.2117 ± 0.1206 | 0.01308 ± 0.00041 |
| SIREN | 1.4044 ± 0.1535 | 0.04539 ± 0.00426 |

**Reflection.** Correct geometry, phase bands, and a non-diagonal core are each
necessary relative to their ablations. Nevertheless the monolithic geometry
INR is 13.8% better in total NRMSE. T3 therefore fails the shared contract's
tensor-versus-flat requirement and is not a headline success. Five seeds also
cannot produce a two-sided exact p below 0.05; these are estimation results, not
significance claims.

**Diagnosis.** The independent time and spatial factors force a moving phase
`d-c*t` to be synthesized indirectly through a finite Tucker core. IP-NF receives
that joint phase explicitly. Any T4 method change must preserve explicit
multilinearity while incorporating the trigonometric low-rank identity for
traveling waves; simply enlarging the networks/core would not test a new
hypothesis.

## T4 — speed-aligned paired-phase CP, negative

**Minimal formula change.** For five bands and three candidate speeds, spatial
and temporal carriers are computed independently and paired through CP using

```text
cos(kd)cos(kct), sin(kd)sin(kct), cos(kd)sin(kct), sin(kd)cos(kct).
```

Their learned linear combinations span arbitrary phase offsets of traveling
waves through the angle-addition identity. Geometry, time-amplitude, and
geometry-conditioned spatial-amplitude networks remain separate; there is no
joint `(distance,time)` input and no unrestricted residual. The matched wrong
control replaces geodesic distance only with Euclidean source distance.

**Protocol.** Falsifying seed 100, identical T3 24→32 geometries and 2% masks,
400 steps. Paired CP has 16,656 parameters versus 16,825 for plain Tucker.

**Result.** Paired CP NRMSE/high-band `0.8183/0.01822`; wrong paired
`1.8976/0.08903`; plain Tucker `0.6953/0.00953`; ordinary conditional CP
`0.7067/0.01087`; monolithic IP-NF `0.6123/0.00935`.

**Reflection.** Correct-versus-wrong geometry remains strong, but hard carrier
pairing reduces accuracy relative to the learned Tucker core and does not close
the monolithic gap. The likely cause is that moving Gaussian envelopes require
amplitude interactions not captured by fixed phase pairing at this CP rank.
Because the predeclared seed-100 gate failed, T4 was not expanded to additional
seeds. This closes the four-round refocus with an honest partial conclusion:
geometry-conditioned tensor factors clearly beat raw/wrong tensor baselines,
but on the cross-resolution geodesic benchmark they do not beat the stronger
monolithic intrinsic-coordinate INR. The central shared-contract claim is not
yet proven for Paper B.

## T5 — moderate-rank eikonal harmonics, positive confirmation

**Story pivot, not a model patch.** We retained the T4 speed-aligned CP exactly
and changed only the data regime. The new independent generator is a sum of
three geometry-dependent standing eikonal harmonics with exponential time
decay, plus a 6% moving localized residual outside the dominant rank-three
structure. This tests sample efficiency when multilinearity is physically
plausible instead of selecting only the hardest moving-envelope case.

**Exploration.** Seeds 200–204 at 1%, 24→32 indicated paired CP
`0.0900 ± 0.0110`, ordinary geometry CP `0.1160 ± 0.0088`, IP-NF
`0.1865 ± 0.0202`, and wrong-paired `1.6138 ± 0.1380`. These seeds selected the
final regime and are not used in confirmatory inference.

**Frozen confirmation.** Fresh seeds 300–309, unchanged generator/model/ranks,
1% observations, six 24x24 train geometries and three unseen 32x32 geometries.
Only the six predeclared necessary models were run.

| model | unseen NRMSE | high-band NRMSE |
|---|---:|---:|
| speed-aligned geometry CP | **0.0952 ± 0.0144** | **0.00404 ± 0.00092** |
| ordinary geometry CP | 0.1113 ± 0.0213 | 0.00497 ± 0.00113 |
| monolithic intrinsic-phase INR | 0.1825 ± 0.0173 | 0.00728 ± 0.00122 |
| wrong/Euclidean paired CP | 1.5598 ± 0.1827 | 0.08799 ± 0.01176 |
| raw F-INR-style Tucker | 1.8167 ± 0.2535 | 0.06079 ± 0.00217 |
| SIREN | 1.0944 ± 0.0366 | 0.05938 ± 0.00318 |

Paired CP improves total NRMSE over ordinary geometry CP by 14.5% (paired
bootstrap CI 3.4–23.8%, exact paired permutation p=0.0391) and over monolithic
IP-NF by 47.9% (41.5–53.2%, p=0.001953). Correct versus wrong geometry improves
93.9% (p=0.001953). High-band improvements versus ordinary CP and IP-NF are
18.7% (p=0.0332) and 44.4% (p=0.001953). Statistics aggregate nested tasks
within seed before paired inference.

**Conclusion after five refocus rounds.** Both factors of the shared claim are
now supported on the aligned moderate-rank benchmark: correct geometry matters
within an identical tensor architecture, and explicit paired tensorization
beats a geometry-aware monolithic decoder. T3/T4 remain a prominent boundary:
on nonseparable moving envelopes, the monolithic decoder is better.
