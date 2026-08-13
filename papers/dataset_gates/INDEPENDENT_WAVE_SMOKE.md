# Independent multi-geometry wave dataset gate

## Purpose

This dataset breaks simulator/learner isomorphism. The finite-difference solver
is implemented in `geoaware.independent_wave_solver` and imports no tensor,
feature, or learner code. Models consume only saved NPZ files.

## Physical problem

For each geometry, source, and resolution, the generator solves

\[
u_{tt}+\gamma u_t+L_c u=f(t,x),
\]

where `L_c` is a symmetric material-weighted graph Laplacian and `f` is a
Ricker pulse with a localized Gaussian spatial profile. Missing graph edges at
outer and obstacle boundaries implement reflecting zero-flux conditions. The
time step is selected independently from the largest wave-operator eigenvalue.

## Smoke-set design

- 8 obstacle geometries: three wall/door layouts, two circles, one ellipse,
  and two double-obstacle layouts;
- 2 source positions;
- 2 resolutions: 24×24 and 32×32 before removing solid nodes;
- 40 recorded times on `[0, 2.0]`; the source-to-obstacle travel time is about
  `0.8--1.0`, so the targets explicitly include transmitted and reflected waves;
- 32 total simulation cases.

Each case stores target fields separately from coordinates, grid indices, fluid
mask, signed distance, material speed, source, edges, unweighted geometry
operator, material-weighted wave operator, and record times.

## Leakage contract

- Operator, geometry, source, and material metadata may be used as side
  information.
- Dense `field` values may only be read through the observation mask during
  training.
- Dataset generation does not import a learner or use a learned parameter.
- Future train/test splits operate at the geometry level; resolutions of the
  same geometry may not be split across train and test unless the experiment is
  explicitly a resolution-transfer test.

## Gate artifacts

- `independent_wave_smoke_summary.json`: compact schema, ranges, and per-case
  checksums;
- `independent_wave_smoke.png`: one post-scattering field per geometry;
- full generated cases under `data/independent_wave_smoke/` (ignored by Git).

## Promotion criteria

The smoke set advances to a pilot dataset only if all fields are finite and
non-degenerate, sparse operators are symmetric PSD within tolerance, source
locations lie in the fluid domain, checksums are stable under a repeated run,
and a geometry-level split manifest is committed.

## Gate result

**Passed (SMOKE), not yet promoted to PILOT.** All 32 cases are finite and
non-degenerate; sources are in the fluid domain; maximum symmetry error is
zero; the smallest computed operator eigenvalue is `-1.62e-6` (within the
`1e-5` numerical tolerance); and a repeated build produced identical SHA-256
checksums. Between 50.1% and 51.1% of field energy occurs after `t=1.0`, so the
targets materially include post-interaction dynamics. The geometry-disjoint
split is frozen in `experiments/dataset_splits/independent_wave_smoke.json`.

The next promotion step is deliberately separate: enlarge each geometry family,
then run three-seed observation-mask pilots at 1%, 2%, 5%, and 10%. This avoids
selecting a model on the eight-case smoke set.
