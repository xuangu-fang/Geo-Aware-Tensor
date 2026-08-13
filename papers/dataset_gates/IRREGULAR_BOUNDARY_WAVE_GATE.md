# Irregular outer-boundary wave gate

## Purpose

This gate corrects the project's geometry semantics. Earlier wave datasets
placed different obstacles inside a common square box. Here the active physical
domain itself changes, including its outer boundary and topology.

## Domains and physical problem

The six domains are an L-shape, a U/notched domain, a three-lobe wavy domain,
a dumbbell, a slanted channel, and a wavy domain with an internal hole. Each is
generated at 24×24 and 32×32 background resolution with two source locations.

On active nodes, the independent solver integrates

\[
u_{tt}+\gamma u_t+L_cu=f(t,x),
\]

with reflecting zero-flux conditions on every outer and hole boundary. It
records 40 times on `[0,2]`. The solver imports no learner or tensor model.

![Irregular outer-boundary fields](irregular_boundary_wave.png)

## Stored geometry contract

Every case contains the active coordinates, grid-to-node map, domain mask,
boundary mask, distance to the nearest boundary, material speed, source,
graph edges, unweighted geometry operator, weighted wave operator and field.
Geometry/operator metadata may be used by a learner; unobserved field values may
not be read during fitting.

## Smoke audit

| Audit | Result |
|---|---:|
| Cases / geometries | 24 / 6 |
| Background resolutions | 24, 32 |
| Active-node range | 236--772 |
| Boundary-node range | 52--146 |
| All fields finite and non-degenerate | yes |
| All sources inside active domain | yes |
| Maximum operator symmetry error | 0 |
| Smallest geometry-operator eigenvalue | `-3.96e-14` |
| Late-half field-energy fraction | 48.6%--50.7% |

The gate passes at **SMOKE** level. The late energy confirms that targets contain
substantial post-boundary interaction rather than only the incident pulse.

## Frozen lightweight split

The first method gate uses four training geometries, one validation geometry and
one untouched test geometry, with all resolutions and sources grouped by
geometry. The exact names are stored in
`experiments/dataset_splits/irregular_boundary_wave_smoke.json`.

This split is only for deciding whether the current A/B methods perceive
irregular boundaries. It is too small for a publication claim.
