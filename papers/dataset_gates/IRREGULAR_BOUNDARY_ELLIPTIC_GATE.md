# Irregular-boundary smooth elliptic gate

This method-matched gate was the final go/no-go check for promoting irregular
boundary geometry to a paper main line. It uses screened elliptic tensors with
semantics `source × diffusivity × irregular-domain node` on six domains and two
resolutions. The solver uses a weighted PDE operator while learners receive only
the unweighted domain operator.

![Smooth irregular-domain tensors](irregular_boundary_elliptic.png)

All twelve cases are finite and non-degenerate. Relative sparse linear-solve
residual is at most `2.97e-14`; tensor shapes are `4 × 14 × N`, where active node
counts range from 236 to 772.

The gate is **NO-GO for a main paper direction**. At 1% observations, correct
operator CP beats wrong and bounding-box operator controls (`0.668` versus
`1.315` and `0.847` macro NRMSE), but coordinate and SDF functional CP both
obtain approximately `0.180`. Correct operator Tucker is `0.895`. In the
cross-domain Paper-B gate, operator-spectral CP obtains `0.278`, while SDF CP
obtains `0.181`.

The result establishes that operator metadata is used, but not that it is the
best or necessary inductive bias. No further boundary-specific method tuning is
authorized under the current research plan.
