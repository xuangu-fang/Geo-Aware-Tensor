# Paper B tensor refocus

## Audit of the current IP-NF

The current positive result is a valid geometry-coordinate result but not yet a
geometry-aware tensor-decomposition result. `IntrinsicPhaseField` concatenates
space, time, geometry descriptors, and intrinsic traveling-phase features and
passes them through one joint MLP. It has no identifiable mode factors, no CP or
Tucker core, no multilinear contraction, and no rank bottleneck tied to tensor
modes. Consequently, its 28→40 result cannot establish that tensor factorization
is useful or necessary.

We retain that result only as evidence that intrinsic travel phase is the right
coordinate on the narrow-door benchmark.

## Refocused object

For geometry instance `g`, time `t`, and spatial query `x`, Paper B now studies

```text
u_hat(g,t,x) = sum_{r=1}^R w_r G_r(g) T_r(t) X_r(x; geometry_g)
```

or a Tucker form

```text
u_hat(g,t,x) = Core ×_1 G(g) ×_2 T(t) ×_3 X(x; geometry_g).
```

The factors have distinct responsibilities:

- `G(g)`: geometry-family descriptor factor, shared across resolutions;
- `T(t)`: temporal/frequency factor;
- `X(x; geometry_g)`: intrinsic spatial factor built from geodesic phase,
  boundary distance, or operator eigenfeatures;
- CP weights or a small Tucker core: cross-mode interactions.

The spatial factor is evaluated continuously/on a new graph rather than stored
as a mesh-specific table. Thus a 28→40 query tests both geometry and resolution
transfer.

## Minimal causal claims

The new headline requires all of:

1. geometry-aware tensor beats traditional discrete CP/Tucker and raw Neural-CP;
2. correct intrinsic geometry beats an identical Euclidean/wrong-geometry tensor;
3. tensor factorization beats or matches a parameter-comparable joint IP-NF in
   the sparse regime, or offers a clear parameter/sample-efficiency advantage;
4. removing the CP/Tucker contraction (joint decoder) or removing phase bands
   degrades performance;
5. results survive unseen geometry and 28→40 resolution across seed-level
   paired statistics.

Claim 3 is essential. If joint IP-NF remains decisively better at matched
capacity, the honest conclusion is that intrinsic phase helps but the proposed
tensor factorization does not.

## Four-round protocol

### Round T1 — naïve geometry-aware neural CP

Independent MLP factors for geometry, time, and intrinsic spatial phase;
rank-`R` CP contraction. Diagnose whether strict separability underfits moving
wave envelopes.

### Round T2 — band-paired phase CP

Use sine/cosine spatial and time factors whose CP pairs implement traveling
phase via the trigonometric addition identity, while amplitudes remain learned.
This is interpretable and directly connects high-frequency recovery to low rank.

### Round T3 — small Tucker core / band adapter

If CP diagonal pairing is too rigid, add a compact bandwise Tucker core to mix
spatial phase, time phase, and geometry amplitudes. Compare parameter-matched
wrong geometry and a diagonal-core ablation.

### Round T4 — frozen cross-resolution confirmation

Select the formulation using pilot seeds only, freeze it, and run fresh seeds.
Report total, high-band, boundary, parameter count, training time, and paired
seed-level statistics. Include same-resolution and observation-ratio slices.

## Benchmark scope

The narrow-wall geodesic wavepacket is retained because it supplies a tensor
indexed by geometry × time × space and has an independently generated eikonal
phase. Easier/moderate settings are allowed: the goal is causal understanding,
not maximizing difficulty. Public Active Matter/cylinder data remain external
stress tests, but the tensor contribution must be established by explicit
mode/core ablations rather than inherited from the prior Geo-NFT result.

## Novelty boundary after 2026 literature audit

We do **not** claim to introduce functional neural tensor decomposition. F-INR
(Vemuri, Büchner, and Denzler, WACV 2026; arXiv:2503.21507) already develops
axis-specific neural subnetworks combined by CP, TT, or Tucker contractions and
evaluates images, video, geometry, and physics simulations. A raw-coordinate
version of our mode-wise neural CP is therefore an F-INR-style baseline.

We also do not claim that Fourier/spectral temporal embeddings or neural Tucker
gating are new. SG-NTF (Wang and Hou, arXiv:2606.00584) combines a continuous
spectra-guided temporal embedding with neural Tucker factorization and
spatiotemporal co-gating for incomplete tensors.

The eligible novelty is narrower: operator/geodesic geometry enters the
*conditional spatial mode factor*, while the CP/Tucker core stays explicit; an
equal-capacity Euclidean/wrong-operator tensor causally isolates that geometry;
and evaluation targets extreme sparse observations plus unseen domain geometry
and mesh resolution. We must show this geometry-conditioned factor adds value
beyond both F-INR-style raw factors and a monolithic intrinsic-coordinate INR.

Primary sources:

- https://arxiv.org/abs/2503.21507
- https://openaccess.thecvf.com/content/WACV2026/papers/Vemuri_F-INR_Functional_Tensor_Decomposition_for_Implicit_Neural_Representations_WACV_2026_paper.pdf
- https://arxiv.org/abs/2606.00584
