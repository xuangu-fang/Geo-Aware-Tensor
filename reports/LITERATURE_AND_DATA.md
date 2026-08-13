# Literature, baseline, and dataset audit

Audit date: 2026-08-12. This note records the sources that materially influenced
the implementation; it does not claim that every related paper was exhaustively
surveyed.

## Closest task and representation baselines

- [F-INR: Functional Tensor Decomposition for Implicit Neural Representations
  (WACV 2026)](https://arxiv.org/abs/2503.21507) already factorizes a monolithic
  INR into axis-specific subnetworks and supports CP, TT, and Tucker contractions,
  including a physics-simulation example.  It is therefore the direct Paper-B
  *tensor architecture* baseline.  Our differentiator cannot be neural functional
  factorization alone: it must be the operator/geodesic prior inside the spatial
  factor, tested by correct-versus-wrong geometry under sparse observations and
  unseen geometry/resolution.
- [Spectra-Guided Neural Tucker Factorization
  (2026 preprint)](https://arxiv.org/abs/2606.00584) combines a continuous Fourier
  time embedding with a learned Tucker core and spatio-temporal gating.  This
  rules out a novelty claim based only on spectral time features or neural Tucker;
  Paper B must isolate domain topology, boundary conditions, and propagation
  distance rather than generic temporal periodicity.
- [Continuous Field Reconstruction from Sparse Observations with Implicit Neural Networks (ICLR 2024)](https://openreview.net/forum?id=kuTZMZdCPZ)
  is the closest task-level reference. It separates spatiotemporal variability
  and learns continuous basis functions from irregular observations. Its official
  [MMGN repository](https://github.com/Xihaier/Continuous-Field-Reconstruction-MMGN)
  targets a supervised multi-field climate setting; this POC therefore implements
  the comparable INR and neural-factor ingredients locally rather than claiming an
  apples-to-apples reproduction on a single-field fitting protocol.
- [SIREN (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html)
  motivates the periodic-activation INR baseline and its initialization.
- [Fourier Features (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)
  motivates the NeRF-style Fourier-feature MLP control.
- [Large-Scale Learning with Fourier Features and Tensor Decompositions (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/92a08bf918f44ccd961477be30023da1-Abstract.html)
  is an important warning that Fourier features plus low-rank tensor structure is
  already a meaningful baseline; geometry-specific boundary conditions and
  uncertainty must provide the extra value.

## Neighboring geometry and Bayesian work

- [Delta-PINNs](https://openreview.net/forum?id=5P96KWeULzE) encode complex-domain
  topology with Laplace-Beltrami eigenfunctions. This supports using operator
  eigenfunctions as coordinates, but their task is PDE solution rather than sparse
  tensor completion.
- [Product Manifold Learning (AISTATS 2021)](https://proceedings.mlr.press/v130/zhang21j.html)
  gives the mathematical product-manifold fact used here: eigenfunctions multiply
  across factors while eigenvalues add.
- [Provable Tensor Completion with Graph Information](https://arxiv.org/abs/2310.02543)
  and [Variational Bayesian inference for CP tensor completion with side information](https://arxiv.org/abs/2206.12486)
  are the closest graph/side-information completion references. They motivate the
  wrong-geometry and side-information controls.  In particular, the latter already
  combines CP, low-dimensional fiber subspaces, variational Bayes, and automatic
  rank determination.  Paper A therefore cannot claim Bayesian CP with side
  information as new; the distinct object is an operator-eigenfunction factor
  prior whose eigenvalues control frequency-dependent shrinkage and whose
  uncertainty is tested on boundary/topology-sensitive fields.
- [Bayesian Sparse Tucker Models for Dimension Reduction and Tensor Completion](https://arxiv.org/abs/1505.02343)
  motivates a Bayesian Tucker core, but does not provide the mode-wise
  operator-eigenvalue prior used here.
- [Uncertainty-aware Continuous Implicit Neural Representations (AISTATS 2024)](https://proceedings.mlr.press/v238/xu24b.html)
  is a useful probabilistic-INR comparator, although its application is remote
  sensing counting rather than geometry-constrained physical-field recovery.
- [Geometric Neural Process Fields](https://openreview.net/forum?id=yvGkEB3C26)
  combines probabilistic neural fields with geometric bases.  As of the audit
  date it is a TMLR submission with a pending decision, so we treat it as a
  close concurrent preprint rather than an established benchmark.  Paper A must
  therefore distinguish itself through operator-resolved posterior calibration,
  extreme-sparsity diagnostics, and uncertainty-driven sensor acquisition.

## Irregular-domain neural operators and geometry transfer

- [GNOT (ICML 2023)](https://proceedings.mlr.press/v202/hao23c.html) handles
  irregular meshes with heterogeneous normalized attention and geometry gating.
- [NUNO (ICML 2023)](https://proceedings.mlr.press/v202/liu23o.html) maps
  non-uniform point data to decomposed regular grids; it is a relevant scalable
  operator baseline but is not designed for fitting a field from below-1% sensors.
- [Beyond Regular Grids (ICML 2024)](https://proceedings.mlr.press/v235/lingsch24a.html)
  directly evaluates Fourier transforms on arbitrary point sets.  It motivates
  an arbitrary-domain Fourier control and makes it insufficient to claim novelty
  from merely replacing FFT evaluation.
- [Geometric Generalization of Neural Operators from a Kernel Integral Perspective](https://arxiv.org/abs/2602.01498)
  studies variable-geometry generalization through multiscale kernel operators.
  It is close in problem setting but does not make sparse single-field spectral
  reconstruction or high-band recovery its central object.
- [Learning Laplacian Eigenspace with Mass-Aware Neural Operators](https://arxiv.org/abs/2605.24390)
  learns invariant eigenspaces and reports resolution transfer.  This recent
  preprint is especially relevant to Paper B: individual eigenvector signs and
  rotations inside repeated eigenspaces cannot be assumed stable.  Our final
  method and ablations must use sign/subspace-invariant quantities or explicitly
  document alignment.

These references sharpen both the common core and the separation between the two
manuscripts.  Both modify a classical tensor decomposition by placing geometric
structure in its mode factors.  Paper A is evaluated as Bayesian rank/frequency
shrinkage, calibrated inference, and experimental design; Paper B is evaluated
as neural functional factorization, high-frequency reconstruction, and transfer
across meshes and domain geometries.  A single aggregate RMSE table is not
sufficient evidence for either claim.

## Datasets selected

### The Well Active Matter (primary public physics result)

The dataset is part of [The Well, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f9a5acd91ac76569f2fe291b1f4772b-Abstract-Datasets_and_Benchmarks_Track.html).
The official [Active Matter dataset card](https://huggingface.co/datasets/polymathic-ai/active_matter)
states that the 2-D spatial boundaries are periodic, the native trajectories have
81 frames at 256x256, and the license is CC-BY-4.0. The local immutable benchmark
used here is a 64x64 downsample with strict train/validation/test provenance:

```text
/home/ubuntu/project/yanjiu/data/active_matter_multi/benchmark_strict_r48.npz
```

The POC selects one held-out trajectory, takes every second frame, and never uses
unobserved values for training or normalization.

### RealPDEBench cylinder wake (secondary real-data result)

[RealPDEBench](https://realpdebench.github.io/) provides paired experimental and
simulated physical systems; the cylinder data are real PIV measurements spanning
multiple Reynolds regimes. The official [repository](https://github.com/AI4Science-WestlakeU/RealPDEBench)
and [Hugging Face release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench)
document the download path. The local locked benchmark is:

```text
/home/ubuntu/project/yanjiu/data/realpde_cylinder_fresh_locked/locked_r64.npz
```

One velocity channel and 48 uniformly spaced frames are used. Unlike Active
Matter, the current tensor-product basis does not encode the cylinder obstacle
itself. Results on this dataset should therefore be read as a partial-geometry
test, not as evidence for obstacle-aware Laplace-Beltrami modeling.

### Controlled geometry sanity checks

`synthetic_boundary` uses a Dirichlet interval times a circle. `synthetic_wave`
uses time-circle x Neumann range x azimuth-circle with a weak nonseparable chirp.
They are generated deterministically in `geoaware.data`, so boundary and topology
claims can be falsified with the wrong-basis ablation.

## Dataset candidates not used in the primary run

- [PDEBench](https://github.com/pdebench/PDEBench) remains suitable for a larger
  multi-trajectory extension, especially shallow-water or reaction-diffusion.
- The Well acoustic-scattering datasets are especially relevant for the proposed
  long-term wave-field story, but a fair supervised multi-instance study is beyond
  this single-field POC and would require substantially more download/storage.
