# The Well acoustic-scattering maze dataset gate

## Question

Can Paper B use an external, independently generated wave dataset without
silently paying a 311 GB download cost or leaking future pressure into the
geometry/source representation?

## Protocol

The gate pins `polymathic-ai/acoustic_scattering_maze` to revision
`8df383a3223f40f7ce66fe77b4ff4d7006dbc272` (CC-BY-4.0). The repository has
20 HDF5 LFS shards, each with 100 trajectories, for 2,000 trajectories total.
Each trajectory contains 202 frames on a 256×256 grid with pressure, velocity,
material density, and speed of sound.

Hugging Face Dataset Viewer cannot index these HDF5 files. We therefore use
HTTP range requests through `fsspec` and inspect the files with `h5py`; the gate
does not download a complete shard. Eight trajectories are read before any
training: four from the official train split, two validation, and two test.
For the visual gate, six fixed time slices are stride-4 sampled to 64×64.

## Geometry and source audit

- Geometry is defined only from static density/speed-of-sound fields.
- Source/initial condition is `pressure(t=0)`, supplied as an explicit input.
- The prediction target is restricted to `pressure(t>0)`.
- Future pressure and all target-time velocity are forbidden from geometry or
  source feature construction.
- Material-field hashes must be distinct across all eight sampled trajectories.
- Connected high-pressure components at `t=0` provide a simple recoverability
  audit for the randomized rings; they are not inferred from future targets.

## Frozen pilot subset

`experiments/dataset_splits/the_well_acoustic_64_16_32.json` selects 64/16/32
trajectories from three official split shards. Hyperparameters may see only the
train/validation selections; test remains confirmation-only. The pilot uses a
fixed 64×64 stride-4 view. A later full-resolution experiment must be registered
separately and cannot retroactively select the pilot method.

## Cost gate

The Hub contains about 319.6 GB (decimal) of HDF5 data. Downloading the minimum
three complete shards for the frozen subset costs about 47.9 GB. Range extraction
is therefore the default pilot path: material fields and required pressure frames
are read directly, and only the downsampled subset is stored locally. Exact LFS
SHA-256 values and byte counts are recorded in
`the_well_acoustic_gate_summary.json`.

## Promotion rule

Advance to PILOT only if the eight fields are finite, all eight material hashes
are unique, the initial-condition components are recoverable, range reads work
at the pinned revision, and the 64/16/32 manifest plus leakage contract are
committed. Passing this gate establishes dataset fitness; it does not establish
model performance.

## Gate result

**Passed (SMOKE), ready for a separate PILOT extraction.** All eight sampled
trajectories are finite and have distinct material hashes. Their maze wall
fractions range from 56.2% to 62.2%; the initial-condition audit recovers 3–5
connected high-pressure components. HTTP range access succeeds at the pinned
revision, a repeated extraction has the same local SHA-256, and all 20 LFS
object hashes are recorded. The next action is to
extract the frozen 64/16/32 subset and run a one-seed data-loader/metric sanity
check before any method selection.
