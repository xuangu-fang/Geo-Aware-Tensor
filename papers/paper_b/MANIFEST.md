# Paper B artifact manifest

- Method/data implementation: `src/geoaware/neural_geometry.py`
- Main runner: `experiments/paper_b_run.py`
- Seed-level analyzer/statistics: `experiments/paper_b_analyze.py`,
  `src/geoaware/statistics.py`
- Frozen raw confirmation: `runs/paper_b_phase_crossres/seed_60.json` through
  `seed_69.json`
- Aggregate statistics: `papers/paper_b/results/summary.json`
- Main result figure: `papers/paper_b/results/cross_resolution.png`
- No-phase ablation: `runs/paper_b_distance_ablation/seed_70.json` through
  `seed_72.json`
- Recomputed external evidence index: `papers/paper_b/results/external_stress.json`
- Tensor implementation/runner: `src/geoaware/neural_tensor.py`,
  `experiments/paper_b_tensor_run.py`
- Tensor refocus contract/audit: `papers/TENSOR_CORE_REFOCUS.md`,
  `papers/paper_b/TENSOR_REFOCUS.md`
- T3 moving-envelope boundary: `runs/paper_b_tensor_t3/seed_100.json` through
  `seed_104.json`, `papers/paper_b/results/tensor_t3_summary.json`
- T5 frozen confirmation: `runs/paper_b_tensor_t5_confirm/seed_300.json`
  through `seed_309.json`
- T5 statistics/figure: `papers/paper_b/results/tensor_t5_confirm_summary.json`,
  `papers/paper_b/results/tensor_t5_confirm_crossres.png`
- Full iteration/failure log: `papers/paper_b/ITERATIONS.md`
- Paper draft: `papers/paper_b/DRAFT.md`
- External stress evidence: `papers/paper_b/EXTERNAL_DATA.md`

The primary campaign is fully specified by each JSON `config` block. All dense
truth values are used only for evaluation; model fitting uses declared observed
indices. High-band metrics use the graph operator only as a diagnostic basis.

## SHA-256 checksums

Regenerate after intentional reruns with:

```bash
sha256sum runs/paper_b_phase_crossres/seed_*.json \
  runs/paper_b_distance_ablation/seed_*.json \
  papers/paper_b/results/summary.json \
  papers/paper_b/results/external_stress.json
```

The generated checksums are stored in `papers/paper_b/results/SHA256SUMS`.
