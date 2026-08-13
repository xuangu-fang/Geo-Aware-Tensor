# External-data stress evidence

Paper B's new intrinsic-phase model requires a known source/travel geometry, so
applying it directly to source-free Active Matter or a short cylinder-PIV clip
would change the method. We therefore treat the existing locked public-data
campaign as a scope stress test for the broader neural geometry hypothesis.

- The Well Active Matter, 5% observations: operator-feature Geo-NFT NRMSE
  0.1218 ± 0.0029 versus SIREN 0.5601 ± 0.0504 and neural CP 0.8209 ± 0.1674.
- RealPDEBench cylinder PIV, 5%: Geo-NFT 0.3788 ± 0.0249 versus neural CP
  0.4152 ± 0.0020 and SIREN 0.5299 ± 0.0164.
- Negative external result, cylinder PIV at 1%: neural CP 0.4867 ± 0.0133 beats
  Geo-NFT 0.8428 ± 0.1284. The rectangular geometry used there does not encode
  the cylinder obstacle; this motivated Paper B's intrinsic domain work.

Authoritative raw artifacts are under `runs/final_active` and
`runs/final_realpde`; aggregate values are in `reports/results/aggregate.json`.
The data sources and provenance are documented in `reports/LITERATURE_AND_DATA.md`.
These are not silently pooled with the confirmatory synthetic campaign.
