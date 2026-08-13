# Tensor-refocus results

Fresh confirmation; mean ± sample SD over seeds.

| mask | ratio | model | NRMSE | NLL | cov95 | selective gain |
|---|---:|---|---:|---:|---:|---:|
| random | 0.005 | geo_bcp_noard | 0.751±0.113 | 0.967±0.139 | 0.933±0.029 | 0.433±0.104 |
| random | 0.005 | wrong_bcp | 1.527±0.109 | 2.281±0.328 | 0.858±0.121 | 0.278±0.072 |
| random | 0.005 | discrete_bcp | 1.424±0.079 | 84.154±16.922 | 0.335±0.020 | 0.105±0.050 |
| random | 0.005 | flat_geo_gp | 0.725±0.038 | 1.047±0.217 | 0.882±0.063 | 0.276±0.042 |
| periodic_gap | 0.005 | geo_bcp_noard | 0.805±0.033 | 1.740±0.880 | 0.821±0.143 | 0.409±0.075 |
| periodic_gap | 0.005 | wrong_bcp | 1.407±0.090 | 2.124±0.526 | 0.842±0.093 | 0.204±0.042 |
| periodic_gap | 0.005 | discrete_bcp | 1.408±0.034 | 99.194±21.163 | 0.443±0.025 | 0.069±0.009 |
| periodic_gap | 0.005 | flat_geo_gp | 0.797±0.039 | 1.236±0.499 | 0.881±0.098 | 0.375±0.047 |
| random_2pct | 0.02 | geo_bcp_noard | 0.198±0.037 | -0.158±0.180 | 0.983±0.011 | 0.330±0.092 |
| random_2pct | 0.02 | wrong_bcp | 2.459±0.381 | 2.193±0.137 | 0.977±0.015 | 0.573±0.059 |
| random_2pct | 0.02 | discrete_bcp | 1.406±0.049 | 29.895±14.615 | 0.265±0.056 | 0.474±0.019 |
| random_2pct | 0.02 | flat_geo_gp | 0.406±0.031 | 0.387±0.085 | 0.948±0.027 | 0.365±0.051 |

## Exact paired seed tests

- `random_geo_vs_wrong_bcp_nrmse`: proposed=0.7514, baseline=1.5273, improvement=50.8% CI [46.7, 55.2], p=0.0625.
- `random_geo_vs_discrete_bcp_nrmse`: proposed=0.7514, baseline=1.4236, improvement=47.2% CI [42.5, 53.6], p=0.0625.
- `random_geo_vs_flat_geo_gp_nrmse`: proposed=0.7514, baseline=0.7247, improvement=-3.7% CI [-12.2, 5.2], p=0.4375.
- `2pct_geo_vs_wrong_bcp_nrmse`: proposed=0.1984, baseline=2.4586, improvement=91.9% CI [90.8, 92.9], p=0.0020.
- `2pct_geo_vs_discrete_bcp_nrmse`: proposed=0.1984, baseline=1.4064, improvement=85.9% CI [84.3, 87.3], p=0.0020.
- `2pct_geo_vs_flat_geo_gp_nrmse`: proposed=0.1984, baseline=0.4064, improvement=51.2% CI [45.0, 57.1], p=0.0020.
- `random_geo_vs_wrong_bcp_nll`: proposed=0.9669, baseline=2.2806, absolute difference=-1.3137, p=0.0625. Relative percentages are omitted because NLL can be negative.
- `random_geo_vs_discrete_bcp_nll`: proposed=0.9669, baseline=84.1536, absolute difference=-83.1867, p=0.0625. Relative percentages are omitted because NLL can be negative.
- `random_geo_vs_flat_geo_gp_nll`: proposed=0.9669, baseline=1.0467, absolute difference=-0.0798, p=0.5000. Relative percentages are omitted because NLL can be negative.
- `2pct_geo_vs_wrong_bcp_nll`: proposed=-0.1577, baseline=2.1925, absolute difference=-2.3502, p=0.0020. Relative percentages are omitted because NLL can be negative.
- `2pct_geo_vs_discrete_bcp_nll`: proposed=-0.1577, baseline=29.8953, absolute difference=-30.0530, p=0.0020. Relative percentages are omitted because NLL can be negative.
- `2pct_geo_vs_flat_geo_gp_nll`: proposed=-0.1577, baseline=0.3871, absolute difference=-0.5448, p=0.0020. Relative percentages are omitted because NLL can be negative.
- `random_geo_vs_wrong_bcp_width95`: proposed=2.5305, baseline=6.3184, improvement=60.0% CI [11.1, 76.5], p=0.1250.
- `random_geo_vs_discrete_bcp_width95`: proposed=2.5305, baseline=2.3246, improvement=-8.9% CI [-39.8, 17.9], p=0.6250.
- `random_geo_vs_flat_geo_gp_width95`: proposed=2.5305, baseline=2.3125, improvement=-9.4% CI [-37.5, 11.3], p=0.5000.
- `2pct_geo_vs_wrong_bcp_width95`: proposed=1.0800, baseline=11.4079, improvement=90.5% CI [89.4, 91.5], p=0.0020.
- `2pct_geo_vs_discrete_bcp_width95`: proposed=1.0800, baseline=0.6480, improvement=-66.6% CI [-95.4, -40.0], p=0.0039.
- `2pct_geo_vs_flat_geo_gp_width95`: proposed=1.0800, baseline=1.5692, improvement=31.2% CI [22.6, 40.1], p=0.0020.
- `random_geo_vs_wrong_bcp_selective_gain50`: proposed=0.4335, baseline=0.2781, improvement=55.9% CI [33.0, 83.0], p=0.0625.
- `random_geo_vs_discrete_bcp_selective_gain50`: proposed=0.4335, baseline=0.1045, improvement=314.8% CI [232.5, 497.7], p=0.0625.
- `random_geo_vs_flat_geo_gp_selective_gain50`: proposed=0.4335, baseline=0.2765, improvement=56.8% CI [44.0, 76.1], p=0.0625.
- `2pct_geo_vs_wrong_bcp_selective_gain50`: proposed=0.3298, baseline=0.5726, improvement=-42.4% CI [-52.0, -32.9], p=0.0020.
- `2pct_geo_vs_discrete_bcp_selective_gain50`: proposed=0.3298, baseline=0.4736, improvement=-30.4% CI [-41.3, -19.2], p=0.0039.
- `2pct_geo_vs_flat_geo_gp_selective_gain50`: proposed=0.3298, baseline=0.3650, improvement=-9.6% CI [-22.8, 4.2], p=0.2129.

## Final 2% efficiency

Predictive coefficient counts: geometry/wrong BCP 256, discrete BCP 680, flat operator GP 512. Mean end-to-end time per seed (including split calibration and initialization): geo_bcp_noard 8.19s, wrong_bcp 8.27s, discrete_bcp 7.22s, flat_geo_gp 0.49s.
