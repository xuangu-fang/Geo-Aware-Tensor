# Aggregated POC results

Metrics are evaluated only on unobserved entries. Values are mean ± sample std across seeds.

## active_matter

| mask | obs | model | NRMSE ↓ | RelL2 ↓ | coverage95 | params |
|---|---:|---|---:|---:|---:|---:|
| random | 0.5% | bayesian_spectral_tensor | 1.5671 ± 0.0865 | 0.0053 ± 0.0003 | 0.5813 ± 0.0312 | 384 |
| random | 0.5% | cp | 1.0861 ± 0.0196 | 0.0037 ± 0.0001 | — | 1,360 |
| random | 0.5% | geo_nft | 1.2337 ± 0.0516 | 0.0042 ± 0.0002 | — | 6,155 |
| random | 0.5% | inr | 1.2962 ± 0.0984 | 0.0044 ± 0.0003 | — | 8,641 |
| random | 0.5% | neural_cp | 1.6615 ± 0.1664 | 0.0056 ± 0.0006 | — | 14,432 |
| random | 0.5% | spectral_cp | 0.6755 ± 0.1808 | 0.0023 ± 0.0006 | — | 496 |
| random | 1% | bayesian_spectral_tensor | 1.1916 ± 0.0550 | 0.0040 ± 0.0002 | 0.7151 ± 0.0241 | 384 |
| random | 1% | cp | 1.1184 ± 0.0317 | 0.0038 ± 0.0001 | — | 1,360 |
| random | 1% | geo_nft | 0.5948 ± 0.0759 | 0.0020 ± 0.0003 | — | 6,155 |
| random | 1% | inr | 1.1690 ± 0.0664 | 0.0039 ± 0.0002 | — | 8,641 |
| random | 1% | neural_cp | 1.0546 ± 0.3597 | 0.0036 ± 0.0012 | — | 14,432 |
| random | 1% | spectral_cp | 0.4161 ± 0.0342 | 0.0014 ± 0.0001 | — | 496 |
| random | 5% | bayesian_spectral_tensor | 1.0136 ± 0.0069 | 0.0034 ± 0.0000 | 0.8377 ± 0.0068 | 384 |
| random | 5% | cp | 0.6871 ± 0.3650 | 0.0023 ± 0.0012 | — | 1,360 |
| random | 5% | geo_nft | 0.1218 ± 0.0029 | 0.0004 ± 0.0000 | — | 6,155 |
| random | 5% | inr | 0.5601 ± 0.0504 | 0.0019 ± 0.0002 | — | 8,641 |
| random | 5% | neural_cp | 0.8209 ± 0.1674 | 0.0028 ± 0.0006 | — | 14,432 |
| random | 5% | spectral_cp | 0.2452 ± 0.0049 | 0.0008 ± 0.0000 | — | 496 |

- Best at random, 0.5%: `spectral_cp` (NRMSE 0.6755).
- Best at random, 1%: `spectral_cp` (NRMSE 0.4161).
- Best at random, 5%: `geo_nft` (NRMSE 0.1218).

## realpde_cylinder

| mask | obs | model | NRMSE ↓ | RelL2 ↓ | coverage95 | params |
|---|---:|---|---:|---:|---:|---:|
| random | 1% | bayesian_spectral_tensor | 0.6228 ± 0.0135 | 0.0639 ± 0.0014 | 0.6745 ± 0.0122 | 384 |
| random | 1% | cp | 1.2845 ± 0.0587 | 0.1317 ± 0.0060 | — | 1,160 |
| random | 1% | geo_nft | 0.8428 ± 0.1284 | 0.0864 ± 0.0131 | — | 5,435 |
| random | 1% | inr | 0.7297 ± 0.0343 | 0.0748 ± 0.0035 | — | 8,641 |
| random | 1% | neural_cp | 0.4867 ± 0.0133 | 0.0499 ± 0.0014 | — | 14,432 |
| random | 1% | spectral_cp | 0.5313 ± 0.0134 | 0.0545 ± 0.0014 | — | 416 |
| random | 5% | bayesian_spectral_tensor | 0.5024 ± 0.0007 | 0.0515 ± 0.0001 | 0.7077 ± 0.0011 | 384 |
| random | 5% | cp | 0.4649 ± 0.0198 | 0.0477 ± 0.0021 | — | 1,160 |
| random | 5% | geo_nft | 0.3788 ± 0.0249 | 0.0388 ± 0.0026 | — | 5,435 |
| random | 5% | inr | 0.5299 ± 0.0164 | 0.0543 ± 0.0016 | — | 8,641 |
| random | 5% | neural_cp | 0.4152 ± 0.0020 | 0.0426 ± 0.0002 | — | 14,432 |
| random | 5% | spectral_cp | 0.4497 ± 0.0010 | 0.0461 ± 0.0001 | — | 416 |

- Best at random, 1%: `neural_cp` (NRMSE 0.4867).
- Best at random, 5%: `geo_nft` (NRMSE 0.3788).

## synthetic_boundary

| mask | obs | model | NRMSE ↓ | RelL2 ↓ | coverage95 | params |
|---|---:|---|---:|---:|---:|---:|
| periodic_gap | 5% | bayesian_spectral_tensor | 0.4904 ± 0.0293 | 0.2881 ± 0.0177 | 0.8456 ± 0.0090 | 168 |
| periodic_gap | 5% | geo_nft | 0.4783 ± 0.0718 | 0.2810 ± 0.0426 | — | 1,378 |
| periodic_gap | 5% | inr | 0.7120 ± 0.0367 | 0.4183 ± 0.0209 | — | 8,577 |
| periodic_gap | 5% | neural_cp | 0.3865 ± 0.0300 | 0.2271 ± 0.0174 | — | 2,508 |
| periodic_gap | 5% | spectral_cp | 0.5960 ± 0.0650 | 0.3501 ± 0.0380 | — | 120 |
| periodic_gap | 5% | wrong_spectral_cp | 0.6299 ± 0.0594 | 0.3701 ± 0.0351 | — | 148 |
| random | 0.5% | bayesian_spectral_tensor | 0.7979 ± 0.0863 | 0.4691 ± 0.0512 | 0.8868 ± 0.0560 | 168 |
| random | 0.5% | cp | 1.0367 ± 0.0409 | 0.6094 ± 0.0235 | — | 516 |
| random | 0.5% | geo_nft | 1.0766 ± 0.0977 | 0.6329 ± 0.0579 | — | 1,378 |
| random | 0.5% | inr | 0.9944 ± 0.1975 | 0.5844 ± 0.1154 | — | 8,577 |
| random | 0.5% | neural_cp | 1.1364 ± 0.1857 | 0.6681 ± 0.1096 | — | 2,508 |
| random | 0.5% | spectral_cp | 0.9559 ± 0.2951 | 0.5620 ± 0.1742 | — | 120 |
| random | 0.5% | wrong_spectral_cp | 1.0098 ± 0.0475 | 0.5936 ± 0.0283 | — | 148 |
| random | 1% | bayesian_spectral_tensor | 0.7939 ± 0.0944 | 0.4668 ± 0.0559 | 0.8967 ± 0.0129 | 168 |
| random | 1% | cp | 1.0318 ± 0.0178 | 0.6066 ± 0.0107 | — | 516 |
| random | 1% | geo_nft | 0.8225 ± 0.0404 | 0.4836 ± 0.0241 | — | 1,378 |
| random | 1% | inr | 0.8169 ± 0.0779 | 0.4803 ± 0.0455 | — | 8,577 |
| random | 1% | neural_cp | 0.9131 ± 0.2553 | 0.5369 ± 0.1503 | — | 2,508 |
| random | 1% | spectral_cp | 0.8713 ± 0.1373 | 0.5122 ± 0.0802 | — | 120 |
| random | 1% | wrong_spectral_cp | 0.9093 ± 0.0606 | 0.5346 ± 0.0353 | — | 148 |
| random | 5% | bayesian_spectral_tensor | 0.5190 ± 0.0440 | 0.3052 ± 0.0254 | 0.8164 ± 0.0236 | 168 |
| random | 5% | cp | 1.1122 ± 0.0662 | 0.6542 ± 0.0382 | — | 516 |
| random | 5% | geo_nft | 0.2212 ± 0.0108 | 0.1301 ± 0.0066 | — | 1,378 |
| random | 5% | inr | 0.4777 ± 0.0427 | 0.2810 ± 0.0247 | — | 8,577 |
| random | 5% | neural_cp | 0.2561 ± 0.0040 | 0.1507 ± 0.0026 | — | 2,508 |
| random | 5% | spectral_cp | 0.4917 ± 0.0399 | 0.2892 ± 0.0240 | — | 120 |
| random | 5% | wrong_spectral_cp | 0.5206 ± 0.0328 | 0.3063 ± 0.0198 | — | 148 |

- Best at periodic_gap, 5%: `neural_cp` (NRMSE 0.3865).
- Best at random, 0.5%: `bayesian_spectral_tensor` (NRMSE 0.7979).
- Best at random, 1%: `bayesian_spectral_tensor` (NRMSE 0.7939).
- Best at random, 5%: `geo_nft` (NRMSE 0.2212).

## synthetic_wave

| mask | obs | model | NRMSE ↓ | RelL2 ↓ | coverage95 | params |
|---|---:|---|---:|---:|---:|---:|
| random | 0.5% | bayesian_spectral_tensor | 1.0705 ± 0.0753 | 1.0705 ± 0.0753 | 0.8267 ± 0.0240 | 192 |
| random | 0.5% | cp | 1.1597 ± 0.2753 | 1.1597 ± 0.2753 | — | 420 |
| random | 0.5% | geo_nft | 1.3707 ± 0.4809 | 1.3707 ± 0.4809 | — | 2,263 |
| random | 0.5% | inr | 0.7981 ± 0.0371 | 0.7981 ± 0.0371 | — | 8,641 |
| random | 0.5% | neural_cp | 0.6494 ± 0.0582 | 0.6494 ± 0.0582 | — | 3,760 |
| random | 0.5% | spectral_cp | 1.0381 ± 0.4215 | 1.0381 ± 0.4215 | — | 200 |
| random | 0.5% | wrong_spectral_cp | 0.6479 ± 0.0019 | 0.6479 ± 0.0019 | — | 228 |
| random | 1% | bayesian_spectral_tensor | 0.7308 ± 0.0635 | 0.7308 ± 0.0635 | 0.5903 ± 0.0270 | 192 |
| random | 1% | cp | 1.0023 ± 0.0007 | 1.0023 ± 0.0007 | — | 420 |
| random | 1% | geo_nft | 1.3716 ± 0.9494 | 1.3716 ± 0.9494 | — | 2,263 |
| random | 1% | inr | 0.7385 ± 0.0130 | 0.7385 ± 0.0130 | — | 8,641 |
| random | 1% | neural_cp | 0.5096 ± 0.0242 | 0.5097 ± 0.0242 | — | 3,760 |
| random | 1% | spectral_cp | 0.4790 ± 0.0118 | 0.4790 ± 0.0118 | — | 200 |
| random | 1% | wrong_spectral_cp | 0.5541 ± 0.0315 | 0.5541 ± 0.0315 | — | 228 |
| random | 5% | bayesian_spectral_tensor | 0.4474 ± 0.0033 | 0.4475 ± 0.0033 | 0.6107 ± 0.0061 | 192 |
| random | 5% | cp | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | — | 420 |
| random | 5% | geo_nft | 0.2602 ± 0.0049 | 0.2602 ± 0.0049 | — | 2,263 |
| random | 5% | inr | 0.4065 ± 0.0148 | 0.4066 ± 0.0148 | — | 8,641 |
| random | 5% | neural_cp | 0.4260 ± 0.0009 | 0.4260 ± 0.0009 | — | 3,760 |
| random | 5% | spectral_cp | 0.4264 ± 0.0006 | 0.4264 ± 0.0006 | — | 200 |
| random | 5% | wrong_spectral_cp | 0.4783 ± 0.0133 | 0.4784 ± 0.0133 | — | 228 |

- Best at random, 0.5%: `wrong_spectral_cp` (NRMSE 0.6479).
- Best at random, 1%: `spectral_cp` (NRMSE 0.4790).
- Best at random, 5%: `geo_nft` (NRMSE 0.2602).
