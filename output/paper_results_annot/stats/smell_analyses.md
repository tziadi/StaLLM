# Smell-classification analyses (RQ3, RQ5)

Computed from stored outputs, no model call.

## 1. Oracle shift per configuration: Δκ = κ(metric oracle) − κ(human labels)

Paired bootstrap, 1000 resamples, seed 42, majority vote over the runs of each configuration.

| Dataset | Oracle | Configuration | κ human | κ metric | Δκ [95% CI] |
|---|---|---|---|---|---|
| DACOS | heuristic | AZ1/binary_oracle_baseline | 0.02 | 0.107 | 0.087 [-0.02, 0.2] |
| DACOS | heuristic | AZ1/conservative_evidence | 0.057 | 0.099 | 0.042 [-0.076, 0.155] |
| DACOS | heuristic | AZ1/definition_driven | 0.073 | 0.04 | -0.033 [-0.1, 0.036] |
| DACOS | heuristic | AZ2/binary_oracle_baseline | 0.084 | 0.296 | 0.212 [0.055, 0.374] |
| DACOS | heuristic | AZ2/conservative_evidence | 0.025 | 0.24 | 0.214 [0.05, 0.373] |
| DACOS | heuristic | AZ2/definition_driven | 0.117 | 0.136 | 0.019 [-0.113, 0.149] |
| DACOS | heuristic | AZ3/binary_oracle_baseline | 0.148 | 0.492 | 0.343 [0.191, 0.484] |
| DACOS | heuristic | AZ3/conservative_evidence | 0.077 | 0.372 | 0.295 [0.13, 0.462] |
| DACOS | heuristic | AZ3/definition_driven | 0.117 | 0.522 | 0.404 [0.253, 0.554] |
| DACOS | heuristic | AZ7/binary_oracle_baseline | 0.07 | 0.063 | -0.007 [-0.171, 0.156] |
| DACOS | heuristic | AZ7/conservative_evidence | 0.102 | 0.084 | -0.018 [-0.183, 0.139] |
| DACOS | heuristic | AZ7/definition_driven | 0.121 | 0.14 | 0.019 [-0.148, 0.187] |
| DACOS | heuristic | AZ8/binary_oracle_baseline | 0.079 | 0.039 | -0.04 [-0.156, 0.083] |
| DACOS | heuristic | AZ8/conservative_evidence | 0.041 | 0.075 | 0.034 [-0.059, 0.133] |
| DACOS | heuristic | AZ8/definition_driven | -0.001 | 0.059 | 0.06 [-0.067, 0.18] |
| MLCQ | heuristic | AZ1/binary_oracle_baseline | 0.365 | 0.283 | -0.083 [-0.3, 0.129] |
| MLCQ | heuristic | AZ1/conservative_evidence | 0.436 | 0.345 | -0.092 [-0.292, 0.096] |
| MLCQ | heuristic | AZ1/definition_driven | 0.35 | 0.236 | -0.114 [-0.257, 0.025] |
| MLCQ | heuristic | AZ2/binary_oracle_baseline | 0.151 | 0.277 | 0.126 [-0.091, 0.361] |
| MLCQ | heuristic | AZ2/conservative_evidence | 0.308 | 0.406 | 0.098 [-0.121, 0.307] |
| MLCQ | heuristic | AZ2/definition_driven | 0.377 | 0.407 | 0.03 [-0.192, 0.265] |
| MLCQ | heuristic | AZ3/binary_oracle_baseline | 0.408 | 0.423 | 0.016 [-0.256, 0.246] |
| MLCQ | heuristic | AZ3/conservative_evidence | 0.547 | 0.421 | -0.125 [-0.361, 0.121] |
| MLCQ | heuristic | AZ3/definition_driven | 0.389 | 0.363 | -0.026 [-0.24, 0.201] |
| MLCQ | heuristic | AZ7/binary_oracle_baseline | 0.158 | 0.169 | 0.011 [-0.241, 0.292] |
| MLCQ | heuristic | AZ7/conservative_evidence | 0.147 | 0.154 | 0.008 [-0.241, 0.277] |
| MLCQ | heuristic | AZ7/definition_driven | 0.221 | 0.156 | -0.065 [-0.277, 0.128] |
| MLCQ | heuristic | AZ8/binary_oracle_baseline | 0.468 | 0.421 | -0.047 [-0.265, 0.184] |
| MLCQ | heuristic | AZ8/conservative_evidence | 0.545 | 0.327 | -0.219 [-0.412, -0.038] |
| MLCQ | heuristic | AZ8/definition_driven | 0.427 | 0.349 | -0.078 [-0.259, 0.113] |
| DACOS | PMD | AZ1/binary_oracle_baseline | 0.02 | 0.045 | 0.025 [-0.082, 0.133] |
| DACOS | PMD | AZ1/conservative_evidence | 0.057 | 0.033 | -0.024 [-0.133, 0.084] |
| DACOS | PMD | AZ1/definition_driven | 0.073 | 0.006 | -0.067 [-0.136, 0.002] |
| DACOS | PMD | AZ2/binary_oracle_baseline | 0.084 | 0.128 | 0.044 [-0.121, 0.198] |
| DACOS | PMD | AZ2/conservative_evidence | 0.025 | 0.099 | 0.073 [-0.075, 0.227] |
| DACOS | PMD | AZ2/definition_driven | 0.117 | 0.106 | -0.011 [-0.16, 0.14] |
| DACOS | PMD | AZ3/binary_oracle_baseline | 0.148 | 0.209 | 0.06 [-0.106, 0.221] |
| DACOS | PMD | AZ3/conservative_evidence | 0.077 | 0.225 | 0.147 [-0.004, 0.312] |
| DACOS | PMD | AZ3/definition_driven | 0.117 | 0.315 | 0.198 [0.04, 0.358] |
| DACOS | PMD | AZ7/binary_oracle_baseline | 0.07 | -0.036 | -0.106 [-0.27, 0.051] |
| DACOS | PMD | AZ7/conservative_evidence | 0.102 | 0.021 | -0.08 [-0.236, 0.07] |
| DACOS | PMD | AZ7/definition_driven | 0.121 | 0.078 | -0.043 [-0.206, 0.12] |
| DACOS | PMD | AZ8/binary_oracle_baseline | 0.079 | 0.032 | -0.046 [-0.169, 0.073] |
| DACOS | PMD | AZ8/conservative_evidence | 0.041 | 0.022 | -0.019 [-0.111, 0.073] |
| DACOS | PMD | AZ8/definition_driven | -0.001 | 0.065 | 0.066 [-0.065, 0.187] |

- DACOS / PMD: 1 of 15 configurations have Δκ > 0 with a CI excluding 0, 0 have Δκ < 0; median Δκ -0.011
- DACOS / heuristic: 5 of 15 configurations have Δκ > 0 with a CI excluding 0, 0 have Δκ < 0; median Δκ 0.042
- MLCQ / heuristic: 0 of 15 configurations have Δκ > 0 with a CI excluding 0, 1 have Δκ < 0; median Δκ -0.047

## 2. Paired bootstrap of Δκ between prompts and configurations

| Comparison | Split | A | B | Δκ (A − B) [95% CI] |
|---|---|---|---|---|
| RQ5 selected vs manual | DACOS selection | calibrated_confidence | binary_oracle_baseline | 0.071 [-0.017, 0.163] |
| RQ5 selected vs manual | DACOS selection | calibrated_confidence | conservative_evidence | 0.053 [-0.034, 0.148] |
| RQ5 selected vs manual | DACOS selection | calibrated_confidence | definition_driven | 0.045 [-0.058, 0.163] |
| RQ5 selected vs manual | DACOS held-out | calibrated_confidence | binary_oracle_baseline | -0.026 [-0.134, 0.082] |
| RQ5 selected vs manual | DACOS held-out | calibrated_confidence | conservative_evidence | 0.053 [-0.055, 0.164] |
| RQ5 selected vs manual | DACOS held-out | calibrated_confidence | definition_driven | 0.088 [-0.04, 0.214] |
| RQ5 selected vs manual | MLCQ | calibrated_confidence | binary_oracle_baseline | 0.023 [-0.101, 0.152] |
| RQ5 selected vs manual | MLCQ | calibrated_confidence | conservative_evidence | 0.036 [-0.111, 0.194] |
| RQ5 selected vs manual | MLCQ | calibrated_confidence | definition_driven | 0.08 [-0.059, 0.209] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ1/binary_oracle_baseline | 0.128 [-0.006, 0.261] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ1/conservative_evidence | 0.091 [-0.048, 0.232] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ1/definition_driven | 0.076 [-0.049, 0.196] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ2/binary_oracle_baseline | 0.064 [-0.039, 0.174] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ2/conservative_evidence | 0.123 [-0.011, 0.25] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ2/definition_driven | 0.032 [-0.116, 0.181] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ3/conservative_evidence | 0.071 [-0.041, 0.184] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ3/definition_driven | 0.031 [-0.052, 0.119] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ7/binary_oracle_baseline | 0.078 [-0.055, 0.21] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ7/conservative_evidence | 0.047 [-0.106, 0.206] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ7/definition_driven | 0.027 [-0.107, 0.17] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ8/binary_oracle_baseline | 0.07 [-0.07, 0.214] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ8/conservative_evidence | 0.107 [-0.019, 0.232] |
| RQ3 best vs other | DACOS | AZ3/binary_oracle_baseline | AZ8/definition_driven | 0.149 [0.009, 0.295] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ1/binary_oracle_baseline | 0.181 [0.014, 0.358] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ1/conservative_evidence | 0.11 [-0.07, 0.286] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ1/definition_driven | 0.196 [0.016, 0.357] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ2/binary_oracle_baseline | 0.396 [0.232, 0.557] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ2/conservative_evidence | 0.239 [0.04, 0.46] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ2/definition_driven | 0.17 [-0.035, 0.362] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ3/binary_oracle_baseline | 0.139 [-0.006, 0.302] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ3/definition_driven | 0.158 [0.009, 0.305] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ7/binary_oracle_baseline | 0.388 [0.21, 0.573] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ7/conservative_evidence | 0.4 [0.224, 0.561] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ7/definition_driven | 0.325 [0.161, 0.493] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ8/binary_oracle_baseline | 0.079 [-0.098, 0.254] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ8/conservative_evidence | 0.001 [-0.18, 0.188] |
| RQ3 best vs other | MLCQ | AZ3/conservative_evidence | AZ8/definition_driven | 0.119 [-0.053, 0.289] |

- RQ5 selected vs manual: 0 of 9 intervals exclude 0
- RQ3 best vs other: 9 of 28 intervals exclude 0

## 3. RQ5 — repeated stratified splits of the selection sample, selected by κ

1000 splits into two halves; the winner by κ on one half is evaluated on the other. The halves resample the same 200 samples, so they are not independent experiments. Null: the predictions of the eight prompts are permuted among prompts within each sample (100 permutations, 100 splits each), so that the prompts are exchangeable and the dependence between halves is kept.

- prompts: 8
- repeats: 1000
- winner_stays_best: 0.063
- chance_1_over_k: 0.125
- null_exchangeable_median: 0.065
- null_exchangeable_ci: [0.0, 0.64]
- optimism_mean: 0.089
- null_optimism_median: 0.084
- selection_gain_kappa: 0.045
- gpt4o_run_ranges: binary_oracle_baseline [0.028, 0.002, 0.028] range 0.026; conservative_evidence [0.046, 0.057, 0.057] range 0.011; definition_driven [0.054, 0.073, 0.054] range 0.018

## 4. Metric heuristic vs LLM prompts, selection and held-out DACOS

| Split | Method | κ | F1 |
|---|---|---|---|
| DACOS | metric heuristic | 0.1811 | 0.5616 |
| DACOS_heldout | metric heuristic | 0.095 | 0.4632 |
| DACOS_heldout | AZ1_binary_oracle_baseline | 0.14 | 0.784 |
| DACOS_heldout | AZ1_calibrated_confidence | 0.114 | 0.778 |
| DACOS_heldout | AZ1_conservative_evidence | 0.061 | 0.787 |
| DACOS_heldout | AZ1_definition_driven | 0.026 | 0.788 |

## 5. MLCQ — row labels vs the reviewers' majority

Majority over all reviewers of the (fragment, smell) pair in the MLCQ export; ties keep the row label.

- labels_changed: 10
- ties_kept_row_label: 3
- positives_row_label: 36
- positives_majority: 26
- rho_rankings: 0.921
- best_row_label: AZ3/conservative_evidence
- best_majority: AZ3/conservative_evidence

| Configuration | κ (row label) | κ (majority) |
|---|---|---|
| AZ1/binary_oracle_baseline | 0.365 | 0.348 |
| AZ1/conservative_evidence | 0.436 | 0.372 |
| AZ1/definition_driven | 0.35 | 0.318 |
| AZ2/binary_oracle_baseline | 0.151 | 0.222 |
| AZ2/conservative_evidence | 0.308 | 0.366 |
| AZ2/definition_driven | 0.377 | 0.406 |
| AZ3/binary_oracle_baseline | 0.408 | 0.488 |
| AZ3/conservative_evidence | 0.547 | 0.558 |
| AZ3/definition_driven | 0.389 | 0.419 |
| AZ7/binary_oracle_baseline | 0.158 | 0.226 |
| AZ7/conservative_evidence | 0.147 | 0.212 |
| AZ7/definition_driven | 0.221 | 0.265 |
| AZ8/binary_oracle_baseline | 0.468 | 0.465 |
| AZ8/conservative_evidence | 0.545 | 0.46 |
| AZ8/definition_driven | 0.427 | 0.439 |

## 6. DACOS inter-annotator agreement (samples selected for the smell, two or more annotators)

| Smell | Units | Agreement | Krippendorff's α | Two-annotator ties |
|---|---|---|---|---|
| complex_method | 2275 | 0.82 | 0.599 | 407 |
| long_parameter_list | 1557 | 0.764 | 0.513 | 366 |

