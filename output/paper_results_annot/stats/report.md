# StaLLM paper — statistical analysis

## RQ3 — human-oracle smell classification (run 1)

| Dataset | Slot | Prompt | P | R | F1 [95% CI] | κ |
|---|---|---|---|---|---|---|
| DACOS | AZ1 | binary_oracle_baseline | 0.659 | 0.901 | 0.761 [0.707, 0.812] | 0.02 |
| DACOS | AZ1 | conservative_evidence | 0.667 | 0.931 | 0.777 [0.722, 0.827] | 0.057 |
| DACOS | AZ1 | definition_driven | 0.668 | 0.985 | 0.796 [0.745, 0.841] | 0.073 |
| DACOS | AZ2 | binary_oracle_baseline | 0.705 | 0.473 | 0.566 [0.488, 0.639] | 0.084 |
| DACOS | AZ2 | conservative_evidence | 0.664 | 0.649 | 0.656 [0.582, 0.719] | 0.025 |
| DACOS | AZ2 | definition_driven | 0.686 | 0.817 | 0.746 [0.686, 0.797] | 0.117 |
| DACOS | AZ3 | binary_oracle_baseline | 0.797 | 0.359 | 0.495 [0.402, 0.571] | 0.148 |
| DACOS | AZ3 | conservative_evidence | 0.693 | 0.534 | 0.603 [0.522, 0.669] | 0.077 |
| DACOS | AZ3 | definition_driven | 0.734 | 0.443 | 0.552 [0.467, 0.632] | 0.117 |
| DACOS | AZ7 | binary_oracle_baseline | 0.698 | 0.458 | 0.553 [0.469, 0.624] | 0.07 |
| DACOS | AZ7 | conservative_evidence | 0.694 | 0.641 | 0.667 [0.601, 0.733] | 0.102 |
| DACOS | AZ7 | definition_driven | 0.698 | 0.687 | 0.692 [0.629, 0.754] | 0.121 |
| DACOS | AZ8 | binary_oracle_baseline | 0.672 | 0.893 | 0.767 [0.713, 0.817] | 0.079 |
| DACOS | AZ8 | conservative_evidence | 0.663 | 0.947 | 0.78 [0.726, 0.828] | 0.041 |
| DACOS | AZ8 | definition_driven | 0.655 | 0.84 | 0.736 [0.678, 0.786] | -0.001 |
| MLCQ | AZ1 | binary_oracle_baseline | 0.56 | 0.389 | 0.459 [0.294, 0.593] | 0.365 |
| MLCQ | AZ1 | conservative_evidence | 0.526 | 0.556 | 0.541 [0.394, 0.66] | 0.436 |
| MLCQ | AZ1 | definition_driven | 0.391 | 0.694 | 0.5 [0.374, 0.612] | 0.35 |
| MLCQ | AZ2 | binary_oracle_baseline | 0.5 | 0.139 | 0.217 [0.051, 0.368] | 0.151 |
| MLCQ | AZ2 | conservative_evidence | 0.625 | 0.278 | 0.385 [0.213, 0.538] | 0.308 |
| MLCQ | AZ2 | definition_driven | 0.583 | 0.389 | 0.467 [0.298, 0.615] | 0.377 |
| MLCQ | AZ3 | binary_oracle_baseline | 0.8 | 0.333 | 0.471 [0.275, 0.625] | 0.408 |
| MLCQ | AZ3 | conservative_evidence | 0.783 | 0.5 | 0.61 [0.439, 0.746] | 0.547 |
| MLCQ | AZ3 | definition_driven | 0.609 | 0.389 | 0.475 [0.304, 0.623] | 0.389 |
| MLCQ | AZ7 | binary_oracle_baseline | 0.8 | 0.111 | 0.195 [0.045, 0.372] | 0.158 |
| MLCQ | AZ7 | conservative_evidence | 0.667 | 0.111 | 0.19 [0.048, 0.356] | 0.147 |
| MLCQ | AZ7 | definition_driven | 0.45 | 0.25 | 0.321 [0.146, 0.48] | 0.221 |
| MLCQ | AZ8 | binary_oracle_baseline | 0.696 | 0.444 | 0.542 [0.367, 0.678] | 0.468 |
| MLCQ | AZ8 | conservative_evidence | 0.6 | 0.667 | 0.632 [0.492, 0.75] | 0.545 |
| MLCQ | AZ8 | definition_driven | 0.5 | 0.583 | 0.538 [0.394, 0.659] | 0.427 |

**McNemar (Holm-adjusted within factor):** 0/90 pairwise comparisons significant at α=0.05.


**Run-to-run stability (temperature 0):**

| Slot | Prompt | Dataset | runs | identical predictions | F1 mean ± sd |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | DACOS | 3 | 96.0% | 0.758 ± 0.004 |
| AZ1 | binary_oracle_baseline | MLCQ | 3 | 97.0% | 0.448 ± 0.031 |
| AZ1 | conservative_evidence | DACOS | 3 | 98.0% | 0.774 ± 0.005 |
| AZ1 | conservative_evidence | MLCQ | 3 | 95.5% | 0.534 ± 0.038 |
| AZ1 | definition_driven | DACOS | 3 | 99.0% | 0.795 ± 0.001 |
| AZ1 | definition_driven | MLCQ | 3 | 94.0% | 0.5 ± 0.014 |
| AZ7 | binary_oracle_baseline | DACOS | 3 | 94.0% | 0.559 ± 0.005 |
| AZ7 | binary_oracle_baseline | MLCQ | 3 | 99.5% | 0.194 ± 0.003 |
| AZ7 | conservative_evidence | DACOS | 3 | 90.5% | 0.663 ± 0.009 |
| AZ7 | conservative_evidence | MLCQ | 3 | 99.0% | 0.175 ± 0.027 |
| AZ7 | definition_driven | DACOS | 3 | 95.0% | 0.691 ± 0.004 |
| AZ7 | definition_driven | MLCQ | 3 | 99.5% | 0.32 ± 0.003 |
| AZ8 | binary_oracle_baseline | DACOS | 3 | 96.5% | 0.763 ± 0.01 |
| AZ8 | binary_oracle_baseline | MLCQ | 3 | 96.0% | 0.542 ± 0.022 |
| AZ8 | conservative_evidence | DACOS | 3 | 99.5% | 0.781 ± 0.002 |
| AZ8 | conservative_evidence | MLCQ | 3 | 98.0% | 0.64 ± 0.015 |
| AZ8 | definition_driven | DACOS | 3 | 97.0% | 0.74 ± 0.006 |
| AZ8 | definition_driven | MLCQ | 3 | 97.0% | 0.551 ± 0.015 |

## RQ3 — oracle substitution (human labels vs. metric heuristic, same samples)

**DACOS** (15 configurations): Spearman ρ(κ human, κ heuristic) = 0.489 (p=0.064); best vs human: AZ3/binary_oracle_baseline, best vs heuristic: AZ3/definition_driven

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.761 | 0.02 | 0.566 | 0.107 |
| AZ1 | conservative_evidence | 0.777 | 0.057 | 0.565 | 0.099 |
| AZ1 | definition_driven | 0.796 | 0.073 | 0.543 | 0.04 |
| AZ2 | binary_oracle_baseline | 0.566 | 0.084 | 0.575 | 0.296 |
| AZ2 | conservative_evidence | 0.656 | 0.025 | 0.59 | 0.24 |
| AZ2 | definition_driven | 0.746 | 0.117 | 0.561 | 0.136 |
| AZ3 | binary_oracle_baseline | 0.495 | 0.148 | 0.656 | 0.492 |
| AZ3 | conservative_evidence | 0.603 | 0.077 | 0.636 | 0.372 |
| AZ3 | definition_driven | 0.552 | 0.117 | 0.702 | 0.522 |
| AZ7 | binary_oracle_baseline | 0.553 | 0.07 | 0.43 | 0.063 |
| AZ7 | conservative_evidence | 0.667 | 0.102 | 0.497 | 0.084 |
| AZ7 | definition_driven | 0.692 | 0.121 | 0.537 | 0.14 |
| AZ8 | binary_oracle_baseline | 0.767 | 0.079 | 0.528 | 0.039 |
| AZ8 | conservative_evidence | 0.78 | 0.041 | 0.556 | 0.075 |
| AZ8 | definition_driven | 0.736 | -0.001 | 0.533 | 0.059 |

**MLCQ** (15 configurations): Spearman ρ(κ human, κ heuristic) = 0.701 (p=0.004); best vs human: AZ3/conservative_evidence, best vs heuristic: AZ3/binary_oracle_baseline

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.459 | 0.365 | 0.35 | 0.283 |
| AZ1 | conservative_evidence | 0.541 | 0.436 | 0.415 | 0.345 |
| AZ1 | definition_driven | 0.5 | 0.35 | 0.329 | 0.236 |
| AZ2 | binary_oracle_baseline | 0.217 | 0.151 | 0.32 | 0.277 |
| AZ2 | conservative_evidence | 0.385 | 0.308 | 0.452 | 0.406 |
| AZ2 | definition_driven | 0.467 | 0.377 | 0.462 | 0.407 |
| AZ3 | binary_oracle_baseline | 0.471 | 0.408 | 0.467 | 0.423 |
| AZ3 | conservative_evidence | 0.61 | 0.547 | 0.474 | 0.421 |
| AZ3 | definition_driven | 0.475 | 0.389 | 0.421 | 0.363 |
| AZ7 | binary_oracle_baseline | 0.195 | 0.158 | 0.2 | 0.169 |
| AZ7 | conservative_evidence | 0.19 | 0.147 | 0.19 | 0.154 |
| AZ7 | definition_driven | 0.321 | 0.221 | 0.229 | 0.156 |
| AZ8 | binary_oracle_baseline | 0.542 | 0.468 | 0.474 | 0.421 |
| AZ8 | conservative_evidence | 0.632 | 0.545 | 0.4 | 0.327 |
| AZ8 | definition_driven | 0.538 | 0.427 | 0.421 | 0.349 |

## RQ3 — non-LLM baselines

| Dataset | Method | P | R | F1 | κ |
|---|---|---|---|---|---|
| DACOS | always_present | 0.655 | 1.0 | 0.7915 | 0.0 |
| DACOS | metric_heuristic | 0.7917 | 0.4351 | 0.5616 | 0.1811 |
| MLCQ | always_present | 0.18 | 1.0 | 0.3051 | 0.0 |
| MLCQ | metric_heuristic | 0.8 | 0.3333 | 0.4706 | 0.4079 |
| DACOS_heldout | always_present | 0.655 | 1.0 | 0.7915 | 0.0 |
| DACOS_heldout | metric_heuristic | 0.7458 | 0.3359 | 0.4632 | 0.095 |

## RQ5 — held-out DACOS test split (disjoint from the selection split)

| Source | Prompt | P | R | F1 [95% CI] | κ | McNemar p vs generated |
|---|---|---|---|---|---|---|
| RQ3_manual | binary_oracle_baseline | 0.686 | 0.916 | 0.784 [0.732, 0.831] | 0.14 | 0.8145 |
| RQ5_generated | calibrated_confidence | 0.68 | 0.908 | 0.778 [0.728, 0.826] | 0.114 | - |
| RQ3_manual | conservative_evidence | 0.667 | 0.962 | 0.787 [0.736, 0.834] | 0.061 | 1.0 |
| RQ3_manual | definition_driven | 0.66 | 0.977 | 0.788 [0.737, 0.833] | 0.026 | 1.0 |
