# StaLLM paper — statistical analysis

## RQ3 — human-oracle smell classification (run 1)

| Dataset | Slot | Prompt | P | R | F1 [95% CI] | κ |
|---|---|---|---|---|---|---|
| DACOS | AZ1 | binary_oracle_baseline | 0.661 | 0.893 | 0.76 [0.705, 0.81] | 0.028 |
| DACOS | AZ1 | conservative_evidence | 0.665 | 0.908 | 0.768 [0.714, 0.817] | 0.046 |
| DACOS | AZ1 | definition_driven | 0.665 | 0.985 | 0.794 [0.744, 0.839] | 0.054 |
| DACOS | AZ2 | binary_oracle_baseline | 0.705 | 0.473 | 0.566 [0.488, 0.639] | 0.084 |
| DACOS | AZ2 | conservative_evidence | 0.664 | 0.649 | 0.656 [0.582, 0.719] | 0.025 |
| DACOS | AZ2 | definition_driven | 0.686 | 0.817 | 0.746 [0.686, 0.797] | 0.117 |
| DACOS | AZ3 | binary_oracle_baseline | 0.797 | 0.359 | 0.495 [0.402, 0.571] | 0.148 |
| DACOS | AZ3 | conservative_evidence | 0.693 | 0.534 | 0.603 [0.522, 0.669] | 0.077 |
| DACOS | AZ3 | definition_driven | 0.734 | 0.443 | 0.552 [0.467, 0.632] | 0.117 |
| DACOS | AZ7 | binary_oracle_baseline | 0.698 | 0.458 | 0.553 [0.471, 0.627] | 0.07 |
| DACOS | AZ7 | conservative_evidence | 0.697 | 0.649 | 0.672 [0.608, 0.737] | 0.109 |
| DACOS | AZ7 | definition_driven | 0.695 | 0.695 | 0.695 [0.631, 0.756] | 0.115 |
| DACOS | AZ8 | binary_oracle_baseline | 0.676 | 0.893 | 0.77 [0.717, 0.818] | 0.095 |
| DACOS | AZ8 | conservative_evidence | 0.663 | 0.947 | 0.78 [0.726, 0.828] | 0.041 |
| DACOS | AZ8 | definition_driven | 0.659 | 0.84 | 0.738 [0.678, 0.789] | 0.016 |
| MLCQ | AZ1 | binary_oracle_baseline | 0.577 | 0.417 | 0.484 [0.32, 0.623] | 0.392 |
| MLCQ | AZ1 | conservative_evidence | 0.486 | 0.5 | 0.493 [0.353, 0.624] | 0.38 |
| MLCQ | AZ1 | definition_driven | 0.39 | 0.639 | 0.484 [0.353, 0.602] | 0.336 |
| MLCQ | AZ2 | binary_oracle_baseline | 0.5 | 0.139 | 0.217 [0.051, 0.368] | 0.151 |
| MLCQ | AZ2 | conservative_evidence | 0.625 | 0.278 | 0.385 [0.213, 0.538] | 0.308 |
| MLCQ | AZ2 | definition_driven | 0.583 | 0.389 | 0.467 [0.298, 0.615] | 0.377 |
| MLCQ | AZ3 | binary_oracle_baseline | 0.8 | 0.333 | 0.471 [0.275, 0.625] | 0.408 |
| MLCQ | AZ3 | conservative_evidence | 0.783 | 0.5 | 0.61 [0.439, 0.746] | 0.547 |
| MLCQ | AZ3 | definition_driven | 0.609 | 0.389 | 0.475 [0.304, 0.623] | 0.389 |
| MLCQ | AZ7 | binary_oracle_baseline | 0.8 | 0.111 | 0.195 [0.045, 0.372] | 0.158 |
| MLCQ | AZ7 | conservative_evidence | 0.667 | 0.111 | 0.19 [0.048, 0.356] | 0.147 |
| MLCQ | AZ7 | definition_driven | 0.429 | 0.25 | 0.316 [0.145, 0.471] | 0.211 |
| MLCQ | AZ8 | binary_oracle_baseline | 0.708 | 0.472 | 0.567 [0.4, 0.7] | 0.494 |
| MLCQ | AZ8 | conservative_evidence | 0.6 | 0.667 | 0.632 [0.492, 0.75] | 0.545 |
| MLCQ | AZ8 | definition_driven | 0.512 | 0.583 | 0.545 [0.4, 0.667] | 0.438 |

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

**DACOS** (15 configurations): Spearman ρ(κ human, κ heuristic) = 0.511 (p=0.052); best vs human: AZ3/binary_oracle_baseline, best vs heuristic: AZ3/definition_driven

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.76 | 0.028 | 0.562 | 0.103 |
| AZ1 | conservative_evidence | 0.768 | 0.046 | 0.558 | 0.091 |
| AZ1 | definition_driven | 0.794 | 0.054 | 0.541 | 0.034 |
| AZ2 | binary_oracle_baseline | 0.566 | 0.084 | 0.575 | 0.296 |
| AZ2 | conservative_evidence | 0.656 | 0.025 | 0.59 | 0.24 |
| AZ2 | definition_driven | 0.746 | 0.117 | 0.561 | 0.136 |
| AZ3 | binary_oracle_baseline | 0.495 | 0.148 | 0.656 | 0.492 |
| AZ3 | conservative_evidence | 0.603 | 0.077 | 0.636 | 0.372 |
| AZ3 | definition_driven | 0.552 | 0.117 | 0.702 | 0.522 |
| AZ7 | binary_oracle_baseline | 0.553 | 0.07 | 0.443 | 0.084 |
| AZ7 | conservative_evidence | 0.672 | 0.109 | 0.505 | 0.096 |
| AZ7 | definition_driven | 0.695 | 0.115 | 0.532 | 0.126 |
| AZ8 | binary_oracle_baseline | 0.77 | 0.095 | 0.539 | 0.062 |
| AZ8 | conservative_evidence | 0.78 | 0.041 | 0.556 | 0.075 |
| AZ8 | definition_driven | 0.738 | 0.016 | 0.536 | 0.065 |

**MLCQ** (15 configurations): Spearman ρ(κ human, κ heuristic) = 0.676 (p=0.006); best vs human: AZ3/conservative_evidence, best vs heuristic: AZ3/binary_oracle_baseline

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.484 | 0.392 | 0.39 | 0.326 |
| AZ1 | conservative_evidence | 0.493 | 0.38 | 0.423 | 0.354 |
| AZ1 | definition_driven | 0.484 | 0.336 | 0.324 | 0.233 |
| AZ2 | binary_oracle_baseline | 0.217 | 0.151 | 0.32 | 0.277 |
| AZ2 | conservative_evidence | 0.385 | 0.308 | 0.452 | 0.406 |
| AZ2 | definition_driven | 0.467 | 0.377 | 0.462 | 0.407 |
| AZ3 | binary_oracle_baseline | 0.471 | 0.408 | 0.467 | 0.423 |
| AZ3 | conservative_evidence | 0.61 | 0.547 | 0.474 | 0.421 |
| AZ3 | definition_driven | 0.475 | 0.389 | 0.421 | 0.363 |
| AZ7 | binary_oracle_baseline | 0.195 | 0.158 | 0.2 | 0.169 |
| AZ7 | conservative_evidence | 0.19 | 0.147 | 0.19 | 0.154 |
| AZ7 | definition_driven | 0.316 | 0.211 | 0.222 | 0.148 |
| AZ8 | binary_oracle_baseline | 0.567 | 0.494 | 0.462 | 0.407 |
| AZ8 | conservative_evidence | 0.632 | 0.545 | 0.4 | 0.327 |
| AZ8 | definition_driven | 0.545 | 0.438 | 0.429 | 0.358 |

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
