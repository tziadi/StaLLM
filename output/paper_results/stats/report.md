# StaLLM paper — statistical analysis

## RQ3 — human-oracle smell classification (run 1)

| Dataset | Slot | Prompt | P | R | F1 [95% CI] | κ |
|---|---|---|---|---|---|---|
| DACOS | AZ1 | binary_oracle_baseline | 0.314 | 0.846 | 0.458 [0.333, 0.577] | 0.332 |
| DACOS | AZ1 | conservative_evidence | 0.262 | 0.846 | 0.4 [0.282, 0.51] | 0.251 |
| DACOS | AZ1 | definition_driven | 0.241 | 0.808 | 0.372 [0.255, 0.486] | 0.214 |
| DACOS | AZ2 | binary_oracle_baseline | 0.381 | 0.308 | 0.34 [0.158, 0.519] | 0.254 |
| DACOS | AZ2 | conservative_evidence | 0.438 | 0.538 | 0.483 [0.319, 0.632] | 0.396 |
| DACOS | AZ2 | definition_driven | 0.414 | 0.462 | 0.436 [0.261, 0.603] | 0.347 |
| DACOS | AZ3 | binary_oracle_baseline | 0.5 | 0.077 | 0.133 [0.0, 0.308] | 0.102 |
| DACOS | AZ3 | conservative_evidence | 0.583 | 0.269 | 0.368 [0.158, 0.556] | 0.312 |
| DACOS | AZ3 | definition_driven | 0.636 | 0.269 | 0.378 [0.171, 0.563] | 0.326 |
| MLCQ | AZ1 | binary_oracle_baseline | 0.577 | 0.417 | 0.484 [0.32, 0.623] | 0.392 |
| MLCQ | AZ1 | conservative_evidence | 0.486 | 0.5 | 0.493 [0.353, 0.624] | 0.38 |
| MLCQ | AZ1 | definition_driven | 0.39 | 0.639 | 0.484 [0.353, 0.602] | 0.336 |
| MLCQ | AZ2 | binary_oracle_baseline | 0.5 | 0.139 | 0.217 [0.051, 0.368] | 0.151 |
| MLCQ | AZ2 | conservative_evidence | 0.625 | 0.278 | 0.385 [0.213, 0.538] | 0.308 |
| MLCQ | AZ2 | definition_driven | 0.583 | 0.389 | 0.467 [0.298, 0.615] | 0.377 |
| MLCQ | AZ3 | binary_oracle_baseline | 0.8 | 0.333 | 0.471 [0.275, 0.625] | 0.408 |
| MLCQ | AZ3 | conservative_evidence | 0.783 | 0.5 | 0.61 [0.439, 0.746] | 0.547 |
| MLCQ | AZ3 | definition_driven | 0.609 | 0.389 | 0.475 [0.304, 0.623] | 0.389 |

**McNemar (Holm-adjusted within factor):** 9/36 pairwise comparisons significant at α=0.05.

- prompt | AZ1 | DACOS: binary_oracle_baseline vs conservative_evidence (17 vs 3 discordant, p_holm=0.0438)
- prompt | AZ1 | DACOS: binary_oracle_baseline vs definition_driven (21 vs 2 discordant, p_holm=0.0012)
- slot | binary_oracle_baseline | DACOS: AZ1 vs AZ2 (14 vs 35 discordant, p_holm=0.0494)
- slot | binary_oracle_baseline | DACOS: AZ1 vs AZ3 (20 vs 46 discordant, p_holm=0.026)
- slot | binary_oracle_baseline | MLCQ: AZ2 vs AZ3 (0 vs 9 discordant, p_holm=0.0494)
- slot | conservative_evidence | DACOS: AZ1 vs AZ2 (8 vs 44 discordant, p_holm=0.0)
- slot | conservative_evidence | DACOS: AZ1 vs AZ3 (15 vs 57 discordant, p_holm=0.0)
- slot | definition_driven | DACOS: AZ1 vs AZ2 (9 vs 49 discordant, p_holm=0.0)
- slot | definition_driven | DACOS: AZ1 vs AZ3 (14 vs 62 discordant, p_holm=0.0)

**Run-to-run stability (temperature 0):**

| Slot | Prompt | Dataset | runs | identical predictions | F1 mean ± sd |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | DACOS | 3 | 95.0% | 0.449 ± 0.008 |
| AZ1 | binary_oracle_baseline | MLCQ | 3 | 97.0% | 0.448 ± 0.031 |
| AZ1 | conservative_evidence | DACOS | 3 | 92.0% | 0.404 ± 0.006 |
| AZ1 | conservative_evidence | MLCQ | 3 | 95.5% | 0.534 ± 0.038 |
| AZ1 | definition_driven | DACOS | 3 | 91.0% | 0.382 ± 0.01 |
| AZ1 | definition_driven | MLCQ | 3 | 94.0% | 0.5 ± 0.014 |

## RQ3 — oracle substitution (human labels vs. metric heuristic, same samples)

**DACOS** (9 configurations): Spearman ρ(κ human, κ heuristic) = 0.567 (p=0.112); best vs human: AZ2/conservative_evidence, best vs heuristic: AZ3/definition_driven

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.458 | 0.332 | 0.25 | 0.178 |
| AZ1 | conservative_evidence | 0.4 | 0.251 | 0.213 | 0.136 |
| AZ1 | definition_driven | 0.372 | 0.214 | 0.206 | 0.128 |
| AZ2 | binary_oracle_baseline | 0.34 | 0.254 | 0.258 | 0.204 |
| AZ2 | conservative_evidence | 0.483 | 0.396 | 0.381 | 0.33 |
| AZ2 | definition_driven | 0.436 | 0.347 | 0.359 | 0.307 |
| AZ3 | binary_oracle_baseline | 0.133 | 0.102 | 0.286 | 0.265 |
| AZ3 | conservative_evidence | 0.368 | 0.312 | 0.364 | 0.327 |
| AZ3 | definition_driven | 0.378 | 0.326 | 0.381 | 0.347 |

**MLCQ** (9 configurations): Spearman ρ(κ human, κ heuristic) = 0.583 (p=0.099); best vs human: AZ3/conservative_evidence, best vs heuristic: AZ3/binary_oracle_baseline

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

## RQ4 — feature location, ArgoUML (8 single features) (mean over features, runs averaged)

| Method | Slot | Prompt | Ordering | runs | Hit@1 | Hit@5 | R@10 | MRR | MAP | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLM | AZ1 | baseline | alphabetical | 3 | 0.5 | 0.833 | 0.035 | 0.603 | 0.026 | 0.371 |
| LLM | AZ1 | baseline | bm25_masked | 3 | 0.75 | 0.875 | 0.061 | 0.812 | 0.054 | 0.562 |
| LLM | AZ1 | baseline | bm25_sorted | 3 | 1.0 | 1.0 | 0.112 | 1.0 | 0.104 | 0.8 |
| LLM | AZ1 | baseline | keyword_sorted | 3 | 0.875 | 1.0 | 0.103 | 0.938 | 0.098 | 0.717 |
| LLM | AZ1 | feature_evidence | alphabetical | 3 | 0.5 | 0.667 | 0.038 | 0.584 | 0.031 | 0.392 |
| LLM | AZ1 | feature_evidence | bm25_masked | 3 | 0.75 | 0.875 | 0.054 | 0.788 | 0.047 | 0.479 |
| LLM | AZ1 | feature_evidence | bm25_sorted | 3 | 0.875 | 1.0 | 0.108 | 0.938 | 0.1 | 0.8 |
| LLM | AZ1 | feature_evidence | keyword_sorted | 3 | 0.875 | 0.958 | 0.103 | 0.917 | 0.099 | 0.683 |
| LLM | AZ2 | baseline | alphabetical | 1 | 0.5 | 0.75 | 0.043 | 0.556 | 0.038 | 0.425 |
| LLM | AZ2 | baseline | bm25_masked | 1 | 0.875 | 1.0 | 0.076 | 0.938 | 0.065 | 0.637 |
| LLM | AZ2 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.122 | 1.0 | 0.114 | 0.838 |
| LLM | AZ2 | baseline | keyword_sorted | 1 | 0.75 | 0.875 | 0.096 | 0.833 | 0.08 | 0.637 |
| LLM | AZ2 | feature_evidence | alphabetical | 1 | 0.5 | 0.5 | 0.037 | 0.512 | 0.036 | 0.35 |
| LLM | AZ2 | feature_evidence | bm25_masked | 1 | 0.875 | 1.0 | 0.077 | 0.938 | 0.068 | 0.637 |
| LLM | AZ2 | feature_evidence | bm25_sorted | 1 | 0.875 | 1.0 | 0.12 | 0.938 | 0.113 | 0.8 |
| LLM | AZ2 | feature_evidence | keyword_sorted | 1 | 0.875 | 1.0 | 0.103 | 0.938 | 0.099 | 0.688 |
| LLM | AZ3 | baseline | alphabetical | 1 | 0.625 | 0.875 | 0.041 | 0.698 | 0.036 | 0.4 |
| LLM | AZ3 | baseline | bm25_masked | 1 | 1.0 | 1.0 | 0.078 | 1.0 | 0.07 | 0.65 |
| LLM | AZ3 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.108 | 1.0 | 0.105 | 0.75 |
| LLM | AZ3 | baseline | keyword_sorted | 1 | 1.0 | 1.0 | 0.109 | 1.0 | 0.107 | 0.738 |
| LLM | AZ3 | feature_evidence | alphabetical | 1 | 0.25 | 0.75 | 0.042 | 0.431 | 0.03 | 0.375 |
| LLM | AZ3 | feature_evidence | bm25_masked | 1 | 0.875 | 1.0 | 0.074 | 0.917 | 0.067 | 0.625 |
| LLM | AZ3 | feature_evidence | bm25_sorted | 1 | 1.0 | 1.0 | 0.119 | 1.0 | 0.114 | 0.838 |
| LLM | AZ3 | feature_evidence | keyword_sorted | 1 | 1.0 | 1.0 | 0.106 | 1.0 | 0.104 | 0.713 |
| LLM | AZ4 | baseline | alphabetical | 1 | 0.25 | 0.375 | 0.01 | 0.302 | 0.007 | 0.188 |
| LLM | AZ4 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.091 | 1.0 | 0.091 | 0.867 |
| LLM | AZ4 | baseline | keyword_sorted | 1 | 0.75 | 1.0 | 0.093 | 0.875 | 0.089 | 0.65 |
| LLM | AZ4 | feature_evidence | alphabetical | 1 | 0.25 | 0.375 | 0.01 | 0.281 | 0.009 | 0.188 |
| LLM | AZ4 | feature_evidence | keyword_sorted | 1 | 0.875 | 1.0 | 0.103 | 0.938 | 0.101 | 0.7 |
| keyword_sorted_only | - | - | - | 1 | 1.0 | 1.0 | 0.093 | 1.0 | 0.09 | 0.613 |
| bm25_only | - | - | - | 1 | 0.5 | 0.875 | 0.026 | 0.592 | 0.012 | 0.237 |
| bm25_masked_only | - | - | - | 1 | 0.375 | 0.5 | 0.016 | 0.473 | 0.008 | 0.163 |

*Paired Wilcoxon tests across features are in `rq4_wilcoxon.csv` (with n=8 features the minimum two-sided p is 0.0078; with n=5 it is 0.0625, so eShop comparisons are descriptive only).*

## RQ4 — feature location, eShopOnContainers (5 features) (mean over features, runs averaged)

| Method | Slot | Prompt | Ordering | runs | Hit@1 | Hit@5 | R@10 | MRR | MAP | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLM | AZ1 | baseline | alphabetical | 3 | 1.0 | 1.0 | 0.223 | 1.0 | 0.223 | 0.907 |
| LLM | AZ1 | baseline | bm25_masked | 3 | 0.8 | 0.8 | 0.104 | 0.8 | 0.104 | 0.8 |
| LLM | AZ1 | baseline | bm25_sorted | 3 | 1.0 | 1.0 | 0.214 | 1.0 | 0.214 | 0.893 |
| LLM | AZ1 | baseline | keyword_sorted | 3 | 1.0 | 1.0 | 0.193 | 1.0 | 0.193 | 0.88 |
| LLM | AZ1 | feature_evidence | alphabetical | 3 | 1.0 | 1.0 | 0.222 | 1.0 | 0.221 | 0.9 |
| LLM | AZ1 | feature_evidence | bm25_masked | 3 | 0.8 | 0.8 | 0.097 | 0.8 | 0.096 | 0.753 |
| LLM | AZ1 | feature_evidence | bm25_sorted | 3 | 1.0 | 1.0 | 0.213 | 1.0 | 0.213 | 0.887 |
| LLM | AZ1 | feature_evidence | keyword_sorted | 3 | 1.0 | 1.0 | 0.197 | 1.0 | 0.196 | 0.867 |
| LLM | AZ2 | baseline | alphabetical | 1 | 1.0 | 1.0 | 0.26 | 1.0 | 0.26 | 0.94 |
| LLM | AZ2 | baseline | bm25_masked | 1 | 0.8 | 0.8 | 0.104 | 0.8 | 0.104 | 0.8 |
| LLM | AZ2 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.215 | 1.0 | 0.204 | 0.9 |
| LLM | AZ2 | baseline | keyword_sorted | 1 | 1.0 | 1.0 | 0.215 | 1.0 | 0.215 | 0.9 |
| LLM | AZ2 | feature_evidence | alphabetical | 1 | 1.0 | 1.0 | 0.257 | 1.0 | 0.255 | 0.92 |
| LLM | AZ2 | feature_evidence | bm25_masked | 1 | 0.8 | 1.0 | 0.167 | 0.84 | 0.122 | 0.84 |
| LLM | AZ2 | feature_evidence | bm25_sorted | 1 | 1.0 | 1.0 | 0.19 | 1.0 | 0.189 | 0.86 |
| LLM | AZ2 | feature_evidence | keyword_sorted | 1 | 1.0 | 1.0 | 0.171 | 1.0 | 0.171 | 0.86 |
| LLM | AZ3 | baseline | alphabetical | 1 | 1.0 | 1.0 | 0.234 | 1.0 | 0.234 | 0.9 |
| LLM | AZ3 | baseline | bm25_masked | 1 | 1.0 | 1.0 | 0.227 | 1.0 | 0.227 | 0.86 |
| LLM | AZ3 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.234 | 1.0 | 0.215 | 0.9 |
| LLM | AZ3 | baseline | keyword_sorted | 1 | 1.0 | 1.0 | 0.227 | 1.0 | 0.227 | 0.86 |
| LLM | AZ3 | feature_evidence | alphabetical | 1 | 1.0 | 1.0 | 0.256 | 1.0 | 0.256 | 0.92 |
| LLM | AZ3 | feature_evidence | bm25_masked | 1 | 1.0 | 1.0 | 0.208 | 1.0 | 0.197 | 0.86 |
| LLM | AZ3 | feature_evidence | bm25_sorted | 1 | 1.0 | 1.0 | 0.237 | 1.0 | 0.237 | 0.92 |
| LLM | AZ3 | feature_evidence | keyword_sorted | 1 | 1.0 | 1.0 | 0.215 | 1.0 | 0.215 | 0.9 |
| keyword_sorted_only | - | - | - | 1 | 0.4 | 0.8 | 0.044 | 0.567 | 0.025 | 0.38 |
| bm25_only | - | - | - | 1 | 0.8 | 1.0 | 0.178 | 0.867 | 0.151 | 0.76 |
| bm25_masked_only | - | - | - | 1 | 0.8 | 1.0 | 0.092 | 0.867 | 0.08 | 0.54 |

*Paired Wilcoxon tests across features are in `rq4_wilcoxon_eshop.csv` (with n=8 features the minimum two-sided p is 0.0078; with n=5 it is 0.0625, so eShop comparisons are descriptive only).*

## RQ3 — non-LLM baselines

| Dataset | Method | P | R | F1 | κ |
|---|---|---|---|---|---|
| DACOS | always_present | 0.13 | 1.0 | 0.2301 | 0.0 |
| DACOS | metric_heuristic | 0.8 | 0.3077 | 0.4444 | 0.4012 |
| MLCQ | always_present | 0.18 | 1.0 | 0.3051 | 0.0 |
| MLCQ | metric_heuristic | 0.8 | 0.3333 | 0.4706 | 0.4079 |
| DACOS_heldout | always_present | 0.135 | 1.0 | 0.2379 | 0.0 |
| DACOS_heldout | metric_heuristic | 0.2 | 0.1111 | 0.1429 | 0.0514 |

## RQ1/RQ2 — span-level reruns (eShop Catalog, path-matched oracle, Top-k=20, Lenient)

*Random placement: same number and lengths of spans, uniformly random lines in the same file; random files: also a uniformly random file of the universe. p = share of draws reaching the LLM F1.*

| Slot | Model | Prompt | runs | P | R | F1 mean ± sd | random placement [95%] | p | random files [95%] | p | LLM errors | median time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AZ1 | azure:gpt-4o-Tewfik | baseline | 3 | 0.112 | 0.167 | 0.134 ± 0.044 | 0.155 [0.122, 0.187] | 0.9041 | 0.113 [0.083, 0.145] | 0.1099 | 0 | 64.0 |
| AZ1 | azure:gpt-4o-Tewfik | hybrid | 3 | 0.069 | 0.213 | 0.104 ± 0.026 | 0.15 [0.129, 0.174] | 1.0 | 0.116 [0.092, 0.138] | 0.8482 | 0 | 103.1 |
| AZ1 | azure:gpt-4o-Tewfik | scanner | 3 | 0.076 | 0.213 | 0.112 ± 0.02 | 0.166 [0.141, 0.19] | 1.0 | 0.123 [0.099, 0.148] | 0.8212 | 0 | 81.2 |
| AZ2 | azure:gpt-4.1-features | scanner | 3 | 0.117 | 0.148 | 0.131 ± 0.021 | 0.151 [0.114, 0.185] | 0.8721 | 0.107 [0.074, 0.142] | 0.0649 | 0 | 87.2 |
| AZ3 | azure:o3-features | scanner | 3 | 0.203 | 0.273 | 0.232 ± 0.074 | 0.174 [0.142, 0.206] | 0.001 | 0.132 [0.099, 0.165] | 0.001 | 0 | 580.6 |

## RQ5 — held-out DACOS test split (disjoint from the selection split)

| Source | Prompt | P | R | F1 [95% CI] | κ | McNemar p vs generated |
|---|---|---|---|---|---|---|
| RQ3_manual | binary_oracle_baseline | 0.291 | 0.593 | 0.39 [0.244, 0.509] | 0.255 | 0.2295 |
| RQ3_manual | conservative_evidence | 0.288 | 0.704 | 0.409 [0.28, 0.524] | 0.268 | 0.0501 |
| RQ3_manual | definition_driven | 0.286 | 0.667 | 0.4 [0.267, 0.516] | 0.26 | 0.0433 |
| RQ5_generated | threshold_guided | 0.31 | 0.481 | 0.377 [0.222, 0.514] | 0.254 | - |

**MLCQ validation** — generated threshold_guided F1=0.548:

- vs binary_oracle_baseline (F1=0.484): 9 vs 10 discordant, McNemar p=1.0
- vs conservative_evidence (F1=0.493): 10 vs 6 discordant, McNemar p=0.4545
- vs definition_driven (F1=0.484): 20 vs 4 discordant, McNemar p=0.0015
