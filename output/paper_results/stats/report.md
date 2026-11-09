# StaLLM paper — statistical analysis

## RQ3 — human-oracle smell classification (run 1)

| Dataset | Slot | Prompt | P | R | F1 [95% CI] | κ |
|---|---|---|---|---|---|---|
| DACOS | AZ1 | binary_oracle_baseline | 0.304 | 0.808 | 0.442 [0.312, 0.558] | 0.312 |
| DACOS | AZ1 | conservative_evidence | 0.259 | 0.846 | 0.396 [0.28, 0.504] | 0.246 |
| DACOS | AZ1 | definition_driven | 0.25 | 0.808 | 0.382 [0.263, 0.496] | 0.229 |
| DACOS | AZ2 | binary_oracle_baseline | 0.381 | 0.308 | 0.34 [0.158, 0.519] | 0.254 |
| DACOS | AZ2 | conservative_evidence | 0.438 | 0.538 | 0.483 [0.319, 0.632] | 0.396 |
| DACOS | AZ2 | definition_driven | 0.414 | 0.462 | 0.436 [0.261, 0.603] | 0.347 |
| DACOS | AZ3 | binary_oracle_baseline | 0.5 | 0.077 | 0.133 [0.0, 0.308] | 0.102 |
| DACOS | AZ3 | conservative_evidence | 0.583 | 0.269 | 0.368 [0.158, 0.556] | 0.312 |
| DACOS | AZ3 | definition_driven | 0.636 | 0.269 | 0.378 [0.171, 0.563] | 0.326 |
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

**McNemar (Holm-adjusted within factor):** 6/63 pairwise comparisons significant at α=0.05.

- prompt | AZ1 | DACOS: binary_oracle_baseline vs definition_driven (18 vs 3 discordant, p_holm=0.0358)
- slot | binary_oracle_baseline | DACOS: AZ1 vs AZ3 (19 vs 46 discordant, p_holm=0.0382)
- slot | conservative_evidence | DACOS: AZ1 vs AZ2 (8 vs 45 discordant, p_holm=0.0)
- slot | conservative_evidence | DACOS: AZ1 vs AZ3 (15 vs 58 discordant, p_holm=0.0)
- slot | definition_driven | DACOS: AZ1 vs AZ2 (10 vs 47 discordant, p_holm=0.0)
- slot | definition_driven | DACOS: AZ1 vs AZ3 (14 vs 59 discordant, p_holm=0.0)

**Run-to-run stability (temperature 0):**

| Slot | Prompt | Dataset | runs | identical predictions | F1 mean ± sd |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | DACOS | 3 | 95.0% | 0.449 ± 0.008 |
| AZ1 | binary_oracle_baseline | MLCQ | 3 | 97.0% | 0.448 ± 0.031 |
| AZ1 | conservative_evidence | DACOS | 3 | 92.0% | 0.404 ± 0.006 |
| AZ1 | conservative_evidence | MLCQ | 3 | 95.5% | 0.534 ± 0.038 |
| AZ1 | definition_driven | DACOS | 3 | 91.0% | 0.382 ± 0.01 |
| AZ1 | definition_driven | MLCQ | 3 | 94.0% | 0.5 ± 0.014 |
| AZ7 | binary_oracle_baseline | MLCQ | 3 | 99.5% | 0.194 ± 0.003 |
| AZ7 | conservative_evidence | MLCQ | 3 | 99.0% | 0.175 ± 0.027 |
| AZ7 | definition_driven | MLCQ | 3 | 99.5% | 0.32 ± 0.003 |
| AZ8 | binary_oracle_baseline | MLCQ | 3 | 96.0% | 0.542 ± 0.022 |
| AZ8 | conservative_evidence | MLCQ | 3 | 98.0% | 0.64 ± 0.015 |
| AZ8 | definition_driven | MLCQ | 3 | 97.0% | 0.551 ± 0.015 |

## RQ3 — oracle substitution (human labels vs. metric heuristic, same samples)

**DACOS** (9 configurations): Spearman ρ(κ human, κ heuristic) = 0.65 (p=0.058); best vs human: AZ2/conservative_evidence, best vs heuristic: AZ3/definition_driven

| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |
|---|---|---|---|---|---|
| AZ1 | binary_oracle_baseline | 0.442 | 0.312 | 0.253 | 0.182 |
| AZ1 | conservative_evidence | 0.396 | 0.246 | 0.211 | 0.133 |
| AZ1 | definition_driven | 0.382 | 0.229 | 0.213 | 0.136 |
| AZ2 | binary_oracle_baseline | 0.34 | 0.254 | 0.258 | 0.204 |
| AZ2 | conservative_evidence | 0.483 | 0.396 | 0.381 | 0.33 |
| AZ2 | definition_driven | 0.436 | 0.347 | 0.359 | 0.307 |
| AZ3 | binary_oracle_baseline | 0.133 | 0.102 | 0.286 | 0.265 |
| AZ3 | conservative_evidence | 0.368 | 0.312 | 0.364 | 0.327 |
| AZ3 | definition_driven | 0.378 | 0.326 | 0.381 | 0.347 |

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

## RQ4 — feature location, ArgoUML (8 single features) (mean over features, runs averaged)

| Method | Slot | Prompt | Ordering | runs | Hit@1 | Hit@5 | R@10 | MRR | MAP | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLM | AZ1 | baseline | alphabetical | 3 | 0.5 | 0.708 | 0.037 | 0.606 | 0.027 | 0.388 |
| LLM | AZ1 | baseline | bm25_masked | 3 | 0.75 | 0.875 | 0.07 | 0.812 | 0.063 | 0.558 |
| LLM | AZ1 | baseline | bm25_sorted | 3 | 0.875 | 1.0 | 0.086 | 0.917 | 0.073 | 0.646 |
| LLM | AZ1 | baseline | keyword_sorted | 3 | 0.875 | 1.0 | 0.099 | 0.938 | 0.095 | 0.679 |
| LLM | AZ1 | feature_evidence | alphabetical | 3 | 0.583 | 0.75 | 0.039 | 0.666 | 0.03 | 0.4 |
| LLM | AZ1 | feature_evidence | bm25_masked | 3 | 0.875 | 0.875 | 0.065 | 0.875 | 0.061 | 0.525 |
| LLM | AZ1 | feature_evidence | bm25_sorted | 3 | 0.75 | 1.0 | 0.093 | 0.875 | 0.078 | 0.683 |
| LLM | AZ1 | feature_evidence | keyword_sorted | 3 | 0.958 | 1.0 | 0.102 | 0.979 | 0.097 | 0.708 |
| LLM | AZ2 | baseline | alphabetical | 1 | 0.5 | 0.75 | 0.046 | 0.573 | 0.041 | 0.45 |
| LLM | AZ2 | baseline | bm25_masked | 1 | 0.875 | 1.0 | 0.081 | 0.938 | 0.073 | 0.625 |
| LLM | AZ2 | baseline | bm25_sorted | 1 | 1.0 | 1.0 | 0.1 | 1.0 | 0.091 | 0.713 |
| LLM | AZ2 | baseline | keyword_sorted | 1 | 1.0 | 1.0 | 0.102 | 1.0 | 0.1 | 0.713 |
| LLM | AZ2 | feature_evidence | alphabetical | 1 | 0.5 | 0.5 | 0.039 | 0.5 | 0.039 | 0.362 |
| LLM | AZ2 | feature_evidence | bm25_masked | 1 | 0.875 | 1.0 | 0.084 | 0.917 | 0.075 | 0.625 |
| LLM | AZ2 | feature_evidence | bm25_sorted | 1 | 0.75 | 1.0 | 0.109 | 0.875 | 0.098 | 0.75 |
| LLM | AZ2 | feature_evidence | keyword_sorted | 1 | 0.75 | 0.875 | 0.09 | 0.833 | 0.073 | 0.637 |
| LLM | AZ3 | baseline | alphabetical | 1 | 0.375 | 0.625 | 0.042 | 0.48 | 0.032 | 0.4 |
| LLM | AZ3 | baseline | bm25_masked | 1 | 1.0 | 1.0 | 0.076 | 1.0 | 0.071 | 0.6 |
| LLM | AZ3 | baseline | bm25_sorted | 1 | 0.75 | 1.0 | 0.101 | 0.875 | 0.093 | 0.713 |
| LLM | AZ3 | baseline | keyword_sorted | 1 | 0.875 | 0.875 | 0.093 | 0.891 | 0.09 | 0.625 |
| LLM | AZ3 | feature_evidence | alphabetical | 1 | 0.375 | 0.625 | 0.041 | 0.469 | 0.035 | 0.388 |
| LLM | AZ3 | feature_evidence | bm25_masked | 1 | 0.875 | 1.0 | 0.079 | 0.938 | 0.074 | 0.613 |
| LLM | AZ3 | feature_evidence | bm25_sorted | 1 | 0.875 | 1.0 | 0.115 | 0.938 | 0.11 | 0.8 |
| LLM | AZ3 | feature_evidence | keyword_sorted | 1 | 1.0 | 1.0 | 0.108 | 1.0 | 0.106 | 0.738 |
| LLM | AZ4 | baseline | alphabetical | 1 | 0.25 | 0.375 | 0.01 | 0.302 | 0.007 | 0.188 |
| LLM | AZ4 | baseline | bm25_sorted | 1 | 0.875 | 1.0 | 0.128 | 0.938 | 0.126 | 0.875 |
| LLM | AZ4 | baseline | keyword_sorted | 1 | 0.875 | 1.0 | 0.092 | 0.938 | 0.091 | 0.637 |
| LLM | AZ4 | feature_evidence | alphabetical | 1 | 0.25 | 0.375 | 0.01 | 0.281 | 0.009 | 0.188 |
| LLM | AZ4 | feature_evidence | bm25_sorted | 1 | 0.875 | 1.0 | 0.126 | 0.938 | 0.123 | 0.838 |
| LLM | AZ4 | feature_evidence | keyword_sorted | 1 | 0.875 | 1.0 | 0.093 | 0.938 | 0.09 | 0.675 |
| keyword_sorted_only | - | - | - | 1 | 1.0 | 1.0 | 0.101 | 1.0 | 0.099 | 0.7 |
| bm25_only | - | - | - | 1 | 0.5 | 0.875 | 0.044 | 0.646 | 0.025 | 0.412 |
| bm25_masked_only | - | - | - | 1 | 0.5 | 0.75 | 0.036 | 0.583 | 0.018 | 0.325 |

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

## RQ1/RQ2 — span-level runs re-scored against the current oracle (eShop Catalog, path matching, line offset checked, Top-k=20, Lenient)

*protocol: original = raw code; numbered = code lines prefixed with their number. f1_as_recorded = F1 computed at run time (before later oracle corrections).*

*Random placement: same number and lengths of spans, uniformly random lines in the same file; random files: also a uniformly random file of the universe. p = share of draws reaching the LLM F1.*

| Protocol | Slot | Model | Prompt | runs | P | R | F1 mean ± sd | random placement [95%] | p | random files [95%] | p | LLM errors | median time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| numbered | AZ1 | azure:gpt-4o-Tewfik | baseline | 3 | 0.172 | 0.296 | 0.217 ± 0.037 | 0.179 [0.146, 0.214] | 0.017 | 0.12 [0.089, 0.15] | 0.001 | 0 | 59.6 |
| numbered | AZ1 | azure:gpt-4o-Tewfik | hybrid | 3 | 0.103 | 0.329 | 0.156 ± 0.012 | 0.158 [0.137, 0.178] | 0.5744 | 0.122 [0.099, 0.146] | 0.002 | 0 | 111.2 |
| numbered | AZ1 | azure:gpt-4o-Tewfik | scanner | 3 | 0.071 | 0.204 | 0.105 ± 0.024 | 0.154 [0.129, 0.179] | 1.0 | 0.116 [0.093, 0.139] | 0.8282 | 0 | 72.1 |
| numbered | AZ2 | azure:gpt-4.1-features | scanner | 3 | 0.153 | 0.176 | 0.163 ± 0.067 | 0.13 [0.099, 0.164] | 0.03 | 0.116 [0.083, 0.15] | 0.009 | 0 | 81.8 |
| numbered | AZ3 | azure:o3-features | scanner | 3 | 0.186 | 0.282 | 0.224 ± 0.028 | 0.168 [0.138, 0.196] | 0.001 | 0.117 [0.086, 0.149] | 0.001 | 2 | 335.1 |
| numbered | AZ4 | ollama:qwen2.5-coder:14b-instruct@localhost:11434 | scanner | 3 | 0.159 | 0.204 | 0.179 ± 0.008 | 0.149 [0.114, 0.183] | 0.0609 | 0.111 [0.081, 0.146] | 0.002 | 0 | 1853.1 |
| numbered | AZ6 | ollama:deepseek-coder:6.7b-instruct@localhost:11434 | scanner | 3 | 0.15 | 0.31 | 0.202 ± 0.005 | 0.137 [0.109, 0.163] | 0.001 | 0.109 [0.081, 0.136] | 0.001 | 0 | 1167.6 |
| numbered | AZ7 | azure:DeepSeek-V3.2 | scanner | 3 | 0.162 | 0.458 | 0.239 ± 0.063 | 0.172 [0.147, 0.199] | 0.001 | 0.12 [0.095, 0.143] | 0.001 | 0 | 72.3 |
| numbered | AZ8 | azure:Mistral-Large-3 | scanner | 3 | 0.131 | 0.375 | 0.194 ± 0.031 | 0.175 [0.153, 0.197] | 0.0569 | 0.139 [0.118, 0.163] | 0.001 | 0 | 123.2 |
| original | AZ1 | azure:gpt-4o-Tewfik | baseline | 3 | 0.116 | 0.171 | 0.138 ± 0.031 | 0.155 [0.123, 0.187] | 0.8342 | 0.112 [0.082, 0.146] | 0.0629 | 0 | 64.0 |
| original | AZ1 | azure:gpt-4o-Tewfik | hybrid | 3 | 0.073 | 0.227 | 0.11 ± 0.032 | 0.15 [0.129, 0.173] | 1.0 | 0.115 [0.091, 0.138] | 0.6543 | 0 | 103.1 |
| original | AZ1 | azure:gpt-4o-Tewfik | scanner | 3 | 0.07 | 0.194 | 0.102 ± 0.021 | 0.165 [0.142, 0.19] | 1.0 | 0.122 [0.098, 0.147] | 0.9491 | 0 | 81.2 |
| original | AZ2 | azure:gpt-4.1-features | scanner | 3 | 0.114 | 0.144 | 0.127 ± 0.011 | 0.151 [0.114, 0.187] | 0.9001 | 0.106 [0.073, 0.142] | 0.0919 | 0 | 87.2 |
| original | AZ3 | azure:o3-features | scanner | 3 | 0.204 | 0.273 | 0.233 ± 0.07 | 0.174 [0.143, 0.206] | 0.001 | 0.131 [0.101, 0.165] | 0.001 | 0 | 580.6 |
| original | AZ4 | ollama:qwen2.5-coder:14b-instruct@localhost:11434 | scanner | 3 | 0.089 | 0.13 | 0.105 ± 0.006 | 0.116 [0.087, 0.147] | 0.7552 | 0.119 [0.09, 0.151] | 0.8082 | 0 | 1247.8 |
| original | AZ6 | ollama:deepseek-coder:6.7b-instruct@localhost:11434 | scanner | 3 | 0.036 | 0.069 | 0.047 ± 0.001 | 0.147 [0.117, 0.177] | 1.0 | 0.11 [0.082, 0.139] | 1.0 | 0 | 1221.3 |
| original | AZ7 | azure:DeepSeek-V3.2 | scanner | 3 | 0.117 | 0.292 | 0.167 ± 0.026 | 0.16 [0.132, 0.188] | 0.3367 | 0.122 [0.098, 0.148] | 0.002 | 0 | 74.6 |
| original | AZ8 | azure:Mistral-Large-3 | scanner | 3 | 0.106 | 0.269 | 0.152 ± 0.01 | 0.179 [0.157, 0.202] | 0.988 | 0.144 [0.118, 0.168] | 0.2597 | 0 | 113.6 |

*Without the empty-answer retry (files answered by the retry count as empty); random placement recomputed on the same spans.*

| Protocol | Slot | Prompt | retry files | F1 with retry | F1 without retry | random placement | p |
|---|---|---|---|---|---|---|---|
| numbered | AZ4 | scanner | 6.7 | 0.179 | 0.181 | 0.154 | 0.0919 |
| numbered | AZ6 | scanner | 2.0 | 0.202 | 0.195 | 0.135 | 0.001 |
| numbered | AZ7 | scanner | 1.3 | 0.239 | 0.151 | 0.121 | 0.004 |
| numbered | AZ8 | scanner | 0.7 | 0.194 | 0.188 | 0.16 | 0.013 |
| original | AZ7 | scanner | 1.3 | 0.167 | 0.151 | 0.142 | 0.2767 |
| original | AZ8 | scanner | 0.7 | 0.152 | 0.145 | 0.165 | 0.952 |

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
