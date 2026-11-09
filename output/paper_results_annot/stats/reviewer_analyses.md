# Additional analyses (review)

Computed from stored outputs, no model call.

RQ3 on DACOS labeled by the annotators' majority (complex method, long parameter list), with the MLCQ runs of output/paper_results/. These are the numbers of the paper.

## 1. RQ3 — ranking stability with the oracle held fixed

**(a) Reruns.** The configurations ranked with run r of the models that have three runs (gpt-4o, DeepSeek-V3.2, Mistral-Large-3; gpt-4.1 and o3 have one run). ρ over all configurations, and over the 3 gpt-4o prompts alone.

| Dataset | Metric | Runs | ρ (all configs) | same best | ρ (3 gpt-4o prompts) |
|---|---|---|---|---|---|
| DACOS | f1 | 1 vs 2 | 0.989 | True | 1.0 |
| DACOS | f1 | 1 vs 3 | 0.989 | True | 1.0 |
| DACOS | f1 | 2 vs 3 | 0.989 | True | 1.0 |
| DACOS | kappa | 1 vs 2 | 0.918 | True | 1.0 |
| DACOS | kappa | 1 vs 3 | 0.961 | True | 0.5 |
| DACOS | kappa | 2 vs 3 | 0.886 | True | 0.5 |
| MLCQ | f1 | 1 vs 2 | 0.964 | True | 1.0 |
| MLCQ | f1 | 1 vs 3 | 0.968 | True | 1.0 |
| MLCQ | f1 | 2 vs 3 | 0.996 | True | 1.0 |
| MLCQ | kappa | 1 vs 2 | 0.932 | False | -0.5 |
| MLCQ | kappa | 1 vs 3 | 0.911 | True | -0.5 |
| MLCQ | kappa | 2 vs 3 | 0.982 | False | 1.0 |

**(b) Paired bootstrap** (1000 resamples of the samples, run 1 of each configuration, seed 42). Median ρ and 95% interval.

| Dataset | ρ(κ human, κ metric) observed | ρ κ human: resample vs original | ρ κ human: two resamples | ρ F1 human: resample vs original | ρ κ metric: resample vs original | ρ(κ human, κ metric) in resamples | same human-best (κ) | same human-best (F1) | same metric-best (κ) | human-best ≠ metric-best |
|---|---|---|---|---|---|---|---|---|---|---|
| DACOS | 0.489 | 0.661 [0.171, 0.896] | 0.432 [-0.157, 0.829] | 0.979 [0.936, 0.996] | 0.925 [0.8, 0.982] | 0.327 [-0.382, 0.782] | 32% | 89% | 69% | 81% |
| MLCQ | 0.701 | 0.9 [0.696, 0.968] | 0.821 [0.55, 0.946] | 0.911 [0.718, 0.975] | 0.81 [0.397, 0.952] | 0.6 [-0.098, 0.889] | 47% | 53% | 24% | 88% |

- DACOS: best by human κ = AZ3/binary_oracle_baseline, best by metric κ = AZ3/definition_driven.
- MLCQ: best by human κ = AZ3/conservative_evidence, best by metric κ = AZ3/binary_oracle_baseline.

