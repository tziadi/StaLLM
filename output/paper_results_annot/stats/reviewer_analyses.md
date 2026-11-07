# Additional analyses (review)

Computed from stored outputs, no model call.

RQ3 on DACOS labeled by the annotators' majority (complex method, long parameter list), with the MLCQ runs of output/paper_results/. These are the numbers of the paper.

## 1. RQ3 — ranking stability with the oracle held fixed

**(a) Reruns.** The nine configurations ranked with gpt-4o's run r (gpt-4.1 and o3 have one run). ρ over 9 configurations, and over the 3 gpt-4o prompts alone.

| Dataset | Metric | Runs | ρ (9 configs) | same best | ρ (3 gpt-4o prompts) |
|---|---|---|---|---|---|
| DACOS | f1 | 1 vs 2 | 1.0 | True | 1.0 |
| DACOS | f1 | 1 vs 3 | 1.0 | True | 1.0 |
| DACOS | f1 | 2 vs 3 | 1.0 | True | 1.0 |
| DACOS | kappa | 1 vs 2 | 0.983 | True | 1.0 |
| DACOS | kappa | 1 vs 3 | 0.983 | True | 0.5 |
| DACOS | kappa | 2 vs 3 | 0.967 | True | 0.5 |
| MLCQ | f1 | 1 vs 2 | 0.9 | True | 1.0 |
| MLCQ | f1 | 1 vs 3 | 0.9 | True | 1.0 |
| MLCQ | f1 | 2 vs 3 | 1.0 | True | 1.0 |
| MLCQ | kappa | 1 vs 2 | 0.767 | True | -0.5 |
| MLCQ | kappa | 1 vs 3 | 0.767 | True | -0.5 |
| MLCQ | kappa | 2 vs 3 | 1.0 | True | 1.0 |

**(b) Paired bootstrap** (1000 resamples of the samples, run 1 of each configuration, seed 42). Median ρ and 95% interval.

| Dataset | ρ(κ human, κ metric) observed | ρ κ human: resample vs original | ρ κ human: two resamples | ρ F1 human: resample vs original | ρ κ metric: resample vs original | ρ(κ human, κ metric) in resamples | same human-best (κ) | same human-best (F1) | same metric-best (κ) | human-best ≠ metric-best |
|---|---|---|---|---|---|---|---|---|---|---|
| DACOS | 0.65 | 0.7 [-0.05, 0.933] | 0.483 [-0.35, 0.887] | 0.983 [0.917, 1.0] | 0.983 [0.917, 1.0] | 0.494 [-0.383, 0.883] | 44% | 97% | 69% | 74% |
| MLCQ | 0.583 | 0.75 [0.3, 0.967] | 0.667 [0.167, 0.933] | 0.733 [0.283, 0.967] | 0.7 [0.1, 0.95] | 0.433 [-0.4, 0.9] | 89% | 83% | 29% | 78% |

- DACOS: best by human κ = AZ3/binary_oracle_baseline, best by metric κ = AZ3/definition_driven.
- MLCQ: best by human κ = AZ3/conservative_evidence, best by metric κ = AZ3/binary_oracle_baseline.

