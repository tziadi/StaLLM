# Additional analyses (review)

Computed from stored outputs, no model call.

**Legacy DACOS labels.** In this root, DACOS is labeled by the `smells` column of the DACOS sample table (preliminary version). The paper reports RQ3 and RQ5 on the annotated DACOS sample: see output/paper_results_annot/stats/reviewer_analyses.md. Analyses 2–4 do not depend on DACOS and are the numbers of the paper.

## 1. RQ3 — ranking stability with the oracle held fixed

**(a) Reruns.** The configurations ranked with run r of the models that have three runs (gpt-4o, DeepSeek-V3.2, Mistral-Large-3; gpt-4.1 and o3 have one run). ρ over all configurations, and over the 3 gpt-4o prompts alone.

| Dataset | Metric | Runs | ρ (all configs) | same best | ρ (3 gpt-4o prompts) |
|---|---|---|---|---|---|
| DACOS | f1 | 1 vs 2 | 0.983 | True | 1.0 |
| DACOS | f1 | 1 vs 3 | 0.983 | True | 1.0 |
| DACOS | f1 | 2 vs 3 | 1.0 | True | 1.0 |
| DACOS | kappa | 1 vs 2 | 0.983 | True | 1.0 |
| DACOS | kappa | 1 vs 3 | 0.967 | True | 1.0 |
| DACOS | kappa | 2 vs 3 | 0.983 | True | 1.0 |
| MLCQ | f1 | 1 vs 2 | 0.964 | True | 1.0 |
| MLCQ | f1 | 1 vs 3 | 0.968 | True | 1.0 |
| MLCQ | f1 | 2 vs 3 | 0.996 | True | 1.0 |
| MLCQ | kappa | 1 vs 2 | 0.932 | False | -0.5 |
| MLCQ | kappa | 1 vs 3 | 0.911 | True | -0.5 |
| MLCQ | kappa | 2 vs 3 | 0.982 | False | 1.0 |

**(b) Paired bootstrap** (1000 resamples of the samples, run 1 of each configuration, seed 42). Median ρ and 95% interval.

| Dataset | ρ(κ human, κ metric) observed | ρ κ human: resample vs original | ρ κ human: two resamples | ρ F1 human: resample vs original | ρ κ metric: resample vs original | ρ(κ human, κ metric) in resamples | same human-best (κ) | same human-best (F1) | same metric-best (κ) | human-best ≠ metric-best |
|---|---|---|---|---|---|---|---|---|---|---|
| DACOS | 0.65 | 0.783 [0.3, 0.967] | 0.617 [-0.05, 0.933] | 0.783 [0.2, 0.983] | 0.768 [-0.067, 0.967] | 0.502 [-0.317, 0.933] | 53% | 52% | 22% | 68% |
| MLCQ | 0.701 | 0.9 [0.696, 0.968] | 0.821 [0.55, 0.946] | 0.911 [0.718, 0.975] | 0.81 [0.397, 0.952] | 0.6 [-0.098, 0.889] | 47% | 53% | 24% | 88% |

- DACOS: best by human κ = AZ2/conservative_evidence, best by metric κ = AZ3/definition_driven.
- MLCQ: best by human κ = AZ3/conservative_evidence, best by metric κ = AZ3/binary_oracle_baseline.

## 2. RQ1/RQ2 — without the empty-answer retry

*Files answered only through the retry are counted as empty. Runs with per-file provenance (`source` in spans_by_file: the numbered runs of AZ4 and AZ6) give an exact value. The other runs record only the number k of such files, so the files are identified from the retry answers saved in llm_debug/ (a file whose predictions equal a saved retry answer and no saved first answer). This reconstruction recovers the recorded k in every run and agrees with the per-file provenance where it exists. When a file matches both a first and a retry answer, low/high give the extreme choices. Random placement and p are recomputed on the same spans (1000 draws, seed 42).*

| Protocol | Slot | Prompt | runs | exact | retry files | LLM errors | F1 with retry | F1 without retry (low–high) | random (low/high) | p (low/high) |
|---|---|---|---|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | 3 | False | 3.3 | 0 | 0.217 | 0.218–0.218 | 0.183/0.183 | 0.035/0.035 |
| numbered | AZ1 | hybrid | 3 | False | 0.7 | 0 | 0.156 | 0.149–0.149 | 0.147/0.147 | 0.4146/0.4146 |
| numbered | AZ1 | scanner | 3 | False | 2.3 | 0 | 0.105 | 0.104–0.104 | 0.157/0.157 | 1.0/1.0 |
| numbered | AZ2 | scanner | 3 | False | 10.3 | 0 | 0.163 | 0.13–0.13 | 0.068/0.068 | 0.001/0.001 |
| numbered | AZ3 | scanner | 3 | False | 1.3 | 2 | 0.224 | 0.226–0.231 | 0.17/0.172 | 0.002/0.001 |
| numbered | AZ4 | scanner | 3 | True | 6.7 | 0 | 0.179 | 0.181–0.181 | 0.154/0.154 | 0.0919/0.0919 |
| numbered | AZ6 | scanner | 3 | True | 2.0 | 0 | 0.202 | 0.195–0.195 | 0.135/0.135 | 0.001/0.001 |
| numbered | AZ7 | scanner | 3 | True | 1.3 | 0 | 0.239 | 0.151–0.151 | 0.121/0.121 | 0.004/0.004 |
| numbered | AZ8 | scanner | 3 | True | 0.7 | 0 | 0.194 | 0.188–0.188 | 0.16/0.16 | 0.013/0.013 |
| original | AZ1 | baseline | 3 | False | 3.3 | 0 | 0.138 | 0.115–0.115 | 0.125/0.125 | 0.7383/0.7383 |
| original | AZ1 | hybrid | 3 | False | 0.7 | 0 | 0.11 | 0.106–0.106 | 0.145/0.145 | 1.0/1.0 |
| original | AZ1 | scanner | 3 | False | 2.0 | 0 | 0.102 | 0.096–0.096 | 0.162/0.162 | 1.0/1.0 |
| original | AZ2 | scanner | 3 | False | 12.0 | 0 | 0.127 | 0.091–0.091 | 0.13/0.13 | 0.97/0.97 |
| original | AZ3 | scanner | 3 | False | 5.7 | 0 | 0.233 | 0.186–0.186 | 0.125/0.125 | 0.001/0.001 |
| original | AZ4 | scanner | 3 | False | 6.7 | 0 | 0.105 | 0.054–0.054 | 0.092/0.092 | 0.997/0.997 |
| original | AZ6 | scanner | 3 | False | 5.0 | 0 | 0.047 | 0.052–0.052 | 0.155/0.155 | 1.0/1.0 |
| original | AZ7 | scanner | 3 | True | 1.3 | 0 | 0.167 | 0.151–0.151 | 0.142/0.142 | 0.2767/0.2767 |
| original | AZ8 | scanner | 3 | True | 0.7 | 0 | 0.152 | 0.145–0.145 | 0.165/0.165 | 0.952/0.952 |

## 3. RQ4 — random ordering of the candidate list

*The model receives only the file paths of the candidate list (StaLLM_tasks.py, `candidate_text = "- path"` lines), never file contents. Expected Precision@10 = gold files in the list / list length (hypergeometric mean of the top ten). Expected MRR = exact expectation of the reciprocal rank of the first gold file within the top ten (0 if none), uniform random permutation.*

| System | Condition | candidates | gold in list (mean) | E[P@10] | E[MRR] |
|---|---|---|---|---|---|
| argouml | alphabetical | 600 | 36.6 | 0.061 | 0.134 |
| argouml | keyword_sorted | 600 | 51.6 | 0.086 | 0.174 |
| argouml | bm25_sorted | 600 | 77.9 | 0.130 | 0.263 |
| argouml | bm25_masked | 600 | 54.2 | 0.090 | 0.193 |
| eshop | alphabetical | 523 | 79.8 | 0.152 | 0.294 |
| eshop | keyword_sorted | 523 | 79.8 | 0.152 | 0.294 |
| eshop | bm25_sorted | 523 | 79.8 | 0.152 | 0.294 |
| eshop | bm25_masked | 523 | 79.8 | 0.152 | 0.294 |

Per feature: reviewer_rq4_random_ordering.csv.

## 4. MLCQ — fragments and reviewers of the 200 rows used

*Presence = severity other than `none`. Agreement over (fragment, smell) pairs reviewed by two or more reviewers in the full MLCQ export; Krippendorff's α (nominal).*

- rows_used: 200
- rows_matched: 200
- distinct_fragments: 101
- distinct_fragment_smell_pairs: 200
- reviewers: {'10': 4, '13': 4, '5': 8, '6': 86, '7': 90, '8': 8}
- pairs_with_2plus_reviewers: 62
- pairs_with_disagreement: 28
- pct_agreement_used: 0.548
- alpha_used: 0.423
- rows_label_differs_from_majority: 10
- alpha_full_dataset: 0.379
- pct_agreement_full_dataset: 0.55

## 5. Checks

- Table 2 time: analyze_paper_stats.analyze_rq12 writes `time_s = statistics.median(elapsed_s of the runs)`, i.e. the median over runs of the wall-clock time of a run.
- Run counts, sensitivity_chance.csv vs rq12_rerun.csv: identical. A difference means sensitivity_analyses.py was last run before all runs were stored and must be rerun.
- RQ5 held-out (stats/rq5_heldout.csv): `generated_only_correct` counts samples where only the generated prompt is correct (mcnemar_exact(generated, manual)).
- RQ5 MLCQ, generated vs binary_oracle_baseline (run 1): 9 vs 10 discordant, p=1.0
- RQ5 MLCQ, generated vs conservative_evidence (run 1): 10 vs 6 discordant, p=0.4545
- RQ5 MLCQ, generated vs definition_driven (run 1): 20 vs 4 discordant, p=0.0015

