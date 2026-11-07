# Claims manifest — IST paper

Where each table, figure, and group of numbers of the paper comes from. Every
command runs offline from the stored outputs (no model call), from the
repository root.

Two result roots exist:

- `output/paper_results/`: RQ1, RQ2, RQ4, and the MLCQ runs of RQ3. Its DACOS
  results use the legacy labels of the preliminary version (the `smells`
  column of the DACOS sample table) and are **not** reported in the paper.
- `output/paper_results_annot/`: RQ3 and RQ5 on DACOS labeled by the
  annotators' majority (`--dacos-labels annotations`, the default of
  `run_paper_experiments.py`). The samples are listed in
  `output/paper_results_annot/stallm_annotated_splits.json`
  (`python run_paper_experiments.py --rq splits` regenerates it).

## Regenerate everything

```bash
python analyze_paper_stats.py
python analyze_paper_stats.py --root output/paper_results_annot
python sensitivity_analyses.py
python sensitivity_analyses.py --root output/paper_results_annot
python line_offset_analysis.py
python reviewer_analyses.py
python reviewer_analyses.py --root output/paper_results_annot
python structured_baselines.py
python span_analyses.py
python smell_analyses.py
python rq4_reranker_analyses.py
python pmd_oracle.py --pmd <path to PMD 7>/bin/pmd          # and --lpl-minimum 5
python generate_paper_figures.py
```

## Tables and figures

| Paper element | File(s) | Command |
|---|---|---|
| Fig. 1 (two activities, code excerpts) | `Publications/.../figures/activities_code.tex` | `python Publications/StaLLM___Journal/figures/make_activities_code.py .` |
| Fig. 5 (exact start lines) | `output/paper_results/stats/line_offsets*.csv`, `images/fig_rq12_exact_lines.pdf` | `line_offset_analysis.py`, `generate_paper_figures.py` |
| Fig. 6 (RQ4 P@10) | `output/paper_results/stats/rq4_conditions*.csv` | `analyze_paper_stats.py`, `generate_paper_figures.py` |
| Table 2 (RQ1–RQ2) | `output/paper_results/stats/rq12_rerun.csv` | `analyze_paper_stats.py` |
| Table 3 (RQ3) | `output/paper_results_annot/summary_rq3.csv`, `.../stats/rq3_oracle_swap.csv`, `.../stats/report.md` (MLCQ rows: `output/paper_results/rq3/`) | `analyze_paper_stats.py --root output/paper_results_annot` |
| Table 3, annotator row | `output/paper_results_annot/stats/smell_dacos_agreement.csv` | `smell_analyses.py` |
| Table 4 (RQ4) | `output/paper_results/stats/rq4_conditions.csv`, `rq4_conditions_eshop.csv`, `summary_rq4_baselines.csv` | `analyze_paper_stats.py` |
| Table 5 (RQ5) | `output/paper_results_annot/stats/rq5_heldout.csv`, `output/paper_results_annot/summary_rq5.csv` | `analyze_paper_stats.py --root output/paper_results_annot` |

## Numeric claims by research question

| Claim group | File(s) | Command |
|---|---|---|
| RQ1–RQ2 random placement, p values, retry counts, median time | `output/paper_results/stats/rq12_rerun.csv` | `analyze_paper_stats.py` |
| RQ1–RQ2 Holm adjustment, 10,000-draw borderline tests | `output/paper_results/stats/span_holm.csv`, `span_borderline.csv` | `span_analyses.py` |
| RQ1–RQ2 without the forced retry | `output/paper_results/stats/reviewer_rq12_no_retry.csv` | `reviewer_analyses.py` |
| RQ1–RQ2 code-line and method-body nulls, balanced/strict presets | `output/paper_results/stats/structured_baselines.csv`, `matching_sensitivity.csv` | `structured_baselines.py` |
| RQ1–RQ2 category audit of lenient matches | `output/paper_results/stats/lenient_category_pairs.csv` | `structured_baselines.py` |
| RQ1–RQ2 category-aware matching | `output/paper_results/stats/span_category_matching.csv` | `span_analyses.py` |
| RQ1–RQ2 seeds, 13-file universe, presets (500 draws) | `output/paper_results/stats/sensitivity_chance.csv` | `sensitivity_analyses.py` |
| RQ1–RQ2 line offsets | `output/paper_results/stats/line_offsets.md` | `line_offset_analysis.py` |
| RQ3 κ, F1, CIs, McNemar, run stability | `output/paper_results_annot/stats/report.md`, `rq3_mcnemar.csv`, `rq3_run_stability.csv` | `analyze_paper_stats.py --root output/paper_results_annot` |
| RQ3 oracle substitution ρ and best configurations | `output/paper_results_annot/stats/rq3_oracle_swap.csv` | same |
| RQ3 bootstrap and rerun stability of the ranking | `output/paper_results_annot/stats/reviewer_rq3_bootstrap.csv`, `reviewer_rq3_rerun_rankings.csv` | `reviewer_analyses.py --root output/paper_results_annot` |
| RQ3 threshold sensitivity | `output/paper_results_annot/stats/sensitivity_rq3_thresholds.csv` | `sensitivity_analyses.py --root output/paper_results_annot` |
| RQ3 PMD oracle | `output/paper_results_annot/stats/pmd_oracle*.md`, `pmd_oracle_summary*.csv`, `pmd_labels_*.json` | `pmd_oracle.py --pmd …` |
| RQ3 Δκ (metric − human) per configuration, paired Δκ between configurations | `output/paper_results_annot/stats/smell_oracle_shift.csv`, `smell_pairwise_kappa.csv` | `smell_analyses.py` |
| RQ3 non-LLM baselines (always present, heuristic; selection and held-out) | `output/paper_results_annot/summary_rq3_baselines.csv`, `stats/smell_heldout_heuristic.csv` | `run_paper_experiments.py --rq 3b`, `smell_analyses.py` |
| RQ3 MLCQ reviewers (α, disagreements) and majority-label sensitivity | `output/paper_results/stats/reviewer_mlcq_agreement.csv`, `output/paper_results_annot/stats/smell_mlcq_majority.csv` | `reviewer_analyses.py`, `smell_analyses.py` |
| RQ3 DACOS ties excluded, eligible population | `docs/dacos_annotated_sample.md` | `run_paper_experiments.py --rq splits` |
| RQ4 MRR/P@10 per list, Wilcoxon, Holm | `output/paper_results/stats/rq4_conditions*.csv`, `rq4_wilcoxon*.csv` | `analyze_paper_stats.py` |
| RQ4 candidate recall | `output/paper_results/stats/sensitivity_rq4_retrieval.csv` | `sensitivity_analyses.py` |
| RQ4 random ordering P@10/MRR | `output/paper_results/stats/reviewer_rq4_random_ordering.csv` | `reviewer_analyses.py` |
| RQ4 non-LLM re-rankers on the BM25 list, duplicated files | `output/paper_results/stats/rq4_rerankers.csv`, `rq4_duplicates.csv`, `rq4_reranker_analyses.md` | `rq4_reranker_analyses.py` |
| RQ5 selection, held-out, MLCQ | `output/paper_results_annot/summary_rq5.csv`, `stats/rq5_heldout.csv` | `analyze_paper_stats.py --root output/paper_results_annot` |
| RQ5 paired Δκ, repeated splits by κ with chance references, run-to-run spread | `output/paper_results_annot/stats/smell_pairwise_kappa.csv`, `smell_rq5_splits.csv` | `smell_analyses.py` |
