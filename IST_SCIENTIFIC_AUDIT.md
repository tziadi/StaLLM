# IST scientific audit — StaLLM journal extension

Audit date: 2026-10-10. Scope: repository (`StaLLM_core.py`, `StaLLM_tasks.py`,
`StaLLM_benchmarks.py`, `run_paper_experiments.py`, `analyze_paper_stats.py`,
`generate_paper_figures.py`), results under `output/paper_results/`, and the
manuscript `Publications/StaLLM___Journal/` (not versioned in git).

Every item was checked against code or artifacts; evidence is given as file:line
or as a command result. "Fixed" items were corrected earlier in this session and
are listed so that the record is complete.

Priority: **C** critical (can change a headline conclusion), **M** major
(reviewer would require a change), **m** minor.

## Summary table

| ID | Priority | RQ | Issue | Status | New LLM runs? |
|---|---|---|---|---|---|
| C1 | C | RQ1–2 | Code shown to the model without line numbers, answers required in line numbers | Verified; experiment running | Yes (approved, ~1 h hosted) |
| C2 | C | RQ1–2 | Oracle matched to files by file name (ENASE pipeline) | Fixed (cf2b415); tests + validation report missing | No |
| M1 | M | RQ1–2 | Empty answers are retried with "return AT LEAST 3 findings", T=0.25 | Verified | Optional |
| M2 | M | RQ1–2 | "Positive ratio 1.00" vs 13/20 positives | Verified: reporting issue, not a defect | No |
| M3 | M | RQ1–2 | Wording of chance-baseline conclusions implies equivalence / incapacity | Verified in manuscript | No |
| M4 | M | RQ1–2 | Chance baseline sensitivity untested (seed, span length, sample composition) | Verified missing | No |
| M5 | M | RQ3 | Metric oracle is a regex approximation; MA lacks Designite's cohesion criterion; no threshold sensitivity | Verified | No |
| M6 | M | RQ4 | Candidate recall before the LLM not reported; ranking not conditioned on visibility | Verified | No |
| M7 | M | RQ4 | Wilcoxon tests not corrected for multiple comparisons | Verified | No |
| M8 | M | RQ5 | One search, one non-random held-out split; conclusion worded generally | Verified | No (offline simulation) |
| M9 | M | all | No automated tests; no end-to-end demo from configuration to report | Verified | No |
| m1 | m | RQ1–2 | GT loader forward-fills missing line numbers | Verified (no effect on C# corpus) | No |
| m2 | m | RQ4 | Masking removes the feature's own name only | Verified, behaves as described | No |
| m3 | m | RQ5 | Prompt-generation call has no fixed temperature | Verified (outputs saved) | No |
| m4 | m | RQ3 | RQ3 table uses gpt-4o run 1 although three runs exist | Verified | No |
| m5 | m | stats | `mcnemar_exact` docstring states the wrong return order | Verified (no effect on results) | No |
| U1 | — | RQ3 | Provenance of each heuristic threshold in Designite | **Unverified** (needs Designite documentation) | No |

## Detailed findings

### C1 — Line numbers are not shown to the model (RQ1, RQ2)
- **Where:** `StaLLM_core.py` `analyze_with_llm`: `code_snippet = code[:20000]` is inserted
  verbatim; the output rules require `line` or `startLine`/`endLine`.
- **Evidence:** the model must count lines to report a location. gpt-4o's Scanner
  and Hybrid prompts score *below* random placement in every draw (p = 1.00,
  `stats/rq12_rerun.csv`), which is more consistent with systematic line offsets than
  with an absence of signal. The reasoning model o3 is the only one above chance.
- **Impact:** the headline RQ1–RQ2 conclusion ("two of three models locate findings no
  better than chance") may be an artefact of the protocol.
- **Correction:** optional `number_lines=True` (commit c54ef88, recorded in
  `metrics.number_lines`); original protocol unchanged. Re-run AZ1–AZ3 numbered
  (`rq2_numbered/`), compare both protocols against their own chance baselines.
- **Experiments:** yes, approved; running. First run: gpt-4o Baseline F1 0.208
  numbered vs 0.161 unnumbered (one run; not yet interpretable).
- **Effort:** small (code done); ~1 h of hosted calls.

### C2 — File-name matching of the SonarQube oracle (RQ1, RQ2; ENASE E1/E2)
- **Where:** `StaLLM_core._build_universe` joined oracle and archive on basename.
- **Evidence:** on the RQ1–RQ2 universe, 157 oracle spans were counted, 72 belong to
  the evaluated files; 85 spans in 5 files came from homonymous files of other
  services (`Program.cs` 65→10, `Startup.cs` 27→3, `CatalogService.cs` 4→2,
  `HomeController.cs` 3→0, `CatalogController.cs` 2→1). ArgoUML and Magento archives
  contain 1,922 and 10,440 duplicate file names.
- **Status:** fixed by path alignment (`_align_gt_to_zip`, cf2b415); old results kept in
  `rq2_basename_matching/`. Missing: regression tests and a validation report.
- **Experiments:** none beyond the re-runs already done.

### M1 — Forced retry on empty answers (RQ1, RQ2)
- **Where:** `analyze_with_llm` retry branch: "Return AT LEAST 3 findings if plausible",
  `temperature=0.25`.
- **Evidence:** share of files answered by the retry (`metrics.file_sources`):
  gpt-4.1 12/20 in all three runs; o3 5–6/20; gpt-4o 0–4/20.
- **Impact:** gpt-4.1's predictions are mostly produced under a "find something"
  instruction at non-zero temperature. Model comparison in RQ2 is confounded with the
  retry rate. The chance baselines remain fair (they reuse the same spans).
- **Correction:** disclose the retry rate per model in RQ2; record per-file source in
  future runs. A no-retry configuration would require new runs (not proposed now).

### M2 — Sample composition (RQ1, RQ2)
- **Where:** `_build_universe`: `want_pos = min(len(pos), round(top_k * pos_ratio))`,
  positives sorted by number of oracle spans, negatives sampled with `random_state=42`.
- **Evidence:** the Catalog service has 13 C# files with findings; the sampler takes
  all 13 and fills the budget with 7 of 45 files without findings. "Positive ratio" is
  the requested value; the achieved ratio is 0.65.
- **Impact:** wording only; the random-file baseline does depend on the 7 negatives.
- **Correction:** describe requested vs achieved composition; add a positives-only
  sensitivity analysis (offline).

### M3 — Calibration of the chance-baseline conclusions
- **Where:** sections 5.1–5.2, abstract, conclusion ("indistinguishable from random
  placement", "places findings no better than chance").
- **Impact:** failure to exceed one baseline under one protocol is not evidence of
  equivalence nor of an absence of localization ability.
- **Correction:** rephrase as "does not exceed random placement of its own spans under
  this protocol"; report the numbered-protocol result (C1).

### M4 — Sensitivity of the chance baseline
- **Evidence:** implementation preserves count, length and file
  (`analyze_paper_stats.random_placement_f1`); offline re-scoring reproduces pipeline
  F1 exactly (0.1613 = 0.1613). Presets Lenient/Balanced/Strict already checked.
  Untested: seed, length-free placement (1-line spans), positives-only universe.
- **Correction:** offline sensitivity table.

### M5 — The metric oracle of RQ3
- **Where:** `run_paper_experiments.smell_heuristic`: regex approximations of
  signatures, fields and decision points; thresholds LPL ≥ 5 params, complex method
  ≥ 8 decision points, long method > 100 non-blank lines, MA ≥ 7 fields and ≥ 7
  methods, god class ≥ 20 methods or > 500 lines, data class ≥ 3 fields and ≥ 70 %
  accessors, feature envy ≥ 5 external calls and > 2× `this.` accesses.
- **Evidence:** MA has no cohesion criterion; on the held-out split MA gives 11 false
  positives and finds 1 of 21 positives (`stats/rq3_heuristic_by_smell.csv`).
- **Impact:** the second oracle is author-defined, not a validated detector; the
  substitution result shows sensitivity to the oracle, not which oracle is correct.
- **Correction:** document definitions and limitations; threshold-sensitivity analysis
  of the oracle-substitution ranking (offline).

### M6 — Retrieval vs ranking in RQ4
- **Where:** `_run_rq4_condition` truncates to `CANDIDATE_BUDGET = 600` and records
  `gold_visible`.
- **Evidence:** masked BM25 on DeploymentDiagram shows 6 of 32 gold files to the model.
  Candidate recall is never reported.
- **Correction:** report candidate Recall@600 per ordering (deterministic, offline) and
  P@10 relative to the number of visible gold files. Comparisons LLM vs "same list
  without LLM" already use identical candidate sets.

### M7 — Multiple comparisons in RQ4
- **Evidence:** six LLM-vs-matched-baseline Wilcoxon tests per system are reported
  without correction (`stats/rq4_wilcoxon.csv`).
- **Correction:** Holm adjustment within the family; report adjusted p.

### M8 — Scope of the RQ5 conclusion
- **Evidence:** selection and held-out splits are disjoint by DACOS code id (0 overlap);
  selection used only the 200 selection samples; the meta-prompt only sees selection
  F1. The held-out split is the next 200 samples in file order, not a random or
  stratified draw; one search was run.
- **Correction:** offline selection-bias simulation (bootstrap the selection split,
  record which prompt wins and its held-out score); narrow the conclusion to the
  evaluated configuration.

### M9 — Engineering evidence
- **Evidence:** the only test file is `test_ollama_tokens.py`; no script turns a
  configuration into an evaluation report without the UI.
- **Correction:** unit tests for oracle alignment, chance baselines and statistics;
  an end-to-end demo that replays recorded predictions (no credentials needed).

### Minor items
- **m1** `load_ground_truth_spans` forward-fills missing start lines, assigning
  file-level findings to the previous row's line. 6 rows with line 0 in the eShop CSV,
  all `.js`/`.ts`, so no effect on RQ1–RQ2; fix by dropping rows without a line.
- **m2** Masking removes the feature's own name (and "critic" for Cognitive); other
  features' names remain in descriptions (e.g. "state diagrams" in ActivityDiagram);
  candidate paths are unchanged by design. Document.
- **m3** `_generate_prompt_candidates` calls the model without a temperature; the
  generated prompts are saved in `rq5/generated_prompts.json`, so results are
  reproducible from artifacts but the generation is not.
- **m4** Table RQ3 reports gpt-4o run 1; run-to-run spread is reported separately.
- **m5** Docstring of `mcnemar_exact` says `(b_only, a_only, p)`; the code returns
  `(a_only, b_only, p)` and callers use it that way.

### Unverified
- **U1** The manuscript calls the thresholds "Designite-inspired". Designite's own
  definitions (e.g. LCOM for multifaceted abstraction) could not be checked offline;
  the paper should state the implemented rules and avoid attributing them to Designite.

## ENASE vs journal

| Aspect | ENASE 2026 | Journal |
|---|---|---|
| Tasks | Span-level findings (E1 prompts, E2 models) on eShopOnContainers; Snyk example | + human smell labels (DACOS, MLCQ), feature location (ArgoUML, eShop), prompt search |
| Oracle matching | File name | Path (C2) |
| Controls | None | Chance placement, oracle substitution, non-LLM baselines + masking, held-out selection |
| Statistics | Single runs | Repeated runs, bootstrap CIs, McNemar/Holm, Wilcoxon |
| Conclusions reversed | — | Span-level prompt ranking, keyword pre-sorting gain, prompt-search gain |

## Prioritised plan

1. C1: finish numbered runs; compare protocols (running).
2. C2/M9: oracle regression tests and validation report.
3. M2–M4: offline sensitivity of the chance baseline; manuscript wording.
4. M5: threshold sensitivity of the oracle substitution.
5. M6–M7: candidate recall, conditional P@10, Holm.
6. M8: offline selection-bias simulation for RQ5.
7. M1: disclose retry rates.
8. m1, m5: small code fixes. M9: demo.
9. Manuscript revision, then revision report, remaining issues, reviewer simulation.
