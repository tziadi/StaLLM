# StaLLM manuscript — presentation changelog (2026-10-10)

Layout kept: Elsevier `elsarticle` 5p, Times, as in the Microservices reference
paper (journal to be chosen later). Before/after page images:
`docs/visual_comparison/*.png`. Audit: `STALLM_VISUAL_AUDIT.md`.

| # | Change | Files | Why |
|---|---|---|---|
| 1 | Times for text with every engine: `fontspec` + TeX Gyre Termes under XeTeX/LuaTeX (Tectonic), class option `times` under pdfLaTeX; Helvetica (TeX Gyre Heros / `phv`) for sans text in figures | `00main.tex` | The body was set in Latin Modern because Tectonic (XeTeX) ignores `times`; the reference is Times throughout |
| 2 | New architecture figure in TikZ (inputs → four numbered pipeline steps → controlled result; built-in controls band; reproducibility record band), replacing an outdated raster image that still showed Snyk and no controls | `figures/architecture.tex`, `03_stallm_framework.tex` | Vector, editable, legible at page width, consistent with the paper's current content |
| 3 | New four-panel figure of the controls (random placement, oracle substitution, baselines and masking, held-out selection): principle only, no results | `figures/controls.tex`, `03_stallm_framework.tex` | The controls are the central contribution and were only described in text and a dense table |
| 4 | Shared figure palette (`stInk`, `stBlue`, `stOrange`, greys) defined once in the preamble | `00main.tex` | One visual vocabulary across figures |
| 5 | Raster localization-workflow figure removed; its content is covered by the architecture figure | `02_background_motivation.tex` | Low-resolution, tiny labels in one column |
| 6 | Listing of oracle records (broken across columns) replaced by an "Example record" column in the oracle-family table | `03_stallm_framework.tex` | Removes a two-column code block; one place for oracle information |
| 7 | Corpus table: redundant "Codebase" column removed; moved to a single-column table with a descriptive caption | `03_stallm_framework.tex` | Less width, no repetition |
| 8 | Prompt example: grey filled box replaced by a thin frame with a sans title and monospace text | `02_background_motivation.tex` | The reference uses no tinted boxes |
| 9 | Research questions: `\rqstatement` (bold label, italic question) in Section 2 and at the head of each results subsection; one `\rqanswer` (thin blue left rule) per RQ, replacing heterogeneous "Key findings" lists | `00main.tex`, `02_…`, `09_…` | Same Question → Design → Results → Interpretation → Answer progression for all RQs |
| 10 | RQ1/RQ2 merged into one table with both protocols (original, numbered), chance baseline, p, retry count and time | `09_results_discussion.tex` | One table instead of two; new numbered-protocol results (commit 7610f46) |
| 11 | RQ3 table: CIs in their own column at the same size with leading zeros; baselines and ρ in separate rule groups | `09_…` | CIs were in 5 pt with ".21" notation |
| 12 | RQ4 table: MRR and P@10 in separate column groups instead of "MRR / P@10" pairs | `09_…` | Values can be scanned per metric |
| 13 | RQ5 table: no `\resizebox`; grouped Manual / Generated; held-out CI in its own column | `09_…` | No shrunken fonts |
| 14 | All table captions above tables, figure captions below, 2–4 self-contained sentences | all sections | Consistency with the reference |
| 15 | RQ4 plot regenerated in Times, full text width (`figure*`), caption explains the encodings | `generate_paper_figures.py`, `09_…` | Was DejaVu at half size |
| 16 | Links coloured (`stBlue`) instead of boxed | `00main.tex` | Cleaner pages |

Values check: all numbers in the RQ3, RQ4 and RQ5 tables were compared
automatically before and after the redesign; they are identical. RQ1/RQ2 values
changed for scientific reasons (oracle line-offset correction and the
numbered-protocol runs), documented in `IST_SCIENTIFIC_AUDIT.md`.

Not changed: the benchmark implementation (only the figure script's fonts),
the five RQs, references and claims other than the RQ1/RQ2 update.
