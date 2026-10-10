# StaLLM manuscript — visual review after redesign (2026-10-10)

All 16 pages of `Publications/StaLLM___Journal/00main.pdf` were rendered and
inspected (contact sheets and zoomed crops of every figure and table).

## Assessment

- **Typography:** Times body, Helvetica in figures, Latin Modern Mono for code,
  txfonts math, matching the reference. No overfull box in the body.
- **Figures:** three figures, all vector. Architecture (p. 4) and controls (p. 6)
  are TikZ with one palette; the RQ4 plot (p. 11) uses Times. Labels are 6–7 pt at
  print size.
- **Tables:** booktabs, no vertical rules, captions above, consistent 3-decimal
  values and `[lo, hi]` intervals, grouped headers.
- **RQs:** each results subsection opens with the question and closes with an
  "Answer to RQn" paragraph.
- **Page composition:** no float-only page, no detached caption; the reference
  list ends on p. 16.

Compared with the reference paper, the manuscript now has the same typographic
base and a comparable process figure. The reference still has more figures per
page of results (feature models and architectures); StaLLM's results are
numerical, so tables carry them.

## Unresolved

1. **Page footer:** "Preprint submitted to Information and Software
   Technology" (from `\journal`) overflows by 1.7 pt and names a journal that is no
   longer decided. Change `\journal{...}` when the venue is chosen.
2. **Page 7** stacks two full-width tables (controls, study overview); they
   partly overlap in content. Merging them would save about a third of a page.
3. **Control panel text** in Fig. 2 is 5 pt in the subtitles; acceptable on
   screen, small in print. Enlarging requires taller panels.
4. **Local-model results** (qwen2.5-coder, deepseek-coder) are still missing in
   RQ2–RQ4 (`TODO` comments); the RQ4 figure already shows partial qwen data, as
   its caption says.
5. **Overleaf check:** the preamble supports pdfLaTeX, but only Tectonic was
   available here; compile once on Overleaf before submission.
6. Unused legacy sources (`04_`–`07_*.tex`, `images/stallm_framework_workflow.pdf`,
   `images/Localization_Workflow.png`, `images/mainInterface.png`) remain in the
   folder.
